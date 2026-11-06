#!/usr/bin/env python3
"""Label-noise robustness experiment (paper Table 4) for RigidGS.

Given a list of scenes, builds the full label-corruption matrix (schemes x
rates x seeds, plus one uncorrupted baseline per scene) and runs every job
through src/train.py with --object_ids_override, spreading jobs across a
pool of GPUs as a continuous work queue. Inpainting is disabled
(--inpaint_at_iter 0): this experiment only measures segmentation.

Each scene must have been preprocessed first (src/preprocessing.py, output
under output/<dataset>/<scene>/). Everything this script produces
(corrupted label JSONs, run outputs, logs, manifest) lives under
--experiment_root. Run from the repository root.

Usage:
  # 1. See the planned job matrix and resolved paths without running anything:
  python experiments/label_noise/run_label_noise_matrix.py \
      --scenes llff:fern,llff:trex,mipnerf360:kitchen,mipnerf360:garden --gpus 0 \
      --mask_2d_bg_margin_map "llff:fern=0.5,llff:trex=0.5,mipnerf360:kitchen=5,mipnerf360:garden=5" \
      --dry_run

  # 2. Launch for real (same arguments, without --dry_run), then evaluate with
  #    experiments/label_noise/evaluate.sh.
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

# Run from anywhere: subprocesses run in the repo root, and the helper
# scripts are addressed by their location next to this file.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = "src/train.py"

# Hardcoded in the trainer's densify_and_prune call site (not a CLI arg).
# Recorded here so suppression_metric.py and this launcher agree on the
# same fixed cutoff, chosen before running (per experiment guardrails).
MIN_OPACITY_PRUNE_THRESHOLD = 0.005

# dataset -> images/ dir template, derived from what's actually on disk
# (see data/*/*/images). Extend this dict if you add a new dataset.
def resolve_scene(dataset, scene):
    """Paths follow the release layout: data/<dataset>/<scene>/images in,
    output/<dataset>/<scene>/ for preprocessing.py's results."""
    out = f"output/{dataset}/{scene}"
    source_path = f"{out}/point_cloud/pcd_data"
    scene_name = f"{dataset}/{scene}"
    # Refined masks when the scene was preprocessed with mask refinement,
    # otherwise the raw SAM2 masks -- the same masks the trainer is given.
    masks_dir = f"{out}/preprocessing/refined/refined_masks"
    if not os.path.isdir(os.path.join(REPO_ROOT, masks_dir)):
        masks_dir = f"{out}/preprocessing/segmentation/masks"
    images_dir = f"data/{dataset}/{scene}/images"

    missing = []
    sparse0 = os.path.join(source_path, "sparse", "0")
    if not os.path.isdir(os.path.join(REPO_ROOT, sparse0)):
        missing.append(f"source_path/sparse/0 not found: {sparse0} (run preprocessing.py first)")
    if not os.path.isdir(os.path.join(REPO_ROOT, masks_dir)):
        missing.append(f"masks_dir not found: {masks_dir}")
    if not os.path.isdir(os.path.join(REPO_ROOT, images_dir)):
        missing.append(f"images dir not found: {images_dir}")

    return {
        "dataset": dataset, "scene": scene,
        "source_path": source_path, "scene_name": scene_name,
        "masks_dir": masks_dir, "images_dir": images_dir,
        "missing": missing,
    }


def run_subprocess(cmd, cwd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def export_init_labels(scene_cfg, experiment_root):
    tag = f"{scene_cfg['dataset']}_{scene_cfg['scene']}"
    out_json = os.path.join(experiment_root, "init_labels", f"{tag}_init_object_metadata.json")
    if os.path.exists(out_json):
        return out_json, None
    cmd = [sys.executable, os.path.join(HERE, "export_init_labels.py"),
           "--source_path", scene_cfg["source_path"],
           "--output_json", out_json]
    log_path = os.path.join(experiment_root, "logs", tag, "export_init_labels.log")
    rc = run_subprocess(cmd, REPO_ROOT, log_path)
    if rc != 0:
        return None, f"export_init_labels.py failed (rc={rc}), see {log_path}"
    with open(out_json) as f:
        arr = json.load(f)["object_id_array"]
    n_fg = sum(1 for x in arr if x != 0)
    if n_fg == 0:
        return None, f"exported init labels have ZERO foreground points — broken init, skipping scene"
    return out_json, None


def make_corrupted_labels(init_json, scheme, rate, seed, experiment_root, tag):
    out_json = os.path.join(
        experiment_root, "corrupted_labels", tag,
        f"{scheme}_r{int(round(rate*100)):03d}_seed{seed}.json")
    if os.path.exists(out_json):
        return out_json
    cmd = [sys.executable, os.path.join(HERE, "flip_labels.py"),
           "--input_json", init_json, "--output_json", out_json,
           "--scheme", scheme, "--rate", str(rate), "--seed", str(seed)]
    log_path = os.path.join(experiment_root, "logs", tag,
                             f"flip_{scheme}_r{int(round(rate*100)):03d}_seed{seed}.log")
    rc = run_subprocess(cmd, REPO_ROOT, log_path)
    if rc != 0:
        raise RuntimeError(f"flip_labels.py failed (rc={rc}), see {log_path}")
    return out_json


def build_jobs(scenes, schemes, rates, seeds, experiment_root, bg_margin_map,
                include_baseline=True):
    jobs = []
    skipped_scenes = []
    for dataset, scene in scenes:
        tag = f"{dataset}_{scene}"

        if (dataset, scene) not in bg_margin_map:
            skipped_scenes.append((tag, [
                f"no --mask_2d_bg_margin_map entry for {dataset}:{scene} — "
                f"refusing to silently default this (that's the exact mistake "
                f"that caused the last rerun). Add '{dataset}:{scene}=<value>' "
                f"to --mask_2d_bg_margin_map."]))
            continue
        bg_margin = bg_margin_map[(dataset, scene)]

        scene_cfg = resolve_scene(dataset, scene)
        if scene_cfg["missing"]:
            skipped_scenes.append((tag, scene_cfg["missing"]))
            continue

        init_json, err = export_init_labels(scene_cfg, experiment_root)
        if err:
            skipped_scenes.append((tag, [err]))
            continue

        run_root = os.path.join(experiment_root, "runs", tag)

        if include_baseline:
            # Baseline: rate=0, NO override arg at all (exercises the untouched
            # init path, not just an all-zero-flip corrupted file).
            jobs.append({
                "dataset": dataset, "scene": scene, "scheme": "baseline",
                "rate": 0.0, "seed": 0, "scene_cfg": scene_cfg,
                "override_path": None, "mask_2d_bg_margin": bg_margin,
                "model_path": os.path.join(run_root, "baseline"),
            })

        for scheme in schemes:
            for rate in rates:
                for seed in seeds:
                    override_path = make_corrupted_labels(
                        init_json, scheme, rate, seed, experiment_root, tag)
                    run_name = f"{scheme}_r{int(round(rate*100)):03d}_seed{seed}"
                    jobs.append({
                        "dataset": dataset, "scene": scene, "scheme": scheme,
                        "rate": rate, "seed": seed, "scene_cfg": scene_cfg,
                        "override_path": override_path, "mask_2d_bg_margin": bg_margin,
                        "model_path": os.path.join(run_root, run_name),
                    })
    return jobs, skipped_scenes


def parse_bg_margin_map(spec):
    """Parse 'dataset:scene=value,dataset:scene=value,...' into a dict."""
    mapping = {}
    if not spec:
        return mapping
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, value = entry.split("=")
        dataset, scene = key.split(":")
        mapping[(dataset.strip(), scene.strip())] = float(value)
    return mapping


def job_cmd(job, args, port):
    scene_cfg = job["scene_cfg"]
    cmd = [
        sys.executable, TRAINER,
        "--source_path", scene_cfg["source_path"],
        "--scene_name", scene_cfg["scene_name"],
        "--model_path", job["model_path"],
        "--_images", os.path.join(REPO_ROOT, scene_cfg["images_dir"]),
        "--masks_dir", scene_cfg["masks_dir"],
        "--eval",
        "--iterations", str(args.iterations),
        "--disable_viewer",
        "--inpaint_at_iter", "0",
        "--mask_2d_bg_margin", str(job["mask_2d_bg_margin"]),
        "--port", str(port),
    ]
    if args.logit_freeze_iter is not None:
        cmd += ["--logit_freeze_iter", str(args.logit_freeze_iter)]
    if job["override_path"]:
        cmd += ["--object_ids_override", job["override_path"]]
    if args.extra_trainer_args:
        cmd += args.extra_trainer_args.split()
    return cmd


def worker(gpu_id, job_q, args, manifest_path, manifest_lock, port_counter, port_lock):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        tag = f"{job['dataset']}_{job['scene']}"
        run_name = os.path.basename(job["model_path"])
        log_path = os.path.join(args.experiment_root, "logs", tag, f"{run_name}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with port_lock:
            port_counter[0] += 1
            port = port_counter[0]

        cmd = job_cmd(job, args, port)
        start = datetime.now().isoformat()
        print(f"[GPU {gpu_id}] START  {tag}/{run_name}  -> {job['model_path']}")

        with open(log_path, "w") as logf:
            logf.write("CMD: " + " ".join(cmd) + "\n\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                   stdout=logf, stderr=subprocess.STDOUT)
        end = datetime.now().isoformat()
        status = "ok" if proc.returncode == 0 else "FAILED"
        print(f"[GPU {gpu_id}] {status}  {tag}/{run_name}  (rc={proc.returncode})")

        row = {
            "dataset": job["dataset"], "scene": job["scene"], "scheme": job["scheme"],
            "rate": job["rate"], "seed": job["seed"], "gpu": gpu_id,
            "model_path": job["model_path"], "override_path": job["override_path"] or "",
            "status": status, "returncode": proc.returncode,
            "start_time": start, "end_time": end, "log_path": log_path,
        }
        with manifest_lock:
            write_header = not os.path.exists(manifest_path)
            with open(manifest_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        job_q.task_done()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True,
                   help="Comma-separated dataset:scene pairs, e.g. llff:fern,llff:horns,mipnerf360:kitchen")
    p.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1,2")
    p.add_argument("--schemes", default="fg2bg,bg2fg,symmetric")
    p.add_argument("--rates", default="0.10,0.20,0.30")
    p.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--logit_freeze_iter", type=int, default=None,
                   help="Not set by default — omitting the flag lets the trainer use its "
                        "own default (opt.densify_from_iter, iter 500), i.e. RigidGS's "
                        "real as-shipped freeze schedule, including its brief learnable-"
                        "logit window. This experiment tests RigidGS's actual resilience, "
                        "not an artificially-frozen-at-init variant, so don't override this "
                        "unless you specifically want a different fixed freeze iteration "
                        "applied uniformly across the matrix.")
    p.add_argument("--experiment_root", default="output/label_noise_experiment")
    p.add_argument("--extra_trainer_args", default="",
                   help="Extra args appended verbatim to every trainer invocation.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the resolved scene configs + full job matrix, run nothing.")
    p.add_argument("--no_baseline", action="store_true",
                   help="Skip the rate=0.0 baseline job per scene (use when you already "
                        "have a trusted baseline run for these scenes, e.g. from the main "
                        "paper runs, and just want the corrupted-label matrix).")
    p.add_argument("--mask_2d_bg_margin_map", required=True,
                   help="REQUIRED, no default on purpose. Comma-separated "
                        "'dataset:scene=value' pairs, e.g. "
                        "'llff:fern=0.5,llff:trex=0.5,mipnerf360:kitchen=5,mipnerf360:garden=5'. "
                        "Every scene in --scenes must have an entry or it is skipped with an "
                        "explicit error rather than silently using the trainer's default (0.0).")
    args = p.parse_args()

    scenes = []
    for pair in args.scenes.split(","):
        dataset, scene = pair.split(":")
        scenes.append((dataset.strip(), scene.strip()))
    gpu_ids = [g.strip() for g in args.gpus.split(",")]
    schemes = [s.strip() for s in args.schemes.split(",")]
    rates = [float(r.strip()) for r in args.rates.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    bg_margin_map = parse_bg_margin_map(args.mask_2d_bg_margin_map)

    os.makedirs(args.experiment_root, exist_ok=True)

    print(f"Resolving {len(scenes)} scene(s)...")
    jobs, skipped_scenes = build_jobs(scenes, schemes, rates, seeds, args.experiment_root,
                                       bg_margin_map, include_baseline=not args.no_baseline)

    if skipped_scenes:
        print("\n=== SKIPPED SCENES (prerequisites not met) ===")
        for tag, reasons in skipped_scenes:
            print(f"  {tag}:")
            for r in reasons:
                print(f"    - {r}")

    freeze_desc = (f"logit_freeze_iter={args.logit_freeze_iter}" if args.logit_freeze_iter is not None
                   else "logit_freeze_iter=<trainer default, iter 500>")
    print(f"\n=== JOB MATRIX ({len(jobs)} jobs, {len(gpu_ids)} GPU(s), "
          f"{freeze_desc}, iterations={args.iterations}) ===")
    for j in jobs:
        ovr = os.path.basename(j["override_path"]) if j["override_path"] else "(none, baseline)"
        print(f"  {j['dataset']}/{j['scene']:<10s} {j['scheme']:<10s} rate={j['rate']:<4} "
              f"seed={j['seed']}  bg_margin={j['mask_2d_bg_margin']}  override={ovr}")
        print(f"      -> {j['model_path']}")

    if args.dry_run:
        print("\n[dry_run] Not launching anything.")
        return

    if not jobs:
        print("\nNo runnable jobs. Exiting.")
        return

    job_q = queue.Queue()
    for j in jobs:
        job_q.put(j)

    manifest_path = os.path.join(args.experiment_root, "run_manifest.csv")
    manifest_lock = threading.Lock()
    port_counter = [6100]
    port_lock = threading.Lock()

    print(f"\nLaunching {len(gpu_ids)} worker(s) across GPUs {gpu_ids}. "
          f"Manifest: {manifest_path}\n")

    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(target=worker, args=(
            gpu_id, job_q, args, manifest_path, manifest_lock, port_counter, port_lock))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\nAll jobs complete.")


if __name__ == "__main__":
    main()
