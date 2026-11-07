#!/usr/bin/env python3
"""Run the preprocess -> train -> eval pipeline for one or more scenes, using
the per-scene settings in configs/.

Each scene has one config, configs/<dataset>/<scene>.json, holding the
preprocessing and training settings used in the paper plus the paper's
segmentation numbers. For every requested scene this script runs, in order:

  preprocess  src/preprocessing.py   -> output/<dataset>/<scene>/
  train       src/train.py           -> output/<dataset>/<scene>/rigidgs/
  eval        eval/render_fg_bg.py + eval/collect_renders.py
              -> output/eval_renders/RigidGS/<dataset>/<scene>/

Preprocessing runs one scene at a time in this terminal, because SAM 2
segmentation needs clicks and object names typed here. As soon as a scene's
preprocessing finishes, its training and eval go to a free GPU from --gpus
(one scene per GPU at a time), so the GPUs fill up while you label the next
scene. The Stage 2 hole click (train, at --inpaint_at_iter) opens a window
for each training run; stay at the display.
When "eval" is in --steps, the collected renders are scored at the end and
compared with each config's "paper" numbers as a target to land near --
preprocessing re-prompts SAM 2 from scratch each run, so results are not
guaranteed to be bit-identical to the paper's run (see the README's "note on
exact reproducibility"). Scoring needs only GT_Masks (shipped in the repo);
each mask's target photo is looked up live in data/<dataset>/<scene>/images/,
so no separately staged GT photo directory is required. Run from the
repository root.

Usage:
  python scripts/run_pipeline.py --scenes llff/fern llff/trex --gpus 0,1,2 --dry_run
  python scripts/run_pipeline.py --scenes all --gpus 0,1,2 --steps train eval
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
STEPS = ("preprocess", "train", "eval")
METRICS = ("iou", "boundary_iou", "accuracy")

sys.path.insert(0, str(REPO_ROOT / "src"))
from config_utils import as_flags, scene_paths  # noqa: E402 -- shared with preprocessing.py/train.py


def load_configs(names):
    if names == ["all"]:
        paths = sorted(CONFIG_DIR.glob("*/*.json"))
    else:
        paths = [CONFIG_DIR / f"{n}.json" for n in names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit(f"No config: {', '.join(missing)}")
    return [json.loads(p.read_text()) for p in paths]


def step_cmds(cfg, step, port):
    py = sys.executable
    p = scene_paths(cfg)
    if step == "preprocess":
        return [[py, "src/preprocessing.py", "--images", cfg["images"], "--output", str(p["out"])]
                + as_flags(cfg["preprocessing"], "--")]
    if step == "train":
        return [[py, "src/train.py", "-s", str(p["pcd"]), "-m", str(p["model"]),
                 "--_images", str(p["images"]), "--masks_dir", str(p["masks"]),
                 "--port", str(port)] + p["split"] + as_flags(cfg["train"], "--")]
    it = str(cfg["eval_iteration"])
    return [[py, "eval/render_fg_bg.py", "-m", str(p["model"]), "--iteration", it,
             "--skip_train", "--quiet"],
            [py, "eval/collect_renders.py", "-m", str(p["model"]),
             "--results_root", "output/eval_renders", "--method", "RigidGS",
             "--dataset", cfg["dataset"], "--scene", cfg["scene"], "--iteration", it]]


def run_step(cfg, step, gpu_id, port, args, manifest, lock, terminal=False):
    """Runs one step's commands, logging to output/<dataset>/<scene>/logs/<step>.log.
    terminal=True also shows the output here and lets the step read from this
    terminal (SAM 2 segmentation asks for each object's name)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"       # input() prompts must reach the terminal at once
    tag = f"{cfg['dataset']}/{cfg['scene']}"
    log_path = REPO_ROOT / scene_paths(cfg)["out"] / "logs" / f"{step}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now().isoformat()
    print(f"[GPU {gpu_id}] START  {tag} {step}  (log: {log_path})", flush=True)
    rc = 0
    with open(log_path, "w") as logf:
        for cmd in step_cmds(cfg, step, port):
            logf.write("CMD: " + " ".join(cmd) + "\n\n")
            logf.flush()
            if not terminal:
                rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                    stdout=logf, stderr=subprocess.STDOUT).returncode
            else:
                proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                # raw chunks, not lines, so a prompt without a newline shows up
                while chunk := os.read(proc.stdout.fileno(), 4096):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                    logf.write(chunk.decode(errors="replace"))
                    logf.flush()
                rc = proc.wait()
            if rc != 0:
                break
    status = "ok" if rc == 0 else "FAILED"
    print(f"[GPU {gpu_id}] {status}  {tag} {step}  (rc={rc})", flush=True)
    row = {"dataset": cfg["dataset"], "scene": cfg["scene"], "step": step,
           "gpu": gpu_id, "status": status, "returncode": rc,
           "start_time": start, "end_time": datetime.now().isoformat(),
           "log_path": str(log_path)}
    with lock:
        write_header = not manifest.exists()
        with open(manifest, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    return rc


def worker(gpu_id, job_q, steps, args, manifest, lock):
    """Takes (cfg, port) jobs until it gets None and runs the GPU steps."""
    while (job := job_q.get()) is not None:
        cfg, port = job
        for step in steps:
            if run_step(cfg, step, gpu_id, port, args, manifest, lock) != 0:
                break                   # later steps need this one's output


def score(configs, args):
    out_dir = REPO_ROOT / "output" / "eval"
    cmd = [sys.executable, "eval/evaluate_segmentation.py",
           "--results_root", "output/eval_renders", "--methods", "RigidGS",
           "--output_dir", str(out_dir), "--scenes"] + [c["scene"] for c in configs]
    if args.gt_root:
        cmd += ["--gt_root", args.gt_root]
    if args.gt_mask_root:
        cmd += ["--gt_mask_root", args.gt_mask_root]
    print("\nScoring:", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
        return
    with open(out_dir / "per_scene.csv") as f:
        rows = {(r["dataset"], r["scene"]): r for r in csv.DictReader(f)}
    print("\n" + f"{'scene':<22}" + "".join(f"{m:>22}" for m in METRICS))
    print(" " * 22 + "".join(f"{'ours / paper':>22}" for _ in METRICS))
    for c in configs:
        r = rows.get((c["dataset"], c["scene"]))
        if r is None:
            print(f"{c['dataset'] + '/' + c['scene']:<22}  (no results)")
            continue
        cells = []
        for m in METRICS:
            ours = float(r[f"{m}_mean"])
            paper = c.get("paper", {}).get(m)
            cells.append(f"{ours:.3f} / {paper:.3f}" if paper is not None else f"{ours:.3f} / -")
        print(f"{c['dataset'] + '/' + c['scene']:<22}" + "".join(f"{x:>22}" for x in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="<dataset>/<scene> names under configs/ (e.g. llff/fern), or 'all'.")
    ap.add_argument("--gpus", default="0", help="Comma-separated GPU ids, one scene per GPU.")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--gt_root", default=None,
                    help="Optional: separately staged GT images, if you have them. "
                         "Not needed otherwise -- each GT_Masks file's target photo "
                         "is found live in data/<dataset>/<scene>/images/.")
    ap.add_argument("--gt_mask_root", default="GT_Masks",
                    help="GT foreground masks, shipped in the repo (default: GT_Masks/).")
    ap.add_argument("--base_port", type=int, default=6100,
                    help="Viewer ports are base_port+1, +2, ... so parallel runs don't clash.")
    ap.add_argument("--dry_run", action="store_true", help="Print the commands and exit.")
    args = ap.parse_args()

    configs = load_configs(args.scenes)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    steps = [s for s in STEPS if s in args.steps]     # always in pipeline order
    args.steps = steps

    if args.dry_run:
        for i, cfg in enumerate(configs):
            print(f"# {cfg['dataset']}/{cfg['scene']}  (GPU {gpus[i % len(gpus)]} if run in order)")
            for step in steps:
                for cmd in step_cmds(cfg, step, args.base_port + i + 1):
                    print("  " + " ".join(cmd))
        return

    manifest = REPO_ROOT / "output" / "run_pipeline_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    gpu_steps = [s for s in steps if s != "preprocess"]
    job_q = queue.Queue()
    threads = [threading.Thread(target=worker, args=(g, job_q, gpu_steps, args, manifest, lock))
               for g in gpus]
    for t in threads:
        t.start()
    # Preprocessing is interactive (SAM 2 clicks + object names typed here), so
    # scenes go through it one at a time in the foreground; each scene starts
    # training on a free GPU as soon as its preprocessing is done.
    for i, cfg in enumerate(configs):
        port = args.base_port + i + 1
        if "preprocess" in steps and run_step(cfg, "preprocess", gpus[0], port, args,
                                              manifest, lock, terminal=True) != 0:
            continue
        if gpu_steps:
            job_q.put((cfg, port))
    for _ in threads:
        job_q.put(None)
    for t in threads:
        t.join()

    if "eval" in steps:
        score(configs, args)


if __name__ == "__main__":
    main()
