"""
One-command SPIn-NeRF object-removal eval.

Orchestrates the two existing pieces this project already has -- doesn't
reimplement either:
  1. render_spinnerf_holdout.py -- renders the 40 held-out no-object
     poses for a trained checkpoint (FG + BG-hole), per scene.
  2. evaluate_spinnerf.py -- scores those renders against the GT
     no-object photos: LPIPS + FID + PSNR + SSIM inside the object-mask
     bbox (Table 5's hole-only protocol).

Skips re-rendering a scene if a holdout render already exists for the
resolved checkpoint iteration (use --force_render to redo it anyway) --
rendering is the slow part; re-scoring already-rendered images is fast.

Usage:
    # All scenes, latest checkpoint each, render + score:
    python run_spinnerf_eval.py --scenes 1 2 3 4

    # One specific checkpoint iteration (same iteration across all scenes):
    python run_spinnerf_eval.py --scenes 4 --iteration 30000

    # Re-score already-rendered results without re-rendering:
    python run_spinnerf_eval.py --scenes 4 --skip_render

    # Force a fresh render even if one already exists for this iteration:
    python run_spinnerf_eval.py --scenes 4 --force_render

    # Compare pre- vs post-Stage-3 for the same scene (two separate calls,
    # since each needs its own --iteration):
    python run_spinnerf_eval.py --scenes 4 --iteration 30000
    python run_spinnerf_eval.py --scenes 4 --iteration 35000
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "external" / "gaussian-splatting"))
from utils.system_utils import searchForMaxIteration

PYTHON = sys.executable


def resolve_iteration(model_path: Path, iteration_arg) -> int:
    """iteration_arg is None or -1 -> latest available under model_path/point_cloud;
    otherwise the explicit value, unchanged."""
    if iteration_arg is not None and iteration_arg != -1:
        return iteration_arg
    pc_dir = model_path / "point_cloud"
    if not pc_dir.is_dir():
        raise FileNotFoundError(f"No point_cloud/ under {model_path} -- has this scene "
                                 f"finished training at all?")
    return searchForMaxIteration(str(pc_dir))


def render_exists(model_path: Path, iteration: int) -> bool:
    d = model_path / "spinnerf_holdout_eval" / "background_hole_renders" / f"iter_{iteration:05d}"
    return d.is_dir() and any(d.glob("*.png"))


def render_scene(scene: str, model_path: Path, iteration_arg, source_path: str,
                  images: str, gt_root: str, force: bool) -> int:
    resolved_iter = resolve_iteration(model_path, iteration_arg)
    if not force and render_exists(model_path, resolved_iter):
        print(f"[scene {scene}] holdout render already exists for iter {resolved_iter} "
              f"-- skipping (--force_render to redo).")
        return resolved_iter

    cmd = [
        PYTHON, str(TRAINING_DIR / "render_spinnerf_holdout.py"),
        "--source_path", source_path,
        "--images", images,
        "--scene", scene,
        "--gt_root", gt_root,
        "--model_path", str(model_path),
        "--iteration", str(iteration_arg if iteration_arg is not None else -1),
        "--quiet",
    ]
    print(f"[scene {scene}] rendering holdout eval (resolves to iter {resolved_iter})...")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return resolved_iter


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="SPIn-NeRF scene ids, e.g. 1 2 3 4.")
    parser.add_argument("--model_path_template", type=str,
                        default="output/spinnerf/{scene}/rigidgs",
                        help="Trained output dir per scene; {scene} is substituted.")
    parser.add_argument("--source_path_template", type=str,
                        default="output/spinnerf/{scene}/point_cloud/pcd_data",
                        help="preprocessing.py output; its sparse/0 holds the held-out poses.")
    parser.add_argument("--images_template", type=str,
                        default="output/spinnerf/{scene}/images_undistorted",
                        help="Undistorted training-resolution images dir per scene "
                             "(needed for camera intrinsics/resolution, not actually "
                             "read pixel-by-pixel for the held-out poses).")
    parser.add_argument("--gt_root", type=str, default="data/ground_truth/images/spinnerf",
                        help="Staged by eval/stage_spinnerf_gt.py.")
    parser.add_argument("--gt_mask_root", type=str, default="data/ground_truth/masks/spinnerf")
    parser.add_argument("--output_json", type=str, default="output/spinnerf/eval_results.json")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Checkpoint iteration to evaluate, applied to every "
                             "requested scene. Default: -1 (latest available, "
                             "resolved independently per scene).")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip rendering entirely -- score whatever's already on "
                             "disk. Fails per-scene if nothing's there yet.")
    parser.add_argument("--force_render", action="store_true",
                        help="Re-render even if a holdout render already exists for "
                             "the resolved iteration.")
    args = parser.parse_args()

    resolved_iters = {}
    for scene in args.scenes:
        model_path = Path(args.model_path_template.format(scene=scene))
        if args.skip_render:
            resolved_iters[scene] = resolve_iteration(model_path, args.iteration)
        else:
            source_path = args.source_path_template.format(scene=scene)
            images = args.images_template.format(scene=scene)
            resolved_iters[scene] = render_scene(
                scene, model_path, args.iteration, source_path, images, args.gt_root,
                args.force_render)

    print(f"\nResolved iterations: {resolved_iters}")

    eval_cmd = [
        PYTHON, str(TRAINING_DIR / "evaluate_spinnerf.py"),
        "--scenes", *args.scenes,
        "--model_path_template", args.model_path_template,
        "--gt_root", args.gt_root,
        "--gt_mask_root", args.gt_mask_root,
        "--output_json", args.output_json,
    ]
    if args.iteration is not None:
        # Uniform iteration across all scenes -- pass it through directly.
        eval_cmd += ["--iteration", str(args.iteration)]
    # else: omit --iteration entirely: evaluate_spinnerf.py auto-
    # detects each scene's own latest iter_* independently, which is
    # exactly right when scenes resolved to different iterations above.

    print(f"\n[run_spinnerf_eval] scoring {len(args.scenes)} scene(s)...\n")
    subprocess.run(eval_cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
