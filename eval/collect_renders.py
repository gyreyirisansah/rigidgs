"""
Arrange one trained run's foreground/background renders into the layout
evaluate_segmentation.py reads:

    <results_root>/<method>/<dataset>/<scene>/fg/        foreground renders
                                             fg_mask/   rendered foreground silhouettes
                                             bg/        background-only renders

Input is what render_fg_bg.py writes for the run:

    <model_path>/<split>/ours_<iter>_foreground/{renders,mask_renders}/
    <model_path>/<split>/ours_<iter>_background/renders/

By default the latest iteration that has both a foreground and a background
render is used.

Example:
    python eval/collect_renders.py -m output/llff/fern/rigidgs \\
        --results_root output/eval_renders --method RigidGS --dataset llff --scene fern
"""

import argparse
import re
import shutil
from pathlib import Path


def latest_complete_iteration(split_dir: Path) -> int:
    iters = set()
    for d in split_dir.glob("ours_*_foreground"):
        m = re.fullmatch(r"ours_(\d+)_foreground", d.name)
        if m and (split_dir / f"ours_{m.group(1)}_background").is_dir():
            iters.add(int(m.group(1)))
    if not iters:
        raise SystemExit(f"No ours_<iter>_foreground + _background pair in {split_dir}. "
                         f"Run eval/render_fg_bg.py first.")
    return max(iters)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model_path", type=Path, required=True)
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--method", required=True, help="Method folder name, e.g. RigidGS.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--iteration", type=int, default=None)
    args = p.parse_args()

    split_dir = args.model_path / args.split
    it = args.iteration if args.iteration is not None else latest_complete_iteration(split_dir)
    fg = split_dir / f"ours_{it}_foreground"
    bg = split_dir / f"ours_{it}_background"
    dest = args.results_root / args.method / args.dataset / args.scene

    for src, name in ((fg / "renders", "fg"), (fg / "mask_renders", "fg_mask"),
                      (bg / "renders", "bg")):
        if not src.is_dir():
            raise SystemExit(f"Missing {src}")
        out = dest / name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(src, out)
        print(f"{src} -> {out} ({sum(1 for _ in out.iterdir())} files)")
    print(f"Collected iteration {it} into {dest}")


if __name__ == "__main__":
    main()
