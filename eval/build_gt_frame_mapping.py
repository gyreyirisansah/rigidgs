#!/usr/bin/env python3
"""
Build a JSON mapping, per (dataset, scene), of the canonical GT target
frame(s) -- the real photo + its GT mask, shared identically across every
method -- to each method's own fg/bg/fg_mask filenames that were matched to
that target. Reuses evaluate_segmentation.py's discovery/matching so the mapping
reflects exactly what a evaluate_segmentation.py run would score.

Usage:
  python src/build_gt_frame_mapping.py \
      --results_root ../RigidGS_Results_Accum \
      --gt_root data/ground_truth/images --gt_mask_root data/ground_truth/masks \
      --output ../RigidGS_Results_Accum_Eval/gt_frame_mapping.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import (
    IMAGE_EXTS, MASK_EXTS,
    _index_directory, _find_gt_dir, _find_gt_mask_dir,
    extend_gt_idx_with_orphan_masks, oracle_match,
    discover_combos,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, required=True)
    ap.add_argument("--gt_root", type=Path, required=True)
    ap.add_argument("--gt_mask_root", type=Path, required=True)
    ap.add_argument("--match_resolution", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    combos = discover_combos(args.results_root)
    by_scene: dict[tuple[str, str], dict[str, Path]] = {}
    for method, dataset, scene, scene_dir in combos:
        by_scene[(method, dataset, scene)] = scene_dir

    scenes = sorted({(dataset, scene) for _, dataset, scene, _ in combos})
    mapping: dict[str, dict] = {}

    for dataset, scene in scenes:
        gt_dir = _find_gt_dir(args.gt_root, dataset, scene)
        gt_mask_dir = _find_gt_mask_dir(args.gt_mask_root, dataset, scene)
        if gt_dir is None:
            continue
        gt_idx = _index_directory(gt_dir, IMAGE_EXTS)
        gt_mask_idx = _index_directory(gt_mask_dir, MASK_EXTS) if gt_mask_dir else {}

        # Same conditional-extension rule as evaluate_segmentation.py's main loop:
        # only reach into the raw capture when the staged photos have no
        # mask coverage of their own.
        if not (set(gt_idx) & set(gt_mask_idx)):
            gt_idx = extend_gt_idx_with_orphan_masks(dataset, scene, gt_idx, gt_mask_idx)

        methods_here = sorted({m for m, d, s, _ in combos if d == dataset and s == scene})

        targets = {g_stem: {
            "gt_photo": gt_idx[g_stem].name,
            "gt_mask": gt_mask_idx[g_stem].name if g_stem in gt_mask_idx else None,
            "methods": {},
        } for g_stem in gt_idx}

        for method in methods_here:
            scene_dir = by_scene[(method, dataset, scene)]
            fg_idx = _index_directory(scene_dir / "fg", IMAGE_EXTS)
            bg_idx = _index_directory(scene_dir / "bg", IMAGE_EXTS)
            pred_idx = _index_directory(scene_dir / "fg_mask", MASK_EXTS)
            common_renders = sorted(set(fg_idx) & set(bg_idx))
            if not common_renders or not gt_idx:
                continue

            # GT-driven, matching evaluate_segmentation.py's content-mode matching
            # exactly -- see its comment for why render-driven is no longer
            # used (it can crowd out the one masked target on small
            # imbalances like fern, once conditional extension already
            # keeps gt_idx small everywhere else).
            pairs = list(oracle_match(gt_idx, common_renders, fg_idx, bg_idx,
                                      match_resolution=args.match_resolution).items())

            # Methods with no fg_mask/ export at all (SAGS, GaussianGrouping
            # on most scenes) still get included: fall back to the fg/
            # render at the SAME matched index -- it's already the render
            # for this exact target, just not thresholded into a mask file.
            # evaluate_segmentation.py itself derives the actual mask from this
            # same fg render at scoring time (_mask_from_fg_render); this
            # just reports which file that derivation would read.
            use_fg_as_masks = not pred_idx and bool(fg_idx)

            for g_stem, r_stem in pairs:
                if g_stem not in targets:
                    continue
                if r_stem in pred_idx:
                    fg_mask_name, fg_mask_derived = pred_idx[r_stem].name, False
                elif use_fg_as_masks and r_stem in fg_idx:
                    fg_mask_name, fg_mask_derived = fg_idx[r_stem].name, True
                else:
                    fg_mask_name, fg_mask_derived = None, False
                targets[g_stem]["methods"][method] = {
                    "fg": fg_idx[r_stem].name,
                    "bg": bg_idx[r_stem].name,
                    "fg_mask": fg_mask_name,
                    "fg_mask_derived_from_fg": fg_mask_derived,
                }

        # Only keep targets that at least one method actually matched to,
        # sorted for stable, readable output.
        kept = {g: t for g, t in targets.items() if t["methods"]}
        mapping.setdefault(dataset, {})[scene] = {
            "gt_targets": [dict(gt_frame_id=g, **kept[g]) for g in sorted(kept)]
        }
        n_methods_agree = sum(
            1 for t in kept.values() if len(t["methods"]) == len(methods_here)
        )
        print(f"{dataset}/{scene}: {len(kept)} GT target(s), "
              f"{n_methods_agree}/{len(kept)} matched by all {len(methods_here)} methods "
              f"({', '.join(methods_here)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
