#!/usr/bin/env bash
# Render, collect and evaluate every label-noise run (paper Table 4).
# Run from the repository root after run_label_noise_matrix.py has finished:
#     bash experiments/label_noise/evaluate.sh [experiment_root]
set -euo pipefail

EXP="${1:-output/label_noise_experiment}"
RESULTS="$EXP/eval_renders"
GT_ROOT="${GT_ROOT:-data/ground_truth/images}"
GT_MASK_ROOT="${GT_MASK_ROOT:-data/ground_truth/masks}"

for scene_dir in "$EXP"/runs/*/; do
    tag=$(basename "$scene_dir")              # <dataset>_<scene>
    dataset=""
    for d in data/*/; do                      # dataset names may contain "_"
        d=$(basename "$d")
        if [[ "$tag" == "${d}_"* && ${#d} -gt ${#dataset} ]]; then dataset="$d"; fi
    done
    [ -n "$dataset" ] || { echo "Cannot tell the dataset of $tag (no data/<dataset>/)"; exit 1; }
    scene="${tag#${dataset}_}"

    for run in "$scene_dir"*/; do
        name=$(basename "$run")
        echo "=== $dataset/$scene: $name ==="
        python eval/render_fg_bg.py -m "$run" --iteration 30000 --skip_train --quiet
        python eval/collect_renders.py -m "$run" --results_root "$RESULTS" \
            --method "$name" --dataset "$dataset" --scene "$scene" --iteration 30000
    done
done

python eval/evaluate_segmentation.py \
    --results_root "$RESULTS" \
    --gt_root "$GT_ROOT" \
    --gt_mask_root "$GT_MASK_ROOT" \
    --output_dir "$EXP/eval"
