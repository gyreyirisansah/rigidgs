#!/usr/bin/env python3
"""
Reconstruct training-time metrics for already-trained runs from their saved
TensorBoard event files -- no retraining needed.

For each run directory (anything containing one or more events.out.tfevents.*
files, including subdirs like stage3_tb/ for a later training stage), reports:
  - total logged iter_time (sum of per-iteration CUDA-event-timed GPU compute,
    the tightest "pure compute" time estimate)
  - wall-clock span (last event timestamp - first, across all its files) --
    a looser estimate that also captures eval/checkpoint save overhead
  - max iteration/step reached

Peak GPU memory is NOT recoverable this way -- the trainer never logs it
periodically during training, only a near-meaningless post-cleanup snapshot
after torch.cuda.empty_cache() has already released cached memory back to the
driver. Getting a real peak reading requires re-running (see script docstring
note below) with instrumentation added.

Usage:
  python eval/extract_training_metrics.py --root output
  python eval/extract_training_metrics.py --root output/llff/fern
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_tfevents(root: Path) -> list[Path]:
    return sorted(root.rglob("events.out.tfevents.*"))


def run_dir_for(tfevents_path: Path) -> Path:
    """
    Group tfevents files by the run they belong to. A file directly in
    <run>/ belongs to <run>; a file in a named subdir (e.g. <run>/stage3_tb/)
    still belongs to <run> -- stage3 is a later phase of the same run, not a
    separate one.
    """
    parent = tfevents_path.parent
    # Heuristic: a "stage" subdir sits directly under the run dir and itself
    # contains no point_cloud/ -- walk up one level in that case.
    if not (parent / "point_cloud").is_dir() and (parent.parent / "point_cloud").is_dir():
        return parent.parent
    return parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None,
                    help="CSV output path (default: <root>/training_metrics.csv)")
    args = ap.parse_args()

    files = find_tfevents(args.root)
    if not files:
        print(f"No tfevents files found under {args.root}")
        return

    by_run: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        by_run[run_dir_for(f)].append(f)

    rows = []
    for run_dir, run_files in sorted(by_run.items()):
        total_iter_time_ms = 0.0
        first_wall, last_wall = None, None
        max_step = 0
        n_points = 0
        for f in run_files:
            ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags()["scalars"]
            if "iter_time" not in tags:
                continue
            events = ea.Scalars("iter_time")
            if not events:
                continue
            n_points += len(events)
            total_iter_time_ms += sum(e.value for e in events)
            max_step = max(max_step, events[-1].step)
            fw, lw = events[0].wall_time, events[-1].wall_time
            first_wall = fw if first_wall is None else min(first_wall, fw)
            last_wall = lw if last_wall is None else max(last_wall, lw)

        if n_points == 0:
            continue

        wall_span_s = (last_wall - first_wall) if (first_wall and last_wall) else float("nan")
        rows.append({
            "run_dir": str(run_dir),
            "n_tfevents_files": len(run_files),
            "n_iter_time_points": n_points,
            "max_step_seen": max_step,
            "compute_time_sec": round(total_iter_time_ms / 1000.0, 1),
            "compute_time_min": round(total_iter_time_ms / 1000.0 / 60.0, 2),
            "wall_clock_span_sec": round(wall_span_s, 1),
            "wall_clock_span_min": round(wall_span_s / 60.0, 2),
        })
        print(f"{run_dir}: {len(run_files)} tfevents file(s), "
              f"{n_points} iter_time points up to step {max_step}, "
              f"compute={total_iter_time_ms/1000:.1f}s, "
              f"wall={wall_span_s:.1f}s")

    out_path = args.output or (args.root / "training_metrics.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
