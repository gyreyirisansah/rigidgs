"""
mask_refinement_timed.py -- thin timing wrapper around
mask_refinement.py (that file is left completely untouched).

Rather than copying ~1600 lines of depth/KNN/SAM2-reprompt refinement math
(real risk of a copy silently drifting from the original and producing
different masks over time), this monkey-patches timing wrappers around the
module's key sub-process functions *before* calling its own run(args)
unmodified -- same idea as segmentation_timed.py's subclassing, just the
free-function equivalent. This works because Python resolves a function's
global names (e.g. run_depth(...) inside process_one_image) through its
*module's* __dict__ at CALL time, not at definition time -- so reassigning
mask_refinement.run_depth before run(args) runs takes effect
for every call inside that module, including calls made from inside
process_one_image, with zero changes to the original file.

Sub-processes timed (each accumulated as a TOTAL across the whole run, since
most run once per grid cell per image rather than once per image -- not
broken down per-call):
  - run_depth_sec               monocular depth inference (Depth Anything V2)
                                 -- called once for full-frame depth (if
                                 ridge_fill or sam2_reprompt enabled) AND once
                                 per grid cell per image, so this is usually
                                 the single largest bucket.
  - compute_knn_cell_sec        per-cell KNN soft-threshold refinement
  - apply_ridge_fill_sec        depth-ridge hole filling (only if --enable_ridge_fill)
  - apply_edge_restoration_sec  boundary-band restoration (only if --enable_edge_restoration)
  - sam2_reprompt_sec           SAM2 re-prompting from mask history (only if
                                 --enable_sam2_reprompt) -- includes whatever
                                 SAM2 inference calls happen inside it, so it
                                 can partially overlap run_depth_sec if that
                                 function calls run_depth internally too
                                 (hierarchical timing, not mutually exclusive
                                 buckets -- flagged here rather than hidden)
  - sam2_runner_load_sec        SAM2RepromptRunner model load (one-time, only
                                 if --enable_sam2_reprompt)
  - setup_and_io_sec            residual: total - sum(all of the above) --
                                 covers Depth Anything model loading, image
                                 reads, mask/debug writes, summary CSV/JSON.

Writes mask_refinement_timing.json into --output_dir alongside the refined
masks -- a new filename, so it can't collide with run()'s own
run_summary.json/per_image.csv/failed.txt.

IMPORTANT: this file is written INCREMENTALLY, after every processed image
(process_one_image is patched the same way as the sub-process functions
above, purely to trigger a write -- it doesn't change what's timed), and
also on any unhandled exception from run() itself (model load failure,
etc.), not just on a clean finish. A run that's still in progress, or one
that crashed or got killed partway through, still leaves a real, current
report on disk -- "partial": true while running/interrupted, "partial":
false only once run() has actually returned normally. Earlier versions of
this file only wrote the report after run() fully returned, so a still-
running or crashed run left nothing on disk at all.

Usage: identical CLI to mask_refinement.py (build_parser is
imported, not redefined).
  python src/mask_refinement_timed.py \
      --image_dir ... --coarse_mask_dir ... --output_dir ...
"""
import time
import json
import functools
import traceback
from pathlib import Path

import mask_refinement as dmg


def _time_accumulate(name, fn, bucket):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        result = fn(*a, **kw)
        bucket[name] = bucket.get(name, 0.0) + (time.perf_counter() - t0)
        return result
    return wrapper


def _build_report(args, timing, total_sec, partial, images_done=None, error=None):
    known_sec = sum(timing.values())
    timing_out = dict(timing)
    timing_out["setup_and_io_sec"] = max(0.0, total_sec - known_sec)
    timing_out = {k: round(v, 3) for k, v in timing_out.items()}

    report = {
        "image_dir": args.image_dir,
        "coarse_mask_dir": args.coarse_mask_dir,
        "output_dir": args.output_dir,
        "enable_ridge_fill": args.enable_ridge_fill,
        "enable_edge_restoration": args.enable_edge_restoration,
        "enable_sam2_reprompt": args.enable_sam2_reprompt,
        "grid_size": args.grid_size,
        "partial": partial,
        "images_done": images_done,
        "total_sec": round(total_sec, 3),
        "total_min": round(total_sec / 60.0, 2),
        "timing_sec": timing_out,
    }
    if error is not None:
        report["error"] = error
    return report


def _write_report(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "mask_refinement_timing.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report_path


def main():
    run_timed(dmg.build_parser().parse_args())


def run_timed(args):
    """Run mask_refinement.run(args) with the timing wrappers;
    returns the final timing report (also written to output_dir)."""
    out_dir = Path(args.output_dir)

    timing: dict[str, float] = {}
    state = {"images_done": 0}

    dmg.run_depth = _time_accumulate("run_depth_sec", dmg.run_depth, timing)
    dmg.compute_knn_cell = _time_accumulate("compute_knn_cell_sec", dmg.compute_knn_cell, timing)
    dmg.apply_ridge_fill = _time_accumulate("apply_ridge_fill_sec", dmg.apply_ridge_fill, timing)
    dmg.apply_edge_restoration = _time_accumulate(
        "apply_edge_restoration_sec", dmg.apply_edge_restoration, timing)
    dmg.apply_sam2_reprompt_from_history = _time_accumulate(
        "sam2_reprompt_sec", dmg.apply_sam2_reprompt_from_history, timing)

    if args.enable_sam2_reprompt:
        _RealSAM2RepromptRunner = dmg.SAM2RepromptRunner

        class _TimedSAM2RepromptRunner(_RealSAM2RepromptRunner):
            def __init__(self, *a, **kw):
                t0 = time.perf_counter()
                super().__init__(*a, **kw)
                timing["sam2_runner_load_sec"] = (
                    timing.get("sam2_runner_load_sec", 0.0) + (time.perf_counter() - t0))

        dmg.SAM2RepromptRunner = _TimedSAM2RepromptRunner

    t_total_start = time.perf_counter()

    # Incremental progress: write mask_refinement_timing.json after every
    # image, not just once at the very end. process_one_image's return
    # value is passed through unchanged -- this only adds a write.
    _real_process_one_image = dmg.process_one_image

    @functools.wraps(_real_process_one_image)
    def _process_one_image_and_report(*a, **kw):
        result = _real_process_one_image(*a, **kw)
        state["images_done"] += 1
        elapsed = time.perf_counter() - t_total_start
        report = _build_report(args, timing, elapsed, partial=True,
                               images_done=state["images_done"])
        _write_report(report, out_dir)
        return result

    dmg.process_one_image = _process_one_image_and_report

    try:
        dmg.run(args)
    except BaseException as e:
        # Still leave a real report on disk for whatever completed before
        # the crash/interrupt, instead of nothing at all.
        elapsed = time.perf_counter() - t_total_start
        report = _build_report(args, timing, elapsed, partial=True,
                               images_done=state["images_done"],
                               error=f"{type(e).__name__}: {e}")
        report_path = _write_report(report, out_dir)
        print(f"\n❌ Mask refinement failed/interrupted after "
              f"{state['images_done']} image(s), {elapsed:.1f}s.")
        print(f"📄 Partial timing report saved to: {report_path}")
        traceback.print_exc()
        raise

    total_sec = time.perf_counter() - t_total_start
    report = _build_report(args, timing, total_sec, partial=False,
                           images_done=state["images_done"])
    report_path = _write_report(report, out_dir)

    print(f"\n⏱  Mask refinement timing (total={total_sec:.1f}s / {total_sec/60:.2f} min, "
          f"{state['images_done']} image(s)):")
    for k, v in sorted(report["timing_sec"].items(), key=lambda kv: -kv[1]):
        pct = (v / total_sec * 100) if total_sec > 0 else 0.0
        print(f"    {k:<28} {v:>8.1f}s ({pct:4.1f}%)")
    print(f"📄 Timing report saved to: {report_path}")
    return report


if __name__ == "__main__":
    main()
