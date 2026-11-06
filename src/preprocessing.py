"""
RigidGS preprocessing: everything before training.

    images/ ──► 1. SfM (COLMAP, PINHOLE, from scratch)
            ──► 2. SAM2 segmentation (interactive clicks on one frame, then propagated)
            ──► 3. Depth-based mask refinement (optional, --no-mask-refinement)
            ──► 4. 3D point labeling (multi-view voting of the raw SAM2 masks)
            ──► 5. Point-label refinement (point_refinement)
            ──► point_cloud/pcd_data/  (what train.py loads with -s)

Output layout under --output (e.g. output/llff/fern):

    sfm/sparse/0/                         COLMAP model
    preprocessing/segmentation/masks/     SAM2 masks, <stem>_mask_<object>.<ext>
    preprocessing/refined/refined_masks/  refined masks (same file names)
    point_cloud/labeled_point_cloud.ply   labeled points (+ _metadata.json)
    point_cloud/pcd_data/sparse/0/        train.py input: cameras.bin, images.bin,
                                          points3D.ply, points3D_metadata.json
    preprocessing_timing.json             per-stage wall-clock (paper Table 7)

Each stage can be skipped to reuse its existing output (--skip-sfm etc.).
Run from the repository root.

SPIn-NeRF (--spinnerf): the photos are distorted (SIMPLE_RADIAL). They are
first undistorted with the dataset's own calibration into
<output>/images_undistorted/, and that folder is used for SfM, segmentation
AND training. After SfM, points observed by any of the 40 held-out
(no-object) views are removed, so the initial point cloud cannot leak the
unoccluded background into the evaluation, and sparse/0/test.txt lists the
held-out views for train.py --eval --llffhold 0.
"""

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _cleanup_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _stage(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


# =====================================================================
# SPIn-NeRF helpers
# =====================================================================
def _spinnerf_with_object_stems(scene_root: Path) -> set:
    """The 60 with-object (training) stems, from the dataset's own label/ folder."""
    label_dir = scene_root / "images_4" / "label"
    return {p.stem for p in label_dir.iterdir() if p.suffix.lower() == ".png"}


def spinnerf_undistort(scene_root: Path, out_dir: Path, overwrite: bool = False) -> Path:
    """Undistort <scene_root>/images/ into out_dir with the dataset's own
    SIMPLE_RADIAL calibration (<scene_root>/sparse/0/cameras.bin). COLMAP's
    PINHOLE fit on the raw distorted photos is warped by up to ~120 px at the
    image edges, enough for point labeling to miss the object entirely."""
    import cv2
    from PIL import Image
    from spinnerf_utils import read_cameras_bin, build_undistort_map, undistort_image

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        print(f"[spinnerf] reusing undistorted images in {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = next(iter(read_cameras_bin(scene_root / "sparse" / "0" / "cameras.bin").values()))
    paths = sorted(p for p in (scene_root / "images").iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with Image.open(paths[0]) as im:
        w, h = im.size
    map_x, map_y, roi, _ = build_undistort_map(cam, w, h)
    print(f"[spinnerf] undistorting {len(paths)} images ({w}x{h}) -> {out_dir}")
    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = undistort_image(img, map_x, map_y, roi, cv2.INTER_LINEAR)
        Image.fromarray(img).save(out_dir / p.name, quality=95)
    return out_dir


def spinnerf_fix_point_cloud_leak(sparse_dir: Path, scene_root: Path) -> None:
    """Keep only points triangulated purely from the 60 training views, and
    write sparse/0/test.txt with the 40 held-out view names. Poses of all
    100 views are kept (one jointly calibrated frame for train and test)."""
    from spinnerf_utils import (
        read_images_bin_full, write_images_bin_full,
        read_points3D_bin_full, write_points3D_bin_full,
        filter_points_to_training_images,
    )
    with_object = _spinnerf_with_object_stems(scene_root)
    images_full = read_images_bin_full(sparse_dir / "images.bin")
    train_ids = {iid for iid, img in images_full.items() if Path(img["name"]).stem in with_object}
    test_names = sorted(img["name"] for img in images_full.values()
                        if Path(img["name"]).stem not in with_object)

    points_full = read_points3D_bin_full(sparse_dir / "points3D.bin")
    kept = filter_points_to_training_images(points_full, train_ids)
    print(f"[spinnerf] leak fix: kept {len(kept)}/{len(points_full)} points "
          f"(dropped {len(points_full) - len(kept)} seen by a held-out view)")
    write_points3D_bin_full(sparse_dir / "points3D.bin", kept)

    kept_ids = set(kept)
    for img in images_full.values():
        img["point3D_ids"] = [pid if pid in kept_ids else -1 for pid in img["point3D_ids"]]
    write_images_bin_full(sparse_dir / "images.bin", images_full)

    with open(sparse_dir / "test.txt", "w") as f:
        f.writelines(f"{n}\n" for n in test_names)
    print(f"[spinnerf] wrote {sparse_dir / 'test.txt'} ({len(test_names)} held-out views)")


# =====================================================================
# Stages
# =====================================================================
def run_sfm(args, images_dir: Path, sfm_dir: Path):
    from sfm import COLMAPWrapper
    wrapper = COLMAPWrapper({"matcher": args.matcher, "camera_model": args.camera_model,
                             "max_features": args.max_features})
    wrapper.run_sfm(str(images_dir), str(sfm_dir), args.max_features)


def run_segmentation(args, images_dir: Path, seg_dir: Path) -> dict:
    from segmentation_timed import SAM2SegmenterTimed
    segmenter = SAM2SegmenterTimed({
        "config_file": args.sam_config,
        "model_path": args.sam_checkpoint,
        "model_type": "hiera_large",
        "device": "cuda",
        "first_frame_idx": args.seg_first_frame_idx,
    })
    results = segmenter.segment_scene(str(images_dir), str(seg_dir))
    return results.get("timing", {})


def run_mask_refinement(args, images_dir: Path, seg_dir: Path, refined_dir: Path) -> dict:
    """Depth-based mask refinement (mask_refinement.py), on full-resolution
    photos. The four --mask-* options were chosen per scene (README); every
    other setting is fixed."""
    import mask_refinement as dmg
    from mask_refinement_timed import run_timed
    argv = [
        "--image_dir", str(images_dir),
        "--coarse_mask_dir", str(seg_dir / "masks"),
        "--output_dir", str(refined_dir),
        "--sam2_checkpoint", args.sam_checkpoint,
        "--grid_size", str(args.mask_grid_size),
        "--knn_allow_growth_outside_sam2",
        "--knn_growth_max_distance", "20",
        "--knn_growth_require_color_match",
        "--knn_growth_color_fg_threshold", "0.55",
        "--enable_ridge_fill",
        "--enable_sam2_reprompt",
        "--prev_mask_window", "5",
        "--sam2_reprompt_trigger_mode", "warp_disagreement",
        "--sam2_reprompt_min_warp_iou", "0.85",
        "--sam2_reprompt_min_warp_coverage", "0.75",
        "--sam2_reprompt_min_missing_fraction", str(args.mask_reprompt_missing_fraction),
        "--sam2_reprompt_current_mask_erode", "5",
        "--sam2_reprompt_pos_points", "12",
        "--sam2_reprompt_prev_area_strategy", "max",
        "--sam2_reprompt_max_search_radius", "250",
        "--sam2_reprompt_depth_filter_mode", "quantile",
        "--sam2_reprompt_depth_quantile_low", "2.0",
        "--sam2_reprompt_depth_quantile_high", str(args.mask_depth_quantile_high),
        "--sam2_reprompt_donor_pos_points", "6",
        "--sam2_reprompt_donor_oversample_factor", "5",
        "--sam2_reprompt_template_score_threshold", "0.7",
        "--sam2_reprompt_max_area_epsilon", "0.15",
        "--overwrite",
    ]
    if args.mask_edge_restoration:
        argv += ["--enable_edge_restoration"]
    if args.save_mask_debug:
        argv += ["--save_debug", "--save_sam2_debug"]
    report = run_timed(dmg.build_parser().parse_args(argv))
    return {"mask_refinement_sec": report["total_sec"]}


def run_point_labeling(seg_dir: Path, sfm_dir: Path, point_cloud_dir: Path,
                       use_refined_masks: bool = False):
    """Labels each COLMAP point by voting over the 2D masks of every view
    that sees it. Uses the raw SAM2 masks, or preprocessing/refined/
    refined_masks with use_refined_masks (--label-with-refined-masks)."""
    from sfm import COLMAPWrapper
    from point_labeling import PointLabeler
    model_dir = sfm_dir / "sparse" / "0"
    wrapper = COLMAPWrapper()
    colmap_results = {
        "cameras": wrapper._read_cameras(model_dir / "cameras.bin"),
        "images": wrapper._read_images(model_dir / "images.bin"),
        "points3D": wrapper._read_points3D(model_dir / "points3D.bin"),
    }
    labeler = PointLabeler()
    labeled = labeler.label_points(colmap_results, str(seg_dir),
                                   use_refined_masks=use_refined_masks)
    labeler.save_labeled_points(labeled, point_cloud_dir)


def run_point_refinement(sfm_dir: Path, point_cloud_dir: Path, method: str = "knn",
                         knn_threshold: float = 0.9, mean_spacing: float = None):
    """Relabels outvoted points (method 'knn': KNN voting, as in the paper;
    'dbscan': DBSCAN + radius voting) and writes point_cloud/pcd_data/sparse/0/
    with the COLMAP model alongside."""
    from point_refinement import PointCloudRefiner
    PointCloudRefiner({"method": method, "knn_threshold": knn_threshold,
                       "mean_spacing": mean_spacing}).refine_labels(
        point_cloud_dir / "labeled_point_cloud.ply",
        point_cloud_dir / "labeled_point_cloud_metadata.json",
        point_cloud_dir, sfm_dir)


def copy_labels_without_refinement(sfm_dir: Path, point_cloud_dir: Path):
    """--skip-point-refinement: pcd_data gets the labeled points unchanged."""
    dest = point_cloud_dir / "pcd_data" / "sparse" / "0"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(point_cloud_dir / "labeled_point_cloud.ply", dest / "points3D.ply")
    shutil.copy2(point_cloud_dir / "labeled_point_cloud_metadata.json",
                 dest / "points3D_metadata.json")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(sfm_dir / "sparse" / "0" / name, dest / name)


# =====================================================================
# Main
# =====================================================================
def build_parser():
    p = argparse.ArgumentParser(description="RigidGS preprocessing (everything before training).")
    p.add_argument("--images", required=True, help="Folder of input photos.")
    p.add_argument("--output", required=True,
                   help="Scene output folder, e.g. output/llff/fern.")

    p.add_argument("--skip-sfm", action="store_true", help="Reuse <output>/sfm.")
    p.add_argument("--skip-segmentation", action="store_true",
                   help="Reuse <output>/preprocessing/segmentation.")
    p.add_argument("--skip-mask-refinement", action="store_true",
                   help="Reuse <output>/preprocessing/refined.")
    p.add_argument("--no-mask-refinement", action="store_true",
                   help="Do not refine the SAM2 masks: training uses the raw masks. For "
                        "scenes where SAM2 is already accurate (SPIn-NeRF).")
    p.add_argument("--label-with-refined-masks", dest="label_with_refined_masks",
                   action="store_true", default=None,
                   help="Vote the 3D point labels with the refined masks instead of the raw "
                        "SAM2 masks. Helps when SAM2 misses large parts of the object; hurts "
                        "thin structures such as fern fronds. Default: on for Tanks and "
                        "Temples (an --images path under tnt/, tandt/ or tanks*/), off "
                        "otherwise.")
    p.add_argument("--label-with-sam2-masks", dest="label_with_refined_masks",
                   action="store_false",
                   help="Vote the 3D point labels with the raw SAM2 masks, also for Tanks "
                        "and Temples.")
    p.add_argument("--skip-labeling", action="store_true",
                   help="Reuse <output>/point_cloud/labeled_point_cloud.ply.")
    p.add_argument("--skip-point-refinement", action="store_true",
                   help="Copy the labeled points into pcd_data unchanged.")
    p.add_argument("--point-refinement", choices=["knn", "dbscan"], default="knn",
                   help="Point-label refinement method. knn (default, used for the paper): "
                        "a point takes the label of its 40 nearest neighbours when 90%% of "
                        "them agree. dbscan: per-object DBSCAN (scattered points -> "
                        "background) followed by radius voting; more aggressive, and on "
                        "thin structures (fern fronds) it moves object points to background.")
    p.add_argument("--knn-threshold", type=float, default=0.9,
                   help="knn: fraction of the 40 neighbours that must agree to relabel a point.")
    p.add_argument("--mean-spacing", type=float, default=None,
                   help="dbscan: fixed mean point spacing (scene units). By default it is "
                        "estimated from a random sample of points; the paper's dbscan scenes "
                        "used an unseeded sample, so their configs give the value it produced.")

    p.add_argument("--spinnerf", action="store_true",
                   help="SPIn-NeRF scene: --images is <scene>/images of the raw dataset, "
                        "with <scene>/sparse/0 and <scene>/images_4/label next to it.")

    p.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    p.add_argument("--camera-model", default="PINHOLE")
    p.add_argument("--max-features", type=int, default=8192)

    p.add_argument("--sam-checkpoint", default="external/sam2/checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--seg-first-frame-idx", type=int, default=None,
                   help="Frame shown for the interactive SAM2 clicks. Default 0, or 40 "
                        "with --spinnerf (its first 40 frames have no object).")
    p.add_argument("--mask-grid-size", type=int, default=1,
                   help="Mask refinement splits the object's bounding box into an N x N grid "
                        "and thresholds depth in each cell separately. 1 = one threshold for "
                        "the whole object. Larger grids remove background trapped between "
                        "thin parts (fern fronds) but can cut into compact objects. "
                        "Per-scene values are in the README.")
    p.add_argument("--mask-edge-restoration", action="store_true",
                   help="Restore boundary pixels the depth threshold removed.")
    p.add_argument("--mask-reprompt-missing-fraction", type=float, default=0.10,
                   help="Re-prompt SAM2 on a frame when at least this fraction of the mask "
                        "warped from neighbouring frames is missing from SAM2's mask.")
    p.add_argument("--mask-depth-quantile-high", type=float, default=99.99,
                   help="Upper depth quantile (percent) for re-prompt candidates; lower "
                        "values drop more far-away background.")
    p.add_argument("--save-mask-debug", action="store_true",
                   help="Write mask-refinement debug images.")
    return p


def is_tanks_and_temples(images_dir: Path) -> bool:
    """True for an --images path inside a Tanks and Temples folder
    (data/tnt/..., data/tandt/..., data/tanks_and_temples/...)."""
    parts = [x.lower() for x in images_dir.absolute().parts]
    return any(x in ("tnt", "tandt") or x.startswith("tanks") for x in parts)


def main():
    args = build_parser().parse_args()
    out = Path(args.output)
    sfm_dir = out / "sfm"
    seg_dir = out / "preprocessing" / "segmentation"
    refined_dir = out / "preprocessing" / "refined"
    point_cloud_dir = out / "point_cloud"
    for d in (sfm_dir, seg_dir.parent, point_cloud_dir):
        d.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)
    if args.label_with_refined_masks is None:
        args.label_with_refined_masks = (is_tanks_and_temples(images_dir)
                                         and not args.no_mask_refinement)
    print(f"Point labels: voted with the "
          f"{'refined' if args.label_with_refined_masks else 'raw SAM2'} masks")
    if args.seg_first_frame_idx is None:
        args.seg_first_frame_idx = 40 if args.spinnerf else 0

    timing_path = out / "preprocessing_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    def timed(key, fn, *a):
        t0 = time.perf_counter()
        extra = fn(*a)
        timing[key] = round(time.perf_counter() - t0, 3)
        if isinstance(extra, dict):
            timing.update(extra)
        timing_path.write_text(json.dumps(timing, indent=2))
        _cleanup_memory()

    if args.spinnerf:
        scene_root = images_dir.parent
        images_dir = spinnerf_undistort(scene_root, out / "images_undistorted")

    if not args.skip_sfm:
        _stage("STEP 1: SfM (COLMAP)")
        if sfm_dir.exists():
            shutil.rmtree(sfm_dir)
        timed("sfm_sec", run_sfm, args, images_dir, sfm_dir)
        if args.spinnerf:
            spinnerf_fix_point_cloud_leak(sfm_dir / "sparse" / "0", scene_root)
    if not (sfm_dir / "sparse" / "0" / "points3D.bin").exists():
        sys.exit(f"No COLMAP model at {sfm_dir / 'sparse' / '0'}")

    if not args.skip_segmentation:
        _stage("STEP 2: SAM2 segmentation (interactive)")
        timed("segmentation_sec", run_segmentation, args, images_dir, seg_dir)

    if args.no_mask_refinement and args.skip_mask_refinement:
        sys.exit("--no-mask-refinement and --skip-mask-refinement contradict each other.")
    if args.no_mask_refinement and args.label_with_refined_masks:
        sys.exit("--label-with-refined-masks needs refined masks, but --no-mask-refinement is set.")
    if args.no_mask_refinement:
        pass                            # labeling and training both use the raw masks
    elif args.skip_mask_refinement:
        if not (refined_dir / "refined_masks").is_dir():
            sys.exit(f"--skip-mask-refinement, but there are no refined masks in {refined_dir}.")
    else:
        _stage("STEP 3: depth-based mask refinement")
        timed("mask_refinement_total_sec", run_mask_refinement,
              args, images_dir, seg_dir, refined_dir)

    if not args.skip_labeling:
        _stage("STEP 4: 3D point labeling")
        timed("point_labeling_sec", run_point_labeling, seg_dir, sfm_dir, point_cloud_dir,
              args.label_with_refined_masks)

    pcd_sparse = point_cloud_dir / "pcd_data" / "sparse" / "0"
    if pcd_sparse.exists():
        shutil.rmtree(pcd_sparse)       # always rebuilt from the current labels
    if args.skip_point_refinement:
        copy_labels_without_refinement(sfm_dir, point_cloud_dir)
    else:
        _stage(f"STEP 5: point-label refinement ({args.point_refinement})")
        timed("point_refinement_sec", run_point_refinement, sfm_dir, point_cloud_dir,
              args.point_refinement, args.knn_threshold, args.mean_spacing)

    if args.spinnerf:
        shutil.copy2(sfm_dir / "sparse" / "0" / "test.txt", pcd_sparse / "test.txt")

    masks_dir = (seg_dir / "masks") if args.no_mask_refinement else (refined_dir / "refined_masks")
    # --eval holds out every 8th view (llffhold 8) for evaluation; SPIn-NeRF
    # instead holds out the 40 no-object views listed in sparse/0/test.txt.
    split = " --eval --llffhold 0" if args.spinnerf else " --eval"
    print("\n" + "=" * 80 + "\nPREPROCESSING COMPLETE\n" + "=" * 80)
    print(f"Timing: {timing_path}")
    print("Train with:")
    print(f"  python src/train.py -s {point_cloud_dir / 'pcd_data'} -m {out / 'rigidgs'} \\\n"
          f"    --_images {images_dir} --masks_dir {masks_dir}{split} \\\n"
          f"    --inpaint_at_iter 30000 "
          f"--refine_after_inpaint --refine_iterations 7000")


if __name__ == "__main__":
    main()
