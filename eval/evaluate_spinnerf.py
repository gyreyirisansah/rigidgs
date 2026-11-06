"""
LPIPS + FID + PSNR/SSIM (+ outside-hole PSNR/SSIM) for SPIn-NeRF
object-removal evaluation.

Matches the SPIn-NeRF paper's own protocol exactly (Mirzaei et al., CVPR
2023, Sec. 5 "Metrics"): "Since our focus is the inpainting, we only
calculate the LPIPS and FID inside the bounding box of the object mask
(using our MVSeg model, we can render the object mask from the test views
that do not contain the object)." That's a rectangular crop to the mask's
bounding box per held-out view -- not full-frame, and not a pixel-precise
masked region either. The masks are data/ground_truth/masks/spinnerf/<scene>/{stem}.png,
the dataset's own per-test-view object masks (test_label/ in the raw
SPIn-NeRF data, staged by spinnerf_utils.py) -- exactly the
"object mask rendered from the test views that do not contain the object"
the paper describes.

This is deliberately different from segmentation_metrics.py's fg_lpips/bg_lpips
(masked by zeroing pixels, not cropped, and built for datasets where the
GT photo still shows the object). This script does not touch rigidgs_
metrics.py or evaluate_segmentation.py -- other datasets' established results
depend on their current behavior.

LPIPS is a per-view metric (mean/std over the 40 views, per scene). FID is
distributional, not pairwise: it compares the set of N render crops against
the set of N GT crops as two batches of Inception features (crops vary in
size across views since each bounding box is different -- fine, FID resizes
every image to its fixed input size internally). This script reports FID
per scene AND pooled across all requested scenes -- 40 crops is a small
sample for a stable Frechet distance estimate (the field computes it this
way anyway for these benchmarks), so the pooled number is worth reporting
alongside the per-scene ones for a less noisy read.

Input: background_hole_renders/iter_<N>/{stem}.png from render_spinnerf_
holdout_eval.py, matched by stem against data/ground_truth/images/spinnerf/<scene>/
and data/ground_truth/masks/spinnerf/<scene>/.

PSNR and SSIM (new) are computed on the SAME bbox crop as LPIPS/FID --
standard complementary fidelity/structure numbers, cheap to add alongside
the paper's own two metrics.

Outside-hole PSNR/SSIM (new) are a different check entirely, not part of
the paper's protocol: computed over the FULL-FRAME complement of the
object mask (not a padded box -- the whole rest of the scene), comparing
render against the "no object" ground truth photo there too. This exists
to directly validate an invariant this pipeline depends on but the paper's
own inside-the-box metrics can't see at all: that refining the hole didn't
degrade anything outside it. Should read as near-perfect (high PSNR, SSIM
near 1) -- a low value means something outside the removed object's own
mask changed, which is a bug (this project has hit exactly that failure
mode twice: a densify/prune step and an opacity-reset call that were each,
before being fixed, scoped to the whole scene instead of the hole).

Usage:
    python evaluate_spinnerf.py --scenes 1 2 3 --iteration 40000
    python evaluate_spinnerf.py --scenes 1 2 3   # auto-detect latest iter
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from segmentation_metrics import _get_lpips_model, _LPIPS_DEVICE
# stage3_refinement's own module-level sys.path.insert for
# external/gaussian-splatting runs as an import side effect, so this also
# makes `from utils.loss_utils import ssim` below resolve correctly.
from stage3_refinement import _ssim_map
from utils.loss_utils import ssim as _ssim_scalar

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_rgb01(path: Path) -> np.ndarray:
    """HxWx3 float32 in [0, 1]."""
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def _load_mask_bool(mask_path: Path) -> np.ndarray:
    """HxW boolean mask (True = object region)."""
    m = np.array(Image.open(mask_path).convert("L"))
    return m > 127 if m.max() > 1 else m > 0


def _mask_bbox_from_bool(m: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1) tight bounding box of the nonzero mask region, or
    None if the mask is empty."""
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _psnr(a01: np.ndarray, b01: np.ndarray) -> float:
    """PSNR between two [0,1] images (max_val=1, so 20*log10(1)=0 drops out)."""
    mse = float(np.mean((a01.astype(np.float64) - b01.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * np.log10(mse)


def _to_chw01_tensor(img01: np.ndarray) -> torch.Tensor:
    """HxWx3 [0,1] -> (1,3,H,W) [0,1] on the LPIPS device -- for SSIM, which
    (unlike LPIPS) expects [0,1], not [-1,1]."""
    return torch.from_numpy(img01).float().permute(2, 0, 1).unsqueeze(0).to(_LPIPS_DEVICE)


def _masked_ssim(pred01: np.ndarray, gt01: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean SSIM over the True pixels of `mask`, computed on the FULL frame
    (SSIM's 11x11 windowed formula needs real neighborhood context, unlike
    plain per-pixel PSNR/L1 -- cropping first would corrupt values right at
    the crop edge). Returns None if the mask is empty."""
    if not mask.any():
        return None
    pred_t = _to_chw01_tensor(pred01)
    gt_t = _to_chw01_tensor(gt01)
    smap = _ssim_map(pred_t, gt_t)[0].mean(dim=0)  # (H,W), channel-averaged
    mask_t = torch.from_numpy(mask).to(smap.device)
    return float(smap[mask_t].mean().item())


def _crop(img01: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return img01[y0:y1, x0:x1]


def _to_lpips_tensor(img01: np.ndarray) -> torch.Tensor:
    """HxWx3 [0,1] -> (1,3,H,W) [-1,1] on the LPIPS device. Same convention
    as segmentation_metrics._masked_lpips, just on a bbox crop instead of a
    zeroed-out full frame."""
    t = torch.from_numpy(img01).float().permute(2, 0, 1).unsqueeze(0)
    return (t * 2.0 - 1.0).to(_LPIPS_DEVICE)


def _find_latest_iter_dir(bg_hole_dir: Path) -> Path:
    iter_dirs = sorted(
        (d for d in bg_hole_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* subdirectories found under {bg_hole_dir}")
    return iter_dirs[-1]


def evaluate_scene(scene: str, model_path_template: str, gt_root: Path,
                    gt_mask_root: Path, iteration: int | None) -> dict:
    model_path = Path(model_path_template.format(scene=scene))
    bg_hole_dir = model_path / "spinnerf_holdout_eval" / "background_hole_renders"
    render_dir = (bg_hole_dir / f"iter_{iteration:05d}") if iteration is not None \
        else _find_latest_iter_dir(bg_hole_dir)

    gt_dir = gt_root / scene
    mask_dir = gt_mask_root / scene
    render_idx = {p.stem: p for p in render_dir.iterdir() if p.suffix.lower() == ".png"}
    gt_idx = {p.stem: p for p in gt_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    mask_idx = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"}

    common = sorted(set(render_idx) & set(gt_idx) & set(mask_idx))
    missing = (set(gt_idx) | set(render_idx)) - set(common)
    if missing:
        print(f"  [scene {scene}] WARNING: {len(missing)} views missing a render, "
              f"GT photo, or GT mask -- scoring the {len(common)} complete views only.")

    lpips_model = _get_lpips_model()
    lpips_scores, psnr_scores, ssim_scores = [], [], []
    outside_psnr_scores, outside_ssim_scores = [], []
    render_crops, gt_crops = [], []
    for stem in common:
        mask_full = _load_mask_bool(mask_idx[stem])
        bbox = _mask_bbox_from_bool(mask_full)
        if bbox is None:
            print(f"  [scene {scene}] WARNING: empty mask for {stem} -- skipping.")
            continue
        render_full = _load_rgb01(render_idx[stem])
        gt_full = _load_rgb01(gt_idx[stem])
        r = _crop(render_full, bbox)
        g = _crop(gt_full, bbox)
        if r.shape != g.shape:
            print(f"  [scene {scene}] WARNING: shape mismatch on {stem} "
                  f"(render {r.shape} vs GT {g.shape}) -- skipping.")
            continue

        # Inside the bbox: same paper-matching crop as LPIPS/FID, plus
        # PSNR/SSIM as standard complementary fidelity/structure numbers.
        with torch.no_grad():
            d = lpips_model(_to_lpips_tensor(r), _to_lpips_tensor(g))
        lpips_scores.append(float(d.item()))
        psnr_scores.append(_psnr(r, g))
        with torch.no_grad():
            ssim_scores.append(float(_ssim_scalar(_to_chw01_tensor(r), _to_chw01_tensor(g)).item()))
        render_crops.append(r)
        gt_crops.append(g)

        # Outside the object mask (full-frame complement, not a padded box
        # -- this is deliberately checking the WHOLE rest of the scene, not
        # just "near" it): did anything outside the hole get degraded?
        # Directly validates the "don't touch outside the hole" invariant
        # this whole pipeline depends on -- would have caught the scene-
        # wide opacity-reset/pruning bugs automatically.
        if render_full.shape == gt_full.shape:
            outside_mask = ~mask_full
            if outside_mask.any():
                outside_psnr_scores.append(_psnr(render_full[outside_mask], gt_full[outside_mask]))
                with torch.no_grad():
                    s = _masked_ssim(render_full, gt_full, outside_mask)
                if s is not None:
                    outside_ssim_scores.append(s)
        else:
            print(f"  [scene {scene}] WARNING: full-frame shape mismatch on {stem} "
                  f"(render {render_full.shape} vs GT {gt_full.shape}) -- "
                  f"skipping outside-hole metrics for this view.")

    def _mean_std(xs):
        arr = np.array(xs, dtype=np.float64)
        return ((float(arr.mean()), float(arr.std())) if len(arr) else (float("nan"), float("nan")))

    lpips_mean, lpips_std = _mean_std(lpips_scores)
    psnr_mean, psnr_std = _mean_std(psnr_scores)
    ssim_mean, ssim_std = _mean_std(ssim_scores)
    outside_psnr_mean, outside_psnr_std = _mean_std(outside_psnr_scores)
    outside_ssim_mean, outside_ssim_std = _mean_std(outside_ssim_scores)
    return {
        "scene": scene,
        "n_views": len(lpips_scores),
        "lpips_mean": lpips_mean, "lpips_std": lpips_std,
        "psnr_mean": psnr_mean, "psnr_std": psnr_std,
        "ssim_mean": ssim_mean, "ssim_std": ssim_std,
        "outside_psnr_mean": outside_psnr_mean, "outside_psnr_std": outside_psnr_std,
        "outside_ssim_mean": outside_ssim_mean, "outside_ssim_std": outside_ssim_std,
        "render_crops": render_crops,
        "gt_crops": gt_crops,
    }


def _fid_score(fid_metric, render_imgs: list, gt_imgs: list) -> float:
    # Crops vary in size across views -- FID's own transform resizes each
    # image individually to its fixed input size, so feed one at a time
    # rather than stacking into a single batch tensor.
    fid_metric.reset()
    with torch.no_grad():
        for g in gt_imgs:
            t = torch.from_numpy(g).float().permute(2, 0, 1).unsqueeze(0).to(_LPIPS_DEVICE)
            fid_metric.update(t, real=True)
        for r in render_imgs:
            t = torch.from_numpy(r).float().permute(2, 0, 1).unsqueeze(0).to(_LPIPS_DEVICE)
            fid_metric.update(t, real=False)
    return float(fid_metric.compute().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="SPIn-NeRF scene ids, e.g. 1 2 3.")
    parser.add_argument("--model_path_template", type=str,
                        default="output/spinnerf/{scene}/rigidgs",
                        help="Model path per scene; {scene} is substituted.")
    parser.add_argument("--gt_root", type=str, default="data/ground_truth/images/spinnerf")
    parser.add_argument("--gt_mask_root", type=str, default="data/ground_truth/masks/spinnerf")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Also write the per-scene and averaged results here.")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Which iter_<N> to evaluate. Default: auto-detect "
                             "the latest available per scene.")
    args = parser.parse_args()

    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(normalize=True).to(_LPIPS_DEVICE)

    gt_root = Path(args.gt_root)
    gt_mask_root = Path(args.gt_mask_root)
    results = []
    all_render_crops, all_gt_crops = [], []
    for scene in args.scenes:
        print(f"[scene {scene}] scoring...")
        res = evaluate_scene(scene, args.model_path_template, gt_root, gt_mask_root,
                              args.iteration)
        scene_fid = _fid_score(fid_metric, res["render_crops"], res["gt_crops"])
        res["fid"] = scene_fid
        all_render_crops.extend(res["render_crops"])
        all_gt_crops.extend(res["gt_crops"])
        results.append(res)
        print(f"[scene {scene}] n={res['n_views']}  "
              f"LPIPS={res['lpips_mean']:.4f} (+/- {res['lpips_std']:.4f})  "
              f"FID={scene_fid:.2f}  "
              f"PSNR={res['psnr_mean']:.2f} (+/- {res['psnr_std']:.2f})  "
              f"SSIM={res['ssim_mean']:.4f} (+/- {res['ssim_std']:.4f})")
        print(f"           outside-hole: "
              f"PSNR={res['outside_psnr_mean']:.2f} (+/- {res['outside_psnr_std']:.2f})  "
              f"SSIM={res['outside_ssim_mean']:.4f} (+/- {res['outside_ssim_std']:.4f})")

    pooled_fid = _fid_score(fid_metric, all_render_crops, all_gt_crops)

    print("\n" + "=" * 100)
    print(f"Inside object-mask bbox (paper protocol) + outside-hole (rest of scene, unmasked check)")
    print("-" * 100)
    print(f"{'scene':<8} {'n':>4} {'LPIPS':>9} {'FID':>8} {'PSNR':>8} {'SSIM':>8} "
          f"{'out-PSNR':>10} {'out-SSIM':>10}")
    for r in results:
        print(f"{r['scene']:<8} {r['n_views']:>4} {r['lpips_mean']:>9.4f} {r['fid']:>8.2f} "
              f"{r['psnr_mean']:>8.2f} {r['ssim_mean']:>8.4f} "
              f"{r['outside_psnr_mean']:>10.2f} {r['outside_ssim_mean']:>10.4f}")
    print("-" * 100)
    macro_lpips_mean = float(np.mean([r["lpips_mean"] for r in results]))
    macro_psnr_mean = float(np.mean([r["psnr_mean"] for r in results]))
    macro_ssim_mean = float(np.mean([r["ssim_mean"] for r in results]))
    macro_out_psnr_mean = float(np.mean([r["outside_psnr_mean"] for r in results]))
    macro_out_ssim_mean = float(np.mean([r["outside_ssim_mean"] for r in results]))
    print(f"{'macro avg':<8} {'':>4} {macro_lpips_mean:>9.4f} {pooled_fid:>8.2f} "
          f"{macro_psnr_mean:>8.2f} {macro_ssim_mean:>8.4f} "
          f"{macro_out_psnr_mean:>10.2f} {macro_out_ssim_mean:>10.4f}")
    print(f"  (FID pooled across all {sum(r['n_views'] for r in results)} views; "
          f"everything else macro-averaged per scene)")
    print("=" * 100)
    if args.output_json:
        keys = ("scene", "n_views", "lpips_mean", "lpips_std", "fid", "psnr_mean", "psnr_std",
                "ssim_mean", "ssim_std", "outside_psnr_mean", "outside_psnr_std",
                "outside_ssim_mean", "outside_ssim_std")
        out = {
            "per_scene": [{k: r[k] for k in keys if k in r} for r in results],
            "macro_avg": {"lpips": macro_lpips_mean, "psnr": macro_psnr_mean,
                          "ssim": macro_ssim_mean, "outside_psnr": macro_out_psnr_mean,
                          "outside_ssim": macro_out_ssim_mean},
            "fid_pooled": pooled_fid,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"Results written to {args.output_json}")
    print("outside-hole PSNR/SSIM should be very high (near-perfect) -- that's the "
          "'rest of the scene is untouched' check. A low value there means something "
          "outside the removed object's own mask changed, which shouldn't happen.")


if __name__ == "__main__":
    main()
