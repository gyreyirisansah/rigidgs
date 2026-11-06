"""
RigidGS evaluation metrics.

Provides three metrics for object-aware 3DGS evaluation:
    - IoU              : binary mask agreement between rendered FG and GT mask
    - BG-PSNR          : PSNR on the BG-only render restricted to GT-background pixels
    - FG-LPIPS         : LPIPS on the FG-only render restricted to GT-foreground pixels

Also provides:
    - Boundary F-score : sharpness of FG boundary vs GT boundary
    - Foreground leakage (Obj -> BG): fraction of predicted FG pixels outside GT FG
    - FG-PSNR / BG-LPIPS as additional convenience metrics

All metrics operate on numpy arrays in HxW or HxWx3 format.
RGB images are float32 in [0, 1]. Masks are bool or {0, 1} uint8.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim
from scipy.ndimage import binary_dilation, binary_erosion


# ----------------------------------------------------------------------------
# Lazy global LPIPS model (loaded once on first use)
# ----------------------------------------------------------------------------
_LPIPS_MODEL = None
_LPIPS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_lpips_model():
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS metrics require the optional 'lpips' package. "
                "Install it or run the evaluator with --skip_lpips."
            ) from exc

        # 'alex' is the standard backbone used in 3DGS papers
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(_LPIPS_DEVICE).eval()
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad_(False)
    return _LPIPS_MODEL


# ----------------------------------------------------------------------------
# Input validation and coercion
# ----------------------------------------------------------------------------
def _as_float_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce HxWx3 RGB to float32 in [0, 1]."""
    assert img.ndim == 3 and img.shape[-1] == 3, f"Expected HxWx3, got {img.shape}"
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Coerce HxW mask to bool."""
    if mask.ndim == 3:
        # Allow HxWx1 or single-channel HxWx3 by collapsing
        mask = mask[..., 0]
    assert mask.ndim == 2, f"Expected HxW mask, got {mask.shape}"
    return mask.astype(bool)


# ----------------------------------------------------------------------------
# Mask metrics
# ----------------------------------------------------------------------------
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection-over-Union between predicted FG mask and GT FG mask.

    Args:
        pred_mask : HxW binary mask rendered from RigidGS foreground subset
                    (True / 1 = predicted foreground)
        gt_mask   : HxW binary GT mask (True / 1 = ground-truth foreground)

    Returns:
        IoU in [0, 1]. Returns NaN if union is empty.
    """
    p = _as_bool_mask(pred_mask)
    g = _as_bool_mask(gt_mask)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def compute_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Pixel-wise accuracy: (TP + TN) / total, over the full frame.

    Unlike IoU, this counts correct background agreement too, so it's
    dominated by the (usually large) true-background region -- report it
    alongside IoU/boundary_f, not as a replacement for either.
    """
    p = _as_bool_mask(pred_mask)
    g = _as_bool_mask(gt_mask)
    correct = (p == g).sum()
    return float(correct) / float(g.size)


def compute_foreground_leakage(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Foreground leakage: fraction of predicted FG pixels that lie outside GT FG.

    L_{obj -> bg} = |P intersect (not G)| / |P|

    Lower is better. NaN if no predicted FG pixels.
    """
    p = _as_bool_mask(pred_mask)
    g = _as_bool_mask(gt_mask)
    if p.sum() == 0:
        return float("nan")
    return float(np.logical_and(p, ~g).sum()) / float(p.sum())


def compute_boundary_fscore(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: int = 2,
) -> float:
    """
    Boundary F-score with `tolerance`-pixel slack on both sides.

    Boundaries are extracted as the pixel-thin contour of each mask (the XOR
    between the mask and its 1-pixel erosion). A predicted boundary pixel is
    counted as TP if it lies within `tolerance` pixels of any GT boundary pixel
    (and vice versa for recall).

    Standard choice in segmentation papers; matches Csurka et al. 2013.
    """
    p = _as_bool_mask(pred_mask)
    g = _as_bool_mask(gt_mask)

    # Extract 1-pixel-wide boundaries via mask XOR with its erosion.
    def _boundary(m):
        eroded = ~binary_dilation(~m, iterations=1)  # erosion by 1
        return m & ~eroded

    pb = _boundary(p)
    gb = _boundary(g)

    if pb.sum() == 0 and gb.sum() == 0:
        return float("nan")

    # Dilate the GT boundary by tolerance and count predictions that fall inside.
    gb_tol = binary_dilation(gb, iterations=tolerance)
    pb_tol = binary_dilation(pb, iterations=tolerance)

    if pb.sum() == 0:
        precision = 0.0
    else:
        precision = float(np.logical_and(pb, gb_tol).sum()) / float(pb.sum())
    if gb.sum() == 0:
        recall = 0.0
    else:
        recall = float(np.logical_and(gb, pb_tol).sum()) / float(gb.sum())

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_boundary_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation_ratio: float = 0.02,
) -> float:
    """
    Boundary IoU (Cheng, Girshick, Dollar, Berg, Kirillov -- "Boundary IoU:
    Improving Object-Centric Image Segmentation Evaluation", CVPR 2021).

    NOT the same metric as compute_boundary_fscore above. That's a
    precision/recall F-measure matching thin boundary contours within a
    tolerance band (standard in salient-object-detection evaluation).
    Boundary IoU instead restricts a standard IoU computation to a band
    near each mask's own boundary: erode each mask by `dilation` pixels,
    subtract to get a boundary strip (mask interior near the edge, not
    a symmetric contour), then IoU those two strips against each other.

    Args:
        pred_mask, gt_mask : HxW binary masks
        dilation_ratio : boundary strip width as a fraction of the image
                         diagonal (paper default 0.02 = 2%)

    Returns:
        Boundary IoU in [0, 1]. NaN if both boundary strips are empty.
    """
    p = _as_bool_mask(pred_mask)
    g = _as_bool_mask(gt_mask)
    H, W = g.shape
    diag = float(np.sqrt(H ** 2 + W ** 2))
    dilation = max(1, int(round(dilation_ratio * diag)))
    # 8-connected 3x3 square, matching eval_all_models.py's cv2.erode(...,
    # np.ones((3,3))) exactly -- scipy's default structure (when
    # unspecified) is a 4-connected cross, which erodes less aggressively
    # per iteration and gives measurably different boundary strips for the
    # same dilation count.
    _square3x3 = np.ones((3, 3), dtype=bool)

    def _boundary_strip(m):
        eroded = binary_erosion(m, structure=_square3x3, iterations=dilation, border_value=0)
        return m & ~eroded

    pb = _boundary_strip(p)
    gb = _boundary_strip(g)

    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


# ----------------------------------------------------------------------------
# RGB-space subset metrics
# ----------------------------------------------------------------------------
def _masked_psnr(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray) -> float:
    """
    PSNR computed only on pixels where `mask` is True.

    Args:
        img_a, img_b : HxWx3 RGB in [0, 1]
        mask         : HxW bool, True = include this pixel

    Returns:
        PSNR in dB. NaN if mask is empty or MSE is zero.
    """
    if mask.sum() == 0:
        return float("nan")
    a = img_a[mask]  # (N, 3)
    b = img_b[mask]
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def _masked_lpips(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    LPIPS on the masked region. Masked pixels are kept, unmasked pixels are
    zeroed in both images so the perceptual network sees the same "background"
    on both sides and reacts only to differences inside the region of interest.

    For LPIPS to be meaningful, the masked region should be reasonably
    spatially coherent (LPIPS uses a CNN, so isolated pixels won't be
    perceived correctly). For object segmentation this is satisfied.
    """
    if mask.sum() == 0:
        return float("nan")

    m3 = mask[..., None].astype(np.float32)  # HxWx1
    a = img_a * m3
    b = img_b * m3

    # LPIPS expects (B, 3, H, W) in [-1, 1].
    def _to_tensor(x):
        t = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)
        return (t * 2.0 - 1.0).to(_LPIPS_DEVICE)

    model = _get_lpips_model()
    with torch.no_grad():
        d = model(_to_tensor(a), _to_tensor(b))
    return float(d.item())


def _masked_ssim(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray) -> float:
    """
    SSIM computed by setting unmasked pixels to zero in both images and running
    standard SSIM. This is the convention used in SPIn-NeRF and related
    inpainting eval pipelines.
    """
    if mask.sum() == 0:
        return float("nan")
    m3 = mask[..., None].astype(np.float32)
    a = img_a * m3
    b = img_b * m3
    return float(_ssim(a, b, channel_axis=2, data_range=1.0))


# ----------------------------------------------------------------------------
# Public evaluation entry points
# ----------------------------------------------------------------------------
def evaluate_view(
    fg_render: np.ndarray,
    bg_render: np.ndarray,
    gt_image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None = None,
    boundary_tolerance: int = 2,
) -> dict:
    """
    Compute all per-view metrics for a single test view.

    Args:
        fg_render  : HxWx3 RGB, foreground-only render (background Gaussians removed,
                     e.g., made fully transparent or set to black)
        bg_render  : HxWx3 RGB, background-only render (foreground Gaussians removed)
        gt_image   : HxWx3 RGB, ground-truth image for this view
        gt_mask    : HxW   bool, ground-truth foreground mask
        pred_mask  : HxW   bool, predicted foreground mask. If None, it is
                     computed from fg_render by thresholding luminance > 0,
                     which works if your fg_render has the BG set to pure black.
                     For correctness it is safer to compute pred_mask by
                     rasterizing the foreground Gaussian subset with hard
                     opacity, separately from the RGB fg_render.
        boundary_tolerance : pixel tolerance for boundary F-score

    Returns:
        dict with keys:
            iou, boundary_f, boundary_iou, accuracy, leakage,
            fg_psnr, fg_ssim, fg_lpips,
            bg_psnr, bg_ssim, bg_lpips
    """
    fg_render = _as_float_rgb(fg_render)
    bg_render = _as_float_rgb(bg_render)
    gt_image = _as_float_rgb(gt_image)
    gt_mask = _as_bool_mask(gt_mask)

    if pred_mask is None:
        # Derive predicted FG mask from fg_render by checking which pixels are
        # actually contributed to (i.e., not background-zero). This assumes
        # fg_render has hard transparent / black background outside FG.
        pred_mask = fg_render.sum(axis=-1) > 1e-6
    else:
        pred_mask = _as_bool_mask(pred_mask)

    # Mask metrics
    iou = compute_iou(pred_mask, gt_mask)
    bf = compute_boundary_fscore(pred_mask, gt_mask, tolerance=boundary_tolerance)
    biou = compute_boundary_iou(pred_mask, gt_mask)
    acc = compute_accuracy(pred_mask, gt_mask)
    leak = compute_foreground_leakage(pred_mask, gt_mask)

    # FG-region RGB metrics (evaluated on the GT FG region)
    fg_psnr = _masked_psnr(fg_render, gt_image, gt_mask)
    fg_ssim = _masked_ssim(fg_render, gt_image, gt_mask)
    fg_lpips = _masked_lpips(fg_render, gt_image, gt_mask)

    # BG-region RGB metrics (evaluated on the GT BG region)
    bg_mask_gt = ~gt_mask
    bg_psnr = _masked_psnr(bg_render, gt_image, bg_mask_gt)
    bg_ssim = _masked_ssim(bg_render, gt_image, bg_mask_gt)
    bg_lpips = _masked_lpips(bg_render, gt_image, bg_mask_gt)

    return {
        "iou": iou,
        "boundary_f": bf,
        "boundary_iou": biou,
        "accuracy": acc,
        "leakage": leak,
        "fg_psnr": fg_psnr,
        "fg_ssim": fg_ssim,
        "fg_lpips": fg_lpips,
        "bg_psnr": bg_psnr,
        "bg_ssim": bg_ssim,
        "bg_lpips": bg_lpips,
    }


def aggregate(per_view_results: list[dict]) -> dict:
    """
    Aggregate per-view metric dicts into mean and std.

    Returns dict with keys like 'iou_mean', 'iou_std', etc.
    NaN values are ignored.
    """
    keys = per_view_results[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in per_view_results], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")
        else:
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_std"] = float(vals.std())
    return agg


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic test: known FG region in the center of a 256x256 image.
    H, W = 256, 256
    gt_image = np.random.RandomState(0).rand(H, W, 3).astype(np.float32)
    gt_mask = np.zeros((H, W), dtype=bool)
    gt_mask[80:176, 80:176] = True  # 96x96 square in the middle

    # Perfect FG render: copy GT inside mask, zero outside
    fg_render = gt_image * gt_mask[..., None]
    # Perfect BG render: copy GT outside mask, zero inside
    bg_render = gt_image * (~gt_mask)[..., None]

    results = evaluate_view(
        fg_render=fg_render,
        bg_render=bg_render,
        gt_image=gt_image,
        gt_mask=gt_mask,
    )

    print("Self-test (should give IoU=1.0, PSNRs high, leakage=0):")
    for k, v in results.items():
        print(f"  {k:12s} = {v:.4f}")

    # Imperfect FG render: shifted by 10 pixels (creates leakage and reduces IoU)
    shifted = np.roll(gt_mask, 10, axis=1)
    fg_render_bad = gt_image * shifted[..., None]
    bg_render_bad = gt_image * (~shifted)[..., None]

    results_bad = evaluate_view(
        fg_render=fg_render_bad,
        bg_render=bg_render_bad,
        gt_image=gt_image,
        gt_mask=gt_mask,
    )

    print("\nImperfect (shifted by 10px, IoU should drop, leakage should rise):")
    for k, v in results_bad.items():
        print(f"  {k:12s} = {v:.4f}")
