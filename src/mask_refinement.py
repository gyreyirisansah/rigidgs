#!/usr/bin/env python3
"""
SAM2 Reprompt Mask Refinement — v2 (multi-object)
==================================================

Extends v1 to handle multiple objects per view.

Naming convention
-----------------
  image_dir/    image_1.JPG  image_2.JPG  ...          (base frames, no suffix)
  coarse_mask_dir/
                image_1_bottle.png  image_1_speaker.png
                image_2_bottle.png  image_2_speaker.png  ...

Object names are auto-discovered by scanning coarse_mask_dir and stripping the
matching base-frame stem prefix.  Use --objects to restrict to a subset.

Single pass over frames — all objects are processed per frame so temporal
history (PrevMaskHistory) is threaded correctly for each object independently.

Outputs
-------
  output_dir/
    <object>/
      refined_masks/<mask_filename>   # drop-in replacement
      debug/<stem>/                   # only with --save_debug
        original_coarse.png
        refined_coarse.png
        side_by_side.png
        summary.json
      debug/<stem>/sam2/              # only with --save_sam2_debug
        prompt_points.png
        search_region.png
        candidate_region.png
        warped_prev_mask.png
        template_verification.png
      per_image.csv
      run_summary.json
    run_summary.json                  # combined, keyed by object then stem
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,
             "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128,
             "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256,
             "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384,
             "out_channels": [1536, 1536, 1536, 1536]},
}


# =====================================================================
# I/O helpers
# =====================================================================
def find_coarse_mask(coarse_mask_dir, image_stem):
    """Find the coarse mask file matching an image stem."""
    if coarse_mask_dir is None or not coarse_mask_dir.exists():
        return None
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        c = coarse_mask_dir / f"{image_stem}{ext}"
        if c.exists():
            return c
    matches = sorted(coarse_mask_dir.glob(f"{image_stem}_mask_*"))
    return matches[0] if matches else None


def list_images(image_dir):
    return sorted([p for p in image_dir.iterdir() if p.suffix in IMG_EXTS],
                  key=lambda p: p.name)


def discover_objects(image_stems, coarse_mask_dir):
    """Auto-discover object names from mask filenames.

    Masks are named {image_stem}_{object_name}.{ext}.
    We recover object_name by stripping the matching image_stem prefix.
    Ties (multiple image stems that are prefixes) are broken by preferring
    the longest matching stem so that e.g. image_10 beats image_1.
    """
    if coarse_mask_dir is None or not coarse_mask_dir.exists():
        return []

    sorted_stems = sorted(image_stems, key=len, reverse=True)
    mask_files = [p for p in coarse_mask_dir.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    objects = set()
    for mask_path in mask_files:
        stem = mask_path.stem
        for img_stem in sorted_stems:
            if stem.startswith(img_stem + "_"):
                obj_name = stem[len(img_stem) + 1:]
                if obj_name:
                    objects.add(obj_name)
                break

    return sorted(objects)


def bounding_box(mask_bool, pad_frac=0.1):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = mask_bool.shape
    pad_y = int((y1 - y0) * pad_frac)
    pad_x = int((x1 - x0) * pad_frac)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    return y0, y1 + 1, x0, x1 + 1


def make_strict_grid(crop_h, crop_w, grid_size):
    allowed = {1, 2, 3, 4, 5, 6, 7, 8}
    if grid_size not in allowed:
        raise ValueError(f"--grid_size must be one of {sorted(allowed)}. Got {grid_size}")
    ys = np.linspace(0, crop_h, grid_size + 1, dtype=int)
    xs = np.linspace(0, crop_w, grid_size + 1, dtype=int)
    cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = int(ys[i]), int(ys[i + 1])
            x0, x1 = int(xs[j]), int(xs[j + 1])
            if y1 > y0 and x1 > x0:
                cells.append((i, j, y0, y1, x0, x1))
    return cells


def dilate_mask(mask, iters):
    if iters <= 0:
        return mask.astype(bool)
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k, iterations=iters).astype(bool)


def erode_mask(mask, iters):
    if iters <= 0:
        return mask.astype(bool)
    k = np.ones((3, 3), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, iterations=iters).astype(bool)


def localize_rgb(rgb, mask, mode="blur_bg", mask_dilate=2, blur_ksize=41,
                 bg_value=0):
    if mode == "none":
        return rgb.copy()
    support = dilate_mask(mask, mask_dilate)
    out = rgb.copy()
    if mode == "black_bg":
        out[~support] = bg_value
        return out
    if mode == "mean_bg":
        if (~support).any():
            mean = rgb[~support].reshape(-1, 3).mean(axis=0)
        else:
            mean = rgb.reshape(-1, 3).mean(axis=0)
        out[~support] = mean.astype(np.uint8)
        return out
    if mode == "blur_bg":
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        k = max(k, 3)
        blurred = cv2.GaussianBlur(rgb, (k, k), 0)
        out[~support] = blurred[~support]
        return out
    if mode == "alpha":
        out[~support] = (0.25 * out[~support]).astype(np.uint8)
        return out
    if mode == "mask_only":
        out[~support] = bg_value
        return out
    raise ValueError(f"Unknown rgb localization mode: {mode}")


def lab_fg_bg_scores(rgb, fg_seed, bg_seed):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_pixels = lab[fg_seed]
    bg_pixels = lab[bg_seed]
    if len(fg_pixels) < 10 or len(bg_pixels) < 10:
        fg = fg_seed.astype(np.float32)
        return fg, 1.0 - fg
    fg_mean = fg_pixels.mean(axis=0)
    bg_mean = bg_pixels.mean(axis=0)
    fg_dist = np.linalg.norm(lab - fg_mean.reshape(1, 1, 3), axis=2)
    bg_dist = np.linalg.norm(lab - bg_mean.reshape(1, 1, 3), axis=2)
    fg_score = bg_dist / (fg_dist + bg_dist + 1e-6)
    bg_score = fg_dist / (fg_dist + bg_dist + 1e-6)
    return fg_score.clip(0, 1).astype(np.float32), bg_score.clip(0, 1).astype(np.float32)


def run_depth(model, bgr_image):
    with torch.inference_mode():
        depth = model.infer_image(bgr_image)
    return depth


def normalize_depth_global(depth, valid_mask=None, robust=True, lo_pct=2.0, hi_pct=98.0):
    d = depth.astype(np.float32)
    vals = d[valid_mask] if valid_mask is not None and valid_mask.any() else d.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(d, dtype=np.float32)
    if robust:
        lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    else:
        lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    return np.clip((d - lo) / (hi - lo), 0, 1).astype(np.float32)


def normalize_depth_cell(cell_depth, cell_mask, args):
    d = cell_depth.astype(np.float32)
    vals = d[cell_mask] if cell_mask.any() else d.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(d, dtype=np.float32)
    if args.soft_robust_normalize:
        lo, hi = np.percentile(vals, [args.soft_norm_lo, args.soft_norm_hi])
    else:
        lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-6:
        return np.zeros_like(d, dtype=np.float32)
    return np.clip((d - lo) / (hi - lo), 0, 1).astype(np.float32)


# =====================================================================
# KNN per-cell depth refinement
# =====================================================================
def percentile_to_alpha(percentile):
    p = np.asarray(percentile, dtype=np.float32)
    alpha = np.zeros_like(p, dtype=np.float32)
    m = p >= 50; alpha[m] = 0.0
    m = (p < 50) & (p >= 25); alpha[m] = 0.67 * (50.0 - p[m]) / 25.0
    m = (p < 25) & (p >= 15); alpha[m] = 0.67 + (1.04 - 0.67) * (25.0 - p[m]) / 10.0
    m = (p < 15) & (p >= 5);  alpha[m] = 1.04 + (1.64 - 1.04) * (15.0 - p[m]) / 10.0
    m = p < 5; alpha[m] = 1.64
    return alpha


def local_knn_soft_threshold(soft, mask, k=31, percentile=25.0, min_neighbors=20,
                              dynamic_percentile=False, dyn_min_percentile=15.0,
                              dyn_max_percentile=40.0, dyn_strength=1.0,
                              decision_region=None):
    k = int(k)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    soft = soft.astype(np.float32)
    m = mask.astype(np.float32)
    kernel = (k, k)
    count = cv2.boxFilter(m, ddepth=-1, ksize=kernel, normalize=False)
    sum_soft = cv2.boxFilter(soft * m, ddepth=-1, ksize=kernel, normalize=False)
    sum_sq = cv2.boxFilter((soft * soft) * m, ddepth=-1, ksize=kernel, normalize=False)
    count_safe = np.maximum(count, 1.0)
    mean = sum_soft / count_safe
    var = np.maximum(sum_sq / count_safe - mean * mean, 0.0)
    std = np.sqrt(var)
    vals = soft[mask]
    if vals.size:
        global_thresh = float(np.percentile(vals, percentile))
        global_mean = float(vals.mean())
        global_std = float(vals.std() + 1e-6)
    else:
        global_thresh = 1.0
        global_mean = 0.5
        global_std = 1.0
    if dynamic_percentile:
        z = (mean - global_mean) / global_std
        z = np.clip(z, -1.0, 1.0)
        center = float(percentile)
        half_range = 0.5 * (float(dyn_max_percentile) - float(dyn_min_percentile))
        percentile_map = center + dyn_strength * half_range * z
        percentile_map = np.clip(percentile_map, dyn_min_percentile, dyn_max_percentile)
    else:
        percentile_map = np.full_like(mean, float(percentile), dtype=np.float32)
    alpha = percentile_to_alpha(percentile_map)
    local_thresh = mean - alpha * std
    local_thresh[count < min_neighbors] = global_thresh

    if decision_region is None:
        decision_region = mask
    keep = (soft >= local_thresh) & decision_region
    return keep, local_thresh, percentile_map


def compute_knn_cell(cell_depth, cell_mask, cell_rgb, args):
    n_fg = int(cell_mask.sum())
    if n_fg < args.min_cell_fg_pixels:
        return cell_mask.copy(), {
            "reason": "too_few_fg",
            "fg_pixels": n_fg,
            "fraction_removed": 0.0,
            "fraction_added": 0.0,
        }

    depth_norm = normalize_depth_cell(cell_depth, cell_mask, args)
    soft = depth_norm if args.depth_keep == "greater" else (1.0 - depth_norm)
    soft = np.clip(soft, 0, 1)
    if args.soft_gamma != 1.0:
        soft = np.power(soft, args.soft_gamma)
    if args.soft_mask_multiply:
        soft = soft * cell_mask.astype(np.float32)

    if args.knn_allow_growth_outside_sam2 and n_fg >= max(args.min_cell_fg_pixels, 50):
        dilated = dilate_mask(cell_mask, args.knn_growth_max_distance)
        decision_region = cell_mask | (dilated & (~cell_mask))
    else:
        decision_region = cell_mask

    cell_keep, _, _ = local_knn_soft_threshold(
        soft=soft, mask=cell_mask,
        k=args.knn_window, percentile=args.knn_percentile,
        min_neighbors=args.knn_min_neighbors,
        dynamic_percentile=args.knn_dynamic_percentile,
        dyn_min_percentile=args.knn_dyn_min_percentile,
        dyn_max_percentile=args.knn_dyn_max_percentile,
        dyn_strength=args.knn_dyn_strength,
        decision_region=decision_region,
    )

    added_outside = cell_keep & (~cell_mask)
    if added_outside.any() and args.knn_allow_growth_outside_sam2:
        if args.knn_growth_require_color_match and cell_mask.any():
            fg_seed = erode_mask(cell_mask, args.knn_growth_color_fg_erode)
            if fg_seed.sum() < 10:
                fg_seed = cell_mask.copy()
            bg_seed = ~dilate_mask(cell_mask, args.knn_growth_color_bg_dilate)
            if bg_seed.sum() < 10:
                bg_seed = ~cell_mask
            fg_score, _ = lab_fg_bg_scores(cell_rgb, fg_seed, bg_seed)
            color_ok = fg_score >= args.knn_growth_color_fg_threshold
            cell_keep = (cell_keep & cell_mask) | (added_outside & color_ok)
            added_outside = cell_keep & (~cell_mask)

        if args.knn_growth_min_component_area > 1 and added_outside.any():
            num, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
                added_outside.astype(np.uint8), 8)
            filtered = np.zeros_like(added_outside, dtype=bool)
            anchor = dilate_mask(cell_mask, 1)
            for lab in range(1, num):
                comp = labels == lab
                area = int(stats_cc[lab, cv2.CC_STAT_AREA])
                if area < args.knn_growth_min_component_area:
                    continue
                if not (comp & anchor).any():
                    continue
                filtered |= comp
            cell_keep = (cell_keep & cell_mask) | filtered

    remove = cell_mask & (~cell_keep)
    added = cell_keep & (~cell_mask)
    return cell_keep, {
        "reason": "depth_knn_soft_mask",
        "fg_pixels": n_fg,
        "growth_enabled": bool(args.knn_allow_growth_outside_sam2),
        "n_removed_from_sam2": int(remove.sum()),
        "n_added_outside_sam2": int(added.sum()),
        "fraction_removed": float(remove.sum() / max(1, n_fg)),
        "fraction_added": float(added.sum() / max(1, n_fg)),
    }


# =====================================================================
# Ridge fill
# =====================================================================
def apply_ridge_fill(rgb_full, full_depth_norm, refined_full, args):
    if not args.enable_ridge_fill or full_depth_norm is None or refined_full.sum() == 0:
        return refined_full, np.zeros_like(refined_full, dtype=bool)

    search_region = dilate_mask(refined_full, args.ridge_fill_search_dilate)

    if args.ridge_fill_depth_blur_sigma > 0:
        k = int(max(3, 2 * round(args.ridge_fill_depth_blur_sigma * 2) + 1))
        depth_blurred = cv2.GaussianBlur(full_depth_norm.astype(np.float32),
                                          (k, k), args.ridge_fill_depth_blur_sigma)
    else:
        k = 3
        depth_blurred = full_depth_norm.astype(np.float32)

    dgx = cv2.Sobel(depth_blurred, cv2.CV_32F, 1, 0, ksize=3)
    dgy = cv2.Sobel(depth_blurred, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad = np.sqrt(dgx * dgx + dgy * dgy)

    if args.ridge_fill_rgb_weight > 0:
        gray = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY)
        if args.ridge_fill_depth_blur_sigma > 0:
            gray = cv2.GaussianBlur(gray, (k, k), args.ridge_fill_depth_blur_sigma)
        rgx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        rgy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        rgb_grad = np.sqrt(rgx * rgx + rgy * rgy)
        depth_grad_n = normalize_depth_global(depth_grad, valid_mask=search_region, robust=True)
        rgb_grad_n = normalize_depth_global(rgb_grad, valid_mask=search_region, robust=True)
        ridge_strength = ((1.0 - args.ridge_fill_rgb_weight) * depth_grad_n
                          + args.ridge_fill_rgb_weight * rgb_grad_n)
    else:
        ridge_strength = normalize_depth_global(depth_grad, valid_mask=search_region, robust=True)

    ridge_strength_in_region = np.where(search_region, ridge_strength, 0.0).astype(np.float32)
    filled = (ridge_strength_in_region >= args.ridge_fill_threshold) & (~refined_full)

    if args.ridge_fill_max_fraction > 0:
        max_fill = int(args.ridge_fill_max_fraction * max(1, refined_full.sum()))
        if filled.sum() > max_fill:
            scores = ridge_strength_in_region[filled]
            if scores.size > max_fill:
                cutoff = np.partition(scores, max(0, scores.size - max_fill))[max(0, scores.size - max_fill)]
                filled = filled & (ridge_strength_in_region >= cutoff)

    return refined_full | filled, filled


# =====================================================================
# Edge restoration
# =====================================================================
def apply_edge_restoration(rgb_full, refined_full, args):
    if not args.enable_edge_restoration or refined_full.sum() == 0:
        return refined_full, np.zeros_like(refined_full, dtype=bool)

    gray = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY)
    if args.edge_restore_blur_sigma > 0:
        k = int(max(3, 2 * round(args.edge_restore_blur_sigma * 2) + 1))
        gray = cv2.GaussianBlur(gray, (k, k), args.edge_restore_blur_sigma)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.sqrt(gx * gx + gy * gy)

    band_width = max(1, int(args.edge_band_width))
    outer = dilate_mask(refined_full, band_width)
    inner = erode_mask(refined_full, band_width)
    boundary_band = outer & (~inner)

    if boundary_band.sum() == 0:
        return refined_full, np.zeros_like(refined_full, dtype=bool)

    edge_in_band = np.where(boundary_band, edge_strength, 0.0)
    edge_norm = normalize_depth_global(
        edge_in_band, valid_mask=boundary_band, robust=True,
    )
    edge_line = boundary_band & (edge_norm >= args.edge_restore_threshold)
    if edge_line.sum() == 0:
        return refined_full, np.zeros_like(refined_full, dtype=bool)

    edge_dilate = max(0, int(args.edge_restore_dilate))
    edge_support = dilate_mask(edge_line, edge_dilate) if edge_dilate > 0 else edge_line
    candidates = edge_support & outer & (~refined_full)
    return refined_full | candidates, candidates


# =====================================================================
# Donor mask history
# =====================================================================
class PrevMaskHistory:
    """Sliding window of recent refined masks + RGB for SAM2 reprompt."""
    def __init__(self, window):
        self.window = max(1, int(window))
        self.entries = []

    def add(self, stem, refined_mask, coarse_mask, rgb=None, accepted=True, reason=""):
        if not accepted:
            return False
        entry = {
            "stem": stem,
            "area": int(refined_mask.sum()),
            "mask": refined_mask.astype(bool).copy(),
            "coarse_mask": coarse_mask.astype(bool).copy(),
            "history_reason": reason,
        }
        if rgb is not None:
            entry["rgb"] = rgb.copy()
        self.entries.append(entry)
        if len(self.entries) > self.window:
            self.entries.pop(0)
        return True


# =====================================================================
# Optical flow warping
# =====================================================================
def warp_prev_mask_to_current(prev_rgb, curr_rgb, prev_mask, args):
    if prev_rgb is None or curr_rgb is None or prev_mask is None:
        return None, None
    if prev_rgb.shape[:2] != curr_rgb.shape[:2] or prev_mask.shape != curr_rgb.shape[:2]:
        return None, None

    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)

    flow_cur_to_prev = cv2.calcOpticalFlowFarneback(
        curr_gray, prev_gray, None,
        float(args.flow_pyr_scale), int(args.flow_levels),
        int(args.flow_winsize), int(args.flow_iterations),
        int(args.flow_poly_n), float(args.flow_poly_sigma), 0,
    )

    h, w = curr_gray.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx + flow_cur_to_prev[..., 0].astype(np.float32)
    map_y = yy + flow_cur_to_prev[..., 1].astype(np.float32)

    warped = cv2.remap(
        prev_mask.astype(np.uint8) * 255, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 127

    if int(args.warp_close) > 0:
        k3 = np.ones((3, 3), np.uint8)
        warped = cv2.morphologyEx(
            warped.astype(np.uint8), cv2.MORPH_CLOSE, k3,
            iterations=int(args.warp_close),
        ).astype(bool)

    return warped, flow_cur_to_prev


def compute_mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter / max(1, union)), inter, union


# =====================================================================
# Per-candidate template verification
# =====================================================================
def verify_candidate_by_template(candidate_xy, current_rgb, donors_info, args):
    cx, cy = int(candidate_xy[0]), int(candidate_xy[1])
    radius = int(args.sam2_reprompt_template_radius)
    search_radius = int(args.sam2_reprompt_template_search_radius)
    score_threshold = float(args.sam2_reprompt_template_score_threshold)

    H, W = current_rgb.shape[:2]
    ty0, ty1 = max(0, cy - radius), min(H, cy + radius + 1)
    tx0, tx1 = max(0, cx - radius), min(W, cx + radius + 1)
    template = current_rgb[ty0:ty1, tx0:tx1]
    if template.shape[0] < 3 or template.shape[1] < 3:
        return False, {"reason": "template_too_small"}

    for d_idx, donor_info in enumerate(donors_info):
        flow = donor_info.get("flow")
        donor_rgb = donor_info.get("donor_rgb")
        donor_mask = donor_info.get("donor_refined_mask")
        if donor_rgb is None or donor_mask is None or flow is None:
            continue

        fx = float(flow[cy, cx, 0]) if 0 <= cy < flow.shape[0] and 0 <= cx < flow.shape[1] else 0.0
        fy = float(flow[cy, cx, 1]) if 0 <= cy < flow.shape[0] and 0 <= cx < flow.shape[1] else 0.0
        src_x = int(round(cx + fx))
        src_y = int(round(cy + fy))

        sx0 = max(0, src_x - search_radius)
        sx1 = min(donor_rgb.shape[1], src_x + search_radius + 1)
        sy0 = max(0, src_y - search_radius)
        sy1 = min(donor_rgb.shape[0], src_y + search_radius + 1)
        search_region = donor_rgb[sy0:sy1, sx0:sx1]

        if (search_region.shape[0] < template.shape[0]
                or search_region.shape[1] < template.shape[1]):
            continue

        try:
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        if result.size == 0:
            continue
        best_score = float(result.max())
        if best_score < score_threshold:
            continue

        best_yx = np.unravel_index(int(result.argmax()), result.shape)
        match_y_donor = sy0 + int(best_yx[0]) + (template.shape[0] // 2)
        match_x_donor = sx0 + int(best_yx[1]) + (template.shape[1] // 2)
        match_y_donor = max(0, min(donor_mask.shape[0] - 1, match_y_donor))
        match_x_donor = max(0, min(donor_mask.shape[1] - 1, match_x_donor))

        if bool(donor_mask[match_y_donor, match_x_donor]):
            return True, {
                "donor_stem": donor_info.get("stem", "?"),
                "donor_index": int(d_idx),
                "score": float(best_score),
                "match_x_donor": int(match_x_donor),
                "match_y_donor": int(match_y_donor),
            }

    return False, {"reason": "no_confirming_donor"}


def sample_positive_points_from_mask(mask, n_points=12, min_dist_from_edge=10):
    mask = mask.astype(bool)
    if mask.sum() == 0 or n_points <= 0:
        return []

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    candidates = mask & (dist >= float(min_dist_from_edge))
    if candidates.sum() == 0:
        candidates = mask.copy()

    H, W = mask.shape
    grid = int(np.ceil(np.sqrt(max(1, n_points))))
    points = []

    for gy in range(grid):
        for gx in range(grid):
            if len(points) >= n_points:
                break
            y0 = int(gy * H / grid); y1 = int((gy + 1) * H / grid)
            x0 = int(gx * W / grid); x1 = int((gx + 1) * W / grid)
            cell = candidates[y0:y1, x0:x1]
            if cell.sum() == 0:
                continue
            cell_dist = dist[y0:y1, x0:x1]
            yy, xx = np.unravel_index(
                np.argmax(cell_dist * cell.astype(np.float32)), cell.shape)
            points.append((x0 + int(xx), y0 + int(yy)))
        if len(points) >= n_points:
            break

    if len(points) < n_points:
        ys, xs = np.where(candidates)
        vals = dist[ys, xs]
        order = np.argsort(-vals)
        existing = set(points)
        for idx in order:
            pt = (int(xs[idx]), int(ys[idx]))
            if pt in existing:
                continue
            points.append(pt)
            existing.add(pt)
            if len(points) >= n_points:
                break

    return points[:n_points]


def draw_prompt_debug(rgb, pos_points, neg_points):
    out = rgb.copy()
    for x, y in pos_points:
        cv2.circle(out, (int(x), int(y)), 7, (0, 255, 0), -1)
        cv2.circle(out, (int(x), int(y)), 9, (255, 255, 255), 2)
    for x, y in neg_points:
        cv2.circle(out, (int(x), int(y)), 7, (255, 0, 0), -1)
        cv2.circle(out, (int(x), int(y)), 9, (255, 255, 255), 2)
    return out


# =====================================================================
# SAM2 runner
# =====================================================================
class SAM2RepromptRunner:
    def __init__(self, checkpoint, model_cfg, device):
        if not checkpoint:
            raise ValueError("--sam2_checkpoint is required when "
                             "--enable_sam2_reprompt is used")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "Could not import SAM2. Install/activate SAM2 or add it to "
                "PYTHONPATH before using --enable_sam2_reprompt.") from e
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def predict(self, rgb, pos_points, neg_points, area_prior=None,
                multimask_output=True):
        if len(pos_points) == 0:
            return None, {"error": "no_positive_points"}
        pts = list(pos_points) + list(neg_points)
        labels = [1] * len(pos_points) + [0] * len(neg_points)
        point_coords = np.asarray(pts, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)

        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=bool(multimask_output),
        )
        masks = np.asarray(masks).astype(bool)
        scores = np.asarray(scores).astype(np.float32)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if scores.ndim == 0:
            scores = scores[None]

        if area_prior is not None and area_prior > 0:
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
            area_err = np.abs(areas - float(area_prior)) / max(1.0, float(area_prior))
            select_score = scores - 0.15 * area_err
            best = int(np.argmax(select_score))
        else:
            best = int(np.argmax(scores))

        info = {
            "scores": [float(x) for x in scores.tolist()],
            "selected_index": best,
            "selected_score": float(scores[best]),
            "selected_area": int(masks[best].sum()),
            "n_pos_points": int(len(pos_points)),
            "n_neg_points": int(len(neg_points)),
        }
        return masks[best].astype(bool), info


# =====================================================================
# SAM2 reprompt from donor history
# =====================================================================
def apply_sam2_reprompt_from_history(rgb_full, coarse_full, history, sam2_runner,
                                     args, full_depth_norm=None):
    H, W = coarse_full.shape
    empty = np.zeros((H, W), dtype=bool)
    stats = {"action": "disabled", "applied": False}

    if not args.enable_sam2_reprompt:
        return coarse_full, empty, [], [], stats, empty, empty
    if sam2_runner is None:
        return coarse_full, empty, [], [], {"action": "no_sam2_runner", "applied": False}, empty, empty
    if history is None or not history.entries:
        return coarse_full, empty, [], [], {"action": "no_history", "applied": False}, empty, empty

    donors = list(reversed(history.entries))
    per_donor_warps = []
    for d in donors:
        if "mask" not in d or "rgb" not in d or d["rgb"] is None:
            per_donor_warps.append({"stem": d.get("stem", "?"), "warped": empty.copy(),
                                     "success": False, "reason": "no_rgb_or_mask"})
            continue
        w, flow = warp_prev_mask_to_current(d["rgb"], rgb_full, d["mask"], args)
        if w is None:
            w = d["mask"].copy()
            flow = None
            ok = False
            reason = "warp_failed_used_unwarped"
        else:
            ok = True
            reason = "warped_ok"

        if int(args.warp_close) > 0:
            k3 = np.ones((3, 3), np.uint8)
            w = cv2.morphologyEx(w.astype(np.uint8), cv2.MORPH_CLOSE, k3,
                                  iterations=int(args.warp_close)).astype(bool)

        per_donor_warps.append({"stem": d.get("stem", "?"), "warped": w, "flow": flow,
                                 "donor_rgb": d.get("rgb"),
                                 "donor_refined_mask": d.get("mask"),
                                 "success": ok, "reason": reason,
                                 "donor_area": int(d.get("area", 0))})

    successful_donors = [p for p in per_donor_warps if p["success"]]
    successful_warps = [p["warped"] for p in successful_donors]
    n_warps = len(successful_warps)
    primary_warp = successful_warps[0] if n_warps >= 1 else per_donor_warps[0]["warped"]
    warped_area = int(primary_warp.sum())

    if n_warps >= 1:
        prev_areas = [int(d["donor_area"]) for d in successful_donors]
        strategy = str(args.sam2_reprompt_prev_area_strategy)
        if strategy == "last":
            prev_area = prev_areas[0]
        elif strategy == "median":
            prev_area = int(np.median(prev_areas))
        else:
            prev_area = int(np.max(prev_areas))
    else:
        prev_area = 0

    current_area = int(coarse_full.sum())
    iou, inter, _ = compute_mask_iou(coarse_full, primary_warp)
    current_over_warp = float(current_area / max(1, warped_area))
    warp_coverage = float(inter / max(1, warped_area))
    missing_from_current = int((primary_warp & (~coarse_full)).sum())
    missing_fraction = float(missing_from_current / max(1, warped_area))

    mode = str(args.sam2_reprompt_trigger_mode)
    if mode == "always":
        trigger = True
        trigger_reason = "always"
    elif mode == "area":
        trigger = current_over_warp < float(args.sam2_reprompt_min_area_ratio_to_warp)
        trigger_reason = "area_to_warp" if trigger else "area_ok"
    else:
        low_iou = iou < float(args.sam2_reprompt_min_warp_iou)
        low_cov = warp_coverage < float(args.sam2_reprompt_min_warp_coverage)
        enough_missing = missing_fraction > float(args.sam2_reprompt_min_missing_fraction)
        small_enough = current_over_warp < float(args.sam2_reprompt_max_current_to_warp_ratio_for_trigger)
        trigger = low_iou and low_cov and enough_missing and small_enough
        trigger_reason = ("warp_disagreement" if trigger else
                          f"not_needed(low_iou={low_iou},low_cov={low_cov},"
                          f"missing={enough_missing},small={small_enough})")

    stats = {
        "action": "checked",
        "applied": False,
        "trigger_mode": mode,
        "triggered": bool(trigger),
        "trigger_reason": trigger_reason,
        "n_donors_total": len(donors),
        "n_warps_successful": n_warps,
        "current_coarse_area": current_area,
        "primary_warped_area": warped_area,
        "prev_donor_area": prev_area,
        "warp_iou": float(iou),
        "warp_coverage": float(warp_coverage),
        "missing_from_current": int(missing_from_current),
        "missing_fraction": float(missing_fraction),
        "current_area_over_warped_area": float(current_over_warp),
    }

    if not trigger:
        stats["action"] = "not_collapsed"
        return coarse_full, primary_warp, [], [], stats, empty, empty

    import math
    if prev_area > 0:
        r_prev = math.sqrt(prev_area / math.pi)
        r_curr = math.sqrt(max(current_area, 1) / math.pi)
        R = max(0.0, r_prev - r_curr)
    else:
        R = 0.0

    max_radius_clamp = int(args.sam2_reprompt_max_search_radius)
    R_int = int(min(round(R), max_radius_clamp))
    stats["search_radius_raw"] = float(R)
    stats["search_radius_used"] = int(R_int)
    stats["search_radius_clamp"] = int(max_radius_clamp)

    if R_int <= 0:
        search_region = coarse_full.copy()
    else:
        search_region = dilate_mask(coarse_full, R_int)
    stats["search_region_area"] = int(search_region.sum())

    depth_mode = str(args.sam2_reprompt_depth_filter_mode)
    can_filter = (full_depth_norm is not None and current_area > 0
                  and depth_mode != "off")
    depth_band = np.ones_like(coarse_full, dtype=bool)
    stats["depth_filter_mode"] = depth_mode

    if can_filter:
        cur_depth_vals = full_depth_norm[coarse_full]
        if cur_depth_vals.size >= 10:
            q_low_pct = float(args.sam2_reprompt_depth_quantile_low)
            q_high_pct = float(args.sam2_reprompt_depth_quantile_high)
            depth_k = float(args.sam2_reprompt_depth_k_mad)
            mad_floor = float(args.sam2_reprompt_depth_mad_floor)
            range_pad = float(args.sam2_reprompt_depth_range_pad)

            q_low = float(np.percentile(cur_depth_vals, q_low_pct))
            q_high = float(np.percentile(cur_depth_vals, q_high_pct))
            d_med = float(np.median(cur_depth_vals))
            d_mad = max(float(np.median(np.abs(cur_depth_vals - d_med))), mad_floor)

            stats["depth_median"] = d_med
            stats["depth_mad"] = d_mad
            stats["depth_quantile_low"] = q_low
            stats["depth_quantile_high"] = q_high

            quantile_band = ((full_depth_norm >= (q_low - range_pad))
                              & (full_depth_norm <= (q_high + range_pad)))
            mad_band = np.abs(full_depth_norm - d_med) <= depth_k * d_mad

            if depth_mode == "quantile":
                depth_band = quantile_band
            elif depth_mode == "mad":
                depth_band = mad_band
            elif depth_mode == "both":
                depth_band = quantile_band | mad_band
            else:
                depth_band = quantile_band

            stats["depth_band_area"] = int(depth_band.sum())
            stats["depth_band_quantile_area"] = int(quantile_band.sum())
            stats["depth_band_mad_area"] = int(mad_band.sum())
        else:
            stats["depth_filter_skipped"] = "too_few_current_pixels"
    else:
        if full_depth_norm is None:
            stats["depth_filter_skipped"] = "no_depth_provided"
        elif depth_mode == "off":
            stats["depth_filter_skipped"] = "disabled"

    candidate_region = search_region & depth_band
    stats["candidate_region_area"] = int(candidate_region.sum())

    erode_iters = int(args.sam2_reprompt_current_mask_erode)
    min_eroded = int(args.sam2_reprompt_min_eroded_pixels)
    min_current = int(args.sam2_reprompt_min_current_mask_pixels)

    if current_area < min_current:
        stats["action"] = "current_mask_too_small_for_reprompt"
        return coarse_full, primary_warp, [], [], stats, search_region, candidate_region

    current_prompt_source = coarse_full.copy()
    if erode_iters > 0:
        eroded = erode_mask(coarse_full, erode_iters)
        if int(eroded.sum()) >= min_eroded:
            current_prompt_source = eroded
            stats["current_prompt_source"] = "eroded_current_coarse"
        else:
            stats["current_prompt_source"] = "uneroded_current_coarse_erosion_too_small"
    else:
        stats["current_prompt_source"] = "uneroded_current_coarse"

    current_pos_points = sample_positive_points_from_mask(
        current_prompt_source,
        n_points=int(args.sam2_reprompt_pos_points),
        min_dist_from_edge=float(args.sam2_reprompt_pos_min_dist),
    )

    donor_pos_n = int(args.sam2_reprompt_donor_pos_points)
    donor_min_area = int(args.sam2_reprompt_donor_min_recovery_area)
    donor_pos_points_raw = []
    if candidate_region.sum() >= donor_min_area and donor_pos_n > 0:
        oversample_factor = int(args.sam2_reprompt_donor_oversample_factor)
        n_raw = max(donor_pos_n, donor_pos_n * oversample_factor)
        donor_pos_points_raw = sample_positive_points_from_mask(
            candidate_region,
            n_points=n_raw,
            min_dist_from_edge=float(args.sam2_reprompt_donor_pos_min_dist),
        )

    donor_pos_points = []
    donor_pos_verification = []
    if donor_pos_points_raw and args.sam2_reprompt_donor_template_verify:
        for pt in donor_pos_points_raw:
            if len(donor_pos_points) >= donor_pos_n:
                break
            accepted, info = verify_candidate_by_template(
                candidate_xy=pt, current_rgb=rgb_full,
                donors_info=successful_donors, args=args,
            )
            info["candidate"] = [int(pt[0]), int(pt[1])]
            info["accepted"] = bool(accepted)
            donor_pos_verification.append(info)
            if accepted:
                donor_pos_points.append(pt)
    elif donor_pos_points_raw:
        donor_pos_points = donor_pos_points_raw[:donor_pos_n]

    pos_points = list(current_pos_points) + list(donor_pos_points)
    neg_points = []

    stats["n_current_pos_points"] = len(current_pos_points)
    stats["n_donor_pos_points_raw"] = len(donor_pos_points_raw)
    stats["n_donor_pos_points_accepted"] = len(donor_pos_points)
    stats["donor_pos_verification"] = donor_pos_verification

    if len(pos_points) == 0:
        stats["action"] = "no_positive_points"
        return coarse_full, primary_warp, [], [], stats, search_region, candidate_region

    sam2_mask, pred_info = sam2_runner.predict(
        rgb_full,
        pos_points=pos_points,
        neg_points=neg_points,
        area_prior=warped_area,
        multimask_output=bool(args.sam2_reprompt_multimask),
    )
    stats.update(pred_info)
    stats["pos_points"] = [[int(x), int(y)] for x, y in pos_points]
    stats["neg_points"] = []

    if sam2_mask is None or sam2_mask.sum() == 0:
        stats["action"] = "sam2_failed_empty"
        return coarse_full, primary_warp, pos_points, neg_points, stats, search_region, candidate_region

    new_iou, _, _ = compute_mask_iou(sam2_mask, primary_warp)
    new_area = int(sam2_mask.sum())
    stats["new_warp_iou"] = float(new_iou)
    stats["new_area"] = new_area
    stats["new_area_over_warped_area"] = float(new_area / max(1, warped_area))

    max_ratio = float(args.sam2_reprompt_max_area_ratio)
    if max_ratio > 0 and new_area > max_ratio * max(1, warped_area):
        stats["action"] = "sam2_rejected_area_too_large"
        return coarse_full, primary_warp, pos_points, neg_points, stats, search_region, candidate_region

    min_improvement = float(args.sam2_reprompt_min_iou_improvement)
    if new_iou + 1e-6 < iou + min_improvement:
        stats["action"] = "sam2_rejected_no_iou_improvement"
        return coarse_full, primary_warp, pos_points, neg_points, stats, search_region, candidate_region

    stats["action"] = "sam2_reprompted"
    stats["applied"] = True
    stats["new_coarse_area"] = new_area
    return sam2_mask.astype(bool), primary_warp, pos_points, neg_points, stats, search_region, candidate_region


# =====================================================================
# Debug visualization helpers
# =====================================================================
def make_side_by_side(rgb, original_mask, refined_mask):
    panel_orig = rgb.copy()
    panel_refi = rgb.copy()
    green = np.array([40, 220, 80], dtype=np.uint8)
    red = np.array([220, 40, 40], dtype=np.uint8)

    panel_orig[original_mask] = (0.45 * panel_orig[original_mask].astype(np.float32)
                                  + 0.55 * red).astype(np.uint8)
    panel_refi[refined_mask] = (0.45 * panel_refi[refined_mask].astype(np.float32)
                                  + 0.55 * green).astype(np.uint8)

    H, W = rgb.shape[:2]
    gap = 8
    canvas = np.full((H + 30, 2 * W + gap, 3), 20, dtype=np.uint8)
    canvas[30:, :W] = panel_orig
    canvas[30:, W + gap:] = panel_refi
    cv2.putText(canvas, "original coarse", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    cv2.putText(canvas, "refined", (W + gap + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    return canvas


# =====================================================================
# Per-image processing (single object)
# =====================================================================
def process_one_image(image_path, coarse_mask_path, model, args,
                       refined_masks_dir, debug_dir,
                       prev_history=None, sam2_runner=None,
                       running_state=None, rgb=None, bgr=None):
    """Process a single frame for one object.

    rgb/bgr may be passed in to avoid re-reading the file when multiple
    objects share the same base frame.
    """
    if running_state is None:
        running_state = {"max_observed_refined_area": 0}

    if bgr is None:
        bgr = cv2.imread(str(image_path))
    if bgr is None:
        return {"error": f"cv2 read failed {image_path}"}
    if rgb is None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    if coarse_mask_path is None:
        return {"error": "no coarse mask"}

    coarse = np.array(Image.open(coarse_mask_path).convert("L"))
    if coarse.shape != (H, W):
        coarse = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_NEAREST)
    coarse_bool_original = coarse > 127
    coarse_bool = coarse_bool_original.copy()

    need_full_depth = args.enable_ridge_fill or args.enable_sam2_reprompt
    full_depth_norm = None
    if need_full_depth:
        full_rgb_for_depth = rgb
        if args.full_depth_localize:
            full_rgb_for_depth = localize_rgb(
                rgb, coarse_bool, mode=args.full_depth_localization,
                mask_dilate=args.full_depth_mask_dilate,
                blur_ksize=args.full_depth_blur_ksize,
                bg_value=args.rgb_bg_value,
            )
        full_bgr_for_depth = cv2.cvtColor(full_rgb_for_depth, cv2.COLOR_RGB2BGR)
        full_depth = run_depth(model, full_bgr_for_depth)
        if full_depth.shape != (H, W):
            full_depth = cv2.resize(full_depth.astype(np.float32), (W, H),
                                     interpolation=cv2.INTER_LINEAR)
        full_depth_norm = normalize_depth_global(
            full_depth,
            valid_mask=coarse_bool if args.full_depth_normalize_inside_mask else None,
            robust=args.full_depth_robust_normalize,
            lo_pct=args.full_depth_norm_lo, hi_pct=args.full_depth_norm_hi,
        )

    sam2_warped_prev_mask = np.zeros((H, W), dtype=bool)
    sam2_search_region = np.zeros((H, W), dtype=bool)
    sam2_candidate_region = np.zeros((H, W), dtype=bool)
    sam2_pos_points = []
    sam2_neg_points = []
    sam2_reprompt_stats = {"action": "disabled", "applied": False}

    if args.enable_sam2_reprompt:
        (coarse_bool, sam2_warped_prev_mask, sam2_pos_points,
         sam2_neg_points, sam2_reprompt_stats,
         sam2_search_region, sam2_candidate_region) = apply_sam2_reprompt_from_history(
            rgb_full=rgb, coarse_full=coarse_bool,
            history=prev_history, sam2_runner=sam2_runner, args=args,
            full_depth_norm=full_depth_norm,
        )

        if (sam2_reprompt_stats.get("applied", False)
                and args.sam2_reprompt_enable_max_area_cap):
            new_area = int(coarse_bool.sum())
            max_area = int(running_state.get("max_observed_refined_area", 0))
            epsilon = float(args.sam2_reprompt_max_area_epsilon)
            if max_area > 0:
                cap = int(max_area * (1.0 + epsilon))
                if new_area > cap:
                    sam2_reprompt_stats["action"] = "sam2_reverted_area_cap"
                    sam2_reprompt_stats["applied"] = False
                    sam2_reprompt_stats["reverted_area_cap"] = {
                        "new_area": new_area,
                        "max_observed_refined_area": max_area,
                        "epsilon": epsilon,
                        "cap": cap,
                    }
                    coarse_bool = coarse_bool_original.copy()

    bbox_pad = args.bbox_pad
    if args.knn_allow_growth_outside_sam2:
        bbox_pad = max(bbox_pad, args.knn_growth_bbox_pad)
    bb = bounding_box(coarse_bool, pad_frac=bbox_pad)
    if bb is None:
        return {"error": "empty coarse mask after reprompt"}
    cy0, cy1, cx0, cx1 = bb

    crop_rgb = rgb[cy0:cy1, cx0:cx1]
    crop_mask = coarse_bool[cy0:cy1, cx0:cx1]
    crop_h, crop_w = crop_mask.shape

    localized_crop_rgb = localize_rgb(
        crop_rgb, crop_mask, mode=args.rgb_localization,
        mask_dilate=args.rgb_mask_dilate, blur_ksize=args.rgb_blur_ksize,
        bg_value=args.rgb_bg_value,
    )

    cells = make_strict_grid(crop_h, crop_w, args.grid_size)
    crop_keep = np.zeros((crop_h, crop_w), dtype=bool)
    cell_summaries = []

    for cell_idx, (gi, gj, y0, y1, x0, x1) in enumerate(cells):
        cell_mask_c = crop_mask[y0:y1, x0:x1]
        cell_rgb_raw = crop_rgb[y0:y1, x0:x1]
        if args.localize_per_cell:
            cell_rgb_input = localize_rgb(
                cell_rgb_raw, cell_mask_c, mode=args.rgb_localization,
                mask_dilate=args.rgb_mask_dilate, blur_ksize=args.rgb_blur_ksize,
                bg_value=args.rgb_bg_value,
            )
        else:
            cell_rgb_input = localized_crop_rgb[y0:y1, x0:x1]

        cell_bgr_input = cv2.cvtColor(cell_rgb_input, cv2.COLOR_RGB2BGR)
        cell_depth = run_depth(model, cell_bgr_input)
        if cell_depth.shape != cell_mask_c.shape:
            cell_depth = cv2.resize(cell_depth.astype(np.float32),
                                     (cell_mask_c.shape[1], cell_mask_c.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

        cell_keep, cell_info = compute_knn_cell(
            cell_depth=cell_depth, cell_mask=cell_mask_c,
            cell_rgb=cell_rgb_raw, args=args,
        )
        crop_keep[y0:y1, x0:x1] = cell_keep
        cell_info.update({
            "cell": cell_idx,
            "grid_ij": [int(gi), int(gj)],
            "window": [int(y0), int(y1), int(x0), int(x1)],
        })
        cell_summaries.append(cell_info)

    if args.morph_open > 0:
        k = np.ones((3, 3), np.uint8)
        crop_keep = cv2.morphologyEx(crop_keep.astype(np.uint8), cv2.MORPH_OPEN, k,
                                      iterations=args.morph_open).astype(bool)
    if args.morph_close > 0:
        k = np.ones((3, 3), np.uint8)
        crop_keep = cv2.morphologyEx(crop_keep.astype(np.uint8), cv2.MORPH_CLOSE, k,
                                      iterations=args.morph_close).astype(bool)

    refined_after_cells = np.zeros((H, W), dtype=bool)
    refined_after_cells[cy0:cy1, cx0:cx1] = crop_keep

    refined_after_fill, ridge_filled_pixels = apply_ridge_fill(
        rgb_full=rgb, full_depth_norm=full_depth_norm,
        refined_full=refined_after_cells, args=args,
    )
    refined_final, edge_restored_pixels = apply_edge_restoration(
        rgb_full=rgb, refined_full=refined_after_fill, args=args,
    )

    running_state["max_observed_refined_area"] = max(
        int(running_state.get("max_observed_refined_area", 0)),
        int(refined_final.sum()),
    )

    refined_filename = coarse_mask_path.name
    refined_path = refined_masks_dir / refined_filename
    Image.fromarray((refined_final.astype(np.uint8) * 255)).save(refined_path)

    if prev_history is not None:
        accept_history = True
        history_reason = "normal"
        if args.enable_sam2_reprompt:
            if (sam2_reprompt_stats.get("triggered", False)
                    and not sam2_reprompt_stats.get("applied", False)):
                accept_history = False
                history_reason = "skip_failed_reprompt"
            elif sam2_reprompt_stats.get("applied", False):
                history_reason = "accepted_after_reprompt"
        prev_history.add(image_path.stem, refined_final, coarse_bool, rgb=rgb,
                         accepted=accept_history, reason=history_reason)

    if args.save_debug:
        per_frame_dir = debug_dir / coarse_mask_path.stem
        per_frame_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((coarse_bool_original.astype(np.uint8) * 255)).save(
            per_frame_dir / "original_coarse.png")
        Image.fromarray((refined_final.astype(np.uint8) * 255)).save(
            per_frame_dir / "refined_coarse.png")
        Image.fromarray(make_side_by_side(rgb, coarse_bool_original, refined_final)).save(
            per_frame_dir / "side_by_side.png")

        if args.save_sam2_debug and args.enable_sam2_reprompt:
            sam2_dir = per_frame_dir / "sam2"
            sam2_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(draw_prompt_debug(rgb, sam2_pos_points, sam2_neg_points)).save(
                sam2_dir / "prompt_points.png")
            Image.fromarray((sam2_search_region.astype(np.uint8) * 255)).save(
                sam2_dir / "search_region.png")
            Image.fromarray((sam2_candidate_region.astype(np.uint8) * 255)).save(
                sam2_dir / "candidate_region.png")
            Image.fromarray((sam2_warped_prev_mask.astype(np.uint8) * 255)).save(
                sam2_dir / "warped_prev_mask.png")
            verification_list = sam2_reprompt_stats.get("donor_pos_verification", [])
            if verification_list:
                tv_img = rgb.copy()
                for v in verification_list:
                    cand = v.get("candidate", [0, 0])
                    accepted = bool(v.get("accepted", False))
                    col = (0, 255, 0) if accepted else (255, 0, 0)
                    cv2.circle(tv_img, (int(cand[0]), int(cand[1])), 8, col, -1)
                    cv2.circle(tv_img, (int(cand[0]), int(cand[1])), 10, (255, 255, 255), 2)
                Image.fromarray(tv_img).save(sam2_dir / "template_verification.png")

        summary_data = {
            "image": image_path.name,
            "coarse_mask": coarse_mask_path.name,
            "refined_mask": refined_filename,
            "shape_HW": [H, W],
            "n_original_coarse_pixels": int(coarse_bool_original.sum()),
            "n_coarse_pixels_after_reprompt": int(coarse_bool.sum()),
            "n_refined_pixels_final": int(refined_final.sum()),
            "n_ridge_filled": int(ridge_filled_pixels.sum()),
            "n_edge_restored": int(edge_restored_pixels.sum()),
            "sam2_reprompt_stats": sam2_reprompt_stats,
            "cell_summaries": cell_summaries,
        }
        with open(per_frame_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

    n_coarse = int(coarse_bool.sum())
    return {
        "image": image_path.name,
        "coarse_mask": coarse_mask_path.name,
        "refined_mask": refined_filename,
        "n_original_coarse_pixels": int(coarse_bool_original.sum()),
        "n_coarse_pixels_after_reprompt": n_coarse,
        "n_refined_pixels_final": int(refined_final.sum()),
        "n_ridge_filled": int(ridge_filled_pixels.sum()),
        "n_edge_restored": int(edge_restored_pixels.sum()),
        "sam2_reprompt_action": sam2_reprompt_stats.get("action", "n/a"),
        "sam2_applied": bool(sam2_reprompt_stats.get("applied", False)),
        "running_max_area": int(running_state["max_observed_refined_area"]),
    }


# =====================================================================
# Run loop — multi-object single pass
# =====================================================================
def run(args):
    if args.depth_anything_path:
        dap = Path(args.depth_anything_path).resolve()
        if not dap.exists():
            raise FileNotFoundError(f"--depth_anything_path missing: {dap}")
        sys.path.insert(0, str(dap))

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("\nERROR: cannot import depth_anything_v2.dpt.\n")
        raise

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n[Refiner] Device: {device}")
    print(f"[Refiner] grid_size={args.grid_size}, "
          f"knn_growth={args.knn_allow_growth_outside_sam2}")
    print(f"[Refiner] ridge_fill={args.enable_ridge_fill}, "
          f"edge_restore={args.enable_edge_restoration}, "
          f"sam2_reprompt={args.enable_sam2_reprompt}")

    cfg = MODEL_CONFIGS[args.encoder]
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    print(f"[Refiner] Depth Anything loaded ({args.encoder}).")

    sam2_runner = None
    if args.enable_sam2_reprompt:
        sam2_runner = SAM2RepromptRunner(
            checkpoint=args.sam2_checkpoint,
            model_cfg=args.sam2_model_cfg,
            device=device,
        )
        print("[Refiner] SAM2 reprompt runner loaded.")

    image_dir = Path(args.image_dir)
    coarse_mask_dir = Path(args.coarse_mask_dir) if args.coarse_mask_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images in {image_dir}")
    if args.max_images and args.max_images < len(images):
        images = images[:args.max_images]

    # Discover objects
    image_stems = {p.stem for p in images}
    objects = discover_objects(image_stems, coarse_mask_dir)
    if args.objects:
        filter_set = set(args.objects)
        objects = [o for o in objects if o in filter_set]
    if not objects:
        raise RuntimeError(
            "No objects discovered from mask filenames. "
            "Expected masks named <image_stem>_<object>.ext in --coarse_mask_dir. "
            "Use --objects to specify object names explicitly.")

    print(f"\n[Refiner] Objects discovered: {objects}")
    print(f"[Refiner] Processing {len(images)} frames × {len(objects)} objects.\n")

    # Shared flat output directories (object names are embedded in mask filenames)
    refined_masks_dir = out_dir / "refined_masks"
    refined_masks_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Per-object state
    histories = {}
    running_states = {}
    for obj in objects:
        histories[obj] = PrevMaskHistory(window=args.prev_mask_window)
        running_states[obj] = {"max_observed_refined_area": 0}

    all_summaries = {obj: {} for obj in objects}
    all_failed = {obj: [] for obj in objects}
    t_start = time.time()

    for idx, img_path in enumerate(images):
        stem = img_path.stem

        # Read the base frame once; share across objects in this frame
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[{idx+1}/{len(images)}] {stem}: FAILED (cv2 read error)")
            for obj in objects:
                all_summaries[obj][stem] = {"error": f"cv2 read failed {img_path}"}
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        print(f"[{idx+1}/{len(images)}] {stem}")

        for obj in objects:
            mask_stem = f"{stem}_{obj}"
            cmp = find_coarse_mask(coarse_mask_dir, mask_stem)
            if cmp is None:
                print(f"    [{obj}] no mask — skip")
                continue

            if not args.overwrite:
                refined_path = refined_masks_dir / cmp.name
                if refined_path.exists():
                    print(f"    [{obj}] SKIP (refined exists)")
                    continue

            try:
                s = process_one_image(
                    image_path=img_path,
                    coarse_mask_path=cmp,
                    model=model,
                    args=args,
                    refined_masks_dir=refined_masks_dir,
                    debug_dir=debug_dir,
                    prev_history=histories[obj],
                    sam2_runner=sam2_runner,
                    running_state=running_states[obj],
                    rgb=rgb,
                    bgr=bgr,
                )
                all_summaries[obj][stem] = s
                if "error" not in s:
                    el = time.time() - t_start
                    total_done = idx * len(objects) + objects.index(obj) + 1
                    total_work = len(images) * len(objects)
                    eta = el / max(1, total_done) * max(0, total_work - total_done)
                    print(f"    [{obj}] coarse={s['n_original_coarse_pixels']}, "
                          f"refined={s['n_refined_pixels_final']}, "
                          f"fill={s['n_ridge_filled']}, edge={s['n_edge_restored']}, "
                          f"sam2={s['sam2_reprompt_action']}, "
                          f"max_area={s['running_max_area']}  [ETA {eta:.0f}s]")
                else:
                    print(f"    [{obj}] {s['error']}")
            except Exception as e:
                print(f"    [{obj}] FAILED: {e}")
                all_failed[obj].append((stem, str(e)))
                all_summaries[obj][stem] = {"error": str(e)}

        if device == "cuda":
            torch.cuda.empty_cache()

    # Combined run summary keyed by object then frame stem
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Single per_image.csv with object column
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "image", "coarse_mask", "refined_mask",
                    "n_original_coarse_pixels", "n_coarse_pixels_after_reprompt",
                    "n_refined_pixels_final", "n_ridge_filled", "n_edge_restored",
                    "sam2_reprompt_action", "sam2_applied"])
        for obj in objects:
            for s_stem, s in all_summaries[obj].items():
                if isinstance(s, dict) and "n_refined_pixels_final" in s:
                    w.writerow([
                        obj, s["image"], s["coarse_mask"], s["refined_mask"],
                        s["n_original_coarse_pixels"], s["n_coarse_pixels_after_reprompt"],
                        s["n_refined_pixels_final"], s["n_ridge_filled"], s["n_edge_restored"],
                        s["sam2_reprompt_action"], s["sam2_applied"],
                    ])

    # Single failed.txt (object \t stem \t error)
    all_failed_flat = [(obj, stem, err)
                       for obj in objects for stem, err in all_failed[obj]]
    if all_failed_flat:
        with open(out_dir / "failed.txt", "w") as f:
            for obj, stem, err in all_failed_flat:
                f.write(f"{obj}\t{stem}\t{err}\n")

    total = time.time() - t_start
    print(f"\n[Refiner] DONE in {total:.0f}s.  Output: {out_dir}")
    for obj in objects:
        n_ok = sum(1 for s in all_summaries[obj].values()
                   if isinstance(s, dict) and "error" not in s)
        n_fail = len(all_failed[obj])
        print(f"  {obj}: {n_ok} refined, {n_fail} failed")
    print(f"  Refined masks → {refined_masks_dir}")


# =====================================================================
# CLI
# =====================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="SAM2-reprompt + KNN-refinement + ridge-fill + edge-restore "
                    "mask refiner — multi-object edition (v2). "
                    "Masks must be named <image_stem>_<object>.ext. "
                    "Objects are auto-discovered from --coarse_mask_dir.")

    # ---- core paths ----
    p.add_argument("--image_dir", required=True,
                   help="Directory of input RGB images (base frames, no object suffix).")
    p.add_argument("--coarse_mask_dir", required=True,
                   help="Directory of coarse masks named <image_stem>_<object>.ext.")
    p.add_argument("--output_dir", required=True,
                   help="Output root. Per-object refined masks → "
                        "<output_dir>/<object>/refined_masks/.")
    p.add_argument("--objects", nargs="+", default=None,
                   help="Restrict processing to these object names (default: all discovered).")
    p.add_argument("--max_images", type=int, default=0,
                   help="Process only the first N images. 0 = all.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocess frames that already have a refined mask.")

    # ---- Depth Anything model ----
    p.add_argument("--depth_anything_path", default="external/Depth-Anything-V2")
    p.add_argument("--encoder", default="vitl", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--checkpoint",
                   default="external/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth")

    # ---- bbox + grid ----
    p.add_argument("--grid_size", type=int, default=4)
    p.add_argument("--bbox_pad", type=float, default=0.1)
    p.add_argument("--min_cell_fg_pixels", type=int, default=200)

    # ---- RGB localization ----
    p.add_argument("--rgb_localization", default="blur_bg",
                   choices=["none", "black_bg", "mean_bg", "blur_bg", "alpha"])
    p.add_argument("--localize_per_cell", action="store_true")
    p.add_argument("--rgb_mask_dilate", type=int, default=2)
    p.add_argument("--rgb_blur_ksize", type=int, default=41)
    p.add_argument("--rgb_bg_value", type=int, default=0)

    # ---- KNN depth tuning ----
    p.add_argument("--depth_keep", default="greater", choices=["greater", "less"])
    p.add_argument("--soft_gamma", type=float, default=1.0)
    p.add_argument("--soft_mask_multiply", action="store_true")
    p.add_argument("--soft_robust_normalize", action="store_true")
    p.add_argument("--soft_norm_lo", type=float, default=2.0)
    p.add_argument("--soft_norm_hi", type=float, default=98.0)
    p.add_argument("--knn_window", type=int, default=31)
    p.add_argument("--knn_percentile", type=float, default=25.0)
    p.add_argument("--knn_min_neighbors", type=int, default=5)
    p.add_argument("--knn_dynamic_percentile", action="store_true")
    p.add_argument("--knn_dyn_min_percentile", type=float, default=15.0)
    p.add_argument("--knn_dyn_max_percentile", type=float, default=40.0)
    p.add_argument("--knn_dyn_strength", type=float, default=1.0)

    # ---- KNN growth ----
    p.add_argument("--knn_allow_growth_outside_sam2", action="store_true")
    p.add_argument("--knn_growth_max_distance", type=int, default=20)
    p.add_argument("--knn_growth_bbox_pad", type=float, default=0.25)
    p.add_argument("--knn_growth_require_color_match", action="store_true")
    p.add_argument("--knn_growth_color_fg_threshold", type=float, default=0.55)
    p.add_argument("--knn_growth_color_fg_erode", type=int, default=1)
    p.add_argument("--knn_growth_color_bg_dilate", type=int, default=3)
    p.add_argument("--knn_growth_min_component_area", type=int, default=5)

    # ---- morphology ----
    p.add_argument("--morph_open", type=int, default=0)
    p.add_argument("--morph_close", type=int, default=0)

    # ---- Ridge fill ----
    p.add_argument("--enable_ridge_fill", action="store_true")
    p.add_argument("--ridge_fill_search_dilate", type=int, default=20)
    p.add_argument("--ridge_fill_threshold", type=float, default=0.65)
    p.add_argument("--ridge_fill_depth_blur_sigma", type=float, default=0.8)
    p.add_argument("--ridge_fill_rgb_weight", type=float, default=0.4)
    p.add_argument("--ridge_fill_max_fraction", type=float, default=0.15)

    # ---- Edge restoration ----
    p.add_argument("--enable_edge_restoration", action="store_true")
    p.add_argument("--edge_band_width", type=int, default=3)
    p.add_argument("--edge_restore_threshold", type=float, default=0.4)
    p.add_argument("--edge_restore_blur_sigma", type=float, default=0.6)
    p.add_argument("--edge_restore_dilate", type=int, default=1)

    # ---- Full-image depth ----
    p.add_argument("--full_depth_localize", action="store_true")
    p.add_argument("--full_depth_localization", default="alpha",
                   choices=["none", "black_bg", "mean_bg", "blur_bg", "alpha", "mask_only"])
    p.add_argument("--full_depth_mask_dilate", type=int, default=4)
    p.add_argument("--full_depth_blur_ksize", type=int, default=61)
    p.add_argument("--full_depth_normalize_inside_mask", action="store_true")
    p.add_argument("--full_depth_robust_normalize", action="store_true")
    p.add_argument("--full_depth_norm_lo", type=float, default=2.0)
    p.add_argument("--full_depth_norm_hi", type=float, default=98.0)

    # ---- SAM2 reprompt ----
    p.add_argument("--enable_sam2_reprompt", action="store_true")
    p.add_argument("--sam2_checkpoint",
                   default="external/sam2/checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--sam2_model_cfg",
                   default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--prev_mask_window", type=int, default=5)

    p.add_argument("--sam2_reprompt_trigger_mode", default="warp_disagreement",
                   choices=["warp_disagreement", "area", "always"])
    p.add_argument("--sam2_reprompt_min_warp_iou", type=float, default=0.50)
    p.add_argument("--sam2_reprompt_min_warp_coverage", type=float, default=0.65)
    p.add_argument("--sam2_reprompt_min_missing_fraction", type=float, default=0.15)
    p.add_argument("--sam2_reprompt_max_current_to_warp_ratio_for_trigger",
                   type=float, default=1.15)
    p.add_argument("--sam2_reprompt_min_area_ratio_to_warp", type=float, default=0.60)
    p.add_argument("--sam2_reprompt_min_iou_improvement", type=float, default=0.03)

    p.add_argument("--sam2_reprompt_current_mask_erode", type=int, default=5)
    p.add_argument("--sam2_reprompt_min_eroded_pixels", type=int, default=100)
    p.add_argument("--sam2_reprompt_min_current_mask_pixels", type=int, default=50)
    p.add_argument("--sam2_reprompt_pos_points", type=int, default=12)
    p.add_argument("--sam2_reprompt_pos_min_dist", type=float, default=10.0)

    p.add_argument("--sam2_reprompt_prev_area_strategy", default="max",
                   choices=["last", "max", "median"])
    p.add_argument("--sam2_reprompt_max_search_radius", type=int, default=250)
    p.add_argument("--sam2_reprompt_depth_filter_mode", default="quantile",
                   choices=["quantile", "mad", "both", "off"])
    p.add_argument("--sam2_reprompt_depth_quantile_low", type=float, default=2.0)
    p.add_argument("--sam2_reprompt_depth_quantile_high", type=float, default=98.0)
    p.add_argument("--sam2_reprompt_depth_range_pad", type=float, default=0.0)
    p.add_argument("--sam2_reprompt_depth_k_mad", type=float, default=3.0)
    p.add_argument("--sam2_reprompt_depth_mad_floor", type=float, default=0.01)
    p.add_argument("--sam2_reprompt_donor_pos_points", type=int, default=6)
    p.add_argument("--sam2_reprompt_donor_pos_min_dist", type=float, default=5.0)
    p.add_argument("--sam2_reprompt_donor_min_recovery_area", type=int, default=200)
    p.add_argument("--sam2_reprompt_donor_oversample_factor", type=int, default=5)
    p.add_argument("--sam2_reprompt_donor_template_verify", action="store_true", default=True)
    p.add_argument("--sam2_reprompt_donor_no_template_verify", action="store_false",
                   dest="sam2_reprompt_donor_template_verify")
    p.add_argument("--sam2_reprompt_template_radius", type=int, default=10)
    p.add_argument("--sam2_reprompt_template_search_radius", type=int, default=50)
    p.add_argument("--sam2_reprompt_template_score_threshold", type=float, default=0.7)

    p.add_argument("--sam2_reprompt_multimask", action="store_true", default=True)
    p.add_argument("--sam2_reprompt_max_area_ratio", type=float, default=2.5)
    p.add_argument("--sam2_reprompt_enable_max_area_cap", action="store_true", default=True)
    p.add_argument("--sam2_reprompt_no_max_area_cap", action="store_false",
                   dest="sam2_reprompt_enable_max_area_cap")
    p.add_argument("--sam2_reprompt_max_area_epsilon", type=float, default=0.15)

    p.add_argument("--flow_pyr_scale", type=float, default=0.5)
    p.add_argument("--flow_levels", type=int, default=3)
    p.add_argument("--flow_winsize", type=int, default=51)
    p.add_argument("--flow_iterations", type=int, default=5)
    p.add_argument("--flow_poly_n", type=int, default=7)
    p.add_argument("--flow_poly_sigma", type=float, default=1.5)
    p.add_argument("--warp_close", type=int, default=1)

    # ---- Debug ----
    p.add_argument("--save_debug", action="store_true")
    p.add_argument("--save_sam2_debug", action="store_true")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
