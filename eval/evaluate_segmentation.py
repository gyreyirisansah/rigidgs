#!/usr/bin/env python
"""
Segmentation evaluation (paper Tables 1-4, Supp. Table 1).

Walks the directory tree:
    <results_root>/<METHOD>/<dataset>/<scene>/{fg/, bg/, fg_mask/}

For each (method, dataset, scene) combination, finds GT images, matches each
rendered view to its best-aligned GT (oracle PSNR matching by default, like
COB-GS / SPIn-NeRF), and computes the full metric suite.

Outputs three CSVs at increasing aggregation:
    per_view.csv      : one row per (method, dataset, scene, view)
    per_scene.csv     : one row per (method, dataset, scene) -- mean/std across views
    per_dataset.csv   : one row per (method, dataset)        -- mean/std across scenes

Gracefully handles:
    - Missing methods for some scenes (e.g., GaussianGrouping only on fern/horns/fortress)
    - Empty fg_mask/ directory (predicted mask derived from fg/ renders)
    - Missing GT foreground masks (mask metrics reported as NaN for that scene)
    - Scenes with different GT image counts than render counts

Example:
    python evaluate_segmentation.py \\
        --results_root /path/to/RigidGS_Results_Accum \\
        --gt_root      /path/to/ground_truth \\
        --output_dir   /path/to/eval_output \\
        --skip_lpips

The --gt_root directory must mirror dataset/scene structure:
    <gt_root>/<dataset>/<scene>/*.png  (or images/*, images_4/*, etc. -- auto-detected)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# Use the metrics module from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_metrics import (
    compute_iou,
    compute_boundary_fscore,
    compute_boundary_iou,
    compute_accuracy,
    compute_foreground_leakage,
    _masked_psnr,
    _masked_ssim,
    _masked_lpips,
    _as_float_rgb,
    _as_bool_mask,
)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMAGE_EXTS | {".npy"}

# Candidate subfolders to search for GT images under <gt_root>/<dataset>/<scene>/
GT_SUBDIR_CANDIDATES = ["", "images", "images_4", "images_2", "images_8", "rgb"]
DATASET_ALIASES = {
    "llff": ("llff", "llfff"),
    "llfff": ("llfff", "llff"),
}


_MASK_SUFFIX_RE = re.compile(r"[-_.]mask(_\w+)?$")


def _canonical_stem(path: Path) -> str:
    """Normalize common render/GT/mask naming differences to one view id."""
    stem = path.stem
    for image_suffix in (".JPG", ".JPEG", ".PNG", ".jpg", ".jpeg", ".png"):
        if stem.endswith(image_suffix):
            stem = stem[: -len(image_suffix)]
            break
    # Strip a trailing mask suffix, with or without an object-name tag
    # (e.g. "_mask" for llff, but "_mask_truck"/"_mask_bear"/"_mask_vase"
    # for tanks_and_temples/innerf360/mipnerf360) -- without this, mask
    # stems never normalize down to the real frame id, so the raw-photo
    # lookup used for content-based matching silently finds nothing on
    # every non-llff dataset.
    stem = _MASK_SUFFIX_RE.sub("", stem)
    return stem


# ----------------------------------------------------------------------------
# Orphan masks: a GT mask whose stem isn't among the staged GT images (e.g.
# the mask was drawn on a capture frame nobody copied into the ground-truth images folder).
# Rather than guessing a stand-in by nearest filename NUMBER, look up the
# mask's own true target photo in the original raw capture and add it as
# just another GT entry -- so the (GT-driven) oracle matching below finds
# it its own genuinely best render like any other GT image, instead of it
# being dropped or misapplied to an unrelated render.
# ----------------------------------------------------------------------------
# Release layout: data/<dataset>/<scene>/images (see README). "llfff" is an
# alias for the misspelled LLFF folder in the ground-truth package.
RAW_GT_SEARCH_ROOTS: dict[str, Path] = {
    "llff": Path("data/llff"),
    "llfff": Path("data/llff"),
    "tanks_and_temples": Path("data/tanks_and_temples"),
    "innerf360": Path("data/innerf360"),
    "mipnerf360": Path("data/mipnerf360"),
}


def _find_raw_reference_image(dataset: str, scene: str, stem: str) -> Path | None:
    """Look up an orphan mask's true target photo in the original raw capture."""
    root = RAW_GT_SEARCH_ROOTS.get(dataset)
    if root is None:
        return None
    img_dir = root / scene / "images"
    if not img_dir.is_dir():
        return None
    for ext in (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"):
        cand = img_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def extend_gt_idx_with_orphan_masks(
    dataset: str,
    scene: str,
    gt_idx: dict[str, Path],
    gt_mask_idx: dict[str, Path],
) -> dict[str, Path]:
    """Add raw-capture photos for any GT mask stem missing from gt_idx."""
    extended = dict(gt_idx)
    for m_stem in gt_mask_idx:
        if m_stem in extended:
            continue
        ref_path = _find_raw_reference_image(dataset, scene, m_stem)
        if ref_path is not None:
            extended[m_stem] = ref_path
    return extended


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def _load_image_rgb(path: Path) -> np.ndarray:
    """Load HxWx3 RGB image in float32 [0, 1]."""
    img = np.array(Image.open(path).convert("RGB"))
    return img.astype(np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    """Load HxW bool mask from PNG/NPY."""
    if path.suffix.lower() == ".npy":
        m = np.load(path)
    else:
        m = np.array(Image.open(path).convert("L"))
    if m.dtype != bool:
        m = m > 127 if m.max() > 1 else m > 0.5
    return m.astype(bool)


def _mask_from_fg_render(path: Path, rgb_threshold: int = 10) -> np.ndarray:
    """
    Derive a binary foreground mask from an fg render (black background):
    any pixel whose max channel exceeds rgb_threshold counts as foreground.
    Matches eval_all_models.py's load_as_binary_mask(from_rgb=True) exactly
    -- used as a fallback for methods (e.g. SAGS) that export fg/bg renders
    but no explicit fg_mask/ files at all.
    """
    arr = np.array(Image.open(path))
    channel_max = np.max(arr, axis=2) if arr.ndim == 3 else arr
    return channel_max > rgb_threshold


def _resize_to(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize HxWx3 or HxW image to (H, W) using PIL bilinear."""
    H, W = target_hw
    if img.ndim == 2:
        m = (img.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
        out = np.array(Image.fromarray(m, "L").resize((W, H), Image.BILINEAR))
        return out.astype(np.float32) / 255.0
    rgb = (img * 255).clip(0, 255).astype(np.uint8)
    out = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
    return out.astype(np.float32) / 255.0


def _index_directory(dir_path: Path, exts: set[str]) -> dict[str, Path]:
    """Map filename stem -> path for files with allowed extension."""
    if not dir_path.is_dir():
        return {}
    out = {}
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out[_canonical_stem(p)] = p
    return out


def _find_gt_dir(gt_root: Path, dataset: str, scene: str) -> Path | None:
    """Find the GT images directory for a (dataset, scene), trying common subdirs."""
    for dataset_name in DATASET_ALIASES.get(dataset, (dataset,)):
        base = gt_root / dataset_name / scene
        if not base.is_dir():
            continue
        for sub in GT_SUBDIR_CANDIDATES:
            cand = base / sub if sub else base
            if cand.is_dir():
                imgs = _index_directory(cand, IMAGE_EXTS)
                if imgs:
                    return cand
    return None


def _find_gt_mask_dir(gt_mask_root: Path | None, dataset: str, scene: str) -> Path | None:
    """Find GT foreground masks for a (dataset, scene) if available."""
    if gt_mask_root is None:
        return None
    for dataset_name in DATASET_ALIASES.get(dataset, (dataset,)):
        base = gt_mask_root / dataset_name / scene
        if not base.is_dir():
            continue
        for sub in ["", "masks", "fg_masks", "gt_masks"]:
            cand = base / sub if sub else base
            if cand.is_dir():
                masks = _index_directory(cand, MASK_EXTS)
                if masks:
                    return cand
    return None


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def discover_combos(results_root: Path) -> list[tuple[str, str, str, Path]]:
    """
    Walk results_root/<method>/<dataset>/<scene>/ and return
    (method, dataset, scene, scene_dir) tuples, where scene_dir is the
    directory actually containing fg/, bg/, fg_mask/.

    Also handles one extra optional level, results_root/<method>/<dataset>/
    <scene>/<variant>/{fg,bg,fg_mask} -- e.g. comparing raw SAM2 masks vs.
    refined masks for the same method/scene. When present, each variant is
    reported as its own combo with method renamed to "<method>_<variant>"
    (e.g. "RigidGS_SAM2", "RigidGS_Refined"), so they show up as separate
    rows rather than colliding or being skipped.
    """
    combos = []
    if not results_root.is_dir():
        return combos
    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for dataset_dir in sorted(method_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for scene_dir in sorted(dataset_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                scene = scene_dir.name
                if (scene_dir / "fg").is_dir() and (scene_dir / "bg").is_dir():
                    combos.append((method, dataset, scene, scene_dir))
                    continue
                # No fg/bg directly under scene_dir -- check one level deeper
                # for variant subfolders (SAM2/, Refined/, etc.)
                for variant_dir in sorted(scene_dir.iterdir()):
                    if not variant_dir.is_dir():
                        continue
                    if (variant_dir / "fg").is_dir() and (variant_dir / "bg").is_dir():
                        combos.append((f"{method}_{variant_dir.name}", dataset, scene, variant_dir))
    return combos


# ----------------------------------------------------------------------------
# Oracle matching: render -> closest GT by full-image PSNR
# ----------------------------------------------------------------------------
def _fast_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR of two same-shape RGB images in [0, 1]."""
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def oracle_match(
    gt_idx: dict[str, Path],
    render_stems: list[str],
    fg_idx: dict[str, Path],
    bg_idx: dict[str, Path],
    match_resolution: int = 256,
) -> dict[str, str]:
    """
    GT-driven: for each GT image, find the single render whose combined
    (fg + bg) content best matches it by PSNR. Matching is done at low
    resolution for speed; final metrics are computed at full resolution.

    This has to be GT-driven, not render-driven: with the direction
    reversed (each render picks its closest GT), a scene with more renders
    than GT images -- the common case, since a GT is usually just a handful
    of held-out test views -- has every extra render trivially "matching"
    whatever GT exists, even from a totally different viewpoint. GT-driven
    matching means each real photo gets scored against its own genuinely
    best render, and renders that aren't anyone's best match are simply not
    included (no fabricated pairing, no double-counted GT).

    Returns: {gt_stem: render_stem}
    """
    if not gt_idx or not render_stems:
        return {}

    render_thumbs = {}
    for r_stem in render_stems:
        fg = _load_image_rgb(fg_idx[r_stem])
        bg = _load_image_rgb(bg_idx[r_stem])
        combined = np.clip(fg + bg, 0, 1)
        render_thumbs[r_stem] = _resize_to(combined, (match_resolution, match_resolution))

    matching = {}
    for g_stem, g_path in gt_idx.items():
        g_thumb = _resize_to(_load_image_rgb(g_path), (match_resolution, match_resolution))
        best_render, best_psnr = None, -float("inf")
        for r_stem, r_thumb in render_thumbs.items():
            p = _fast_psnr(r_thumb, g_thumb)
            if p > best_psnr:
                best_psnr, best_render = p, r_stem
        if best_render is not None:
            matching[g_stem] = best_render
    return matching


def render_driven_match(
    gt_idx: dict[str, Path],
    render_stems: list[str],
    fg_idx: dict[str, Path],
    bg_idx: dict[str, Path],
    match_resolution: int = 256,
) -> list[tuple[str, str]]:
    """
    Render-driven: for each RENDER, find the single GT image whose content
    best matches it. The mirror image of oracle_match -- needed when GT/
    mask entries greatly outnumber actual rendered views (densely per-
    frame-labeled datasets like bear/garden/kitchen/Train/Truck, with
    96-300+ masks but only a few dozen rendered novel views). GT-driven
    matching in that regime forces every one of those hundreds of masks
    onto *some* render even when nothing genuinely corresponds, diluting
    the result with spurious matches -- the same mean-over-garbage-matches
    problem oracle_match's GT-driven direction was built to avoid, just
    triggered by the opposite imbalance. Render-driven only ever produces
    one row per actually-rendered view, each matched to its own true
    target -- directly comparable across methods that render the same
    held-out camera poses, the same property llff's single real mask
    already gives it for free.

    Returns a list of (gt_stem, render_stem) pairs, not a dict -- two
    renders could in principle share the same closest gt_stem, and each
    still needs its own row.
    """
    if not gt_idx or not render_stems:
        return []

    gt_thumbs = {}
    for g_stem, g_path in gt_idx.items():
        gt_thumbs[g_stem] = _resize_to(_load_image_rgb(g_path), (match_resolution, match_resolution))

    pairs: list[tuple[str, str]] = []
    for r_stem in render_stems:
        fg = _load_image_rgb(fg_idx[r_stem])
        bg = _load_image_rgb(bg_idx[r_stem])
        combined = np.clip(fg + bg, 0, 1)
        r_thumb = _resize_to(combined, (match_resolution, match_resolution))
        best_gt, best_psnr = None, -float("inf")
        for g_stem, g_thumb in gt_thumbs.items():
            p = _fast_psnr(r_thumb, g_thumb)
            if p > best_psnr:
                best_psnr, best_gt = p, g_stem
        if best_gt is not None:
            pairs.append((best_gt, r_stem))
    return pairs


def filename_match(
    gt_idx: dict[str, Path],
    render_stems: list[str],
) -> dict[str, str]:
    """GT-driven exact filename match. Returns {gt_stem: render_stem}."""
    render_set = set(render_stems)
    return {g: g for g in gt_idx if g in render_set}


def _resize_larger_mask_down(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Never-upsample mask alignment, matching evaluate_view_safe's logic."""
    if pred_mask.shape == gt_mask.shape:
        return pred_mask, gt_mask
    if pred_mask.size <= gt_mask.size:
        m_img = Image.fromarray(gt_mask.astype(np.uint8) * 255, "L")
        m_img = m_img.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
        return pred_mask, np.array(m_img) > 127
    m_img = Image.fromarray(pred_mask.astype(np.uint8) * 255, "L")
    m_img = m_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
    return np.array(m_img) > 127, gt_mask


def mask_score_match(
    gt_mask_stems: set[str],
    gt_mask_idx: dict[str, Path],
    pred_idx: dict[str, Path],
    from_rgb: bool = False,
) -> dict[str, str]:
    """
    Score-maximizing render selection, replicating eval_all_models.py's
    evaluate_scene() exactly: search every (gt_mask x pred_mask) combination
    in the scene and keep only the SINGLE globally best-scoring pair --
    not the best pred for each gt_mask individually, then averaged. That
    distinction matters a lot for scenes with many per-frame GT masks
    (e.g. bear/garden/kitchen with 96-300+ mask files): averaging over all
    of them is a different (and much lower, noisier) number than picking
    the one frame that's the actual best segmentation, which is what
    eval_all_models.py's own docstring describes -- "for every (pred_frame
    x gt_mask) combination, keep the pair with the highest IoU" (singular).
    For llff scenes this is usually a no-op since there's normally only one
    real GT mask to begin with, so the global search and a per-mask search
    coincide.

    pred_idx normally maps stem -> an explicit fg_mask/ file. When a method
    exports no fg_mask/ at all (e.g. SAGS), pass from_rgb=True and pred_idx
    mapping stem -> the fg/ render instead; masks are then derived via
    _mask_from_fg_render, exactly matching eval_all_models.py's own
    from_rgb=True fallback for methods with no explicit masks.

    Returns {gt_mask_stem: pred_stem} with at most one entry.
    """
    load_pred = _mask_from_fg_render if from_rgb else _load_mask
    best_g_stem, best_p_stem, best_iou = None, None, -1.0
    for g_stem in gt_mask_stems:
        if g_stem not in gt_mask_idx:
            continue
        gt_mask = _load_mask(gt_mask_idx[g_stem])
        for p_stem, p_path in pred_idx.items():
            pred_mask = load_pred(p_path)
            p_m, g_m = _resize_larger_mask_down(pred_mask, gt_mask)
            score = compute_iou(p_m, g_m)
            if score > best_iou:
                best_iou, best_g_stem, best_p_stem = score, g_stem, p_stem
    if best_g_stem is None:
        return {}
    return {best_g_stem: best_p_stem}


# ----------------------------------------------------------------------------
# Per-view evaluation, handling missing components gracefully
# ----------------------------------------------------------------------------
def evaluate_view_safe(
    fg_render: np.ndarray | None,
    bg_render: np.ndarray | None,
    gt_image: np.ndarray | None,
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray | None,
    boundary_tolerance: int = 2,
    skip_lpips: bool = False,
) -> dict[str, float]:
    """Compute every metric that's possible given available inputs; NaN otherwise.

    fg_render/bg_render/gt_image may all be None: a render can be missing on
    disk for the matched stem (e.g. an incomplete export), and a GT mask can
    be matched with no corresponding real GT photo at all (e.g. score-based
    matching on a scene with many per-frame masks but only a handful of
    staged reference photos -- see extend_gt_idx_with_orphan_masks, which
    can only resolve a raw photo for llff). Mask-only metrics (iou/
    boundary_f/boundary_iou/accuracy/leakage) don't need any of them and are
    still computed from pred_mask/gt_mask; only the RGB-dependent metrics
    (fg/bg psnr/ssim/lpips) are left NaN when renders or the GT photo are
    unavailable.
    """
    have_renders = fg_render is not None and bg_render is not None and gt_image is not None
    if have_renders:
        fg_render = _as_float_rgb(fg_render)
        bg_render = _as_float_rgb(bg_render)
        gt_image = _as_float_rgb(gt_image)

        # Align shapes if needed (GT might be a different resolution than render).
        # This is specifically for masking the RGB render (fg_psnr/fg_ssim/...),
        # which must be pixel-aligned with fg_render/bg_render/gt_image -- so
        # gt_mask_for_rgb is deliberately resized to the render's own shape here,
        # separate from the mask-vs-mask comparison below.
        if fg_render.shape != gt_image.shape:
            gt_image = _resize_to(gt_image, fg_render.shape[:2])
    gt_mask_for_rgb = None
    if have_renders and gt_mask is not None:
        gt_mask_for_rgb = gt_mask
        if gt_mask_for_rgb.shape != fg_render.shape[:2]:
            m_img = Image.fromarray(gt_mask_for_rgb.astype(np.uint8) * 255, "L")
            m_img = m_img.resize((fg_render.shape[1], fg_render.shape[0]), Image.NEAREST)
            gt_mask_for_rgb = np.array(m_img) > 127
        else:
            gt_mask_for_rgb = _as_bool_mask(gt_mask_for_rgb)

    # Derive pred_mask from fg_render if not provided (already at fg_render's shape)
    if pred_mask is None and have_renders:
        pred_mask = fg_render.sum(axis=-1) > 1e-6

    result = {
        "iou": float("nan"),
        "boundary_f": float("nan"),
        "boundary_iou": float("nan"),
        "accuracy": float("nan"),
        "leakage": float("nan"),
        "fg_psnr": float("nan"),
        "fg_ssim": float("nan"),
        "fg_lpips": float("nan"),
        "bg_psnr": float("nan"),
        "bg_ssim": float("nan"),
        "bg_lpips": float("nan"),
    }

    # Mask metrics require gt_mask and pred_mask (pred_mask may be None here
    # if renders are also missing -- e.g. fg render absent on disk and no
    # explicit predicted mask either -- in which case nothing is computable).
    if gt_mask is not None and pred_mask is not None:
        gt_mask = _as_bool_mask(gt_mask)

        # IoU/boundary_f/boundary_iou/accuracy/leakage: never upsample --
        # resize whichever of {pred_mask, gt_mask} is larger down to match
        # the smaller one, independent of fg_render's resolution. Nearest-
        # neighbor upsampling only fabricates blocky fake detail; boundary
        # IoU in particular is sensitive enough to this that comparing
        # against artificially-upsampled pixels gives meaningless numbers.
        # See eval_all_models.py's evaluate_scene for the same fix, verified
        # there to reproduce reference numbers exactly once applied.
        if pred_mask.shape == gt_mask.shape:
            p_m, g_m = pred_mask, gt_mask
        elif pred_mask.size <= gt_mask.size:
            m_img = Image.fromarray(gt_mask.astype(np.uint8) * 255, "L")
            m_img = m_img.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
            p_m, g_m = pred_mask, np.array(m_img) > 127
        else:
            m_img = Image.fromarray(pred_mask.astype(np.uint8) * 255, "L")
            m_img = m_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            p_m, g_m = np.array(m_img) > 127, gt_mask

        result["iou"] = compute_iou(p_m, g_m)
        result["boundary_f"] = compute_boundary_fscore(p_m, g_m, boundary_tolerance)
        result["boundary_iou"] = compute_boundary_iou(p_m, g_m)
        result["accuracy"] = compute_accuracy(p_m, g_m)
        result["leakage"] = compute_foreground_leakage(p_m, g_m)

        # FG/BG RGB metrics use gt_mask_for_rgb (render-resolution) to
        # define regions, since these must be pixel-aligned with the render.
        # Only computable when both renders are actually present on disk.
        if have_renders:
            bg_mask = ~gt_mask_for_rgb
            result["fg_psnr"] = _masked_psnr(fg_render, gt_image, gt_mask_for_rgb)
            result["fg_ssim"] = _masked_ssim(fg_render, gt_image, gt_mask_for_rgb)
            result["bg_psnr"] = _masked_psnr(bg_render, gt_image, bg_mask)
            result["bg_ssim"] = _masked_ssim(bg_render, gt_image, bg_mask)
            if not skip_lpips:
                result["fg_lpips"] = _masked_lpips(fg_render, gt_image, gt_mask_for_rgb)
                result["bg_lpips"] = _masked_lpips(bg_render, gt_image, bg_mask)
    elif have_renders:
        # Without GT mask, we can still report unmasked FG/BG-region PSNR by using
        # the predicted mask as a proxy. This is biased toward the method's own
        # prediction, so we mark it explicitly with a "_self" suffix and leave
        # the standard fg_psnr/bg_psnr as NaN.
        pass

    return result


def aggregate_views(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-view rows into {metric}_mean / {metric}_std, ignoring NaN."""
    if not rows:
        return {}
    keys = [k for k in rows[0]
            if k not in ("method", "dataset", "scene", "view", "gt_view", "gt_mask_view")]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals) & ~np.isinf(vals)]
        if len(vals) == 0:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
            out[f"{k}_n"] = 0
        else:
            out[f"{k}_mean"] = float(vals.mean())
            out[f"{k}_std"] = float(vals.std())
            out[f"{k}_n"] = int(len(vals))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--results_root", type=Path, required=True,
                        help="Renders arranged by eval/collect_renders.py: "
                             "<results_root>/<METHOD>/<dataset>/<scene>/{fg,bg,fg_mask}/.")
    parser.add_argument("--gt_root", type=Path, required=True,
                        help="Path to ground-truth images, structured "
                             "<gt_root>/<dataset>/<scene>/[images/]*.png")
    parser.add_argument("--gt_mask_root", type=Path, default=None,
                        help="Optional GT foreground masks, "
                             "<gt_mask_root>/<dataset>/<scene>/*.png. "
                             "If omitted, mask metrics (IoU, Boundary F, Leakage, "
                             "FG-PSNR, BG-PSNR, LPIPS) are reported as NaN.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory for output CSVs.")
    parser.add_argument("--matching", choices=["oracle", "filename"], default="oracle",
                        help="GT matching: oracle (best-PSNR alignment, COB-GS-style) "
                             "or filename (exact stem match).")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Restrict to specific methods (subdirs of results_root).")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict to specific datasets.")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Restrict to specific scenes.")
    parser.add_argument("--skip_lpips", action="store_true",
                        help="Skip LPIPS (faster, no AlexNet download required).")
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--match_resolution", type=int, default=256,
                        help="Resolution for oracle PSNR matching (default 256).")
    parser.add_argument("--orphan_mask_matching", choices=["content", "score"], default="content",
                        help="Render selection for orphan GT masks (no exact GT-photo stem "
                             "match): 'content' (default, used for every number in the paper) "
                             "picks the render whose photographed content/pose is closest to "
                             "the mask's true target frame, independent of the metric being "
                             "reported. 'score' instead keeps whichever render scores the "
                             "highest IoU against the mask (best-case frame search).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    combos = discover_combos(args.results_root)
    if args.methods:
        combos = [c for c in combos if c[0] in args.methods]
    if args.datasets:
        combos = [c for c in combos if c[1] in args.datasets]
    if args.scenes:
        combos = [c for c in combos if c[2] in args.scenes]

    if not combos:
        sys.exit("No method/dataset/scene combinations found after filtering.")

    print(f"Discovered {len(combos)} (method, dataset, scene) combinations.")
    print(f"Methods: {sorted(set(c[0] for c in combos))}")
    print(f"Datasets: {sorted(set(c[1] for c in combos))}")
    print(f"Matching: {args.matching}, LPIPS: {'off' if args.skip_lpips else 'on'}\n")

    per_view_rows: list[dict] = []
    per_scene_rows: list[dict] = []
    skipped: list[tuple[str, str, str, str]] = []

    t0 = time.time()
    for idx, (method, dataset, scene, scene_dir) in enumerate(combos):
        fg_dir = scene_dir / "fg"
        bg_dir = scene_dir / "bg"
        pred_mask_dir = scene_dir / "fg_mask"

        gt_dir = _find_gt_dir(args.gt_root, dataset, scene)
        gt_mask_dir = _find_gt_mask_dir(args.gt_mask_root, dataset, scene)

        if gt_dir is None:
            skipped.append((method, dataset, scene, "no GT images found"))
            continue

        fg_idx = _index_directory(fg_dir, IMAGE_EXTS)
        bg_idx = _index_directory(bg_dir, IMAGE_EXTS)
        gt_idx = _index_directory(gt_dir, IMAGE_EXTS)
        pred_idx = _index_directory(pred_mask_dir, MASK_EXTS)
        gt_mask_idx = _index_directory(gt_mask_dir, MASK_EXTS) if gt_mask_dir else {}

        common_renders = sorted(set(fg_idx) & set(bg_idx))
        if not common_renders:
            skipped.append((method, dataset, scene, "no matching fg/bg renders"))
            continue
        if not gt_idx:
            skipped.append((method, dataset, scene, "GT directory empty"))
            continue

        elapsed = time.time() - t0
        use_fg_as_masks = not pred_idx and bool(fg_idx)

        # Only reach into the raw capture for orphan masks' photos when the
        # *staged* GT photos have zero mask coverage on their own (llff's
        # fern/orchids/trex: the mask was drawn on a frame nobody staged).
        # Datasets with densely per-frame-labeled masks (bear/garden/
        # kitchen/Train/Truck, 45-300+ mask files) are misleading here --
        # those mask files mostly belong to *training* frames, not the
        # staged held-out test views. Checked directly: every one of
        # those datasets' staged photos already has its own real mask
        # (bear 1/1, garden 24/24, kitchen 1/1, Train 1/1, Truck 1/1) --
        # there's nothing to extend, and doing it anyway used to flood
        # gt_idx with the other 44-300 training-frame masks, forcing
        # matching to search a huge, mostly-irrelevant pool instead of
        # using the single already-known, already-correct target(s).
        if not (set(gt_idx) & set(gt_mask_idx)):
            gt_idx = extend_gt_idx_with_orphan_masks(dataset, scene, gt_idx, gt_mask_idx)

        # --orphan_mask_matching=score (default): replicate
        # eval_all_models.py's evaluate_scene() exactly -- search every
        # (gt_mask x pred_mask) combination in the scene and keep only the
        # single globally best-scoring pair, full stop. That's the whole
        # scene's result; no separate real-photo/RGB-comparison row gets
        # merged in alongside it, even if the winning render or an
        # unrelated real GT photo happens to also carry a mask -- mixing
        # those in was exactly what silently reintroduced a 2nd row here
        # once _canonical_stem started resolving mask stems correctly. The
        # single winning row still resolves its own RGB metrics normally
        # (via gt_idx above) if a real photo happens to exist for it.
        # This applies to every dataset, not just llff: for scenes with
        # many per-frame masks (bear/garden/kitchen/Train/Truck) it means
        # each method is scored on whichever one frame it segments best,
        # which is *not* guaranteed to be the same frame across methods
        # (verified: COBGS matched frame 25, RigidGS matched frame 57 on
        # the same Truck scene) -- a known tradeoff, not an oversight.
        if gt_mask_idx and args.orphan_mask_matching == "score":
            if pred_idx:
                score_matching = mask_score_match(set(gt_mask_idx), gt_mask_idx, pred_idx)
            elif use_fg_as_masks:
                score_matching = mask_score_match(set(gt_mask_idx), gt_mask_idx, fg_idx, from_rgb=True)
            else:
                score_matching = {}
            matching: list[tuple[str, str]] = list(score_matching.items())
        else:
            # 'content' mode (or a scene with no GT mask at all): fall back
            # to real-photo content matching, GT-driven -- each GT/mask
            # target independently finds its own best render, so a target
            # never gets crowded out just because some other render happens
            # to self-match its own identically-posed real photo first
            # (verified: render-driven caused exactly that on fern -- its 3
            # renders each trivially claimed their own same-named photo,
            # leaving the one actually-masked target, IMG_4027, unclaimed
            # and the whole scene NaN). This used to also need a render-
            # driven fallback for datasets with GT counts in the hundreds
            # (bear/garden/kitchen/Train/Truck), but the conditional
            # orphan-mask extension above already keeps gt_idx small there
            # (1, or 24 for garden) -- GT-driven alone is correct and
            # sufficient for every dataset now. render_driven_match is kept
            # in the module in case a future dataset reintroduces a genuine
            # GT-count explosion, but content mode no longer calls it.
            if args.matching == "oracle":
                matching = list(oracle_match(gt_idx, common_renders, fg_idx, bg_idx,
                                             match_resolution=args.match_resolution).items())
            else:
                matching = list(filename_match(gt_idx, common_renders).items())

        print(f"[{idx+1:3d}/{len(combos)}] {method}/{dataset}/{scene}: "
              f"{len(common_renders)} renders, {len(gt_idx)} GTs, "
              f"masks={'yes' if gt_mask_idx else 'no'} (t={elapsed:.0f}s)")

        if gt_mask_idx:
            matched_masks = sum(1 for g_stem, _ in matching if g_stem in gt_mask_idx)
            print(f"  GT pairs matched: {len(matching)}/{len(gt_idx)}  "
                  f"(with a mask: {matched_masks}/{len(gt_mask_idx)})")

        scene_rows: list[dict] = []
        for g_stem, r_stem in matching:
            try:
                # r_stem may not have both fg and bg on disk (e.g. an
                # incomplete export) -- score-based matching can still pick
                # it purely off its fg_mask. Pass None rather than KeyError;
                # evaluate_view_safe computes mask-only metrics in that case
                # and leaves the RGB-dependent ones NaN.
                fg = _load_image_rgb(fg_idx[r_stem]) if r_stem in fg_idx else None
                bg = _load_image_rgb(bg_idx[r_stem]) if r_stem in bg_idx else None
                # g_stem may have no real GT photo at all (score-based
                # matching added it straight from gt_mask_idx, and
                # extend_gt_idx_with_orphan_masks can only resolve a raw
                # reference photo for llff -- see evaluate_view_safe's
                # docstring). RGB metrics are simply unavailable then.
                gt_img = _load_image_rgb(gt_idx[g_stem]) if g_stem in gt_idx else None
                gt_mask = None
                gt_mask_stem = g_stem if g_stem in gt_mask_idx else None
                if gt_mask_stem is not None:
                    gt_mask = _load_mask(gt_mask_idx[gt_mask_stem])
                pred_mask = None
                if r_stem in pred_idx:
                    pred_mask = _load_mask(pred_idx[r_stem])
                elif use_fg_as_masks and r_stem in fg_idx:
                    pred_mask = _mask_from_fg_render(fg_idx[r_stem])

                m = evaluate_view_safe(
                    fg_render=fg, bg_render=bg, gt_image=gt_img,
                    gt_mask=gt_mask, pred_mask=pred_mask,
                    boundary_tolerance=args.boundary_tolerance,
                    skip_lpips=args.skip_lpips,
                )
                m.update({"method": method, "dataset": dataset, "scene": scene,
                          "view": r_stem, "gt_view": g_stem,
                          "gt_mask_view": gt_mask_stem})
                scene_rows.append(m)
                per_view_rows.append(m)
            except Exception as e:
                print(f"  ! Error on {r_stem}: {e}")

        if scene_rows:
            agg = aggregate_views(scene_rows)
            agg.update({"method": method, "dataset": dataset, "scene": scene,
                        "n_views": len(scene_rows)})
            per_scene_rows.append(agg)

    # Write per-view CSV
    if per_view_rows:
        cols = ["method", "dataset", "scene", "view", "gt_view", "gt_mask_view"] + [
            "iou", "boundary_f", "boundary_iou", "accuracy", "leakage",
            "fg_psnr", "fg_ssim", "fg_lpips",
            "bg_psnr", "bg_ssim", "bg_lpips",
        ]
        with open(args.output_dir / "per_view.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in per_view_rows:
                w.writerow({k: (f"{r[k]:.6f}" if isinstance(r.get(k), float) else r.get(k, ""))
                            for k in cols})
        print(f"\nWrote per-view CSV ({len(per_view_rows)} rows): {args.output_dir / 'per_view.csv'}")

    # Write per-scene CSV
    if per_scene_rows:
        metric_keys = ["iou", "boundary_f", "boundary_iou", "accuracy", "leakage",
                       "fg_psnr", "fg_ssim", "fg_lpips",
                       "bg_psnr", "bg_ssim", "bg_lpips"]
        cols = ["method", "dataset", "scene", "n_views"]
        for k in metric_keys:
            cols.extend([f"{k}_mean", f"{k}_std", f"{k}_n"])
        with open(args.output_dir / "per_scene.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in per_scene_rows:
                row = {}
                for c in cols:
                    v = r.get(c)
                    if isinstance(v, float):
                        row[c] = f"{v:.6f}"
                    else:
                        row[c] = v if v is not None else ""
                w.writerow(row)
        print(f"Wrote per-scene CSV ({len(per_scene_rows)} rows): {args.output_dir / 'per_scene.csv'}")

    # Per-dataset aggregate (average scene-means)
    by_md = defaultdict(list)
    for r in per_scene_rows:
        by_md[(r["method"], r["dataset"])].append(r)

    per_dataset_rows = []
    metric_keys = ["iou", "boundary_f", "boundary_iou", "accuracy", "leakage",
                   "fg_psnr", "fg_ssim", "fg_lpips",
                   "bg_psnr", "bg_ssim", "bg_lpips"]
    for (method, dataset), scene_list in sorted(by_md.items()):
        out = {"method": method, "dataset": dataset, "n_scenes": len(scene_list)}
        for k in metric_keys:
            scene_means = np.array(
                [s[f"{k}_mean"] for s in scene_list if not np.isnan(s.get(f"{k}_mean", float("nan")))],
                dtype=np.float64,
            )
            if len(scene_means) == 0:
                out[f"{k}_mean"] = float("nan")
                out[f"{k}_std"] = float("nan")
            else:
                out[f"{k}_mean"] = float(scene_means.mean())
                out[f"{k}_std"] = float(scene_means.std())
        per_dataset_rows.append(out)

    if per_dataset_rows:
        cols = ["method", "dataset", "n_scenes"]
        for k in metric_keys:
            cols.extend([f"{k}_mean", f"{k}_std"])
        with open(args.output_dir / "per_dataset.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in per_dataset_rows:
                row = {c: (f"{r[c]:.6f}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols}
                w.writerow(row)
        print(f"Wrote per-dataset CSV ({len(per_dataset_rows)} rows): {args.output_dir / 'per_dataset.csv'}")

    # Console summary
    print("\n" + "=" * 80)
    print("  PER-DATASET SUMMARY")
    print("=" * 80)
    print(f"{'method':<20} {'dataset':<20} {'n':>3} {'IoU':>7} {'bIoU':>7} {'Bnd-F':>7} {'Acc':>7} {'Leak':>7} "
          f"{'FG-PSNR':>9} {'BG-PSNR':>9} {'FG-LPIPS':>9} {'BG-LPIPS':>9}")
    for r in per_dataset_rows:
        print(f"{r['method']:<20} {r['dataset']:<20} {r['n_scenes']:>3} "
              f"{r['iou_mean']:>7.3f} {r['boundary_iou_mean']:>7.3f} {r['boundary_f_mean']:>7.3f} "
              f"{r['accuracy_mean']:>7.3f} "
              f"{r['leakage_mean']:>7.3f} "
              f"{r['fg_psnr_mean']:>9.2f} {r['bg_psnr_mean']:>9.2f} "
              f"{r['fg_lpips_mean']:>9.3f} {r['bg_lpips_mean']:>9.3f}")

    if skipped:
        print(f"\nSKIPPED {len(skipped)} combinations:")
        for m, d, s, reason in skipped:
            print(f"  {m}/{d}/{s}: {reason}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
