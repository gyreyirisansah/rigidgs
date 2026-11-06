"""
2D Mask Rendering Loss for VR-RigidGS  —  CLEAN VERSION

Changes from `mask_loss_2d_fixed_gpt.py`:

1. `render_id_mask` now renders the SOFT FOREGROUND PROBABILITY derived from
   `gaussians.obj_assignment._object_logits` (learnable). Gradient flows
   through `override_color` → rasterizer → rendered_mask → BCE → logits.
   This means the mask loss can now CORRECT label assignments, not just
   adjust geometry of the currently-labeled Gaussians.

2. `mask_loss_2d` accepts an optional per-pixel `confidence` map. In
   practice this is always `MaskLoader.get_bg_eroded_weight`'s {0,1}
   background-erosion weight (see `erode_background_supervision` below):
   background-labeled pixels within `bg_margin_px` of the foreground
   silhouette are excluded from BCE, since depth-refined masks place a
   ring of background-labeled pixels right outside the true silhouette
   (a depth-discontinuity artifact) that would otherwise push fg
   Gaussians inward and leave debris.
"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# =====================================================================
# Mask loading (unchanged from the fixed_gpt version)
# =====================================================================
class MaskLoader:
    """
    Loads GT segmentation masks from a flat directory.

    Filenames must follow:
        {image_stem}_mask_{object_name}.{ext}

    Masks are binary: 255 = object, 0 = background.
    """

    def __init__(self, masks_dir, object_name_to_id_fn=None, extension=None):
        self.masks_dir = masks_dir
        self.object_name_to_id = object_name_to_id_fn
        self.extension = extension.lower() if isinstance(extension, str) else None
        self.cache = {}
        self.confidence_cache = {}  # NEW: cache for boundary confidence maps
        self.index = {}
        self.object_names = set()

        if not os.path.isdir(masks_dir):
            print(f"[MaskLoader] masks_dir does not exist: {masks_dir}")
            return

        pattern = os.path.join(masks_dir, "*_mask_*.*")
        files = glob.glob(pattern)
        n_empty_skipped = 0

        for fp in files:
            name = os.path.basename(fp)
            if "_mask_" not in name:
                continue
            stem_part, obj_part = name.split("_mask_", 1)
            obj_name = os.path.splitext(obj_part)[0]
            if not stem_part or not obj_name:
                continue

            if self._is_empty_mask(fp):
                n_empty_skipped += 1
                continue

            key = (stem_part, obj_name)

            if key in self.index and self.extension:
                cur_ext = os.path.splitext(self.index[key])[1].lower().lstrip(".")
                new_ext = os.path.splitext(fp)[1].lower().lstrip(".")
                if cur_ext == self.extension:
                    continue
                if new_ext != self.extension:
                    continue

            self.index[key] = fp
            self.object_names.add(obj_name)

        print(f"[MaskLoader] Indexed {len(self.index)} masks for "
              f"{len(self.object_names)} objects "
              f"({n_empty_skipped} empty masks skipped).")
        if len(self.object_names) > 0:
            objs = sorted(list(self.object_names))
            print(f"[MaskLoader] Objects: {objs[:20]}"
                  + (" ..." if len(objs) > 20 else ""))

    @staticmethod
    def _is_empty_mask(fp) -> bool:
        """True if the mask file has no foreground pixels.

        Raw SAM2 output writes one file per frame even when tracking found
        nothing (all-black). Indexing those would supervise "object is
        definitely absent here" on frames where it may simply be occluded
        or briefly lost, fighting the photometric loss on frames where the
        object is actually visible. Skipping them makes a failed detection
        behave like a missing file: no evidence, no supervision -- the same
        semantics the depth-refinement stage already enforces by not
        writing a file for an empty coarse mask.
        """
        with Image.open(fp) as img:
            return img.convert("L").getbbox() is None

    @staticmethod
    def _stem(image_name: str) -> str:
        return os.path.splitext(os.path.basename(image_name))[0]

    def get_mask(self, image_name, object_name, H=None, W=None):
        image_stem = self._stem(image_name)
        cache_key = (image_stem, object_name, H, W)
        if cache_key in self.cache:
            return self.cache[cache_key].to(device="cuda", non_blocking=True)

        fp = self.index.get((image_stem, object_name), None)
        if fp is None or (not os.path.exists(fp)):
            return None

        img = Image.open(fp).convert("L")
        mask_np = (np.array(img, dtype=np.float32) / 255.0)
        mask = torch.from_numpy(mask_np)
        mask = (mask > 0.5).float()

        if H is not None and W is not None and (mask.shape[0] != H or mask.shape[1] != W):
            mask = F.interpolate(mask[None, None], (H, W), mode="nearest")[0, 0]

        self.cache[cache_key] = mask.cpu()
        return self.cache[cache_key].to(device="cuda", non_blocking=True)

    def get_composite_mask(self, image_name, H=None, W=None):
        image_stem = self._stem(image_name)
        composite = None
        for (stem, obj_name), _fp in self.index.items():
            if stem != image_stem:
                continue
            mask = self.get_mask(image_name, obj_name, H, W)
            if mask is None:
                continue
            composite = mask if composite is None else torch.clamp(composite + mask, 0, 1)
        return composite

    def objects_for_frame(self, image_name):
        """Names of the objects that actually have a mask on this frame.

        The per-object loss supervises only these: an object absent from a
        frame has no evidence there, and inventing an all-zero target for it
        would push its Gaussians toward background every time it happens to
        be out of view.
        """
        stem = self._stem(image_name)
        return sorted(obj for (s, obj) in self.index if s == stem)

    def get_bg_eroded_weight(self, image_name, gt_mask, bg_margin_px=15,
                             object_name=None):
        """
        Returns a binary {0,1} pixel weight (H, W) that zeroes out BCE
        for background-labeled pixels within bg_margin_px of the foreground
        silhouette.  Foreground pixels always get weight 1.

        Cached by (image_stem, H, W, bg_margin_px) so distanceTransform
        only runs once per image per session. The per-object loss passes
        object_name, since each object's mask on the same frame has its own
        weight map; the composite path leaves it None.
        """
        H, W = gt_mask.shape
        image_stem = self._stem(image_name)
        cache_key = (image_stem, H, W, int(bg_margin_px), "bg_eroded", object_name)
        if cache_key in self.confidence_cache:
            return self.confidence_cache[cache_key].to(device="cuda", non_blocking=True)

        weight = erode_background_supervision(gt_mask, bg_margin_px=bg_margin_px)
        self.confidence_cache[cache_key] = weight.cpu()
        return self.confidence_cache[cache_key].to(device="cuda", non_blocking=True)


# =====================================================================
# Background-erosion weight (v2 silhouette supervision)
# =====================================================================
def erode_background_supervision(gt_mask, bg_margin_px=15):
    """
    Binary {0,1} pixel weight that silences BCE for background-labeled
    pixels within bg_margin_px of the foreground silhouette.

    Depth-refined masks place a ring of background-labeled pixels right
    outside the true silhouette (depth discontinuity artifact).  Supervising
    those pixels as background pushes fg Gaussians at the edge inward and
    leaves debris.  Setting their weight to 0 avoids this without touching
    the rest of the supervision.

    - Foreground pixels (GT=1):              weight = 1  (always supervised)
    - Background pixels far from silhouette: weight = 1  (correctly background)
    - Background pixels within bg_margin_px: weight = 0  (artifact-prone zone)
    """
    gt_np = gt_mask.detach().cpu().numpy().astype(np.uint8)
    # distance of each pixel to the nearest foreground pixel
    outside_dist = cv2.distanceTransform(1 - gt_np, cv2.DIST_L2, 3)
    weight = np.where(
        (gt_np == 0) & (outside_dist < float(bg_margin_px)),
        0.0,
        1.0,
    ).astype(np.float32)
    return torch.from_numpy(weight).to(gt_mask.device)


def mask_loss_2d_v2(rendered_mask, gt_mask, pixel_weight, loss_type="bce"):
    """
    BCE (or L1) with a pre-computed background-erosion pixel weight.

    pixel_weight is a {0,1} map from MaskLoader.get_bg_eroded_weight or
    erode_background_supervision.  Passes straight through to mask_loss_2d
    so the old function is untouched and the v1 path is always reachable.
    """
    return mask_loss_2d(rendered_mask, gt_mask,
                        confidence=pixel_weight, loss_type=loss_type)


# =====================================================================
# Foreground-probability rendering (CHANGED)
# =====================================================================
def render_object_prob_masks(viewpoint_cam, gaussians, pipe, render_fn, columns):
    """Soft PER-OBJECT probability masks: one (H, W) map per requested class.

    `render_id_mask` renders 1 - P(background), a single foreground/background
    map, compared against MaskLoader.get_composite_mask -- the OR of every
    object. With ONE object that already is per-object supervision. With
    several it collapses them: BCE cannot tell one object from another, and
    cannot pull a background-labelled Gaussian sitting on an object back
    into it.

    Three classes are packed into the RGB channels of a single rasterizer
    pass: alpha compositing is linear and independent per channel, so channel
    j of the result is exactly class j's own soft mask. Only the objects
    visible in a frame are requested, so this is usually one pass.

    Returns {column_index: (H, W) tensor}.
    """
    assignment = getattr(gaussians, "obj_assignment", None)
    if assignment is None or getattr(assignment, "_object_logits", None) is None:
        raise RuntimeError("per-object mask loss needs obj_assignment logits; "
                           "it cannot fall back to frozen ids (no gradient).")
    probs = assignment.get_object_probs()           # (N, K)

    black_bg = torch.zeros(3, device="cuda")
    out = {}
    cols = list(columns)
    for i in range(0, len(cols), 3):
        chunk = cols[i:i + 3]
        chans = [probs[:, c] for c in chunk]
        while len(chans) < 3:
            chans.append(torch.zeros_like(chans[0]))
        override_color = torch.stack(chans, dim=1)      # (N, 3)
        pkg = render_fn(viewpoint_cam, gaussians, pipe, black_bg,
                        override_color=override_color,
                        use_trained_exp=False, separate_sh=False)
        rendered = pkg["render"]                        # (3, H, W)
        for j, c in enumerate(chunk):
            out[c] = rendered[j]
    return out


def mask_loss_per_object(rendered_by_col, gt_by_col, weight_by_col=None,
                         loss_type="bce"):
    """Mean BCE over each object's own mask.

    For class k every pixel of a DIFFERENT object is a zero target, which is
    what supplies the inter-object discrimination the composite mask cannot.
    With a single object this is the composite loss: same target, same pixel
    weight, and with K = 2 classes P(object) = 1 - P(background).
    """
    losses = []
    for col, rendered in rendered_by_col.items():
        gt = gt_by_col.get(col)
        if gt is None:
            continue
        w = None if weight_by_col is None else weight_by_col.get(col)
        losses.append(mask_loss_2d(rendered, gt, confidence=w, loss_type=loss_type))
    if not losses:
        return None
    return torch.stack(losses).mean()


def render_id_mask(viewpoint_cam, gaussians, pipe, background, render_fn,
                   background_id=0):
    """
    Render a soft foreground probability mask through the Gaussian
    rasterizer, using learnable per-Gaussian object_logits.

    Per Gaussian: fg_prob_i = 1 - softmax(logits_i)[background_id]
    Render with each Gaussian's color set to (fg_prob_i, fg_prob_i, fg_prob_i)
    on a black background. Alpha compositing produces the soft FG mask.

    GRADIENT FLOW:
      object_logits → softmax → fg_prob → override_color → rasterizer
      → rendered_mask → BCE → backprop to object_logits.

    This lets BCE supervision CORRECT wrong labels, not just adjust geometry.

    Falls back to legacy frozen `_object_ids` path if obj_assignment isn't
    initialized (e.g. resumed from a checkpoint that pre-dates logits).
    """
    assignment = getattr(gaussians, "obj_assignment", None)

    if (assignment is not None
            and getattr(assignment, "_object_logits", None) is not None):
        # Learnable path: gradient flows.
        probs = assignment.get_object_probs()  # (N, K)
        # Guard against background_id out of range (e.g. if K < expected)
        bg_id = int(background_id)
        if bg_id >= probs.shape[1]:
            # If the logits don't model the bg class explicitly, fall back
            # to frozen labels — gradient won't flow, but at least we won't
            # crash. Should not happen in practice.
            fg_float = (gaussians._object_ids != background_id).float()
        else:
            fg_float = 1.0 - probs[:, bg_id]  # (N,)
    else:
        # Legacy path (frozen labels, no gradient).
        fg_float = (gaussians._object_ids != background_id).float()

    override_color = fg_float.unsqueeze(1).expand(-1, 3)  # (N, 3)
    black_bg = torch.zeros(3, device="cuda")

    render_pkg = render_fn(
        viewpoint_cam, gaussians, pipe, black_bg,
        override_color=override_color,
        use_trained_exp=False,
        separate_sh=False,
    )

    rendered_mask = render_pkg["render"][0]  # (H, W)
    return rendered_mask


# =====================================================================
# Loss (CHANGED: optional confidence weighting)
# =====================================================================
def mask_loss_2d(rendered_mask, gt_mask, confidence=None, loss_type="bce"):
    """
    Pixelwise BCE (or L1) between rendered foreground probability and GT mask.

    Optional `confidence` is a per-pixel weight (H, W) in [0, 1]. Boundary-band
    pixels (where the refined mask is unreliable) should have low weight so
    the loss doesn't aggressively assert incorrect labels there.

    Args:
        rendered_mask: (H, W) foreground probability in [0, 1]
        gt_mask:       (H, W) binary GT (0 or 1)
        confidence:    (H, W) per-pixel weight, or None for uniform weighting
        loss_type:     "bce" or "l1"
    """
    rendered_clamped = rendered_mask.clamp(1e-6, 1 - 1e-6)
    gt = gt_mask.to(device="cuda")

    if loss_type == "bce":
        if confidence is not None:
            # F.binary_cross_entropy supports per-element `weight`.
            conf = confidence.to(device="cuda")
            return F.binary_cross_entropy(rendered_clamped, gt, weight=conf)
        return F.binary_cross_entropy(rendered_clamped, gt)

    if loss_type == "l1":
        loss_per_pixel = (rendered_clamped - gt).abs()
        if confidence is not None:
            conf = confidence.to(device="cuda")
            return (loss_per_pixel * conf).mean()
        return loss_per_pixel.mean()

    raise ValueError(f"Unknown loss_type: {loss_type}")
