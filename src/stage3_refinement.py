"""
stage3_refinement.py -- Stage 3 of train.py: background refinement on the
LaMa-inpainted images (Sec. 3.5 of the paper).

  1. Hole seeding: before training, background Gaussians are inserted on the
     surface hidden behind the removed object, at a depth interpolated from
     the observed background around the hole (seed_hole_from_boundary_depth).
  2. Training alternates two steps, one per iteration, gradients restricted
     to background Gaussians in both (_zero_non_background_grads):
       odd  iterations: full-scene render, Stage 1's photometric loss
                        (L1 + D-SSIM) against the original image;
       even iterations: background-only render (foreground hidden),
                        L_bg = L1 against the inpainted image over the whole
                        frame.
     Outside the hole the inpainted image equals the Stage 1 background, so
     L_bg anchors the rest of the scene while it fills the hole.
  3. Densification: during the first half, background Gaussians inside a
     padded box around the removed object are cloned, split and pruned with
     Stage 1's thresholds, driven by the L_bg gradients.

The foreground Gaussians never move.
"""

import os
import sys
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, 'external/gaussian-splatting')
from gaussian_renderer import render
from utils.sh_utils import RGB2SH
from utils.loss_utils import l1_loss, ssim, create_window
from online_inpainting import render_depth_map, unproject_camera_depth


# =====================================================================
# 2D mask utilities
# =====================================================================
def _binary_dilate_torch(mask, radius_px):
    if radius_px is None or int(radius_px) <= 0:
        return mask.bool()
    r = int(radius_px)
    x = mask.float()[None, None]
    y = F.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)
    return (y[0, 0] > 0.5)


def _load_refinement_mask_file(refinement_mask_dir, cam, H, W):
    """
    Load the hole mask LaMa was given for this camera, saved by
    online_inpainting.batch_inpaint_all_cameras as {inpaint_cache_dir}/
    temp_input/iter_<inpaint_at_iter>/{stem}_mask.png. Returns a boolean
    tensor at (H, W), or None if the file doesn't exist.
    """
    if refinement_mask_dir is None:
        return None
    stem = Path(cam.image_name).stem
    p = Path(refinement_mask_dir) / f"{stem}_mask.png"
    if not p.exists():
        return None
    m = np.array(Image.open(p).convert("L"))
    t = torch.from_numpy(m).to(device="cuda", dtype=torch.float32)
    if t.shape[-2:] != (H, W):
        t = F.interpolate(t[None, None], size=(H, W), mode="nearest")[0, 0]
    return t > 127.0


# =====================================================================
# Image / inpainting lookup
# =====================================================================
def _find_inpainted_image(inpainted_dir, cam):
    """Locate the LaMa-inpainted image for a camera.

    LaMa often appends the input mask suffix to its output filename
    (e.g. IMG_4038.JPG + mask IMG_4038_mask001.png  →  IMG_4038_mask001.png).
    We check common naming patterns first, then fall back to a stem-prefix
    glob match.
    """
    if inpainted_dir is None:
        return None
    stem = Path(cam.image_name).stem            # e.g. "IMG_4038"
    name = Path(cam.image_name).name            # e.g. "IMG_4038.JPG"

    candidates = [
        Path(inpainted_dir) / f"{stem}.png",
        Path(inpainted_dir) / f"{stem}.jpg",
        Path(inpainted_dir) / f"{stem}.jpeg",
        Path(inpainted_dir) / name,
        Path(inpainted_dir) / f"{name}.png",
        Path(inpainted_dir) / f"{stem}_mask.png",
        Path(inpainted_dir) / f"{stem}_mask.jpg",
        Path(inpainted_dir) / f"{stem}_mask.jpeg",
        Path(inpainted_dir) / f"{stem}_inpainted.png",
        Path(inpainted_dir) / f"{stem}_inpaint.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # Fallback: any image whose stem starts with the camera stem
    # (case-insensitive), preferring the shortest stem.
    root = Path(inpainted_dir)
    if root.exists():
        stem_lower = stem.lower()
        matches = [p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and p.stem.lower().startswith(stem_lower)]
        if matches:
            matches.sort(key=lambda p: len(p.stem))
            return str(matches[0])
    return None


def _as_cuda_image(path):
    """Load an RGB image as a CHW float tensor in [0, 1] on cuda."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


# =====================================================================
# Gradient handling
# =====================================================================
def _zero_non_background_grads(gaussians, bg_mask):
    """Zero out FG gradients so only BG Gaussians actually move."""
    fg_mask = ~bg_mask
    for name in ("_xyz", "_features_dc", "_features_rest",
                 "_opacity", "_scaling", "_rotation"):
        t = getattr(gaussians, name, None)
        if (t is not None and getattr(t, "grad", None) is not None
            and t.grad.shape[0] == bg_mask.shape[0]):
            t.grad[fg_mask] = 0
    assignment = getattr(gaussians, "obj_assignment", None)
    logits = getattr(assignment, "_object_logits", None) if assignment is not None else None
    if logits is not None and getattr(logits, "grad", None) is not None:
        logits.grad = None


def _ensure_gaussian_params_grad(gaussians, verbose=True):
    """
    cleanup_before_inpainting detaches the Gaussian parameters to free memory
    before LaMa. Re-wrap each one as an nn.Parameter with requires_grad=True
    so Stage 3 can backpropagate; a no-op for parameters that are already
    trainable. Must run before training_setup, which builds the optimizer
    over these tensors.
    """
    rewrapped, enabled_grad = [], []
    for pname in ("_xyz", "_features_dc", "_features_rest",
                  "_opacity", "_scaling", "_rotation"):
        t = getattr(gaussians, pname, None)
        if t is None:
            continue
        if not isinstance(t, nn.Parameter):
            setattr(gaussians, pname, nn.Parameter(t.detach().requires_grad_(True)))
            rewrapped.append(pname)
        elif not t.requires_grad:
            t.requires_grad_(True)
            enabled_grad.append(pname)

    # _object_logits lives on obj_assignment, not gaussians
    assignment = getattr(gaussians, "obj_assignment", None)
    logits = getattr(assignment, "_object_logits", None) if assignment is not None else None
    if logits is not None:
        if not isinstance(logits, nn.Parameter):
            assignment._object_logits = nn.Parameter(logits.detach().requires_grad_(True))
            rewrapped.append("_object_logits")
        elif not logits.requires_grad:
            logits.requires_grad_(True)
            enabled_grad.append("_object_logits")

    if verbose and rewrapped:
        print(f"[Stage 3] Re-wrapped as nn.Parameter: {', '.join(rewrapped)}")
    if verbose and enabled_grad:
        print(f"[Stage 3] Re-enabled requires_grad: {', '.join(enabled_grad)}")


# =====================================================================
# Background-only renders
# =====================================================================
def _get_hard_labels(gaussians):
    """Return learned argmax labels if logits exist, else frozen _object_ids."""
    assignment = getattr(gaussians, "obj_assignment", None)
    if (assignment is not None
            and getattr(assignment, "_object_logits", None) is not None):
        return assignment.get_object_ids()
    return gaussians._object_ids


def save_background_hole_renders(gaussians, background_id, cameras,
                                  render_func, pipe, background,
                                  output_dir, iteration):
    """
    Render BG-only views (FG Gaussians masked out) for evaluating the hole.

    This is the counterpart to save_foreground_renders. Whereas FG renders
    show the extracted object alone, BG renders show what's behind the
    object after removal — directly evaluating hole/background quality.
    """
    labels = _get_hard_labels(gaussians)
    bg_keep = (labels == int(background_id))  # True for BG (keep these)
    out = os.path.join(output_dir, f"iter_{iteration:05d}")
    os.makedirs(out, exist_ok=True)
    for cam in tqdm(cameras, desc="BG hole renders"):
        with torch.no_grad():
            pkg = render_func(cam, gaussians, pipe, background,
                              object_mask=bg_keep,
                              use_trained_exp=False, separate_sh=False)
            img = pkg["render"].permute(1, 2, 0).cpu().numpy()
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            name = cam.image_name.split(".")[0]
            Image.fromarray(img).save(os.path.join(out, f"{name}.png"))
    print(f"  Saved BG hole renders → {out}")


# =====================================================================
# 3D region around the removed object
#
# Densification and pruning only touch background Gaussians inside a padded
# box around the foreground Gaussians, which do not move in Stage 3, so the
# rest of the scene keeps its Stage 1 geometry.
# =====================================================================
def compute_hole_region_box(gaussians, background_id, pad_factor=0.5):
    """
    Padded axis-aligned bounding box around the foreground Gaussians, padded
    by pad_factor times the box's largest half-extent. Returns
    (region_min, region_max), each a (3,) tensor, or None if there are no
    foreground Gaussians.
    """
    fg_mask = gaussians._object_ids != int(background_id)
    if int(fg_mask.sum().item()) == 0:
        return None
    fg_xyz = gaussians._xyz[fg_mask].detach()
    fg_min = fg_xyz.min(dim=0).values
    fg_max = fg_xyz.max(dim=0).values
    half_extent = (fg_max - fg_min) / 2.0
    pad = pad_factor * half_extent.max().clamp_min(1e-6)
    return fg_min - pad, fg_max + pad


def in_hole_region(xyz, region):
    """Boolean mask: which rows of `xyz` fall inside the padded box from
    compute_hole_region_box. region=None (no FG Gaussians) -> no
    restriction (everything passes)."""
    if region is None:
        return torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    region_min, region_max = region
    return ((xyz >= region_min) & (xyz <= region_max)).all(dim=1)


def _ssim_map(img1, img2, window_size=11):
    """Per-pixel SSIM map, not spatially averaged (utils.loss_utils.ssim
    always reduces to a scalar). Same formula as loss_utils._ssim; used by
    eval/evaluate_spinnerf.py to average SSIM over the hole mask only."""
    channel = img1.size(-3)
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


# =====================================================================
# Hole seeding
# =====================================================================
def _diffuse_into_hole(depth, known, iters, init=None):
    """Harmonic fill: repeatedly replace unknown pixels by their 3x3 mean,
    keeping known pixels fixed."""
    d = torch.where(known, depth, depth[known].mean() if init is None else init)
    for _ in range(iters):
        avg = F.avg_pool2d(d[None, None], 3, stride=1, padding=1,
                           count_include_pad=False)[0, 0]
        d = torch.where(known, depth, avg)
    return d


def seed_hole_from_boundary_depth(
    gaussians, cameras, pipe, background, background_id,
    targets, refinement_mask_dir, pixel_stride=4, ring_px=20,
    initial_opacity=0.3, verbose=True,
):
    """
    Insert new background Gaussians on the surface hidden behind the removed
    object, without an external depth prior. For each view: render the
    Stage 1 background depth, take it from a ring around the hole (where the
    background is actually observed), interpolate it smoothly across the
    hole, and unproject one hole pixel every `pixel_stride` px, coloured
    from that view's inpainted target. Points from all views are then
    voxel-merged with a voxel the size of one stride footprint at the median
    hole depth, so the number of seeds scales with the hole's 3D area.

    targets: {image_name: inpainted image (3, H, W) on cuda}.
    Returns the number of Gaussians inserted.
    """
    import open3d as o3d
    fg_obj_mask = (gaussians._object_ids != int(background_id))
    ones = torch.ones((gaussians.get_xyz.shape[0], 3), device="cuda")
    all_xyz, all_rgb, footprints = [], [], []

    with torch.no_grad():
        saved_fg_opacity = gaussians._opacity.data[fg_obj_mask].clone()
        gaussians._opacity.data[fg_obj_mask] = -50.0
        try:
            for cam in cameras:
                target = targets.get(cam.image_name)
                if target is None:
                    continue
                H, W = target.shape[-2:]
                hole = _load_refinement_mask_file(refinement_mask_dir, cam, H, W)
                if hole is None or int(hole.sum()) == 0:
                    continue
                alpha = render(cam, gaussians, pipe, background, override_color=ones,
                               use_trained_exp=False, separate_sh=False)["render"][0]
                depth = render_depth_map(cam, gaussians, pipe, background, render)
                depth = depth / alpha.clamp_min(1e-6)

                hole_d = _binary_dilate_torch(hole, 3)
                # Some scenes reach the photo's colours with many faint
                # Gaussians (accumulated alpha ~0.1), so "observed" is judged
                # against this view's own alpha level, not an absolute one.
                alpha_ref = float(alpha[~hole_d].median())
                known = ~hole_d & (alpha > min(0.9, 0.5 * alpha_ref))
                ring = _binary_dilate_torch(hole, ring_px) & known
                if int(ring.sum()) < 10:
                    if verbose and not footprints:
                        print(f"[Stage 3] Hole seeding: {cam.image_name}: ring "
                              f"{int(ring.sum())} px, hole {int(hole.sum())} px, "
                              f"observed outside hole {int(known.sum())} px, "
                              f"alpha median {float(alpha.median()):.3f}")
                    continue
                # Coarse-to-fine harmonic fill of the hole from the observed
                # background around it.
                s = 8
                kf = known.float()[None, None]
                d_c = (F.avg_pool2d(depth[None, None] * kf, s)
                       / F.avg_pool2d(kf, s).clamp_min(1e-6))[0, 0]
                d_c = _diffuse_into_hole(d_c, F.avg_pool2d(kf, s)[0, 0] > 0.5, iters=1500)
                d_h = (F.avg_pool2d(depth[None, None] * kf, 2)
                       / F.avg_pool2d(kf, 2).clamp_min(1e-6))[0, 0]
                d_c = F.interpolate(d_c[None, None], size=d_h.shape, mode="bilinear",
                                    align_corners=False)[0, 0]
                d_h = _diffuse_into_hole(d_h, F.avg_pool2d(kf, 2)[0, 0] > 0.99,
                                         iters=200, init=d_c)
                d_f = F.interpolate(d_h[None, None], size=(H, W), mode="bilinear",
                                    align_corners=False)[0, 0]

                grid = torch.zeros_like(hole)
                grid[pixel_stride // 2::pixel_stride, pixel_stride // 2::pixel_stride] = True
                xyz, vs, us = unproject_camera_depth(cam, d_f, mask=hole_d & grid)
                if xyz.shape[0] == 0:
                    if verbose and not footprints:
                        print(f"[Stage 3] Hole seeding: {cam.image_name}: no valid "
                              f"depth in hole (filled depth range "
                              f"{float(d_f.min()):.3f}..{float(d_f.max()):.3f})")
                    continue
                fx = W / (2.0 * np.tan(cam.FoVx / 2.0))
                footprints.append(float(d_f[vs, us].median()) * pixel_stride / fx)
                all_xyz.append(xyz)
                all_rgb.append(target[:, vs, us].T)
        finally:
            gaussians._opacity.data[fg_obj_mask] = saved_fg_opacity

    if not all_xyz:
        if verbose:
            print("[Stage 3] Hole seeding: no usable views; skipping.")
        return 0

    voxel = float(np.median(footprints))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(torch.cat(all_xyz).cpu().numpy().astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        torch.cat(all_rgb).clamp(0, 1).cpu().numpy().astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel)
    new_xyz = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
    new_rgb = torch.from_numpy(np.asarray(pcd.colors)).float().cuda()
    n = new_xyz.shape[0]

    sh_rest_dim = gaussians._features_rest.shape[1]
    new_scaling = torch.log(torch.full((n, 3), 0.5 * voxel, device="cuda"))
    new_rotation = torch.zeros((n, 4), device="cuda")
    new_rotation[:, 0] = 1.0
    gaussians.densification_postfix(
        new_xyz, RGB2SH(new_rgb)[:, None, :],
        torch.zeros((n, sh_rest_dim, 3), device="cuda"),
        gaussians.inverse_opacity_activation(
            initial_opacity * torch.ones((n, 1), device="cuda")),
        new_scaling, new_rotation, torch.zeros(n, device="cuda"),
        torch.full((n,), int(background_id), dtype=torch.long, device="cuda"),
    )
    if verbose:
        print(f"[Stage 3] Hole seeding: inserted {n:,} BG Gaussians from "
              f"{len(all_xyz)} views (stride {pixel_stride}px, voxel {voxel:.4f}, "
              f"opacity {initial_opacity}).")
    return n


# =====================================================================
# Stage 3
# =====================================================================
def run_stage3_refinement(
    gaussians, scene, opt, pipe, background, background_id,
    inpainted_dir, refinement_mask_dir,
    refine_iterations=7000,
    base_iter_offset=0,
    densify_until_frac=0.5,
    hole_region_pad_factor=0.5,
    seed_pixel_stride=4,
    seed_opacity=0.3,
    verbose=True,
):
    """
    Args:
      gaussians, scene, opt, pipe, background, background_id:
        standard 3DGS trainer arguments.
      inpainted_dir: LaMa outputs from Stage 2 (one image per training view).
      refinement_mask_dir: the hole masks LaMa was given (one per view), used
        to place the seeds.
      refine_iterations: total iterations, split evenly between the two steps.
      base_iter_offset: Stage 1's last iteration; the learning-rate schedule
        continues from it.
      densify_until_frac: fraction of the iterations during which background
        Gaussians near the removed object are densified and pruned, with
        opt's densify_from_iter, densification_interval and
        densify_grad_threshold.
      hole_region_pad_factor: padding of the box around the removed object
        that limits densification and pruning (see compute_hole_region_box).
      seed_pixel_stride, seed_opacity: one seed per seed_pixel_stride px of
        hole, inserted at opacity seed_opacity.

    Returns:
      dict with keys ran, used_steps, ema_loss_full, ema_loss_bg,
      final_n_gaussians.
    """
    if refine_iterations <= 0:
        return {"ran": False}

    torch.set_grad_enabled(True)

    N = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    for name, shape in [("max_radii2D",        (N,)),
                        ("xyz_gradient_accum", (N, 1)),
                        ("denom",              (N, 1)),
                        ("tmp_radii",          (N,))]:
        attr = getattr(gaussians, name, None)
        if attr is None or attr.shape[0] != N:
            setattr(gaussians, name, torch.zeros(shape, device=device))

    _ensure_gaussian_params_grad(gaussians, verbose=verbose)
    gaussians.training_setup(opt)

    cameras = scene.getTrainCameras().copy()
    if not cameras:
        print("[Stage 3] No training cameras available.")
        return {"ran": False}

    inpainted = {}
    for cam in cameras:
        p = _find_inpainted_image(inpainted_dir, cam)
        if p is not None:
            inpainted[cam.image_name] = p
    bg_cameras = [c for c in cameras if c.image_name in inpainted]
    if not bg_cameras:
        print(f"[Stage 3] ERROR: no inpainted images found in {inpainted_dir}.")
        return {"ran": False}
    if verbose:
        print(f"[Stage 3] {refine_iterations} iterations alternating full-scene "
              f"photometric (original images) and BG-only L1 (inpainted images); "
              f"{len(bg_cameras)}/{len(cameras)} views have an inpainted image.")

    missing = [c.image_name for c in bg_cameras
               if _load_refinement_mask_file(refinement_mask_dir, c, 1, 1) is None]
    if missing:
        print(f"[Stage 3] ERROR: hole seeding needs a hole mask per view; "
              f"{len(missing)} missing in {refinement_mask_dir} (e.g. {missing[0]}).")
        return {"ran": False}

    target_cache = {}

    def _target(cam):
        if cam.image_name not in target_cache:
            target = _as_cuda_image(inpainted[cam.image_name])
            H, W = cam.original_image.shape[-2:]
            if target.shape[-2:] != (H, W):
                target = F.interpolate(target[None], size=(H, W),
                                       mode="bilinear", align_corners=False)[0]
            target_cache[cam.image_name] = target
        return target_cache[cam.image_name]

    seed_hole_from_boundary_depth(
        gaussians, bg_cameras, pipe, background, background_id,
        {c.image_name: _target(c) for c in bg_cameras}, refinement_mask_dir,
        pixel_stride=seed_pixel_stride, initial_opacity=seed_opacity,
        verbose=verbose)

    grad_threshold = opt.densify_grad_threshold
    densify_until_local_iter = int(refine_iterations * densify_until_frac)
    hole_region = compute_hole_region_box(gaussians, background_id,
                                          pad_factor=hole_region_pad_factor)
    region_min, region_max = hole_region if hole_region is not None else (None, None)
    # Start from fresh statistics, not whatever Stage 1 left behind.
    gaussians.xyz_gradient_accum.zero_()
    gaussians.denom.zero_()
    gaussians.max_radii2D.zero_()
    if verbose:
        print(f"[Stage 3] Densify + prune near the hole: iters "
              f"{opt.densify_from_iter}-{densify_until_local_iter}, "
              f"interval {opt.densification_interval}, grad_threshold {grad_threshold}, "
              f"pad {hole_region_pad_factor}")

    ema_full = ema_bg = 0.0
    used = 0

    progress = tqdm(range(1, refine_iterations + 1), desc="Stage 3")
    for local_iter in progress:
        gaussians.update_learning_rate(base_iter_offset + local_iter)
        bg_step = (local_iter % 2 == 0)
        # Recomputed every step: densification changes the Gaussian count.
        bg_obj_mask = (gaussians._object_ids == int(background_id))
        fg_obj_mask = ~bg_obj_mask

        if not bg_step:
            cam = cameras[randint(0, len(cameras) - 1)]
            render_pkg = render(cam, gaussians, pipe, background,
                                use_trained_exp=False, separate_sh=False)
            pred = render_pkg["render"]
            if cam.alpha_mask is not None:
                pred = pred * cam.alpha_mask.cuda()
            gt = cam.original_image.cuda()
            loss = ((1.0 - opt.lambda_dssim) * l1_loss(pred, gt)
                    + opt.lambda_dssim * (1.0 - ssim(pred, gt)))
            loss.backward()
            # TEMPORARILY DISABLED for A/B comparison against the frond-residue
            # report (2026-11-06) -- re-enable before any real/paper run.
            # _zero_non_background_grads(gaussians, bg_obj_mask)
            ema_full = 0.4 * float(loss.item()) + 0.6 * ema_full
        else:
            cam = bg_cameras[randint(0, len(bg_cameras) - 1)]
            target = _target(cam)

            with torch.no_grad():
                saved_fg_opacity = gaussians._opacity.data[fg_obj_mask].clone()
                gaussians._opacity.data[fg_obj_mask] = -50.0
            try:
                render_pkg = render(cam, gaussians, pipe, background,
                                    use_trained_exp=False, separate_sh=False)
                pred = render_pkg["render"]
                loss = l1_loss(pred, target)
                loss.backward()
            finally:
                with torch.no_grad():
                    gaussians._opacity.data[fg_obj_mask] = saved_fg_opacity
            _zero_non_background_grads(gaussians, bg_obj_mask)
            ema_bg = 0.4 * float(loss.item()) + 0.6 * ema_bg

            if local_iter < densify_until_local_iter:
                visible = render_pkg["visibility_filter"].flatten()
                radii = render_pkg["radii"]
                gaussians.max_radii2D[visible] = torch.max(
                    gaussians.max_radii2D[visible], radii[visible])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], visible)
                if (local_iter > opt.densify_from_iter
                        and local_iter % opt.densification_interval == 0):
                    densify_mask = bg_obj_mask & in_hole_region(gaussians._xyz, hole_region)
                    n_before = int(gaussians._xyz.shape[0])
                    gaussians.densify_and_prune_background(
                        grad_threshold, 0.005, scene.cameras_extent, 20, radii,
                        int(background_id), densify_mask,
                        hole_region_min=region_min, hole_region_max=region_max)
                    if verbose:
                        print(f"[Stage 3] iter {local_iter}: Gaussians "
                              f"{n_before:,} -> {int(gaussians._xyz.shape[0]):,}")

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        used += 1
        if local_iter % 10 == 0:
            progress.set_postfix({"full": f"{ema_full:.4f}", "bg": f"{ema_bg:.4f}"})

    progress.close()
    if verbose:
        print(f"[Stage 3] Done. EMA full-scene loss={ema_full:.4f}, "
              f"BG loss={ema_bg:.4f}, Gaussians={len(gaussians._xyz):,}")
    return {
        "ran": True,
        "used_steps": used,
        "ema_loss_full": ema_full,
        "ema_loss_bg": ema_bg,
        "final_n_gaussians": len(gaussians._xyz),
    }
