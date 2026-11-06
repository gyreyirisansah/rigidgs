"""
online_inpainting.py -- Stage 2 of train.py: hole detection on the
background-only renders and LaMa inpainting. Also provides the depth render
and unprojection that stage3_refinement.py's hole seeding uses.

Hole masks come from SAM2 prompt + propagation (if enabled), else fall back
to intensity-based detection.

Key memory changes vs. original:
  - render_all_views: SKIPS the "full" render (not used anywhere in inpainting)
  - render_all_views: fg render is used only to compute fg_mask, then deleted immediately
  - batch_inpaint_all_cameras: renders dict is deleted after hole detection to free GPU memory
    before the LaMa subprocess runs
"""

import os
import sys
import time
import math
import tempfile
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from sam2_hole_detection import SAM2HoleDetector


# =====================================================================
# Stage 2 timing (new): populated by batch_inpaint_all_cameras() each time
# it runs, read back afterward via get_last_inpainting_timings(). A module-
# level dict rather than a return value / out-parameter, so this doesn't
# require touching batch_inpaint_all_cameras's signature or its one call
# site's unpacking -- the trainer just calls the getter after the call
# instead. Fine for this codebase's single-process, synchronous training
# loop (not meant to be thread-safe).
# =====================================================================
_LAST_STAGE2_TIMINGS: Dict[str, float] = {}


def get_last_inpainting_timings() -> Dict[str, float]:
    """Seconds spent in each Stage 2 sub-phase during the most recent
    batch_inpaint_all_cameras() call: hole_click_sec (interactive SAM2
    prompt UI -- wall-clock includes however long the user took to click),
    hole_prediction_sec (SAM2 mask propagation across frames, or the
    non-SAM2 detect_holes_batch fallback), inpainting_sec (the actual LaMa
    RGB inpainting call). Returns {} if batch_inpaint_all_
    cameras hasn't run yet (or skipped everything via skip_inpainting_
    forever, in which case none of these sub-phases ran)."""
    return dict(_LAST_STAGE2_TIMINGS)


class OnlineInpainter:
    """
    Batch inpainting using existing SAM masks (and optional SAM2 propagation).

    Supports ONLY two hole-mask modes:
      - "sam2"
      - "convex"
    """

    def __init__(
        self,
        lama_model_path: str = "external/lama/big-lama",
        sam_masks_dir: str = None,
        cache_dir: str = None,
        save_to_disk: bool = True,
        save_debug_images: bool = False,
        device: str = "cuda",
        use_direct_inference: bool = True,
        sam2_checkpoint: str = None,
        sam2_config: str = None,
        use_sam2_propagation: bool = False,
        mask_dilation_pixels: int = 20,
    ):
        self.lama_model_path      = lama_model_path
        self.sam_masks_dir        = Path(sam_masks_dir) if sam_masks_dir else None
        self.cache_dir            = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
        self.save_to_disk         = save_to_disk
        self.save_debug_images    = save_debug_images
        self.device               = device
        self.use_direct_inference = use_direct_inference
        self.use_sam2_propagation = use_sam2_propagation
        self.mask_dilation_pixels = mask_dilation_pixels

        self.skip_inpainting_forever = False
        self.skip_reason             = None

        if use_sam2_propagation:
            self.sam2_detector = SAM2HoleDetector(
                sam2_checkpoint=sam2_checkpoint,
                sam2_config=sam2_config,
                device=device,
                use_interactive_ui=True,
                ui_backend="opencv",
            )
        else:
            self.sam2_detector = None

        self.temp_input_dir  = self.cache_dir / "temp_input"
        self.temp_output_dir = self.cache_dir / "temp_output"
        self.debug_dir       = self.cache_dir / "debug_visualizations"

        self.temp_input_dir.mkdir(parents=True, exist_ok=True)
        self.temp_output_dir.mkdir(parents=True, exist_ok=True)
        if save_debug_images:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.sam_masks_cache: Dict[str, torch.Tensor] = {}

        self.lama_model = None
        if use_direct_inference:
            try:
                self._load_lama_model()
                print("[OnlineInpainter] ✓ Loaded LaMa model")
            except Exception as e:
                print(f"[OnlineInpainter] ⚠️  Failed to load LaMa: {e}")
                self.use_direct_inference = False

        dil_msg = (f", mask_dilation={mask_dilation_pixels}px"
                   if mask_dilation_pixels > 0 else "")
        print(f"[OnlineInpainter] Initialized{dil_msg}")

    # ------------------------------------------------------------------
    # Mask dilation helper
    # ------------------------------------------------------------------

    def dilate_mask(self, mask: torch.Tensor, pixels: int) -> torch.Tensor:
        if pixels <= 0:
            return mask
        import cv2
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        ksize   = 2 * pixels + 1
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        return torch.from_numpy(dilated).float().to(mask.device)

    # ------------------------------------------------------------------
    # Hole detection: SAM2
    # ------------------------------------------------------------------

    def detect_holes_batch_with_sam2(
        self,
        renders: Dict[str, Dict],
        hole_threshold: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        if self.sam2_detector is None:
            raise RuntimeError("SAM2 detector not initialized")

        camera_names = sorted(renders.keys())
        first_cam    = camera_names[0]
        bg_renders   = {cam: renders[cam]["bg"] for cam in camera_names}

        hole_masks = self.sam2_detector.propagate_holes_through_frames(
            bg_renders=bg_renders,
            first_cam_name=first_cam,
            hole_threshold=hole_threshold,
            temp_dir=self.cache_dir,
            initial_prompts=None,
        )

        if self.mask_dilation_pixels > 0:
            print(f"[OnlineInpainter] Dilating SAM2 masks by {self.mask_dilation_pixels}px...")
            hole_masks = {
                cam: self.dilate_mask(mask, self.mask_dilation_pixels)
                for cam, mask in hole_masks.items()
            }

        return hole_masks

    # ------------------------------------------------------------------
    # Hole detection: intensity
    # ------------------------------------------------------------------

    def detect_holes_batch_intensity(
        self,
        renders: Dict[str, Dict],
        hole_threshold: float = 0.05,
        min_hole_size: int = 1000,
        save_debug: bool = True
    ) -> Dict[str, torch.Tensor]:
        holes = {}
        print(f"[OnlineInpainter] Detecting main holes (intensity) ...")

        for cam_name, render_data in renders.items():
            sam_mask = self.load_sam_mask(cam_name)

            if sam_mask is None:
                holes[cam_name] = torch.zeros_like(render_data['fg_mask'])
                continue

            bg_image = render_data['bg']

            if sam_mask.shape != bg_image.shape[1:]:
                sam_mask = torch.nn.functional.interpolate(
                    sam_mask.unsqueeze(0).unsqueeze(0),
                    size=bg_image.shape[1:], mode='nearest'
                ).squeeze(0).squeeze(0)

            bg_intensity = bg_image.mean(dim=0)
            bg_is_empty  = (bg_intensity < hole_threshold).float()
            raw_holes    = (sam_mask > 0.5).float() * bg_is_empty

            cleaned_holes = self.clean_mask_keep_main_hole(
                hole_mask=raw_holes,
                sam_mask=sam_mask,
                min_hole_size=min_hole_size,
                connectivity_threshold=500
            )

            if self.mask_dilation_pixels > 0:
                cleaned_holes = self.dilate_mask(cleaned_holes, self.mask_dilation_pixels)
                cleaned_holes = cleaned_holes * (sam_mask > 0.5).float()

            holes[cam_name] = cleaned_holes

            if save_debug and raw_holes.sum() > 0:
                try:
                    debug_path = self.debug_dir / f"{cam_name}_holes_vs_texture.png"
                    self.visualize_hole_vs_texture(
                        cam_name, bg_image, bg_intensity, sam_mask,
                        raw_holes, cleaned_holes, str(debug_path)
                    )
                except Exception:
                    pass

        total_holes = sum(mask.sum().item() for mask in holes.values())
        print(f"[OnlineInpainter] ✓ Final holes (intensity): {total_holes:.0f} pixels")
        return holes

    # ------------------------------------------------------------------
    # Hole detection
    # ------------------------------------------------------------------

    def detect_holes_batch(
        self,
        renders,
        hole_threshold=0.05,
        min_hole_size: int = 1000,
        save_debug: bool = True,
        **kwargs
    ):
        if self.use_sam2_propagation and self.sam2_detector is not None:
            return self.detect_holes_batch_with_sam2(renders, hole_threshold)

        return self.detect_holes_batch_intensity(
            renders=renders,
            hole_threshold=hole_threshold,
            min_hole_size=min_hole_size,
            save_debug=save_debug,
        )

    # ------------------------------------------------------------------
    # Render all views
    # KEY CHANGE: skip "full" render (not used); delete fg render after
    # extracting fg_mask to immediately free that VRAM.
    # ------------------------------------------------------------------

    def render_all_views(
        self,
        gaussians,
        background_id: int,
        cameras: List,
        render_func,
        pipe,
        background,
        iteration: int
    ) -> Dict[str, Dict]:
        renders = {}
        print(f"\n[OnlineInpainter] Rendering all {len(cameras)} views "
              f"(bg + fg_mask only — full render skipped)...")

        bg_mask = gaussians._object_ids == background_id
        fg_mask = ~bg_mask

        for camera in tqdm(cameras, desc="Rendering views"):
            with torch.no_grad():
                # ── Background render (needed by hole detection + LaMa input) ──
                bg_pkg   = render_func(camera, gaussians, pipe, background,
                                       object_mask=bg_mask,
                                       use_trained_exp=False, separate_sh=False)
                image_bg = bg_pkg["render"].detach().clone()
                del bg_pkg

                # ── Foreground render — only to derive fg_mask ──────────────
                fg_pkg        = render_func(camera, gaussians, pipe, background,
                                            object_mask=fg_mask,
                                            use_trained_exp=False, separate_sh=False)
                fg_intensity      = fg_pkg["render"].mean(dim=0)
                foreground_mask   = (fg_intensity > 0.01).float().detach().clone()
                del fg_pkg, fg_intensity   # free immediately

                renders[camera.image_name] = {
                    'camera':  camera,
                    'bg':      image_bg,
                    'fg_mask': foreground_mask,
                    # NOTE: 'full' and 'fg' are intentionally absent
                }

        torch.cuda.empty_cache()
        print(f"[OnlineInpainter] ✓ All views rendered")
        return renders

    # ------------------------------------------------------------------
    # Mask cleanup
    # ------------------------------------------------------------------

    def clean_mask_keep_main_hole(
        self,
        hole_mask: torch.Tensor,
        sam_mask: torch.Tensor,
        min_hole_size: int = 1000,
        connectivity_threshold: int = 500
    ) -> torch.Tensor:
        import cv2

        hole_np = hole_mask.detach().cpu().numpy().astype(np.uint8)
        sam_np  = sam_mask.detach().cpu().numpy().astype(np.uint8)

        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        connected = cv2.morphologyEx(hole_np, cv2.MORPH_CLOSE, kernel_connect)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            connected, connectivity=8
        )

        cleaned = np.zeros_like(hole_np)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_hole_size:
                cleaned[labels == i] = 1

        kernel_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned       = cv2.erode(cleaned, kernel_erode, iterations=1)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned       = cv2.dilate(cleaned, kernel_dilate, iterations=1)

        cleaned = cleaned * sam_np
        return torch.from_numpy(cleaned).float().to(hole_mask.device)

    def visualize_hole_vs_texture(
        self, cam_name, bg_render, bg_intensity, sam_mask,
        raw_holes, cleaned_holes, output_path
    ):
        from PIL import Image as PILImage, ImageDraw, ImageFont

        removed       = torch.clamp(raw_holes - cleaned_holes, 0, 1)
        bg_pil        = self.tensor_to_pil(bg_render)
        intensity_pil = self.tensor_to_pil(bg_intensity.unsqueeze(0))
        sam_pil       = self.tensor_to_pil(sam_mask.unsqueeze(0))
        raw_pil       = self.tensor_to_pil(raw_holes.unsqueeze(0))
        removed_pil   = self.tensor_to_pil(removed.unsqueeze(0))
        clean_pil     = self.tensor_to_pil(cleaned_holes.unsqueeze(0))

        width, height = bg_pil.size
        label_height  = 40
        combined = PILImage.new('RGB', (width * 6, height + label_height))
        draw     = ImageDraw.Draw(combined)
        labels   = ["BG Render", "Intensity", "SAM Mask",
                    "Raw Holes", "Removed", "Final Holes"]

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        for i, label in enumerate(labels):
            draw.text((width * i + 5, 5), label, fill="white", font=font)

        combined.paste(bg_pil,                       (0,       label_height))
        combined.paste(intensity_pil.convert('RGB'), (width,   label_height))
        combined.paste(sam_pil.convert('RGB'),       (width*2, label_height))
        combined.paste(raw_pil.convert('RGB'),       (width*3, label_height))
        combined.paste(removed_pil.convert('RGB'),   (width*4, label_height))
        combined.paste(clean_pil.convert('RGB'),     (width*5, label_height))
        combined.save(output_path)

    # ------------------------------------------------------------------
    # LaMa IO
    # ------------------------------------------------------------------

    def _load_lama_model(self):
        sys.path.insert(0, 'external/lama')
        from omegaconf import OmegaConf
        import yaml

        config_path = Path(self.lama_model_path) / "config.yaml"
        with open(config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        from saicinpainting.training.trainers import load_checkpoint
        checkpoint_path = Path(self.lama_model_path) / "models" / "best.ckpt"
        self.lama_model = load_checkpoint(
            train_config, checkpoint_path,
            strict=False, map_location=self.device
        )
        self.lama_model.eval()
        self.lama_model.to(self.device)

    def load_sam_mask(self, camera_name: str) -> Optional[torch.Tensor]:
        if camera_name in self.sam_masks_cache:
            return self.sam_masks_cache[camera_name]
        if self.sam_masks_dir is None:
            return None

        camera_basename = Path(camera_name).stem
        for name in [f"{camera_basename}_mask.png", f"{camera_basename}_mask.jpg"]:
            mask_path = self.sam_masks_dir / name
            if mask_path.exists():
                mask_pil    = Image.open(mask_path).convert('L')
                mask_np     = np.array(mask_pil) > 128
                mask_tensor = torch.from_numpy(mask_np).float().to(self.device)
                self.sam_masks_cache[camera_name] = mask_tensor
                return mask_tensor

        print(f"⚠️  No SAM mask found for {camera_name} in {self.sam_masks_dir}")
        return None

    def preload_all_sam_masks(self, camera_names: List[str]) -> None:
        print(f"[OnlineInpainter] Preloading {len(camera_names)} SAM masks...")
        for cam_name in tqdm(camera_names, desc="Loading SAM masks"):
            self.load_sam_mask(cam_name)
        print(f"[OnlineInpainter] ✓ Loaded {len(self.sam_masks_cache)}/{len(camera_names)} masks")

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor is None:
            raise ValueError("Cannot convert None tensor to PIL Image")
        img_np = torch.clamp(tensor, 0, 1).detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        elif img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np[0]
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.ndim == 2:
            return Image.fromarray(img_np, mode='L')
        return Image.fromarray(img_np)

    def pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (2, 0, 1))
        else:
            img_np = img_np[np.newaxis, ...]
        return torch.from_numpy(img_np).to(self.device)

    def save_inpainting_input_images(self, renders, holes, iteration):
        print(f"\n[OnlineInpainter] Saving inpainting input images...")
        temp_iter_dir = self.temp_input_dir / f"iter_{iteration:05d}"
        temp_iter_dir.mkdir(parents=True, exist_ok=True)

        for cam_name, render_data in renders.items():
            hole_mask    = holes[cam_name]
            new_cam_name = cam_name.split(".")[0]
            image_path   = temp_iter_dir / f"{new_cam_name}.jpg"
            mask_path    = temp_iter_dir / f"{new_cam_name}_mask.png"

            self.tensor_to_pil(render_data['bg']).save(image_path)
            self.tensor_to_pil(hole_mask.unsqueeze(0)).convert('L').save(mask_path)

    def inpaint_all_images(self, indir, outdir) -> None:
        print(f"\n[OnlineInpainter] Running LaMa subprocess...")
        # Hydra's @hydra.main changes the process's working directory
        # before predict.py's main() body runs, so any RELATIVE indir/
        # outdir/model.path silently resolves against Hydra's own managed
        # directory instead of wherever this was invoked from -- no crash,
        # just make_default_val_dataset() finding zero (image, mask) pairs
        # there and the per-image loop simply never running. Confirmed
        # directly: the same indir/model.path, run manually, produced 0
        # output files as relative paths and 60/60 as absolute paths.
        # os.path.abspath() makes this immune to that regardless of what
        # the caller passed in.
        cmd = [
            sys.executable, "./external/lama/bin/predict.py",
            f"model.path={os.path.abspath(self.lama_model_path)}",
            f"indir={os.path.abspath(indir)}",
            f"outdir={os.path.abspath(outdir)}",
            "dataset.img_suffix=.jpg",
        ]
        subprocess.run(cmd, check=True)
        print(f"[OnlineInpainter] ✓ LaMa complete")

    def get_inpainting_results(self, inpainted_output_dir, iteration):
        inpainted = {}
        # temp_iter_output_dir = Path(inpainted_output_dir) / f"iter_{iteration:05d}"
        temp_iter_output_dir = Path(inpainted_output_dir) 
        print(f"\n[OnlineInpainter] Loading inpainted images from {temp_iter_output_dir}...")

        for image_file in tqdm(os.listdir(temp_iter_output_dir), desc="Loading inpainted"):
            if image_file.endswith(".png") or image_file.endswith(".jpg"):
                ext      = image_file.split('.')[-1]
                cam_name = image_file.split('_mask')[0] + f".{ext}"
                inpainted_pil    = Image.open(temp_iter_output_dir / image_file).convert('RGB')
                inpainted[cam_name] = self.pil_to_tensor(inpainted_pil)

        print(f"[OnlineInpainter] ✓ Loaded {len(inpainted)} inpainted images")
        return inpainted

    def inpaint_batch(self, renders, holes, inpainted_output_dir, iteration):
        print(f"\n[OnlineInpainter] Inpainting {len(renders)} views...")
        start_time = time.time()

        self.save_inpainting_input_images(renders, holes, iteration)

        temp_iter_input_dir  = self.temp_input_dir / f"iter_{iteration:05d}"
        temp_iter_output_dir = Path(inpainted_output_dir)
        temp_iter_output_dir.mkdir(parents=True, exist_ok=True)

        self.inpaint_all_images(indir=str(temp_iter_input_dir),
                                outdir=str(temp_iter_output_dir))

        inpainted = self.get_inpainting_results(
            inpainted_output_dir=str(inpainted_output_dir),
            iteration=iteration)

        elapsed = time.time() - start_time
        print(f"[OnlineInpainter] ✓ Done in {elapsed:.2f}s "
              f"({elapsed / max(len(renders), 1):.2f}s/image)")
        return inpainted

    def clean_ghost_from_background(self, render_bg, original_mask, method='nlm', threshold=0.05):
        import cv2
        ghost_intensity        = render_bg.abs().mean(dim=0)
        original_mask_squeezed = original_mask.squeeze()
        if original_mask_squeezed.shape != ghost_intensity.shape:
            original_mask_squeezed = torch.nn.functional.interpolate(
                original_mask_squeezed.unsqueeze(0).unsqueeze(0),
                size=ghost_intensity.shape, mode='nearest'
            ).squeeze(0).squeeze(0)
        ghost_mask = ((ghost_intensity > threshold) & (original_mask_squeezed > 0.5))
        if ghost_mask.sum() < 10:
            return render_bg
        img_np = render_bg.permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        if method == 'nlm':
            cleaned = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10,
                                                      templateWindowSize=7, searchWindowSize=21)
        elif method == 'bilateral':
            cleaned = cv2.bilateralFilter(img_np, 15, 75, 75)
        elif method == 'median':
            cleaned = cv2.medianBlur(img_np, 11)
        elif method == 'morph':
            k_o     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, k_o)
            k_c     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_c)
        else:
            cleaned = cv2.bilateralFilter(img_np, 15, 75, 75)
        cleaned_bg = torch.from_numpy(cleaned).permute(2, 0, 1).float().to(render_bg.device)
        if cleaned_bg.max() > 1.0:
            cleaned_bg = cleaned_bg / 255.0
        return cleaned_bg


# ------------------------------------------------------------------
# Standalone background-save helpers
# ------------------------------------------------------------------

def save_backgrounds_only(renders, inpainted_output_dir, iteration, backgrounds_to_save=None):
    temp_iter_output_dir = Path(inpainted_output_dir) / f"iter_{iteration:05d}"
    temp_iter_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OnlineInpainter] Saving backgrounds...")
    inpainter_stub = OnlineInpainter.__new__(OnlineInpainter)
    for cam_name, render_data in renders.items():
        output_path = temp_iter_output_dir / f"{cam_name}.jpg"
        bg_to_save  = backgrounds_to_save[cam_name] if backgrounds_to_save else render_data['bg']
        inpainter_stub.tensor_to_pil(bg_to_save).save(output_path)
    print(f"[OnlineInpainter] ✓ Saved {len(renders)} backgrounds")


# =============================================================================
# DEPTH PRIOR (new -- see module docstring)
# =============================================================================
def render_depth_map(camera, gaussians, pipe, background, render_func, object_mask=None):
    """
    Alpha-composited camera-space depth via the existing rasterizer's
    override_color path: per-Gaussian camera-space Z is splatted exactly
    like RGB color (broadcast to 3 identical channels), so this reuses the
    real, tested rasterizer/compositing code -- no new CUDA kernel needed.
    Depth convention: camera-space Z from world_view_transform, matching
    what Camera.world_view_transform already uses everywhere else in this
    codebase (world_view_transform = getWorld2View2(...).transpose(0,1), so
    p_cam = p_world_homogeneous @ world_view_transform).

    Does NOT wrap in torch.no_grad() -- callers control that (prior
    generation wants no_grad; the Stage 3 depth loss needs gradients to
    flow back into the Gaussians being refined).

    IMPORTANT: render()'s last line unconditionally does
    `rendered_image = rendered_image.clamp(0, 1)`
    (external/gaussian-splatting/gaussian_renderer/__init__.py) -- it has
    no raw/unclamped mode, and that file isn't ours to change. Real per-
    Gaussian camera-space depth in scene units is essentially never <=1,
    so splatting it directly as override_color used to get silently
    squashed by that clamp into a near-binary 0/1 map (empty space vs.
    "some surface", no actual depth variation at all) -- confirmed on
    disk: two real depth-prior runs both produced d_min=0.0, d_range=1.0
    exactly, which is what you get when depth.max() >> 1 gets clamped to
    exactly 1.0 instead of reflecting the scene's real depth range.

    Fix: normalize by this camera's own max depth (a cheap per-Gaussian
    max over already-computed camera-space Z, no extra render needed)
    before compositing as color, so the value that survives render()'s
    clamp is still <=1; multiply the clamped output back up by the same
    factor afterward to recover real depth. max_depth is detached (a
    plain scale constant, not something to backprop into) but the
    rendered depth itself keeps its gradient chain through `depth`/`xyz`,
    so this doesn't change the "needs gradients" contract above.

    Returns (H, W) -- all 3 rendered channels are identical, take one.
    """
    xyz = gaussians.get_xyz
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)
    cam_space = xyz_h @ camera.world_view_transform
    depth = cam_space[:, 2:3].clamp_min(0.0)

    max_depth = float(depth.max().item())
    if max_depth < 1e-6:
        max_depth = 1.0
    depth_color = (depth / max_depth).repeat(1, 3)
    pkg = render_func(camera, gaussians, pipe, background, object_mask=object_mask,
                       use_trained_exp=False, separate_sh=False, override_color=depth_color)
    return pkg["render"][0] * max_depth


def unproject_camera_depth(camera, depth: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    Inverse of render_depth_map's camera-space Z convention: given a per-
    pixel depth map (raw camera-space Z, same units render_depth_map
    uses -- NOT normalized), unproject the
    pixels selected by `mask` (or all pixels if None) into world-space XYZ.

    Pinhole intrinsics are derived from camera.FoVx/FoVy and the depth map's
    own (H, W) -- this repo's SPIn-NeRF cameras are all PINHOLE with a
    centered principal point, consistent with how spinnerf_utils.py
    and full_run5_spin_nerf.py construct them.

    Inverts world_view_transform directly rather than assuming a closed-form
    R/T split, since Camera.world_view_transform already bakes in whatever
    convention (translate/scale) getWorld2View2 used -- same reasoning
    render_depth_map's own docstring gives for reusing that matrix as-is:
    p_cam_h = p_world_h @ world_view_transform, so p_world_h = p_cam_h @
    inverse(world_view_transform) (row-vector convention).

    Returns (world_xyz [K,3], vs [K], us [K]) -- pixel row/col indices of
    each returned point, so callers can index a co-registered RGB image.
    """
    device = depth.device
    H, W = depth.shape[-2:]
    fx = W / (2.0 * math.tan(camera.FoVx / 2.0))
    fy = H / (2.0 * math.tan(camera.FoVy / 2.0))
    cx, cy = W / 2.0, H / 2.0

    if mask is not None:
        vs, us = torch.nonzero(mask.bool(), as_tuple=True)
    else:
        grid_v, grid_u = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        vs, us = grid_v.reshape(-1), grid_u.reshape(-1)

    d = depth[vs, us]
    valid = d > 1e-6
    vs, us, d = vs[valid], us[valid], d[valid]
    if d.numel() == 0:
        return (torch.zeros((0, 3), device=device), vs, us)

    x_cam = (us.float() - cx) * d / fx
    y_cam = (vs.float() - cy) * d / fy
    ones = torch.ones_like(d)
    p_cam_h = torch.stack([x_cam, y_cam, d, ones], dim=1)  # (K, 4)

    w2v = camera.world_view_transform.to(device=device, dtype=p_cam_h.dtype)
    v2w = torch.inverse(w2v)
    p_world_h = p_cam_h @ v2w
    p_world = p_world_h[:, :3] / p_world_h[:, 3:4].clamp_min(1e-8)
    return p_world, vs, us


# =============================================================================
# BATCH PROCESSING FUNCTION
# =============================================================================

def batch_inpaint_all_cameras(
    inpainter: OnlineInpainter,
    gaussians,
    background_id: int,
    train_cameras: List,
    render_func,
    pipe,
    background,
    inpainted_output_dir: str,
    iteration: int,
    hole_threshold: float = 0.05,
    clean_ghosts: bool = False,
    ghost_clean_method: str = 'nlm',
    min_hole_size: int = 1000,
) -> None:
    """
    Batch process ALL cameras.

    After this call, inpainted images are saved to disk.
    The trainer exits immediately after — camera.original_image is NOT
    updated here (Phase B is disabled).
    """

    print("\n" + "=" * 80)
    print(f"BATCH INPAINTING ALL CAMERAS (Iteration {iteration})")
    print("=" * 80)

    # Reset the module-level timing accumulator for this call. Sub-phases
    # not reached this call (e.g. skip_inpainting_forever) simply leave
    # their key absent.
    _LAST_STAGE2_TIMINGS.clear()

    try:
        # ================================================================
        # CASE: Skip forever
        # ================================================================
        if inpainter.skip_inpainting_forever:
            print(f"  Reason: {inpainter.skip_reason}")
            renders     = inpainter.render_all_views(
                gaussians=gaussians, background_id=background_id,
                cameras=train_cameras, render_func=render_func,
                pipe=pipe, background=background, iteration=iteration)
            cleaned_bgs = {n: d['bg'] for n, d in renders.items()}
            save_backgrounds_only(renders, inpainted_output_dir, iteration, cleaned_bgs)
            del renders
            torch.cuda.empty_cache()
            return

        # ================================================================
        # Step 1: Render all views
        # ================================================================
        renders = inpainter.render_all_views(
            gaussians=gaussians, background_id=background_id,
            cameras=train_cameras, render_func=render_func,
            pipe=pipe, background=background, iteration=iteration)

        # ================================================================
        # Step 2: Hole detection
        # ================================================================
        if (inpainter.use_sam2_propagation
                and inpainter.sam2_detector is not None):

            camera_names    = sorted(renders.keys())
            first_cam       = camera_names[0]
            first_bg_render = renders[first_cam]['bg']

            print(f"\n{'=' * 80}")
            print(f"{'🖱️  INTERACTIVE HOLE SELECTION (SAM2)':^80}")
            print(f"  Click on holes to inpaint. Remove ALL prompts to skip.")
            print(f"{'=' * 80}\n")

            _t_click_start = time.time()
            hole_prompts = inpainter.sam2_detector.get_hole_prompts(
                first_bg_render=first_bg_render,
                hole_threshold=hole_threshold)
            _t_click_end = time.time()
            # Time from the FIRST actual click, not from when the window
            # opened -- the window can sit open a while before anyone's
            # available to click it, and that idle wait shouldn't count as
            # "hole click" time. Falls back to the full window-open->close
            # duration if there was no click at all (e.g. cancelled).
            _first_click = getattr(inpainter.sam2_detector, 'last_first_click_time', None)
            _LAST_STAGE2_TIMINGS['hole_click_sec'] = (
                _t_click_end - _first_click if _first_click is not None
                else _t_click_end - _t_click_start)

            if not hole_prompts:
                print("  No prompts — skipping inpainting permanently.")
                inpainter.skip_inpainting_forever = True
                inpainter.skip_reason = f"User removed all prompts at iteration {iteration}"
                cleaned_bgs = {n: d['bg'] for n, d in renders.items()}
                save_backgrounds_only(renders, inpainted_output_dir, iteration, cleaned_bgs)
                del renders
                torch.cuda.empty_cache()
                return

            print(f"\n[SAM2] Propagating {len(hole_prompts)} holes through {len(camera_names)} frames...")
            bg_renders = {cam: renders[cam]['bg'] for cam in camera_names}
            _t_predict_start = time.time()
            holes = inpainter.sam2_detector.propagate_holes_through_frames(
                bg_renders=bg_renders,
                first_cam_name=first_cam,
                hole_threshold=hole_threshold,
                temp_dir=inpainter.cache_dir,
                initial_prompts=hole_prompts)
            _LAST_STAGE2_TIMINGS['hole_prediction_sec'] = time.time() - _t_predict_start
            del bg_renders

            if inpainter.mask_dilation_pixels > 0:
                print(f"[OnlineInpainter] Dilating SAM2 masks by {inpainter.mask_dilation_pixels}px...")
                holes = {cam: inpainter.dilate_mask(mask, inpainter.mask_dilation_pixels)
                         for cam, mask in holes.items()}
        else:
            _t_predict_start = time.time()
            holes = inpainter.detect_holes_batch(
                renders=renders,
                hole_threshold=hole_threshold,
                min_hole_size=min_hole_size,
                save_debug=True)
            _LAST_STAGE2_TIMINGS['hole_prediction_sec'] = time.time() - _t_predict_start

        total_holes = sum(mask.sum().item() for mask in holes.values())

        # ================================================================
        # Step 2b: Free fg_mask tensors from renders (not needed by LaMa)
        # ================================================================
        for render_data in renders.values():
            if 'fg_mask' in render_data:
                del render_data['fg_mask']
        torch.cuda.empty_cache()

        # ================================================================
        # Step 3: LaMa inpainting
        # ================================================================
        print(f"\n{'=' * 80}")
        print(f"{'🎨 RUNNING INPAINTING':^80}")
        print(f"  Holes: {total_holes:.0f} pixels | Cameras: {len(renders)}")
        print(f"{'=' * 80}\n")

        _t_inpaint_start = time.time()
        updated_bgs = inpainter.inpaint_batch(
            renders=renders,
            holes=holes,
            inpainted_output_dir=inpainted_output_dir,
            iteration=iteration)
        _LAST_STAGE2_TIMINGS['inpainting_sec'] = time.time() - _t_inpaint_start

        # Free renders dict after LaMa is done (bg tensors no longer needed)
        del renders
        del holes
        torch.cuda.empty_cache()

        # ================================================================
        # Step 4: Re-save (ghost cleaning disabled; clean_ghosts=False)
        #
        # cam_name here already carries an extension -- get_inpainting_
        # results() (called inside inpaint_batch() above) reconstructs it as
        # image_file.split('_mask')[0] + f".{ext}", e.g. "20220819_105840.jpg"
        # from LaMa's own raw output file "20220819_105840_mask.jpg". Simply
        # appending ".jpg" here again (the historical bug) wrote a SECOND
        # copy of the same content as "20220819_105840.jpg.jpg", sitting
        # right next to LaMa's still-present raw "..._mask.jpg" -- both
        # readable, same pixels, cluttering inpainted_output_dir with two
        # files per camera. Strip cam_name's own extension first (matching
        # the convention used everywhere else in this file), then also
        # remove LaMa's raw "*_mask.*" leftovers so only the one clean,
        # correctly-named file per camera remains.
        # ================================================================
        print(f"\n[OnlineInpainter] Saving final inpainted backgrounds...")
        temp_iter_output_dir = Path(inpainted_output_dir)

        for cam_name, inpainted_bg in updated_bgs.items():
            clean_name = cam_name.split(".")[0]
            output_path = temp_iter_output_dir / f"{clean_name}.jpg"
            inpainter.tensor_to_pil(inpainted_bg).save(output_path)

        for stale in temp_iter_output_dir.glob("*_mask.*"):
            stale.unlink()

        del updated_bgs
        torch.cuda.empty_cache()

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print(f"{'BATCH INPAINTING COMPLETE':^80}")
        print("=" * 80)
        print(f"  Cameras processed  : {len(train_cameras)}")
        print(f"  Holes filled       : {total_holes:.0f} pixels")
        print(f"  Results saved to   : {inpainted_output_dir}")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
