"""
sam2_hole_detection.py - Use SAM2 to track holes through rendered frames
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from PIL import Image
import cv2
import shutil

from sam2_interactive_prompts import InteractiveHoleSelector
from sam2_interactive_prompts_cv import InteractiveHoleSelectorCV


class SAM2HoleDetector:
    """
    Use SAM2 to detect and track holes across rendered camera views
    """

    def __init__(
        self,
        sam2_checkpoint: str = "external/sam2/checkpoints/sam2.1_hiera_large.pt",
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
        use_interactive_ui: bool = True,
        ui_backend: str = "opencv",
    ):
        self.device = device
        self.use_interactive_ui = use_interactive_ui
        self.ui_backend = ui_backend
        # Timestamp of the first click in the most recent get_hole_prompts()
        # interactive round, or None if that round never got a click (e.g.
        # automatic detection, or the UI was cancelled with no clicks).
        # Callers time "hole_click_sec" from here, not from when the window
        # opened, since the window can sit open a while before anyone's
        # available to click it.
        self.last_first_click_time = None

        if use_interactive_ui:
            if ui_backend == "opencv":
                self.ui = InteractiveHoleSelectorCV()
            else:
                self.ui = InteractiveHoleSelector()

        import sys
        sys.path.insert(0, "external/sam2")

        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(
            sam2_config,
            sam2_checkpoint,
            device=device,
        )

        print("[SAM2HoleDetector] Initialized")
        print(f"  Model: {sam2_checkpoint}")

    # ------------------------------------------------------------------

    def get_hole_prompts(
        self,
        first_bg_render: torch.Tensor,
        hole_threshold: float = 0.05,
        max_holes: int = 10,
    ) -> List[Dict]:
        if self.use_interactive_ui:
            print("[SAM2] Opening interactive UI...")
            result = self.ui.select_holes(first_bg_render)
            self.last_first_click_time = getattr(self.ui, "first_click_time", None)
            return result
        else:
            print("[SAM2] Using automatic hole detection...")
            self.last_first_click_time = None
            return self.detect_holes_in_first_frame(
                first_bg_render, hole_threshold
            )

    # ------------------------------------------------------------------

    def detect_holes_in_first_frame(
        self,
        bg_render: torch.Tensor,
        hole_threshold: float = 0.05,
    ) -> List[Dict]:

        intensity = bg_render.mean(dim=0).cpu().numpy()
        dark_mask = (intensity < hole_threshold).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8
        )

        hole_prompts = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 1000:
                cx, cy = centroids[i]
                hole_prompts.append(
                    {
                        "point": np.array([cx, cy], dtype=np.float32),
                        "label": 1,
                        "object_id": i,
                    }
                )
                print(
                    f"  Hole {i}: {area:.0f} pixels at ({cx:.0f}, {cy:.0f})"
                )

        return hole_prompts

    # ------------------------------------------------------------------

    def propagate_holes_through_frames(
        self,
        bg_renders: Dict[str, torch.Tensor],
        first_cam_name: str,
        hole_threshold: float = 0.05,
        temp_dir: Path = None,
        initial_prompts: List[Dict] = None,
    ) -> Dict[str, torch.Tensor]:

        if temp_dir is None:
            raise ValueError("temp_dir must be provided")

        first_bg_render = bg_renders[first_cam_name]

        if initial_prompts is None:
            initial_prompts = self.get_hole_prompts(
                first_bg_render=first_bg_render,
                hole_threshold=hole_threshold,
            )

        if not initial_prompts:
            print("⚠️  No holes selected/detected!")
            return {
                cam: torch.zeros_like(bg_renders[cam][0])
                for cam in bg_renders
            }

        frames_dir = temp_dir / "sam2_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        camera_names = sorted(bg_renders.keys())

        print(f"[SAM2HoleDetector] Saving {len(camera_names)} frames...")

        for idx, cam_name in enumerate(camera_names):
            frame = bg_renders[cam_name]
            frame_np = frame.cpu().numpy()
            frame_np = np.transpose(frame_np, (1, 2, 0))
            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
            Image.fromarray(frame_np).save(frames_dir / f"{idx:05d}.jpg")

        inference_state = self.predictor.init_state(
            video_path=str(frames_dir)
        )

        print(f"[SAM2HoleDetector] Adding {len(initial_prompts)} hole prompts...")

        for prompt in initial_prompts:
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=prompt["object_id"],
                points=prompt["point"].reshape(1, 2),
                labels=np.array([prompt["label"]], dtype=np.int32),
            )

        print(
            f"[SAM2HoleDetector] Propagating through {len(camera_names)} frames..."
        )

        hole_masks: Dict[str, torch.Tensor] = {}

        for frame_idx, out_obj_ids, out_mask_logits in (
            self.predictor.propagate_in_video(inference_state)
        ):
            cam_name = camera_names[frame_idx]

            combined_mask = torch.zeros(
                out_mask_logits[0].shape[-2:],
                device=out_mask_logits[0].device,
            )

            for i, _ in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0).squeeze()
                combined_mask = torch.maximum(
                    combined_mask, mask.float()
                )

            hole_masks[cam_name] = combined_mask

        print("[SAM2HoleDetector] ✓ Propagation complete!")

        if frames_dir.exists():
            shutil.rmtree(frames_dir)

        return hole_masks

    # ------------------------------------------------------------------

    def save_prompts(self, prompts: List[Dict], filepath: str):
        import json

        prompts_json = []
        for p in prompts:
            prompts_json.append(
                {
                    "point": p["point"].tolist(),
                    "label": int(p["label"]),
                    "object_id": int(p["object_id"]),
                }
            )

        with open(filepath, "w") as f:
            json.dump(prompts_json, f, indent=2)

        print(f"✓ Saved prompts to: {filepath}")

    # ------------------------------------------------------------------

    def load_prompts(self, filepath: str) -> List[Dict]:
        import json

        with open(filepath, "r") as f:
            prompts_json = json.load(f)

        prompts = []
        for p in prompts_json:
            prompts.append(
                {
                    "point": np.array(p["point"], dtype=np.float32),
                    "label": p["label"],
                    "object_id": p["object_id"],
                }
            )

        print(f"✓ Loaded {len(prompts)} prompts from: {filepath}")
        return prompts
