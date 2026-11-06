"""
RigidGS Segmentation Module (REFINED)
Uses SAM 2 Video Predictor for interactive multi-view segmentation
Saves masks with consistent naming and improved visibility
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt


class SAM2Segmenter:
    """
    Interactive segmentation using SAM 2 Video Predictor
    Tracks objects across multiple views with consistent IDs
    """
    
    def __init__(self, config: dict):
        """
        Initialize SAM 2 segmenter
        
        Args:
            config: Dict with:
                - model_path: Path to SAM 2 checkpoint
                - model_type: Model architecture (e.g., 'hiera_large')
                - device: 'cuda' or 'cpu'
                - first_frame_idx: frame to show for the interactive click
                  prompt (default 0). Some scenes start with frames the
                  target object isn't even in -- e.g. SPIn-NeRF's first 40
                  frames (dictionary order) are the no-object captures, the
                  object only appears from frame 40 on -- so frame 0 has
                  nothing to click on. Set this to the first frame that
                  actually contains the object.
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.first_frame_idx = config.get('first_frame_idx', 0)
        
        print(f"Loading SAM 2 model: {config['model_type']}")
        self.predictor = build_sam2_video_predictor(
            config_file=config['config_file'],
            ckpt_path=config['model_path'],
            device=self.device
        )
        print("✓ SAM 2 loaded successfully")
        
        # Storage for object annotations
        self.object_ids = []
        self.object_names = {}  # {object_id: "bottle", "glass", etc.}
        
    def segment_scene(self, image_dir: str, output_dir: str) -> Dict:
        """
        Interactive segmentation of all objects in scene
        Allows multiple clicks per object for better coverage of large objects
        
        Args:
            image_dir: Directory with input images
            output_dir: Directory to save masks and metadata
            
        Returns:
            Dictionary with segmentation results and metadata
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        
        # Create output directories
        masks_dir = output_dir / 'masks'
        viz_dir = output_dir / 'visualizations'
        masks_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # # Load all images
        # image_paths = sorted(list(image_dir.glob('*.jpg')) + 
        #                     list(image_dir.glob('*.png')) +
        #                     list(image_dir.glob('*.jpeg'))+
        #                     list(image_dir.glob('*.JPG'))+
        #                     list(image_dir.glob('*.PNG'))+
        #                     list(image_dir.glob('*.JPEG')))
        
        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])


        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        if not (0 <= self.first_frame_idx < len(image_paths)):
            raise ValueError(
                f"first_frame_idx={self.first_frame_idx} out of range for "
                f"{len(image_paths)} images"
            )

        print(f"\nFound {len(image_paths)} images")
        print(f"First image: {image_paths[0].name}")
        if self.first_frame_idx != 0:
            print(f"Prompt frame (first_frame_idx={self.first_frame_idx}): "
                  f"{image_paths[self.first_frame_idx].name}")

        # SAM2's own video predictor re-scans video_path itself (see
        # sam2.utils.misc.load_video_frames_from_jpg_images) and ONLY
        # accepts .jpg/.jpeg -- unlike the glob above, it silently ignores
        # .png and raises "no images found" even when this directory is
        # full of them (hit on LLFF's trex, whose images_4/ is PNG-only).
        # Convert into a sibling JPEG cache so PNG-sourced scenes work
        # without a manual pre-conversion step; stems are preserved so
        # frame order stays aligned with image_paths (out_frame_idx below
        # is used to index back into image_paths via _save_masks_with_naming).
        video_path_for_sam2 = str(image_dir)
        non_jpg = [p for p in image_paths if p.suffix.lower() not in ('.jpg', '.jpeg')]
        if non_jpg:
            jpg_cache_dir = image_dir.parent / f"{image_dir.name}_jpg_cache"
            jpg_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{len(non_jpg)}/{len(image_paths)} images aren't JPEG -- "
                  f"SAM2's video predictor only scans for .jpg/.jpeg, so "
                  f"converting into {jpg_cache_dir} for it (source images "
                  f"untouched).")
            for p in image_paths:
                jpg_path = jpg_cache_dir / f"{p.stem}.jpg"
                if not jpg_path.exists():
                    img = cv2.imread(str(p))
                    cv2.imwrite(str(jpg_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            video_path_for_sam2 = str(jpg_cache_dir)

        # Initialize video predictor with image directory
        inference_state = self.predictor.init_state(
            video_path=video_path_for_sam2
        )

        # Interactive annotation on the prompt frame
        print("\n" + "="*60)
        print("INTERACTIVE SEGMENTATION")
        print("="*60)
        print("Instructions:")
        print("  1. Click on objects you want to segment")
        print("  2. You can click MULTIPLE times on the same object")
        print("     (e.g., click different parts of a large table)")
        print("  3. After each click, you'll be asked for the object name")
        print("  4. Use the SAME name for multiple clicks on the same object")
        print("  5. Close window when done with all objects")
        print("="*60)

        # Load the prompt frame for annotation (not necessarily frame 0 --
        # see first_frame_idx docstring in __init__)
        prompt_image = cv2.imread(str(image_paths[self.first_frame_idx]))
        prompt_image_rgb = cv2.cvtColor(prompt_image, cv2.COLOR_BGR2RGB)
        
        # Get user clicks on the prompt frame (can have multiple clicks per object)
        points, labels, click_names = self._get_user_clicks_with_names(prompt_image_rgb)
        
        if len(points) == 0:
            raise ValueError("No objects annotated!")
        
        print(f"\n✓ Annotated {len(points)} clicks")
        
        # Group clicks by object name
        object_name_to_clicks = {}  # {name: [(point, label), ...]}
        for point, label, name in zip(points, labels, click_names):
            if name not in object_name_to_clicks:
                object_name_to_clicks[name] = []
            object_name_to_clicks[name].append((point, label))
        
        print(f"\n✓ Grouped into {len(object_name_to_clicks)} unique objects:")
        for name, clicks in object_name_to_clicks.items():
            print(f"  - {name}: {len(clicks)} click(s)")
        
        # Add each unique object to SAM 2
        # For objects with multiple clicks, we'll merge the masks later
        click_id_to_obj_name = {}  # {sam_click_id: object_name}
        sam_click_id = 1
        
        for obj_name, clicks in object_name_to_clicks.items():
            for point, label in clicks:
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=self.first_frame_idx,
                    obj_id=sam_click_id,
                    points=np.array([point], dtype=np.float32),
                    labels=np.array([label], dtype=np.int32)
                )
                
                click_id_to_obj_name[sam_click_id] = obj_name
                sam_click_id += 1
        
        # Propagate annotations across all frames
        print("\nPropagating segmentation across all frames...")
        
        # Run propagation
        video_segments_raw = {}  # Raw segments from SAM (one per click)
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            inference_state
        ):
            video_segments_raw[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        print(f"✓ Segmented {len(video_segments_raw)} frames")
        
        # Merge masks with same object name
        print("\nMerging masks for objects with multiple clicks...")
        video_segments_merged = self._merge_masks_by_name(
            video_segments_raw, 
            click_id_to_obj_name,
            object_name_to_clicks
        )
        
        # Update object tracking with merged objects
        self.object_names = {
            idx + 1: name 
            for idx, name in enumerate(object_name_to_clicks.keys())
        }
        self.object_ids = list(self.object_names.keys())
        
        # Save masks with consistent naming
        print("\nSaving masks with consistent naming...")
        self._save_masks_with_naming(
            video_segments_merged, 
            image_paths, 
            masks_dir, 
            viz_dir
        )
        
        # Save metadata
        metadata = {
            'num_objects': len(self.object_ids),
            'object_names': self.object_names,
            'object_ids': self.object_ids,
            'num_frames': len(image_paths),
            'image_files': [p.name for p in image_paths],
            'clicks_per_object': {
                name: len(clicks) 
                for name, clicks in object_name_to_clicks.items()
            }
        }
        
        metadata_path = output_dir / 'segmentation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return {
            'masks_dir': masks_dir,
            'metadata': metadata,
            'video_segments': video_segments_merged
        }
    
    def _merge_masks_by_name(self, video_segments_raw, click_id_to_obj_name, 
                            object_name_to_clicks):
        """
        Merge masks from multiple clicks that belong to the same object
        
        Args:
            video_segments_raw: Raw segments from SAM {frame_idx: {click_id: mask}}
            click_id_to_obj_name: Mapping {click_id: object_name}
            object_name_to_clicks: Dict {name: [(point, label), ...]}
        
        Returns:
            Merged segments {frame_idx: {obj_id: mask}}
        """
        # Create mapping from object name to new unified object ID
        unique_names = list(object_name_to_clicks.keys())
        name_to_unified_id = {name: idx + 1 for idx, name in enumerate(unique_names)}
        
        video_segments_merged = {}
        
        for frame_idx, masks_dict in video_segments_raw.items():
            # Group masks by object name
            name_to_masks = {}  # {object_name: [mask1, mask2, ...]}
            
            for click_id, mask in masks_dict.items():
                obj_name = click_id_to_obj_name[click_id]
                if obj_name not in name_to_masks:
                    name_to_masks[obj_name] = []
                name_to_masks[obj_name].append(mask[0])  # Extract (H, W) from (1, H, W)
            
            # Merge masks for each object (union of all clicks)
            merged_frame = {}
            for obj_name, masks_list in name_to_masks.items():
                # Union: logical OR of all masks
                merged_mask = np.zeros_like(masks_list[0], dtype=bool)
                for mask in masks_list:
                    merged_mask = np.logical_or(merged_mask, mask)
                
                # Store with unified object ID
                unified_id = name_to_unified_id[obj_name]
                merged_frame[unified_id] = merged_mask[np.newaxis, ...]  # Add batch dim back
                
                if frame_idx == 0:
                    print(f"  {obj_name}: Merged {len(masks_list)} mask(s)")
            
            video_segments_merged[frame_idx] = merged_frame
        
        return video_segments_merged

    def _get_user_clicks_with_names(self, image: np.ndarray) -> Tuple[List, List, List]:
        """
        Interactive tool to get user clicks on objects WITH names
        Allows multiple clicks per object
        """
        points = []
        labels = []
        names = []
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title("Click on objects to segment\n"
                    "You can click multiple times on same object (e.g., large table)\n"
                    "Close window when done", 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        click_count = 0
        
        def onclick(event):
            nonlocal click_count
            
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                
                # Pause the plot to get user input
                plt.pause(0.1)
                
                # Ask for object name
                print(f"\n--- Click {click_count + 1} ---")
                print(f"Position: ({x}, {y})")
                obj_name = input("Enter object name (or press Enter to use previous name): ").strip()
                
                # Use previous name if empty (convenient for multiple clicks on same object)
                if not obj_name and len(names) > 0:
                    obj_name = names[-1]
                    print(f"  Using previous name: '{obj_name}'")
                elif not obj_name:
                    obj_name = f"object_{click_count + 1}"
                    print(f"  Using default name: '{obj_name}'")
                
                points.append([x, y])
                labels.append(1)  # Positive label
                names.append(obj_name)
                
                # Mark the point with color based on object name
                # Use consistent color for same object name
                unique_names = list(set(names))
                color_idx = unique_names.index(obj_name) % 10
                color = plt.cm.tab10(color_idx)
                
                # Mark the point
                ax.plot(x, y, 'o', markersize=15, markeredgewidth=3, 
                    markerfacecolor=color, markeredgecolor='white')
                ax.text(x+15, y-15, f'{obj_name}\n(click {names.count(obj_name)})', 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, 
                                edgecolor='white', linewidth=1, alpha=0.3))
                fig.canvas.draw()
                
                click_count += 1
                print(f"  Recorded: {obj_name}")
                print(f"  Total clicks: {click_count}")
                print(f"  Unique objects: {len(set(names))}")
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()
        
        return points, labels, names
    def _save_masks_with_naming(self, video_segments, image_paths, 
                                masks_dir, viz_dir):
        """
        Save masks with naming convention: {original_name}_mask_{object_name}.png
        All masks and visualizations saved as PNG with consistent dimensions
        
        Example:
            IMG_001.jpg → IMG_001_mask_bottle.png, IMG_001_mask_glass.png
                       → IMG_001_visualization.png
        """
        for frame_idx, image_path in enumerate(image_paths):
            if frame_idx not in video_segments:
                continue
            
            # Get original image name without extension
            image_stem = image_path.stem  # e.g., "IMG_001"
            
            # Load original image for visualization
            original_img = cv2.imread(str(image_path))
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            img_height, img_width = original_img_rgb.shape[:2]
            
            # Create visualization - start with white background for better visibility
            viz_img = original_img_rgb.copy()
            
            # Create a combined overlay for all masks
            combined_overlay = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            # Save each object mask separately
            masks_for_frame = video_segments[frame_idx]
            
            for obj_id in self.object_ids:
                if obj_id not in masks_for_frame:
                    # Create empty mask if object not detected in this frame
                    mask = np.zeros((img_height, img_width), dtype=bool)
                else:
                    mask = masks_for_frame[obj_id][0]  # (H, W)
                    
                    # Verify mask matches image dimensions (should always match)
                    if mask.shape != (img_height, img_width):
                        raise ValueError(
                            f"Mask dimension mismatch for frame {frame_idx}, object {obj_id}:\n"
                            f"  Image: {(img_height, img_width)}\n"
                            f"  Mask: {mask.shape}\n"
                            f"This should not happen - check SAM2 configuration."
                        )
                
                obj_name = self.object_names[obj_id]
                
                # Save individual mask: {image_stem}_mask_{object_name}.png
                mask_filename = f"{image_stem}_mask_{obj_name}.JPG"
                mask_path = masks_dir / mask_filename
                
                # Save as binary mask (0 or 255) - always PNG format
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(mask_path), mask_uint8)
                
                # Add to visualization with distinct, vibrant colors
                if mask.any():  # Only visualize if mask exists
                    # Use distinct colors from a better colormap
                    color_idx = (obj_id - 1) % 10
                    color = plt.cm.tab10(color_idx)[:3]
                    color = (np.array(color) * 255).astype(np.uint8)
                    
                    # Create colored overlay for this object
                    combined_overlay[mask] = color
                    
                    # Add border around mask for better visibility
                    mask_uint8_border = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask_uint8_border, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Draw thick border in white
                    cv2.drawContours(viz_img, contours, -1, (255, 255, 255), 3)
                    # Draw thinner border in the object color
                    cv2.drawContours(viz_img, contours, -1, color.tolist(), 2)
                    
                    # Add text label with background for better readability
                    y, x = np.where(mask)
                    if len(y) > 0:
                        center_y, center_x = int(y.mean()), int(x.mean())
                        
                        # Create text background
                        text = obj_name
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, font, font_scale, thickness
                        )
                        
                        # Draw filled rectangle as background
                        rect_x1 = center_x - text_width // 2 - 5
                        rect_y1 = center_y - text_height - 5
                        rect_x2 = center_x + text_width // 2 + 5
                        rect_y2 = center_y + 5
                        cv2.rectangle(viz_img, (rect_x1, rect_y1), (rect_x2, rect_y2),
                                    (0, 0, 0), -1)
                        cv2.rectangle(viz_img, (rect_x1, rect_y1), (rect_x2, rect_y2),
                                    color.tolist(), 2)
                        
                        # Draw text
                        text_x = center_x - text_width // 2
                        text_y = center_y
                        cv2.putText(viz_img, text, (text_x, text_y),
                                  font, font_scale, (255, 255, 255), thickness)
            
            # # Blend the combined overlay with original image for better visibility
            # # Higher alpha (0.5) makes masks more visible
            mask_exists = combined_overlay.sum(axis=2) > 0
            # viz_img[mask_exists] = cv2.addWeighted(
            #     viz_img[mask_exists], 0.4, 
            #     combined_overlay[mask_exists], 0.6, 0
            # )

            # Uses numpy operations instead of cv2.addWeighted for sliced arrays
            alpha = 0.6  # mask opacity  
            beta = 0.4   # original image visibility
            viz_img[mask_exists] = (
                beta * viz_img[mask_exists].astype(np.float32) + 
                alpha * combined_overlay[mask_exists].astype(np.float32)
            ).astype(np.uint8)
            
            # Save visualization with consistent naming - always PNG
            viz_filename = f"{image_stem}.JPG"
            viz_path = viz_dir / viz_filename
            viz_bgr = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(viz_path), viz_bgr)
            
            if frame_idx == 0 or frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}: Saved {len(masks_for_frame)} masks")
        
        print(f"\n✓ All masks saved to: {masks_dir}")
        print(f"✓ All masks are PNG format with size matching input images")
        print(f"✓ Visualizations saved to: {viz_dir}")
    
    # def _get_user_clicks(self, image: np.ndarray) -> Tuple[List, List]:
    #     """
    #     Interactive tool to get user clicks on objects
    #     """
    #     points = []
    #     labels = []
        
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.imshow(image)
    #     ax.set_title("Click on objects to segment (close window when done)", 
    #                 fontsize=14, fontweight='bold')
    #     ax.axis('off')
        
    #     def onclick(event):
    #         if event.xdata is not None and event.ydata is not None:
    #             x, y = int(event.xdata), int(event.ydata)
    #             points.append([x, y])
    #             labels.append(1)  # Positive label
                
    #             # Mark the point with a larger, more visible marker
    #             ax.plot(x, y, 'r*', markersize=20, markeredgewidth=2, 
    #                    markeredgecolor='white')
    #             ax.text(x+15, y-15, f'Obj {len(points)}', 
    #                    color='red', fontsize=14, fontweight='bold',
    #                    bbox=dict(boxstyle='round', facecolor='white', 
    #                             edgecolor='red', linewidth=2, alpha=0.9))
    #             fig.canvas.draw()
                
    #             print(f"  Point {len(points)}: ({x}, {y})")
        
    #     cid = fig.canvas.mpl_connect('button_press_event', onclick)
    #     plt.tight_layout()
    #     plt.show()
        
    #     return points, labels


# Standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SAM 2 Interactive Segmentation')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for masks')
    parser.add_argument('--model', type=str, 
                       default='external/segment-anything-2/checkpoints/sam2.1_hiera_large.pt',
                       help='Path to SAM 2 checkpoint')
    parser.add_argument('--model-type', type=str, default='hiera_large',
                       help='SAM 2 model type')
    parser.add_argument('--config-file', type=str,
                       default='external/segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                       help='Path to SAM 2 config file')
    
    args = parser.parse_args()
    
    config = {
        'model_path': args.model,
        'model_type': args.model_type,
        'config_file': args.config_file,
        'device': 'cuda'
    }
    
    segmenter = SAM2Segmenter(config)
    results = segmenter.segment_scene(args.images, args.output)
    
    print("\n" + "="*60)
    print("SEGMENTATION COMPLETE")
    print("="*60)
    print(f"Objects: {results['metadata']['object_names']}")
    print(f"Masks directory: {results['masks_dir']}")
    print(f"Format: All masks and visualizations saved as PNG")
    print("="*60)