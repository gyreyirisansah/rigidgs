"""
RigidGS Point Labeling Module (CORRECTED)
Label 3D points with object IDs using multi-view masks
Compatible with our segmentation output format
"""

import os
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import cv2
import struct
from collections import namedtuple
import open3d as o3d


# COLMAP data structures (from sfm.py)
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])


class CameraProjector:
    """Helper class to project 3D points to 2D image coordinates"""
    
    def __init__(self, camera, image):
        """
        Initialize projector with camera intrinsics and extrinsics
        
        Args:
            camera: Camera namedtuple from COLMAP
            image: Image namedtuple from COLMAP (contains pose)
        """
        self.camera = camera
        self.image = image
        
        # Build intrinsic matrix
        self.K = self._build_intrinsic_matrix()
        
        # Build extrinsic matrix (world to camera)
        self.R, self.t = self._build_extrinsic_matrix()
    
    def _build_intrinsic_matrix(self):
        """Build 3x3 intrinsic matrix from camera params"""
        params = self.camera.params
        
        if self.camera.model == "PINHOLE":
            # params: [fx, fy, cx, cy]
            fx, fy, cx, cy = params
        elif self.camera.model == "SIMPLE_PINHOLE":
            # params: [f, cx, cy]
            f, cx, cy = params
            fx = fy = f
        elif self.camera.model == "SIMPLE_RADIAL":
            # params: [f, cx, cy, k]
            f, cx, cy, _ = params
            fx = fy = f
        else:
            raise ValueError(f"Unsupported camera model: {self.camera.model}")
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return K
    
    def _build_extrinsic_matrix(self):
        """Build rotation and translation from quaternion and tvec"""
        # COLMAP stores rotation as quaternion (qw, qx, qy, qz)
        qvec = self.image.qvec
        tvec = self.image.tvec
        
        # Convert quaternion to rotation matrix
        R = self._qvec_to_rotation_matrix(qvec)
        t = np.array(tvec).reshape(3, 1)
        
        return R, t
    
    def _qvec_to_rotation_matrix(self, qvec):
        """Convert quaternion to 3x3 rotation matrix"""
        qw, qx, qy, qz = qvec
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R
    
    def project(self, point_3d):
        """
        Project 3D point to 2D pixel coordinates
        
        Args:
            point_3d: [x, y, z] in world coordinates
            
        Returns:
            [u, v] pixel coordinates, or None if behind camera
        """
        # World to camera coordinates
        point_cam = self.R @ point_3d.reshape(3, 1) + self.t
        
        # Check if point is behind camera
        if point_cam[2] <= 0:
            return None
        
        # Camera to image coordinates
        point_2d_homog = self.K @ point_cam
        point_2d = point_2d_homog[:2] / point_2d_homog[2]
        
        return point_2d.flatten()


class PointLabeler:
    """Label 3D points with object IDs using multi-view segmentation masks"""
    
    def __init__(self):
        pass
    
    def label_points(self, colmap_results: dict, segmentation_dir: str,
                     use_refined_masks: bool = False) -> list:
        """
        Label each 3D point with object ID by projecting to views
        
        Args:
            colmap_results: Dict with 'cameras', 'images', 'points3D' from COLMAP
            segmentation_dir: Directory with segmentation results
            use_refined_masks: vote with preprocessing/refined/refined_masks
                instead of the raw SAM2 masks in segmentation_dir/masks
            
        Returns:
            List of labeled points with structure:
                [{
                    'point_id': int,
                    'xyz': [x, y, z],
                    'rgb': [r, g, b],
                    'object_id': str or int,
                    'object_name': str,
                    'confidence': float
                }, ...]
        """
        segmentation_dir = Path(segmentation_dir)
        masks_dir = segmentation_dir / "masks"
        refined_masks_dir = segmentation_dir.parent / "refined/refined_masks"

        
        # Load metadata
        metadata_path = segmentation_dir / "segmentation_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # object_names is a dict: {obj_id: name}
        # e.g., {"1": "bottle", "2": "glass", "3": "bowl"}
        object_id_to_name = {int(k): v for k, v in metadata['object_names'].items()}
        image_files = metadata['image_files']
        
        print("\n" + "="*60)
        print("POINT CLOUD LABELING")
        print("="*60)
        print(f"Objects: {list(object_id_to_name.values())}")
        print(f"3D Points: {len(colmap_results['points3D'])}")
        print(f"Images: {len(colmap_results['images'])}")
        print("="*60)
        
        # Vote with the raw SAM2 masks by default, as the paper did. Depth
        # refinement trims thin structures (fern fronds), and SfM points sit
        # on exactly those textured edges: on fern, voting with the refined
        # masks labels ~20% of the object's points as background, and those
        # labels are frozen during training. The refined masks are still the
        # target of the 2D mask loss.
        print("\nLoading segmentation masks...")
        if use_refined_masks:
            if not refined_masks_dir.is_dir():
                raise FileNotFoundError(f"use_refined_masks set, but there are no refined masks in {refined_masks_dir}")
            print(f"  Using refined masks: {refined_masks_dir}")
            all_masks = self._load_masks(refined_masks_dir, object_id_to_name, image_files)
        else:
            print(f"  Using raw SAM2 masks: {masks_dir}")
            all_masks = self._load_masks(masks_dir, object_id_to_name, image_files)
        
        # Create image name to ID mapping
        image_name_to_id = {
            img.name: img_id 
            for img_id, img in colmap_results['images'].items()
        }
        
        # Label each point
        print("\nLabeling 3D points...")
        labeled_points = []
        
        cameras = colmap_results['cameras']
        images = colmap_results['images']
        points3D = colmap_results['points3D']
        
        for point_id, point in tqdm(points3D.items(), desc="Processing points"):
            # Vote: which object does this point belong to?
            votes = {obj_id: 0 for obj_id in object_id_to_name.keys()}
            votes['background'] = 0
            total_votes = 0
            
            # Check each view where this point is visible
            for img_id in point.image_ids:
                if img_id not in images:
                    continue
                
                image = images[img_id]
                image_name = image.name
                
                # Get camera for this image
                camera = cameras[image.camera_id]
                
                # Project 3D point to this view
                projector = CameraProjector(camera, image)
                pixel = projector.project(point.xyz)
                
                if pixel is None:
                    continue
                
                u, v = int(round(pixel[0])), int(round(pixel[1]))
                
                # Check bounds
                if u < 0 or v < 0 or u >= camera.width or v >= camera.height:
                    continue
                
                total_votes += 1
                
                # Check which object mask contains this pixel
                found = False
                if image_name in all_masks:
                    for obj_id, mask in all_masks[image_name].items():
                        if mask[v, u]:  # Note: mask is [y, x]
                            votes[obj_id] += 1
                            found = True
                            break
                
                if not found:
                    votes['background'] += 1
            
            # Assign label based on majority vote
            if total_votes == 0:
                object_id = 'background'
                object_name = 'background'
                confidence = 0.0
            else:
                winner = max(votes, key=votes.get)
                confidence = votes[winner] / total_votes

                # Occluded parts (chain beneath bulldozer, underside of objects)
                # may only be visible in a single view. If that one view says it's
                # an object, trust it — COLMAP only includes this point because it
                # was genuinely matched there.
                single_view_object = (total_votes == 1 and winner != 'background')

                if winner == 'background' or (not single_view_object and confidence < 0.5):
                    object_id = 'background'
                    object_name = 'background'
                else:
                    object_id = winner
                    object_name = object_id_to_name[winner]
            
            labeled_points.append({
                'point_id': point_id,
                'xyz': point.xyz,
                'rgb': point.rgb,
                'object_id': object_id,
                'object_name': object_name,
                'confidence': confidence
            })
        
        # Print statistics
        self._print_statistics(labeled_points, object_id_to_name)
        
        return labeled_points
    
    def _load_masks(self, masks_dir: Path, object_id_to_name: dict, 
                   image_files: list) -> dict:
        """
        Load all masks into memory
        
        Returns:
            Dict: {image_name: {obj_id: mask_array}}
            Example: {"IMG_001.jpg": {1: mask1, 2: mask2, ...}}
        """
        all_masks = {}
        
        for image_name in tqdm(image_files, desc="Loading masks"):
            image_stem = Path(image_name).stem
            masks_for_image = {}
            
            # Load each object's mask for this image
            for obj_id, obj_name in object_id_to_name.items():
                mask_filename = f"{image_stem}_mask_{obj_name}.JPG"
                mask_path = masks_dir / mask_filename
                
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask_binary = (mask > 127)
                    masks_for_image[obj_id] = mask_binary
            
            if masks_for_image:
                all_masks[image_name] = masks_for_image
        
        return all_masks
    
    def _print_statistics(self, labeled_points: list, object_id_to_name: dict):
        """Print labeling statistics"""
        print("\n" + "="*60)
        print("LABELING STATISTICS")
        print("="*60)
        
        # Count points per object
        for obj_id, obj_name in object_id_to_name.items():
            count = sum(1 for p in labeled_points if p['object_id'] == obj_id)
            avg_conf = np.mean([p['confidence'] for p in labeled_points 
                               if p['object_id'] == obj_id]) if count > 0 else 0
            print(f"  {obj_name:20s}: {count:6d} points (avg conf: {avg_conf:.2f})")
        
        bg_count = sum(1 for p in labeled_points if p['object_id'] == 'background')
        bg_conf = np.mean([p['confidence'] for p in labeled_points 
                          if p['object_id'] == 'background']) if bg_count > 0 else 0
        print(f"  {'background':20s}: {bg_count:6d} points (avg conf: {bg_conf:.2f})")
        
        print("="*60)
    
    def save_labeled_points(self, labeled_points: list, output_path: str):
        """
        Save labeled point cloud to PLY format with object IDs using Open3D
        
        Args:
            labeled_points: List of labeled points
            output_path: Path to save PLY file
        """
        
        os.makedirs(output_path, exist_ok=True)
        output_path = Path(output_path) / "labeled_point_cloud.ply"
        
        print(f"\nSaving labeled point cloud to: {output_path}")
        
        # Prepare data for Open3D
        points_xyz = np.array([p['xyz'] for p in labeled_points])
        points_rgb = np.array([p['rgb'] for p in labeled_points])
        
        # Normalize RGB to [0, 1] for Open3D
        colors_normalized = points_rgb.astype(np.float64) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        # Save using Open3D (binary format for better compatibility)
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        
        print(f"✓ Point cloud saved: {output_path}")
        print(f"  Format: Binary PLY with XYZ + RGB")
        print(f"  Note: Labels and confidence stored in metadata JSON")
        
        # Save object IDs, names, and confidence separately
        metadata_path = output_path.parent / (output_path.stem + "_metadata.json")
        metadata = {
            'point_labels': [
                {
                    'point_id': p['point_id'],
                    'object_id': p['object_id'],
                    'object_name': p['object_name'],
                    'confidence': p['confidence']
                }
                for p in labeled_points
            ],
            'num_points': len(labeled_points)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {metadata_path}")


# Standalone script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label 3D points with object IDs')
    # parser.add_argument('--colmap-dir', type=str, default="output/sfm/garden",#required=True,
    #                    help='COLMAP output directory (with cameras.bin, images.bin, points3D.bin)')
    parser.add_argument('--colmap-dir', type=str, required=True,
                       help='COLMAP output directory (with cameras.bin, images.bin, points3D.bin)')
    parser.add_argument('--segmentation-dir', type=str,default="output/preprocessing/garden/segmentation",# required=True,
                       help='Segmentation output directory (with masks/ and metadata)')
    parser.add_argument('--output', type=str,default="output/point_cloud/garden",# required=True,
                       help='Output path for labeled point cloud (.ply)')
    parser.add_argument('--use-refined-masks', action='store_true',
                       help='Vote with ../refined/refined_masks instead of the raw SAM2 masks')
    
    args = parser.parse_args()
    
    # Load COLMAP results
    print("Loading COLMAP results...")
    colmap_dir = Path(args.colmap_dir)
    
    # Import the COLMAP reader from sfm
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from sfm import COLMAPWrapper
    
    wrapper = COLMAPWrapper()
    
    # Read binary files
    model_dir = colmap_dir / "sparse" / "0"
    if not model_dir.exists():
        # Try to find the model directory
        sparse_dir = colmap_dir / "sparse"
        model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if model_dirs:
            model_dir = model_dirs[0]
        else:
            raise FileNotFoundError(f"No model found in {sparse_dir}")
    
    cameras = wrapper._read_cameras(model_dir / "cameras.bin")
    images = wrapper._read_images(model_dir / "images.bin")
    points3D = wrapper._read_points3D(model_dir / "points3D.bin")
    
    colmap_results = {
        'cameras': cameras,
        'images': images,
        'points3D': points3D
    }
    
    # Label points
    labeler = PointLabeler()
    labeled_points = labeler.label_points(colmap_results, args.segmentation_dir,
                                          use_refined_masks=args.use_refined_masks)
    
    # Save results
    labeler.save_labeled_points(labeled_points, args.output)
    
    print("\n✓ Point labeling complete!")