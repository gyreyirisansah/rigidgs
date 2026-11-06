"""
RigidGS COLMAP SfM Initialization Module
Runs COLMAP to extract 3D points and camera parameters
"""

import os
import subprocess
import numpy as np
from pathlib import Path
import struct
import collections
import argparse
from tqdm import tqdm
import pycolmap


# COLMAP camera model (from COLMAP binary format)
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
CAMERA_MODELS = {
    0: CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    2: CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    3: CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    4: CameraModel(model_id=4, model_name="OPENCV", num_params=8),
}

# Camera intrinsics
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

# Camera pose (extrinsics)
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

# 3D point
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class COLMAPWrapper:
    """
    Wrapper for running COLMAP structure from motion
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize COLMAP wrapper
        
        Args:
            config: Optional configuration dict with:
                - matcher: 'exhaustive' or 'sequential'
                - camera_model: 'PINHOLE', 'SIMPLE_RADIAL', etc.
        """
        self.config = config or {}
        self.matcher = self.config.get('matcher', 'exhaustive')
        self.camera_model = self.config.get('camera_model', 'PINHOLE')
        
    def run_sfm(self, image_dir: str, output_dir: str, 
                max_num_features: int = 8192) -> dict:
        """
        Run complete COLMAP SfM pipeline
        
        Args:
            image_dir: Directory with input images
            output_dir: Output directory for COLMAP results
            max_num_features: Maximum features to extract per image
            
        Returns:
            Dict with cameras, images, and points3D
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        database_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("COLMAP STRUCTURE FROM MOTION")
        print("="*60)
        print(f"Images: {image_dir}")
        print(f"Output: {output_dir}")
        print(f"Matcher: {self.matcher}")
        print("="*60)
        
        # Step 1: Feature extraction
        print("\n[1/4] Extracting features...")
        self._run_feature_extraction(image_dir, database_path, max_num_features)
        
        # Step 2: Feature matching
        print("\n[2/4] Matching features...")
        self._run_feature_matching(database_path)
        
        # Step 3: Sparse reconstruction
        print("\n[3/4] Running sparse reconstruction...")
        self._run_mapper(image_dir, database_path, sparse_dir)
        
        # Step 4: Parse results
        print("\n[4/4] Parsing COLMAP results...")
        model_dir = self._find_model_dir(sparse_dir)
        
        cameras = self._read_cameras(model_dir / "cameras.bin")
        images = self._read_images(model_dir / "images.bin")
        points3D = self._read_points3D(model_dir / "points3D.bin")
        
        # Also export to text format for easier debugging
        print("\nExporting text format...")
        self._export_text(model_dir, output_dir)
        
        print("\n" + "="*60)
        print("COLMAP COMPLETE")
        print("="*60)
        print(f"Cameras: {len(cameras)}")
        print(f"Images: {len(images)}")
        print(f"3D Points: {len(points3D)}")
        print(f"Model saved to: {model_dir}")
        print("="*60)
        
        return {
            'cameras': cameras,
            'images': images,
            'points3D': points3D,
            'model_dir': model_dir
        }
    
    def _run_feature_extraction2(self, image_dir, database_path, max_num_features):
        colmap_exe = r"C:\Users\pgyreyir\COLMAP-3.9.1-windows-cuda\bin\colmap.exe"  # Adjust this path
        """Run COLMAP feature extraction"""
        cmd = [
            colmap_exe, 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(image_dir),
            '--ImageReader.camera_model', self.camera_model,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.max_num_features', str(max_num_features),
            '--SiftExtraction.use_gpu', '0'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError("Feature extraction failed")
        print("✓ Feature extraction complete")

    def _run_feature_extraction(self, image_dir, database_path, max_num_features):
        """Run COLMAP feature extraction"""
        import shutil
        
        # Find colmap executable
        # colmap_path = shutil.which('colmap')
        # if colmap_path and colmap_path.endswith('.BAT'):
        #     colmap_path = colmap_path.replace('.BAT', '.exe')
        
        # if not colmap_path:
        #     colmap_path = r"C:\Users\pgyreyir\COLMAP-3.9.1-windows-cuda\bin\colmap.exe"

        colmap_path = "colmap"
        
        print(f"Using COLMAP at: {colmap_path}")
        print(f"Image directory: {image_dir}")
        print(f"Database path: {database_path}")
        
        # Verify inputs exist
        if not os.path.exists(image_dir):
            raise RuntimeError(f"Image directory does not exist: {image_dir}")
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        cmd = [
            colmap_path, 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(image_dir),
            '--ImageReader.camera_model', self.camera_model,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.max_num_features', str(max_num_features),
            # '--SiftExtraction.use_gpu', 'true'
            # '--SiftExtraction.gpu_index', '0'
            # '--SiftExtraction.use_gpu', '1'  # Changed to 0 (CPU mode)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print full output for debugging
        if result.stdout:
            print(f"COLMAP stdout:\n{result.stdout}")
        if result.stderr:
            print(f"COLMAP stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            # Convert to hex for debugging
            hex_code = hex(result.returncode & 0xFFFFFFFF)
            raise RuntimeError(
                f"Feature extraction failed with return code {result.returncode} ({hex_code})\n"
                f"This may indicate a crash or access violation.\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )
        
        print("✓ Feature extraction complete")

    
    

    def _run_feature_extraction0(self, image_dir, database_path, max_num_features):
        """Run COLMAP feature extraction using PyColmap"""
        
        # PyColmap API
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_model=self.camera_model,
            sift_options={
                'max_num_features': max_num_features,
                'use_gpu': True  # or False
            }
        )
        print("✓ Feature extraction complete")
    
    def _run_feature_matching(self, database_path):
        """Run COLMAP feature matching"""
        
        colmpat_dir = "colmap"

        if self.matcher == 'exhaustive':
            cmd = [
                colmpat_dir, 'exhaustive_matcher',
                '--database_path', str(database_path),
                # '--SiftMatching.use_gpu', '1'
            ]
        else:  # sequential
            cmd = [
                colmpat_dir, 'sequential_matcher',
                '--database_path', str(database_path),
                # '--SiftMatching.use_gpu', '1'
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError("Feature matching failed")
        print("✓ Feature matching complete")

    def _run_feature_matching1(self, database_path, use_gpu=True):
        """Run COLMAP feature matching using PyColmap"""
        
        sift_options = {
            'use_gpu': use_gpu,
            'max_num_matches': 32768,  # Maximum number of matches
            'max_distance': 0.7,        # Maximum descriptor distance
            'cross_check': True,        # Enable cross-checking
            'max_error': 4.0            # Maximum epipolar error in pixels
        }
        
        try:
            if self.matcher == 'exhaustive':
                pycolmap.match_exhaustive(
                    database_path=database_path,
                    sift_options=sift_options
                )
            else:  # sequential
                pycolmap.match_sequential(
                    database_path=database_path,
                    overlap=10,  # Number of overlapping image pairs
                    sift_options=sift_options
                )
            
            print("✓ Feature matching complete")
            
        except Exception as e:
            print(f"Error: {e}")
            raise RuntimeError(f"Feature matching failed: {e}")
        
    def _run_feature_matching0(self, database_path, use_gpu=True):
        """Run COLMAP feature matching using PyColmap"""
        import pycolmap
        
        print(f"Running {self.matcher} matching...")
        print(f"Database: {database_path}")
        print(f"GPU enabled: {use_gpu}")
        
        try:
            # Convert WindowsPath to string
            database_path = str(database_path)
            
            # Create proper PyColmap options objects
            sift_options = pycolmap.SiftMatchingOptions()
            sift_options.max_num_matches = 32768
            sift_options.max_distance = 0.7
            sift_options.cross_check = True
            # sift_options.max_error = 4.0
            
            # Set device (GPU or CPU)
            device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
            
            if self.matcher == 'exhaustive':
                matching_options = pycolmap.ExhaustiveMatchingOptions()
                
                pycolmap.match_exhaustive(
                    database_path=database_path,
                    sift_options=sift_options,
                    matching_options=matching_options,
                    device=device
                )
            else:  # sequential
                matching_options = pycolmap.SequentialMatchingOptions()
                matching_options.overlap = 10  # Number of overlapping images
                matching_options.loop_detection = True
                
                pycolmap.match_sequential(
                    database_path=database_path,
                    sift_options=sift_options,
                    matching_options=matching_options,
                    device=device
                )
            
            print("✓ Feature matching complete")
            
        except Exception as e:
            # If GPU fails, retry with CPU
            if use_gpu and ("CUDA" in str(e) or "GPU" in str(e)):
                print(f"GPU matching failed, retrying with CPU...")
                return self._run_feature_matching(database_path, use_gpu=False)
            else:
                print(f"Error: {e}")
                raise RuntimeError(f"Feature matching failed: {e}")
    
    def _run_mapper(self, image_dir, database_path, sparse_dir):
        """Run COLMAP sparse reconstruction"""
        colmpat_dir = "colmap"
        cmd = [
            colmpat_dir, 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(image_dir),
            '--output_path', str(sparse_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError("Sparse reconstruction failed")
        print("✓ Sparse reconstruction complete")
    
    def _find_model_dir(self, sparse_dir):
        """Find the best model directory (usually '0')"""
        model_dirs = [d for d in Path(sparse_dir).iterdir() if d.is_dir()]
        if not model_dirs:
            raise RuntimeError("No model found in sparse directory")
        
        # Usually the first model is the best
        return sorted(model_dirs)[0]
    
    def _export_text(self, model_dir, output_dir):
        """Export COLMAP model to text format"""
        colmpat_dir = "colmap"
        cmd = [
            colmpat_dir, 'model_converter',
            '--input_path', str(model_dir),
            '--output_path', str(output_dir / 'text'),
            '--output_type', 'TXT'
        ]
        
        subprocess.run(cmd, capture_output=True, text=True)
        print(f"✓ Text format saved to: {output_dir / 'text'}")
    
    # Binary file readers
    def _read_cameras(self, path):
        """Read cameras.bin"""
        cameras = {}
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                num_params = CAMERA_MODELS[model_id].num_params
                params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
                
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=CAMERA_MODELS[model_id].model_name,
                    width=width,
                    height=height,
                    params=np.array(params)
                )
        return cameras
    
    def _read_images(self, path):
        """Read images.bin"""
        images = {}
        with open(path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qvec = struct.unpack("<4d", f.read(32))
                tvec = struct.unpack("<3d", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]
                
                # Read image name
                name = b""
                while True:
                    char = f.read(1)
                    if char == b"\x00":
                        break
                    name += char
                name = name.decode("utf-8")
                
                # Read 2D points
                num_points2D = struct.unpack("<Q", f.read(8))[0]
                xys = np.zeros((num_points2D, 2))
                point3D_ids = np.zeros(num_points2D, dtype=np.int64)
                
                for i in range(num_points2D):
                    xys[i] = struct.unpack("<2d", f.read(16))
                    point3D_ids[i] = struct.unpack("<q", f.read(8))[0]
                
                images[image_id] = Image(
                    id=image_id,
                    qvec=np.array(qvec),
                    tvec=np.array(tvec),
                    camera_id=camera_id,
                    name=name,
                    xys=xys,
                    point3D_ids=point3D_ids
                )
        return images
    
    def _read_points3D(self, path):
        """Read points3D.bin"""
        points3D = {}
        with open(path, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_points):
                point_id = struct.unpack("<Q", f.read(8))[0]
                xyz = struct.unpack("<3d", f.read(24))
                rgb = struct.unpack("<3B", f.read(3))
                error = struct.unpack("<d", f.read(8))[0]
                
                # Read track
                track_length = struct.unpack("<Q", f.read(8))[0]
                image_ids = []
                point2D_idxs = []
                for _ in range(track_length):
                    image_ids.append(struct.unpack("<I", f.read(4))[0])
                    point2D_idxs.append(struct.unpack("<I", f.read(4))[0])
                
                points3D[point_id] = Point3D(
                    id=point_id,
                    xyz=np.array(xyz),
                    rgb=np.array(rgb),
                    error=error,
                    image_ids=np.array(image_ids),
                    point2D_idxs=np.array(point2D_idxs)
                )
        return points3D


# Standalone script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run COLMAP SfM')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--output', type=str, default="output/sfm/garden",#required=True,
                       help='Output directory for COLMAP results')
    parser.add_argument('--matcher', type=str, default='exhaustive',
                       choices=['exhaustive', 'sequential'],
                       help='Feature matching strategy')
    parser.add_argument('--camera-model', type=str, default='PINHOLE',
                       help='Camera model (PINHOLE, SIMPLE_RADIAL, etc.)')
    parser.add_argument('--max-features', type=int, default=8192,
                       help='Maximum features per image')
    
    args = parser.parse_args()
    
    config = {
        'matcher': args.matcher,
        'camera_model': args.camera_model
    }
    
    colmap = COLMAPWrapper(config)
    results = colmap.run_sfm(args.images, args.output, args.max_features)

    
    
    print(f"\n✓ SfM complete!")
    print(f"   Results saved to: {results['model_dir']}")