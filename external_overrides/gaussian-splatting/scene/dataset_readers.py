#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modified for VR-RigidGS: Support both labeled (legacy) and unlabeled point clouds.
# When no labels exist, the system learns identities from 2D masks during training.
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    # ===== Identity/segmentation fields =====
    num_objects: int = 0                       # Number of foreground objects (0 = no segmentation)
    object_id_array: Optional[np.ndarray] = None       # Pre-labeled IDs (legacy, from point labeling)
    object_name_to_newid: Optional[dict] = None
    object_newid_to_name: Optional[dict] = None
    # ===== Camera sets =====
    inpainted_cameras: Optional[list] = None
    test_inpainted_cameras: Optional[list] = None


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, 
                      depths_folder, test_cam_names_list, inpainted_folder=""):
    cam_infos = []
    inpainted_cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

        if inpainted_folder != "":
            inpainted_path = os.path.join(inpainted_folder, extr.name)
            inpainted_cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                                image_path=inpainted_path, image_name=image_name, depth_path=depth_path,
                                width=width, height=height, is_test=image_name in test_cam_names_list)
            inpainted_cam_infos.append(inpainted_cam_info)

    sys.stdout.write('\n')
    return cam_infos, inpainted_cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def load_labeled_points(ply_path: str, metadata_path: str):
    """Load points, colors and labels from labeled point cloud (legacy pipeline)"""
    if OPEN3D_AVAILABLE:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
    else:
        # Fallback to plyfile
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    labels = metadata['point_labels']
    
    return points, colors, labels


def fetchRigidGSPly(path, metadata_path):
    """Load labeled point cloud with object IDs (legacy pipeline with point labeling)"""
    points, colors, point_labels = load_labeled_points(path, metadata_path)
    normals = np.zeros_like(points)
    
    unique_names = sorted(set(p["object_name"] for p in point_labels))
    unique_names = ["background"] + [n for n in unique_names if n != "background"]
    object_name_to_newid = {name: i for i, name in enumerate(unique_names)}
    object_newid_to_name = {i: name for name, i in object_name_to_newid.items()}
    object_id_array = np.array([
        object_name_to_newid[p["object_name"]] for p in point_labels
    ], dtype=np.int32)

    num_objects = len(unique_names) - 1  # exclude background

    return (BasicPointCloud(points=points, colors=colors, normals=normals), 
            object_id_array, object_name_to_newid, object_newid_to_name, num_objects)


def count_objects_from_masks(masks_dir: str) -> int:
    """
    Detect how many foreground objects exist by examining the masks directory.
    
    Supports two layouts:
      1. Subfolder per view: masks_dir/{view_name}/ with object_001.png, object_002.png, ...
      2. Flat: masks_dir/ with {view_name}_mask_1.png, {view_name}_mask_2.png, ...
    
    Returns:
        Number of foreground objects detected (0 if can't determine)
    """
    masks_path = Path(masks_dir)
    if not masks_path.exists():
        return 0
    
    max_obj_id = 0
    
    # Check subfolder layout first
    subdirs = [d for d in masks_path.iterdir() if d.is_dir()]
    if subdirs:
        # Sample first subfolder
        sample_dir = subdirs[0]
        for f in sample_dir.iterdir():
            if f.suffix.lower() == '.png':
                # Try to extract object ID from filename
                name = f.stem
                for pattern_fn in [
                    lambda n: int(n.split('_')[-1]) if n.split('_')[-1].isdigit() else 0,
                    lambda n: int(n) if n.isdigit() else 0,
                ]:
                    try:
                        obj_id = pattern_fn(name)
                        max_obj_id = max(max_obj_id, obj_id)
                    except (ValueError, IndexError):
                        continue
    else:
        # Flat layout — scan filenames
        for f in masks_path.iterdir():
            if f.suffix.lower() == '.png':
                name = f.stem
                # Try patterns like "DSC_001_mask_2" or "image001_obj3"
                parts = name.split('_')
                for part in reversed(parts):
                    if part.isdigit():
                        obj_id = int(part)
                        if obj_id > 0:  # object IDs are 1-indexed
                            max_obj_id = max(max_obj_id, obj_id)
                        break

    if max_obj_id > 0:
        print(f"[count_objects_from_masks] Detected {max_obj_id} foreground objects from {masks_dir}")
    else:
        print(f"[count_objects_from_masks] Could not auto-detect object count from {masks_dir}")
    
    return max_obj_id


def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8,
                        inpainted_folder="", masks_dir="", num_objects=0):
    """
    Read COLMAP scene info.
    
    Now supports two modes:
      1. Legacy (with point labeling): reads labeled PLY + metadata
      2. New (Gaussian Grouping): reads standard PLY, num_objects from masks or argument
    
    Args:
        masks_dir: Path to SAM masks directory (for auto-detecting num_objects)
        num_objects: Number of foreground objects (overrides auto-detection)
    """
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
        except FileNotFoundError:
            print(f"Error: depth_params.json not found at '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error reading depth_params.json: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images is None else images
    cam_infos_unsorted, inpainted_cam_infos = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=reading_dir,
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list,
        inpainted_folder=inpainted_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    if len(inpainted_cam_infos) > 0:
        inpainted_cam_infos = sorted(inpainted_cam_infos.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    train_inpainted_cam_infos = [c for c in inpainted_cam_infos if train_test_exp or not c.is_test]
    test_inpainted_cam_infos = [c for c in inpainted_cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ================================================================
    # Point cloud loading — try labeled first, fall back to standard
    # ================================================================
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    label_meta_data_path = os.path.join(path, "sparse/0/points3D_metadata.json")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    # Defaults for unlabeled case
    pcd = None
    object_id_array = None
    object_name_to_newid = None
    object_newid_to_name = None
    detected_num_objects = num_objects

    # Try loading labeled point cloud (legacy pipeline)
    if os.path.exists(label_meta_data_path):
        try:
            pcd, object_id_array, object_name_to_newid, object_newid_to_name, detected_num_objects = \
                fetchRigidGSPly(ply_path, label_meta_data_path)
            print(f"[SceneInfo] Loaded LABELED point cloud: {len(object_id_array)} points, "
                  f"{detected_num_objects} foreground objects")
        except Exception as e:
            print(f"[SceneInfo] Could not read labeled PLY ({e}), falling back to standard PLY")
            pcd = None

    # Fall back to standard PLY (no labels — identities will be learned)
    if pcd is None:
        try:
            pcd = fetchPly(ply_path)
            print(f"[SceneInfo] Loaded UNLABELED point cloud: {len(pcd.points)} points")
            print(f"[SceneInfo] Object identities will be learned from 2D masks during training")
        except Exception as e:
            print(f"[SceneInfo] ERROR: Could not read PLY: {e}")
            pcd = None

    # Auto-detect num_objects from masks if not provided and not from labels
    if detected_num_objects == 0 and masks_dir and os.path.isdir(masks_dir):
        detected_num_objects = count_objects_from_masks(masks_dir)

    if detected_num_objects == 0:
        print(f"[SceneInfo] WARNING: num_objects=0. Set --num_objects or provide masks_dir for auto-detection.")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
        num_objects=detected_num_objects,
        object_id_array=object_id_array,
        object_name_to_newid=object_name_to_newid,
        object_newid_to_name=object_newid_to_name,
        inpainted_cameras=train_inpainted_cam_infos,
        test_inpainted_cameras=test_inpainted_cam_infos,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], 
                            depth_path=depth_path, depth_params=None, is_test=is_test))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):
    depths_folder = os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
