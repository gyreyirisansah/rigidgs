"""
SPIn-NeRF helpers: COLMAP binary readers/writers, lens undistortion and the
training-view point filter used by preprocessing.py --spinnerf and
eval/stage_spinnerf_gt.py.
"""

import struct
from pathlib import Path

import cv2
import numpy as np

CAMERA_MODEL_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}

PINHOLE_MODEL_ID = 1


def _read(fid, n, fmt):
    return struct.unpack("<" + fmt, fid.read(n))


def read_cameras_bin(path: Path) -> dict:
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(fid, 24, "iiQQ")
            model_name, num_params = CAMERA_MODEL_PARAMS[model_id]
            params = np.array(_read(fid, 8 * num_params, "d" * num_params))
            cameras[camera_id] = dict(model=model_name, width=width,
                                       height=height, params=params)
    return cameras


def write_pinhole_cameras_bin(path: Path, cameras: dict):
    """cameras: {camera_id: dict(width, height, fx, fy, cx, cy)}"""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera_id, cam in cameras.items():
            fid.write(struct.pack("<iiQQ", camera_id, PINHOLE_MODEL_ID,
                                   cam["width"], cam["height"]))
            fid.write(struct.pack("<dddd", cam["fx"], cam["fy"],
                                   cam["cx"], cam["cy"]))


def read_images_bin_full(path: Path) -> dict:
    """Full images.bin parse, keyed by image_id, preserving file order.

    {image_id: dict(qvec, tvec, camera_id, name, xys, point3D_ids)}
    """
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = props[1:5]
            tvec = props[5:8]
            camera_id = props[8]
            name = ""
            c = fid.read(1)
            while c != b"\x00":
                name += c.decode()
                c = fid.read(1)
            num_points2D = _read(fid, 8, "Q")[0]
            raw = _read(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = list(zip(raw[0::3], raw[1::3]))
            point3D_ids = list(raw[2::3])
            images[image_id] = dict(
                qvec=qvec, tvec=tvec, camera_id=camera_id, name=name,
                xys=xys, point3D_ids=point3D_ids,
            )
    return images


def write_images_bin_full(path: Path, images: dict):
    """Write images.bin in the same schema read_images_bin_full returns."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id, img in images.items():
            fid.write(struct.pack("<i", image_id))
            fid.write(struct.pack("<dddd", *img["qvec"]))
            fid.write(struct.pack("<ddd", *img["tvec"]))
            fid.write(struct.pack("<i", img["camera_id"]))
            fid.write(img["name"].encode() + b"\x00")
            fid.write(struct.pack("<Q", len(img["xys"])))
            for (x, y), pid in zip(img["xys"], img["point3D_ids"]):
                fid.write(struct.pack("<ddq", x, y, pid))


def read_points3D_bin_full(path: Path) -> dict:
    """Full points3D.bin parse including tracks, preserving file order.

    {point_id: dict(xyz, rgb, error, image_ids, point2D_idxs)}
    """
    points = {}
    with open(path, "rb") as fid:
        num_points = _read(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read(fid, 43, "QdddBBBd")
            point_id = props[0]
            track_length = _read(fid, 8, "Q")[0]
            track = _read(fid, 8 * track_length, "ii" * track_length)
            points[point_id] = dict(
                xyz=props[1:4], rgb=props[4:7], error=props[7],
                image_ids=list(track[0::2]), point2D_idxs=list(track[1::2]),
            )
    return points


def write_points3D_bin_full(path: Path, points: dict):
    """Write points3D.bin in the same schema read_points3D_bin_full returns."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points)))
        for point_id, p in points.items():
            fid.write(struct.pack("<Q", point_id))
            fid.write(struct.pack("<ddd", *p["xyz"]))
            fid.write(struct.pack("<BBB", *(int(v) for v in p["rgb"])))
            fid.write(struct.pack("<d", p["error"]))
            fid.write(struct.pack("<Q", len(p["image_ids"])))
            for iid, idx in zip(p["image_ids"], p["point2D_idxs"]):
                fid.write(struct.pack("<ii", iid, idx))


def filter_points_to_training_images(points: dict, train_image_ids: set) -> dict:
    """
    Keep only 3D points triangulated exclusively from training-image
    observations -- drop any point whose track includes even a single
    test-image observation, so the initial Gaussian point cloud carries zero
    information from the 40 held-out no-object photos. Some background
    points are only visible there (occluded by the object in every training
    view); keeping them would leak the true unoccluded background geometry
    straight into the eval.
    """
    return {
        point_id: p for point_id, p in points.items()
        if len(p["image_ids"]) >= 2 and all(iid in train_image_ids for iid in p["image_ids"])
    }


def build_undistort_map(cam_full_res: dict, sample_w: int, sample_h: int):
    """
    Scale a SIMPLE_RADIAL camera (registered at full resolution) down to the
    actual images_4 resolution, then compute the undistort remap + crop ROI
    for a new PINHOLE camera at that resolution.

    Returns (map_x, map_y, roi, new_cam) where new_cam has width/height/fx/fy/cx/cy.
    """
    assert cam_full_res["model"] == "SIMPLE_RADIAL", (
        f"Unexpected camera model {cam_full_res['model']}, "
        "this script only handles SPIn-NeRF's SIMPLE_RADIAL cameras."
    )
    f, cx, cy, k1 = cam_full_res["params"]
    scale = sample_w / cam_full_res["width"]
    assert abs(scale - sample_h / cam_full_res["height"]) < 1e-3, (
        "images_4 aspect ratio doesn't match cameras.bin aspect ratio"
    )

    f, cx, cy = f * scale, cx * scale, cy * scale
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, dist, (sample_w, sample_h), alpha=0, newImgSize=(sample_w, sample_h)
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (sample_w, sample_h), cv2.CV_32FC1
    )

    x, y, w, h = roi
    new_cam = dict(
        width=w, height=h,
        fx=float(new_K[0, 0]), fy=float(new_K[1, 1]),
        cx=float(new_K[0, 2] - x), cy=float(new_K[1, 2] - y),
    )
    return map_x, map_y, roi, new_cam


def undistort_image(img: np.ndarray, map_x, map_y, roi, interpolation) -> np.ndarray:
    remapped = cv2.remap(img, map_x, map_y, interpolation=interpolation,
                          borderMode=cv2.BORDER_CONSTANT)
    x, y, w, h = roi
    return remapped[y:y + h, x:x + w]
