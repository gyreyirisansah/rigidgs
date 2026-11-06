"""
Render foreground and background-hole images of a trained RigidGS model at
the 40 held-out (no-object) SPIn-NeRF camera poses, for evaluate_spinnerf.py.

The held-out poses come from <source_path>/test_poses.json if present,
otherwise from <source_path>/sparse/0 (all poses) filtered by test.txt, as
written by preprocessing.py --spinnerf. Both are solved in the same joint
COLMAP reconstruction as the training views, so they share the trained
scene's coordinate frame. Cameras are built with the same CameraInfo ->
loadCam path the dataset readers use, with images pointed at the staged
ground-truth photos (eval/stage_spinnerf_gt.py).

The checkpoint is loaded normally via Scene(..., load_iteration=N), which
restores per-Gaussian object ids, so foreground/background separation is
the same as during training.

Usage (normally called by run_spinnerf_eval.py):
    python eval/render_spinnerf_holdout.py \
        --source_path output/spinnerf/2/point_cloud/pcd_data \
        --images output/spinnerf/2/images_undistorted \
        --scene 2 --model_path output/spinnerf/2/rigidgs --iteration -1

Output:
    <output_dir>/foreground_renders/iter_<N>/{stem}.png
    <output_dir>/background_hole_renders/iter_<N>/{stem}.png
File names match data/ground_truth/images/spinnerf/<scene>/{stem}.*.
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "external/gaussian-splatting")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scene import Scene, GaussianModel
from scene.dataset_readers import CameraInfo
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import focal2fov
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render
from utils.general_utils import safe_state
from rigidgs_arguments import ModelParams, PipelineParams

from stage3_refinement import save_background_hole_renders
# save_foreground_renders is defined in the trainer -- import it directly
# so this reuses the exact rendering logic of training-time checkpoints.
# Loaded by file path: gaussian-splatting has its own train.py on sys.path.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rigidgs_train", Path(__file__).resolve().parents[1] / "src" / "train.py")
_rigidgs_train = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rigidgs_train)
save_foreground_renders = _rigidgs_train.save_foreground_renders

REPO_ROOT = Path(__file__).resolve().parents[1]


def _test_poses_from_colmap(sparse_dir: Path) -> dict:
    """The held-out poses in test_poses.json's format, read from the
    preprocessing output: pcd_data/sparse/0/{cameras,images}.bin keep all
    100 poses and test.txt (written by preprocessing.py --spinnerf) lists
    the 40 held-out view names."""
    from spinnerf_utils import read_cameras_bin, read_images_bin_full
    cam = next(iter(read_cameras_bin(sparse_dir / "cameras.bin").values()))
    assert cam["model"] == "PINHOLE", f"Unexpected camera model {cam['model']}"
    fx, fy, cx, cy = (float(v) for v in cam["params"])
    test_names = {l.strip() for l in open(sparse_dir / "test.txt") if l.strip()}
    images = sorted((img for img in read_images_bin_full(sparse_dir / "images.bin").values()
                     if img["name"] in test_names), key=lambda img: img["name"])
    assert len(images) == len(test_names), (
        f"{len(test_names)} names in test.txt but {len(images)} matching poses")
    return {
        "camera": dict(model="PINHOLE", width=int(cam["width"]), height=int(cam["height"]),
                       fx=fx, fy=fy, cx=cx, cy=cy),
        "images": [dict(name=img["name"], qvec=list(map(float, img["qvec"])),
                        tvec=list(map(float, img["tvec"]))) for img in images],
    }


def _build_test_cameras(test_poses_path: Path, gt_dir: Path, dataset) -> list:
    """CameraInfo objects for the 40 held-out poses, images pointed at the
    already-undistorted GT photos staged by spinnerf_utils.py --
    same pixel grid as the trained scene, just never used for training.
    Falls back to the poses in <source_path>/sparse/0 when there is no
    test_poses.json (the release preprocessing does not write one)."""
    if test_poses_path.exists():
        with open(test_poses_path) as f:
            data = json.load(f)
    else:
        data = _test_poses_from_colmap(Path(dataset.source_path) / "sparse" / "0")
    cam = data["camera"]
    assert cam["model"] == "PINHOLE", f"Unexpected camera model {cam['model']}"
    FovX = focal2fov(cam["fx"], cam["width"])
    FovY = focal2fov(cam["fy"], cam["height"])

    cam_infos = []
    for uid, entry in enumerate(data["images"]):
        qvec = np.array(entry["qvec"])
        tvec = np.array(entry["tvec"])
        R = np.transpose(qvec2rotmat(qvec))
        T = tvec
        image_path = str(gt_dir / entry["name"])
        cam_infos.append(CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=None,
            image_path=image_path, image_name=entry["name"], depth_path="",
            width=cam["width"], height=cam["height"], is_test=True,
        ))
    return cameraList_from_camInfos(cam_infos, 1.0, dataset,
                                     is_nerf_synthetic=False, is_test_dataset=True)


def main():
    parser = ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Checkpoint iteration to load (-1 = latest).")
    parser.add_argument("--test_poses", type=str, default=None,
                        help="Path to test_poses.json. Default: "
                             "<source_path>/test_poses.json.")
    parser.add_argument("--gt_root", type=str, default="data/ground_truth/images/spinnerf",
                        help="Root of the staged GT no-object photos "
                             "(spinnerf_utils.py's data/ground_truth/images/spinnerf/<scene>/).")
    parser.add_argument("--scene", type=str, required=True,
                        help="SPIn-NeRF scene id, to find <gt_root>/<scene>/ "
                             "for the GT photos referenced by test_poses.json.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write foreground_renders/ and "
                             "background_hole_renders/. Default: "
                             "<model_path>/spinnerf_holdout_eval.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    # readColmapCameras joins images_folder with a filename directly (no
    # join against source_path), so this must be resolved to an absolute
    # path.
    dataset.images = str(Path(dataset.images).resolve())

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        iteration = scene.loaded_iter
        background_id = scene.getBackgroundObjectID()

        test_poses_path = Path(args.test_poses) if args.test_poses else \
            Path(dataset.source_path) / "test_poses.json"
        gt_dir = REPO_ROOT / args.gt_root / args.scene
        test_cameras = _build_test_cameras(test_poses_path, gt_dir, dataset)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        output_dir = Path(args.output_dir) if args.output_dir else \
            Path(dataset.model_path) / "spinnerf_holdout_eval"

        print(f"[render_spinnerf_holdout] {len(test_cameras)} held-out cameras "
              f"(from {test_poses_path}), iteration={iteration}, background_id={background_id}")

        save_foreground_renders(
            gaussians, background_id, test_cameras, render, pipe, background,
            str(output_dir / "foreground_renders"), iteration,
        )
        save_background_hole_renders(
            gaussians, background_id, test_cameras, render, pipe, background,
            str(output_dir / "background_hole_renders"), iteration,
        )


if __name__ == "__main__":
    main()
