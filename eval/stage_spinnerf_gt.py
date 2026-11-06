"""
Stage the SPIn-NeRF evaluation ground truth (paper Table 5).

For each scene, from the raw dataset (data/spinnerf/<scene>/):

    <gt_root>/<scene>/<stem>.<ext>       the 40 no-object photos (images_4/,
                                         undistorted with the scene's own
                                         SIMPLE_RADIAL calibration)
    <gt_mask_root>/<scene>/<stem>.png    the matching images_4/test_label/
                                         region mask, undistorted the same way

test_label/out_0000N.png carry no filename correspondence to the no-object
photos; they are paired by sorted order (sorted(no-object) <-> sorted(test_label)).
This pairing was spot-checked on all scenes by comparing mask centroids
against the nearest-pose training view's label mask.

Example:
    python eval/stage_spinnerf_gt.py --scenes 2 4 10
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spinnerf_utils import (  # noqa: E402
    read_cameras_bin, read_images_bin_full, build_undistort_map, undistort_image,
)


def stage_scene(scene_root: Path, gt_dir: Path, gt_mask_dir: Path):
    img4_dir = scene_root / "images_4"
    with_object = {p.name for p in (img4_dir / "label").iterdir() if p.suffix.lower() == ".png"}
    all_images = sorted(p for p in img4_dir.iterdir()
                        if p.is_file() and p.suffix.lower() == ".png")
    no_object = [p.name for p in all_images if p.name not in with_object]
    test_labels = sorted(p.name for p in (img4_dir / "test_label").iterdir()
                         if p.suffix.lower() == ".png")
    assert len(test_labels) == len(no_object), (
        f"{scene_root}: {len(test_labels)} test_label masks != {len(no_object)} no-object images")

    images_full = read_images_bin_full(scene_root / "sparse" / "0" / "images.bin")
    name_to_camid = {img["name"]: img["camera_id"] for img in images_full.values()}
    ext = next(iter(name_to_camid)).rsplit(".", 1)[-1]
    cameras = read_cameras_bin(scene_root / "sparse" / "0" / "cameras.bin")
    with Image.open(all_images[0]) as im:
        w, h = im.size
    maps = {cid: build_undistort_map(cam, w, h)[:3] for cid, cam in cameras.items()}

    for d in (gt_dir, gt_mask_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for img_name, mask_name in zip(no_object, test_labels):
        stem = Path(img_name).stem
        map_x, map_y, roi = maps[name_to_camid[f"{stem}.{ext}"]]

        img = cv2.cvtColor(cv2.imread(str(img4_dir / img_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = undistort_image(img, map_x, map_y, roi, cv2.INTER_LINEAR)
        Image.fromarray(img).save(gt_dir / f"{stem}.{ext}", "JPEG", quality=95)

        mask = np.array(Image.open(img4_dir / "test_label" / mask_name))
        mask = (mask.astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)
        mask = undistort_image(mask, map_x, map_y, roi, cv2.INTER_NEAREST)
        Image.fromarray(mask, mode="L").save(gt_mask_dir / f"{stem}.png")

    print(f"{scene_root.name}: {len(no_object)} GT photos -> {gt_dir}, masks -> {gt_mask_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--data_root", type=Path, default=Path("data/spinnerf"))
    p.add_argument("--gt_root", type=Path, default=Path("data/ground_truth/images/spinnerf"))
    p.add_argument("--gt_mask_root", type=Path, default=Path("data/ground_truth/masks/spinnerf"))
    args = p.parse_args()
    for scene in args.scenes:
        stage_scene(args.data_root / scene, args.gt_root / scene, args.gt_mask_root / scene)


if __name__ == "__main__":
    main()
