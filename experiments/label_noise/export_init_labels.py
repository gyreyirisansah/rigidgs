#!/usr/bin/env python3
"""Export the TRUE initial per-Gaussian labels for a RigidGS scene.

RigidGS's fresh-training init (Scene.__init__ -> create_from_pcd, see
external/gaussian-splatting/scene/dataset_readers.py:fetchRigidGSPly) reads
per-point object labels from <source_path>/sparse/0/points3D_metadata.json,
which uses a {point_id, object_id, object_name, confidence} schema keyed by
list position (position i in point_labels == vertex i in points3D.ply).

flip_labels.py operates on the OTHER label schema used by the codebase --
object_metadata.json's flat integer object_id_array (0 = background,
nonzero = object), which is what GaussianModel.save_ply()/load_ply() read
and write for a Gaussian checkpoint.

This script bridges the two: it replays fetchRigidGSPly's exact name->id
assignment logic (background always id 0; other object names sorted
alphabetically from 1) and writes the result as an object_metadata.json
(object_id_array) file that is index-aligned with points3D.ply -- i.e.
identical to what gaussians._object_ids will be immediately after a fresh
create_from_pcd() init for this scene. That makes it a valid input to
flip_labels.py, and the corrupted output can be fed back in via the
label-noise trainer's --object_ids_override.
"""
import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Export true initial object_id_array for a RigidGS scene.")
    p.add_argument("--source_path", required=True,
                    help="Scene pcd_data dir, e.g. output/llff/fern/point_cloud/pcd_data")
    p.add_argument("--output_json", required=True)
    args = p.parse_args()

    metadata_path = os.path.join(args.source_path, "sparse", "0", "points3D_metadata.json")
    ply_path = os.path.join(args.source_path, "sparse", "0", "points3D.ply")

    with open(metadata_path) as f:
        metadata = json.load(f)
    point_labels = metadata["point_labels"]

    # Sanity-check alignment with the actual point cloud the trainer will load.
    if os.path.exists(ply_path):
        from plyfile import PlyData
        n_vertices = len(PlyData.read(ply_path).elements[0])
        assert n_vertices == len(point_labels), (
            f"points3D.ply has {n_vertices} vertices but points3D_metadata.json "
            f"has {len(point_labels)} point_labels -- these must be index-aligned "
            f"1:1 (see fetchRigidGSPly). Refusing to export a mismatched file."
        )
    else:
        print(f"WARNING: {ply_path} not found, skipping vertex-count cross-check")

    # Exactly replicates fetchRigidGSPly's id assignment
    # (external/gaussian-splatting/scene/dataset_readers.py).
    unique_names = sorted(set(p_["object_name"] for p_ in point_labels))
    unique_names = ["background"] + [n for n in unique_names if n != "background"]
    object_name_to_newid = {name: i for i, name in enumerate(unique_names)}

    object_id_array = np.array(
        [object_name_to_newid[p_["object_name"]] for p_ in point_labels],
        dtype=np.int64,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"object_id_array": object_id_array.tolist()}, f)

    uniq, counts = np.unique(object_id_array, return_counts=True)
    id_to_name = {v: k for k, v in object_name_to_newid.items()}
    print("=" * 60)
    print(f"source: {metadata_path}")
    print(f"output: {args.output_json}")
    print(f"object_name_to_newid: {object_name_to_newid}")
    for u, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  id {u} ({id_to_name.get(u, '?')}): {c}")
    print(f"total points: {len(object_id_array)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
