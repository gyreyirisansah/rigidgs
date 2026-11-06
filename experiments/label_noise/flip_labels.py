#!/usr/bin/env python3
"""Corrupt per-Gaussian object-id labels in a RigidGS object_metadata.json.

Standalone label-noise injection tool for the label-noise-robustness experiment.
Reads object_id_array (0 = background, nonzero = object), flips a per-class
fraction of labels according to a scheme, and writes a corrupted copy plus a
sidecar of the flipped indices.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


def load_object_id_array(input_json):
    with open(input_json) as f:
        data = json.load(f)
    arr = np.array(data["object_id_array"], dtype=np.int64)
    return data, arr


def pick_object_id(arr, background_id):
    nonzero_vals = arr[arr != background_id]
    if nonzero_vals.size == 0:
        raise ValueError("No foreground (nonzero) labels found in object_id_array; "
                          "cannot determine object id for bg2fg.")
    uniq, counts = np.unique(nonzero_vals, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    print("Nonzero object ids present:", dict(zip(uniq.tolist(), counts.tolist())))
    chosen = int(uniq[0])
    print(f"Using most frequent nonzero label as object id for bg2fg: {chosen}")
    return chosen


def flip_indices(rng, candidate_indices, rate):
    n_flip = round(rate * len(candidate_indices))
    if n_flip == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(candidate_indices, size=n_flip, replace=False)


def main():
    p = argparse.ArgumentParser(description="Flip a per-class fraction of RigidGS object labels.")
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--scheme", required=True, choices=["fg2bg", "bg2fg", "symmetric"])
    p.add_argument("--rate", type=float, required=True, help="Per-class fraction to flip, e.g. 0.10")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--background_id", type=int, default=0)
    args = p.parse_args()

    data, arr = load_object_id_array(args.input_json)
    bg_id = args.background_id

    orig_fg_idx = np.where(arr != bg_id)[0]
    orig_bg_idx = np.where(arr == bg_id)[0]
    n_fg_before, n_bg_before = len(orig_fg_idx), len(orig_bg_idx)

    rng = np.random.default_rng(args.seed)
    out_arr = arr.copy()

    fg2bg_flipped = np.array([], dtype=np.int64)
    bg2fg_flipped = np.array([], dtype=np.int64)

    if args.scheme in ("fg2bg", "symmetric"):
        fg2bg_flipped = flip_indices(rng, orig_fg_idx, args.rate)
        out_arr[fg2bg_flipped] = bg_id

    if args.scheme in ("bg2fg", "symmetric"):
        object_id = pick_object_id(arr, bg_id)
        bg2fg_flipped = flip_indices(rng, orig_bg_idx, args.rate)
        out_arr[bg2fg_flipped] = object_id

    injected_indices = np.concatenate([fg2bg_flipped, bg2fg_flipped]).astype(np.int64)
    injected_indices.sort()

    n_fg_after = int(np.sum(out_arr != bg_id))
    n_bg_after = int(np.sum(out_arr == bg_id))

    # --- sanity assertions ---
    if args.rate == 0.0:
        assert np.array_equal(out_arr, arr), "rate=0.0 must be a no-op on labels"
        assert injected_indices.size == 0, "rate=0.0 must inject zero indices"

    if args.scheme == "fg2bg":
        expected_n_flip = round(args.rate * n_fg_before)
        assert len(fg2bg_flipped) == expected_n_flip, \
            f"expected {expected_n_flip} fg2bg flips, got {len(fg2bg_flipped)}"
        assert len(bg2fg_flipped) == 0
        assert n_bg_after == n_bg_before + expected_n_flip
        assert n_fg_after == n_fg_before - expected_n_flip

    if args.scheme == "bg2fg":
        expected_n_flip = round(args.rate * n_bg_before)
        assert len(bg2fg_flipped) == expected_n_flip, \
            f"expected {expected_n_flip} bg2fg flips, got {len(bg2fg_flipped)}"
        assert len(fg2bg_flipped) == 0
        assert n_fg_after == n_fg_before + expected_n_flip
        assert n_bg_after == n_bg_before - expected_n_flip

    if args.scheme == "symmetric":
        expected_fg2bg = round(args.rate * n_fg_before)
        expected_bg2fg = round(args.rate * n_bg_before)
        assert len(fg2bg_flipped) == expected_fg2bg
        assert len(bg2fg_flipped) == expected_bg2fg

    # --- write outputs ---
    out_data = copy.deepcopy(data)
    out_data["object_id_array"] = out_arr.tolist()

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(out_data, f)

    sidecar_path = output_json_path.with_name(output_json_path.stem + "_injected_indices.npy")
    np.save(sidecar_path, injected_indices)

    # --- summary ---
    print("=" * 60)
    print(f"scheme={args.scheme} rate={args.rate} seed={args.seed} background_id={bg_id}")
    print(f"input:  {args.input_json}")
    print(f"output: {output_json_path}")
    print(f"sidecar: {sidecar_path}")
    print(f"original counts:  foreground={n_fg_before}  background={n_bg_before}")
    print(f"flipped fg->bg: {len(fg2bg_flipped)}   flipped bg->fg: {len(bg2fg_flipped)}")
    print(f"resulting counts: foreground={n_fg_after}  background={n_bg_after}")
    print(f"total injected (corrupted) indices: {len(injected_indices)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
