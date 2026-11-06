#!/usr/bin/env bash
# Clone the third-party code RigidGS builds on, at the exact commits used for
# the paper, copy our modified files over it (external_overrides/), build the
# CUDA extensions, and download the model weights.
#
# Run once from the repository root, inside the activated conda environment:
#     bash scripts/setup_external.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
OVR="$ROOT/external_overrides"
mkdir -p "$EXT"

clone_at() {   # clone_at <url> <dir> <commit>
    local url="$1" dir="$2" commit="$3"
    if [ ! -d "$dir/.git" ]; then
        git clone "$url" "$dir"
    fi
    git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$commit"
}

overlay() {    # overlay <override-subdir> <target-dir>
    cp -rv "$OVR/$1/." "$2/"
}

# ---- 3D Gaussian Splatting (graphdeco) + its CUDA submodules -----------
GS="$EXT/gaussian-splatting"
clone_at https://github.com/graphdeco-inria/gaussian-splatting.git "$GS" \
    54c035f7834b564019656c3e3fcc3646292f727d
clone_at https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
    "$GS/submodules/diff-gaussian-rasterization" 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
git -C "$GS/submodules/diff-gaussian-rasterization" submodule update --init --recursive
clone_at https://gitlab.inria.fr/bkerbl/simple-knn.git \
    "$GS/submodules/simple-knn" 86710c2d4b46680c02301765dd79e465819c8f19
clone_at https://github.com/rahul-goel/fused-ssim.git \
    "$GS/submodules/fused-ssim" 1272e21a282342e89537159e4bad508b19b34157
overlay gaussian-splatting "$GS"
overlay diff-gaussian-rasterization "$GS/submodules/diff-gaussian-rasterization"
pip install --no-build-isolation \
    "$GS/submodules/diff-gaussian-rasterization" \
    "$GS/submodules/simple-knn" \
    "$GS/submodules/fused-ssim"

# ---- SAM 2 ---------------------------------------------------------------
SAM2="$EXT/sam2"
clone_at https://github.com/facebookresearch/sam2.git "$SAM2" \
    2b90b9f5ceec907a1c18123530e92e794ad901a4
overlay sam2 "$SAM2"
pip install --no-build-isolation -e "$SAM2"
mkdir -p "$SAM2/checkpoints"
[ -f "$SAM2/checkpoints/sam2.1_hiera_large.pt" ] || wget -O "$SAM2/checkpoints/sam2.1_hiera_large.pt" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# ---- LaMa (inpainting) ---------------------------------------------------
LAMA="$EXT/lama"
clone_at https://github.com/advimman/lama.git "$LAMA" \
    786f5936b27fb3dacd2b1ad799e4de968ea697e7
overlay lama "$LAMA"
if [ ! -f "$LAMA/big-lama/models/best.ckpt" ]; then
    wget -O "$LAMA/big-lama.zip" \
        https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
    unzip -o "$LAMA/big-lama.zip" -d "$LAMA"
    rm "$LAMA/big-lama.zip"
fi

# ---- Depth Anything V2 (mask refinement) ---------------------------------
DA="$EXT/Depth-Anything-V2"
clone_at https://github.com/DepthAnything/Depth-Anything-V2.git "$DA" \
    a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
mkdir -p "$DA/checkpoints"
[ -f "$DA/checkpoints/depth_anything_v2_vitl.pth" ] || wget -O "$DA/checkpoints/depth_anything_v2_vitl.pth" \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

echo
echo "External dependencies ready in $EXT"
