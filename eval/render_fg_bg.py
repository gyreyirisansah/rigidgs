#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import torch
sys.path.insert(0, 'external/gaussian-splatting')
sys.path.insert(0, 'src')
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _render_with_optional_override(view, gaussians, pipeline, background, train_test_exp,
                                   separate_sh, object_mask=None, override_color=None):
    """Small compatibility wrapper around the renderer.

    Some 3DGS forks support override_color, while others do not. For RGB renders
    we do not need it. For mask renders, we try to render selected Gaussians as
    white on a black background so the output becomes a clean silhouette.
    """
    render_kwargs = dict(
        use_trained_exp=train_test_exp,
        separate_sh=separate_sh,
        object_mask=object_mask,
    )

    if override_color is not None:
        render_kwargs["override_color"] = override_color

    try:
        return render(view, gaussians, pipeline, background, **render_kwargs)["render"]
    except TypeError:
        # Fallback for renderer versions that do not expose override_color.
        render_kwargs.pop("override_color", None)
        return render(view, gaussians, pipeline, background, **render_kwargs)["render"]


def _rgb_to_binary_mask(rendering, threshold=1e-4):
    """Convert an RGB object-only/background-only render to a single-channel mask."""
    if rendering.dim() == 3 and rendering.shape[0] >= 3:
        mask = rendering[0:3].abs().amax(dim=0, keepdim=True)
    elif rendering.dim() == 3:
        mask = rendering.abs().amax(dim=0, keepdim=True)
    else:
        mask = rendering.unsqueeze(0)

    return (mask > threshold).float()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, object_mask=None, suffix="",
               render_mask=False, mask_threshold=0.5):
    """
    Render a set of views with optional object masking.

    Args:
        object_mask: Boolean tensor indicating which Gaussians to render.
        suffix: String to append to output folder name, e.g., "_foreground".
        render_mask: If True, also save a binary rendered silhouette to mask_renders/.
    """
    folder_name = "ours_{}{}".format(iteration, suffix)
    render_path = os.path.join(model_path, name, folder_name, "renders")
    gts_path = os.path.join(model_path, name, folder_name, "gt")
    mask_path = os.path.join(model_path, name, folder_name, "mask_renders")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if render_mask:
        makedirs(mask_path, exist_ok=True)

    # For mask rendering, selected Gaussians are rendered as white on black if
    # the renderer supports override_color. If not, the fallback binarizes the
    # normal selected-Gaussian RGB render.
    mask_background = torch.zeros_like(background)
    white_override = None
    if object_mask is not None:
        white_override = torch.ones((int(object_mask.sum().item()), 3), device=background.device)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}{suffix}")):
        rendering = _render_with_optional_override(
            view, gaussians, pipeline, background,
            train_test_exp=train_test_exp,
            separate_sh=separate_sh,
            object_mask=object_mask,
            override_color=None,
        )
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if render_mask:
            if object_mask is None:
                # Full-scene mask: anything rendered is foreground in the silhouette.
                mask_rendering = rendering
            else:
                mask_rendering = _render_with_optional_override(
                    view, gaussians, pipeline, mask_background,
                    train_test_exp=False,
                    separate_sh=separate_sh,
                    object_mask=object_mask,
                    override_color=white_override,
                )

                if train_test_exp:
                    mask_rendering = mask_rendering[..., mask_rendering.shape[-1] // 2:]

            binary_mask = _rgb_to_binary_mask(mask_rendering, threshold=mask_threshold)
            torchvision.utils.save_image(binary_mask, os.path.join(mask_path, view.image_name + ".png"))


def render_per_object(model_path, name, iteration, views, gaussians, pipeline, background,
                      train_test_exp, separate_sh, object_id_to_name, suffix="",
                      mask_threshold=0.5):
    """
    Render + mask each foreground object in isolation: select only that
    object's own Gaussians (object_mask) and render exactly those, nothing
    else -- no other objects present for that pass, no full-scene compositing,
    no post-hoc cropping. This is the exact same object_mask-based isolation
    `render_set` already uses for the combined `_foreground`/`_background`
    outputs, just scoped to one object at a time instead of "all foreground".
    It directly tests what the paper claims (how well a per-Gaussian label
    segments each object on its own), with no extra processing layered on
    top, and stays consistent with how baseline comparison methods isolate
    Gaussians for their own per-object renders.

    Output layout, one subfolder per object under ours_{iteration}{suffix}:
        <name>/ours_{iteration}{suffix}/renders/<object_name>/<image>.png
        <name>/ours_{iteration}{suffix}/mask_renders/<object_name>/<image>.png
    """
    folder_name = "ours_{}{}".format(iteration, suffix)
    mask_background = torch.zeros_like(background)

    for obj_id, obj_name in sorted(object_id_to_name.items()):
        object_mask = gaussians._object_ids == obj_id
        render_path = os.path.join(model_path, name, folder_name, "renders", obj_name)
        mask_path = os.path.join(model_path, name, folder_name, "mask_renders", obj_name)
        makedirs(render_path, exist_ok=True)
        makedirs(mask_path, exist_ok=True)

        white_override = torch.ones((int(object_mask.sum().item()), 3), device=background.device)

        for view in tqdm(views, desc=f"Rendering {name}{suffix} {obj_name}"):
            rendering = _render_with_optional_override(
                view, gaussians, pipeline, background,
                train_test_exp=train_test_exp,
                separate_sh=separate_sh,
                object_mask=object_mask,
                override_color=None,
            )
            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

            mask_rendering = _render_with_optional_override(
                view, gaussians, pipeline, mask_background,
                train_test_exp=False,
                separate_sh=separate_sh,
                object_mask=object_mask,
                override_color=white_override,
            )
            if train_test_exp:
                mask_rendering = mask_rendering[..., mask_rendering.shape[-1] // 2:]
            binary_mask = _rgb_to_binary_mask(mask_rendering, threshold=mask_threshold)
            torchvision.utils.save_image(binary_mask, os.path.join(mask_path, view.image_name + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool,
                render_background_only: bool = False, per_object: bool = False):
    """
    Render train and test sets with optional background-only rendering.

    Args:
        render_background_only: If True, also render background-only views
        per_object: If True, also render + mask each foreground object
            individually (see `render_per_object`), in addition to the
            combined full-scene / foreground / background renders above.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Get background object ID if rendering background only
        bg_mask = None

        backgroundId = scene.getBackgroundObjectID()
        print(f"Background ID: {backgroundId}")
        bg_mask = gaussians._object_ids == backgroundId
        fg_mask = ~bg_mask
        print(f"Background ID: {backgroundId}, Background Gaussians: {bg_mask.sum().item()}/{len(bg_mask)}")

        object_id_to_name = None
        if per_object:
            object_id_to_name = {int(k): v for k, v in scene.object_id_to_name.items()
                                 if int(k) != backgroundId}
            print(f"Per-object rendering: {len(object_id_to_name)} objects -> "
                  f"{sorted(object_id_to_name.values())}")

        if not skip_train:
            # Render full scene
            render_set(dataset.model_path, "train", scene.loaded_iter, 
                      scene.getTrainCameras(), gaussians, pipeline, background, 
                      dataset.train_test_exp, separate_sh)
            
            # Optionally render background only
            if render_background_only:
                render_set(dataset.model_path, "train", scene.loaded_iter, 
                          scene.getTrainCameras(), gaussians, pipeline, background, 
                          dataset.train_test_exp, separate_sh, 
                          object_mask=bg_mask, suffix="_background",
                          render_mask=True)
                
            render_set(dataset.model_path, "train", scene.loaded_iter,
                      scene.getTrainCameras(), gaussians, pipeline, background,
                      dataset.train_test_exp, separate_sh,
                      object_mask=fg_mask, suffix="_foreground",
                      render_mask=True)

            if per_object:
                render_per_object(dataset.model_path, "train", scene.loaded_iter,
                                  scene.getTrainCameras(), gaussians, pipeline, background,
                                  dataset.train_test_exp, separate_sh, object_id_to_name,
                                  suffix="_objects")

        if not skip_test:
            # Render full scene
            print(f"Length of test cameras: {len(scene.getTestCameras())}")
            render_set(dataset.model_path, "test", scene.loaded_iter, 
                      scene.getTestCameras(), gaussians, pipeline, background, 
                      dataset.train_test_exp, separate_sh)
            
            # Optionally render background only
            if render_background_only:
                print(f"Length of inpainted test cameras: {len(scene.getTestInpaintedCameras())}")
                render_set(dataset.model_path, "test", scene.loaded_iter, 
                          scene.getTestInpaintedCameras(), gaussians, pipeline, background, 
                          dataset.train_test_exp, separate_sh,
                          object_mask=bg_mask, suffix="_background",
                          render_mask=True)
            
            render_set(dataset.model_path, "test", scene.loaded_iter,
                      scene.getTestInpaintedCameras(), gaussians, pipeline, background,
                      dataset.train_test_exp, separate_sh,
                      object_mask=fg_mask, suffix="_foreground",
                      render_mask=True)

            if per_object:
                # getTestInpaintedCameras() is empty for scenes without
                # inpainting-eval camera metadata (e.g. these LERF runs) --
                # same reason the _background/_foreground calls above fall
                # through to a second pass on getTestCameras() below.
                test_cams_for_objects = scene.getTestInpaintedCameras() or scene.getTestCameras()
                render_per_object(dataset.model_path, "test", scene.loaded_iter,
                                  test_cams_for_objects, gaussians, pipeline, background,
                                  dataset.train_test_exp, separate_sh, object_id_to_name,
                                  suffix="_objects")

            USETESTINPAINTED = True
            if USETESTINPAINTED:
                render_set(dataset.model_path, "test", scene.loaded_iter, 
                          scene.getTestCameras(), gaussians, pipeline, background, 
                          dataset.train_test_exp, separate_sh,
                          object_mask=bg_mask, suffix="_background",
                          render_mask=True)
            

                print(f"Length of inpainted test cameras: {len(scene.getTestCameras())}")
                render_set(dataset.model_path, "test", scene.loaded_iter, 
                          scene.getTestCameras(), gaussians, pipeline, background, 
                          dataset.train_test_exp, separate_sh,
                          object_mask=fg_mask, suffix="_foreground",
                          render_mask=True)
                


def render_sets_with_inpainted(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                                separate_sh: bool):
    """
    Render both original and inpainted camera sets for comparison.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        backgroundId = scene.getBackgroundObjectID()
        bg_mask = gaussians._object_ids == backgroundId
        fg_mask = ~bg_mask
        print(f"Background ID: {backgroundId}, Background Gaussians: {bg_mask.sum().item()}/{len(bg_mask)}")

        # Render original training views (full scene)
        print("\n=== Rendering Original Training Views (Full Scene) ===")
        render_set(dataset.model_path, "train_original", scene.loaded_iter, 
                  scene.getTrainCameras(), gaussians, pipeline, background, 
                  dataset.train_test_exp, separate_sh)

        # Render inpainted views (background only)
        print("\n=== Rendering Inpainted Training Views (Background Only) ===")
        inpainted_cams = scene.getInpaintedCameras()
        if inpainted_cams:
            render_set(dataset.model_path, "train_inpainted", scene.loaded_iter, 
                      inpainted_cams, gaussians, pipeline, background, 
                      dataset.train_test_exp, separate_sh,
                      object_mask=bg_mask, suffix="_background",
                      render_mask=True)
            render_set(dataset.model_path, "train_inpainted", scene.loaded_iter,
                      inpainted_cams, gaussians, pipeline, background,
                      dataset.train_test_exp, separate_sh,
                      object_mask=fg_mask, suffix="_foreground",
                      render_mask=True)
        else:
            print("No inpainted cameras found!")

        # Render test views (full scene)
        print("\n=== Rendering Test Views (Full Scene) ===")
        render_set(dataset.model_path, "test", scene.loaded_iter, 
                  scene.getTestCameras(), gaussians, pipeline, background, 
                  dataset.train_test_exp, separate_sh)
        
        # Render inpainted test views (background only)
        print("\n=== Rendering Inpainted Test Views (Background Only) ===")
        inpainted_test_cams = scene.getTestInpaintedCameras()
        if inpainted_test_cams:
            render_set(dataset.model_path, "test_inpainted", scene.loaded_iter, 
                      inpainted_test_cams, gaussians, pipeline, background, 
                      dataset.train_test_exp, separate_sh,
                      object_mask=bg_mask, suffix="_background",
                      render_mask=True)
            render_set(dataset.model_path, "test_inpainted", scene.loaded_iter,
                      inpainted_test_cams, gaussians, pipeline, background,
                      dataset.train_test_exp, separate_sh,
                      object_mask=fg_mask, suffix="_foreground",
                      render_mask=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_background_only", action="store_false", 
                       help="Also render background-only views")
    parser.add_argument("--render_with_inpainted", action="store_true",
                       help="Render both original and inpainted camera sets")
    parser.add_argument("--per_object", action="store_true",
                       help="Also render + mask each foreground object individually "
                            "(one subfolder per object name), in addition to the "
                            "combined full-scene/foreground/background renders")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.render_with_inpainted:
        render_sets_with_inpainted(model.extract(args), args.iteration,
                                   pipeline.extract(args), SPARSE_ADAM_AVAILABLE)
    else:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                   args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,
                   args.render_background_only, args.per_object)