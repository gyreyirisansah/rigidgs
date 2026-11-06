"""
RigidGS trainer.

Input is preprocessing.py's output: -s <scene>/point_cloud/pcd_data (COLMAP
poses + a labeled point cloud), --_images (the photos) and --masks_dir (the
2D object masks). Run from the repository root.

Stage 1  (iter 1 .. --inpaint_at_iter)
    3DGS photometric loss (L1 + D-SSIM)
    + 2D mask loss: rendered object probability vs. the 2D masks (BCE),
      --mask_2d_weight; composite foreground/background by default,
      per object with --mask_2d_per_object (multi-object scenes)
    + 3D boundary regularizer on Gaussians whose kNN neighbourhood has a
      different label: flatten, shrink and repulsion terms,
      --boundary_reg_weight, every --boundary_reg_interval iterations
    Labels come from the point cloud and are committed at --logit_freeze_iter
    (default: densify_from_iter).
Stage 2  (at --inpaint_at_iter): SAM2 hole detection on background-only
    renders (one interactive click) and LaMa inpainting.
    The state is checkpointed to <model_path>/chkpnt<inpaint_at_iter>.pth,
    so Stage 3 can be rerun with --start_checkpoint ... --skip_reinpaint.
Stage 3  (--refine_after_inpaint, --refine_iterations): background
    refinement (stage3_refinement.py). Background Gaussians are seeded in
    the hole, then a full-scene photometric step on the original images
    alternates with a background-only L1 step on the inpainted images
    (foreground frozen), densifying background Gaussians near the hole.

Writes to --model_path: point_cloud/iteration_*/, foreground_renders/,
background_hole_renders/, online_inpainting/, training_cost.json
(wall-clock, CPU time and peak GPU memory per stage) and TensorBoard logs.

Experiment flags:
    --learnable_logits                ablation: labels keep training through
                                      densification (with a later --logit_freeze_iter)
    --object_ids_override, --enable_mechanism_trace, --true_labels_path,
    --trace_csv                       label-noise study (experiments/label_noise)
"""

import os
import sys
import json
import shutil
import time
import resource
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from pathlib import Path

sys.path.insert(0, 'external/gaussian-splatting')

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser, Namespace
from rigidgs_arguments import ModelParams, PipelineParams, OptimizationParams, TrainingParams

# === CLEAN IMPORTS ===
import csv
from tracked_gaussian_model import TrackedGaussianModel
from boundary_regularization import (
    boundary_regularization_loss,
    postprocess_cleanup,
)
from mask_loss import (MaskLoader, render_id_mask, mask_loss_2d, mask_loss_2d_v2,
                                render_object_prob_masks, mask_loss_per_object)
from stage3_refinement import run_stage3_refinement, save_background_hole_renders

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from online_inpainting import (
        OnlineInpainter, batch_inpaint_all_cameras, get_last_inpainting_timings)
    INPAINTING_AVAILABLE = True
except ImportError:
    INPAINTING_AVAILABLE = False


# =====================================================================
# Cost-tracking helpers
# =====================================================================
def _cpu_time_sec() -> float:
    """User+sys CPU seconds consumed by this process (all threads) since
    process start -- cumulative and diffable, unlike RSS below, so it can
    be broken down per pipeline stage by taking snapshots at each boundary."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _peak_rss_gb() -> float:
    """Peak resident set size since process start (Linux: ru_maxrss is KB).
    Monotonic non-decreasing and NOT resettable mid-process (unlike torch's
    CUDA peak trackers), so this is only meaningful as a single overall
    figure, not something to diff between stage boundaries."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


# =====================================================================
# Helpers
# =====================================================================
def prepare_output_and_logger(dataset):
    if not dataset.model_path:
        raise ValueError("--model_path (-m) is required: the run's output folder.")
    print(f"Output folder: {dataset.model_path}")
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(dataset))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    return tb_writer


def cleanup_before_inpainting(scene, gaussians, tb_writer=None):
    print("[CLEANUP] Preparing for inpainting — freeing training state...")
    torch.set_grad_enabled(False)
    for attr in ("optimizer", "exposure_optimizer"):
        if hasattr(gaussians, attr) and getattr(gaussians, attr) is not None:
            try:
                getattr(gaussians, attr).zero_grad(set_to_none=True)
            except Exception:
                pass
            setattr(gaussians, attr, None)
            print(f"[CLEANUP] deleted gaussians.{attr}")
    training_only_attrs = [
        "xyz_gradient_accum", "denom", "max_radii2D",
        "tmp_radii", "tmp_visibility", "optimizer_state",
        "densification_stats", "opacity_accum", "grad_accum",
        "means2D_gradient_accum", "radii2D_max",
    ]
    for a in training_only_attrs:
        if hasattr(gaussians, a):
            try:
                delattr(gaussians, a)
                print(f"[CLEANUP] deleted gaussians.{a}")
            except Exception:
                pass
    try:
        for cam in scene.getTrainCameras():
            for tensor_attr in ("original_image", "alpha_mask", "invdepthmap", "depth_mask"):
                if hasattr(cam, tensor_attr) and torch.is_tensor(getattr(cam, tensor_attr)):
                    setattr(cam, tensor_attr, getattr(cam, tensor_attr).cpu())
    except Exception:
        pass
    if tb_writer is not None:
        try:
            tb_writer.close()
            print("[CLEANUP] TensorBoard writer closed")
        except Exception:
            pass
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    print(f"[CLEANUP] Done. GPU: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def _get_hard_labels_for_mask(gaussians):
    """Use LEARNED labels if available, else fall back to frozen _object_ids."""
    assignment = getattr(gaussians, "obj_assignment", None)
    if (assignment is not None
            and getattr(assignment, "_object_logits", None) is not None):
        return assignment.get_object_ids()
    return gaussians._object_ids


def training_report(tb_writer, iteration, Ll1, loss_full, loss_bg,
                    l_boundary, l_mask, elapsed, testing_iterations,
                    scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss/l1_loss',      Ll1.item(),       iteration)
        tb_writer.add_scalar('train_loss/total_full',   loss_full.item(), iteration)
        tb_writer.add_scalar('train_loss/total_bg',     loss_bg,          iteration)
        tb_writer.add_scalar('train_loss/boundary_reg', l_boundary,       iteration)
        tb_writer.add_scalar('train_loss/mask_2d',      l_mask,           iteration)
        tb_writer.add_scalar('iter_time',               elapsed,          iteration)
        # Running peak, not current usage -- survives empty_cache() calls,
        # so this is a true peak-over-time curve, not a snapshot.
        tb_writer.add_scalar('gpu/peak_memory_allocated_gb',
                             torch.cuda.max_memory_allocated() / 1024**3, iteration)
        tb_writer.add_scalar('gpu/peak_memory_reserved_gb',
                             torch.cuda.max_memory_reserved() / 1024**3, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        configs = [
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                            for idx in range(5, 30, 5)]},
        ]
        for cfg in configs:
            if not cfg['cameras']:
                continue
            l1_test = psnr_test = 0.0
            for vp in cfg['cameras']:
                img = torch.clamp(renderFunc(vp, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt = torch.clamp(vp.original_image.to("cuda"), 0.0, 1.0)
                if train_test_exp:
                    img = img[..., img.shape[-1] // 2:]
                    gt  = gt[..., gt.shape[-1]  // 2:]
                l1_test   += l1_loss(img, gt).mean().double()
                psnr_test += psnr(img, gt).mean().double()
            n = len(cfg['cameras'])
            l1_test /= n; psnr_test /= n
            print(f"\n[ITER {iteration}] {cfg['name']}: L1={l1_test:.5f}  PSNR={psnr_test:.2f}")
            if tb_writer:
                tb_writer.add_scalar(f"{cfg['name']}/l1",   l1_test,   iteration)
                tb_writer.add_scalar(f"{cfg['name']}/psnr", psnr_test, iteration)
        torch.cuda.empty_cache()


def save_foreground_renders(gaussians, background_id, cameras, render_func,
                            pipe, background, output_dir, iteration):
    """Renders foreground using LEARNED labels (not frozen COLMAP labels)."""
    labels = _get_hard_labels_for_mask(gaussians)
    bg_mask = labels == background_id
    fg_mask = ~bg_mask
    out = os.path.join(output_dir, f"iter_{iteration:05d}")
    os.makedirs(out, exist_ok=True)
    for cam in tqdm(cameras, desc="FG renders"):
        with torch.no_grad():
            pkg = render_func(cam, gaussians, pipe, background,
                              object_mask=fg_mask, use_trained_exp=False,
                              separate_sh=False)
            img = pkg["render"].permute(1, 2, 0).cpu().numpy()
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            name = cam.image_name.split(".")[0]
            Image.fromarray(img).save(os.path.join(out, f"{name}.png"))
    print(f"  Saved FG renders → {out}")


def _save_checkpoint_state(gaussians, scene, backgroundId, render_func, pipe,
                            background, foreground_save_dir, bg_hole_save_dir,
                            cameras, iteration, label):
    """Save .ply + FG renders + BG hole renders at one checkpoint."""
    print(f"\n[{label}] Saving Gaussians + FG + BG renders at iter {iteration}")
    scene.save(iteration)
    save_foreground_renders(gaussians, backgroundId, cameras, render_func,
                             pipe, background, foreground_save_dir, iteration)
    save_background_hole_renders(gaussians, backgroundId, cameras, render_func,
                                  pipe, background, bg_hole_save_dir, iteration)


# =====================================================================
# Main training loop
# =====================================================================
def training(dataset, opt, pipe, training_params, testing_iterations,
             saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, extra_args=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("sparse_adam not installed")

    # ---- Training-cost tracking, broken down per pipeline stage:
    # Stage 1 (ordinary 3DGS training, iter 1 -> inpaint_at_iter) vs.
    # Stage 2/3 (inpainting + BG refinement, triggered once at
    # inpaint_at_iter). GPU peak is reset HERE (Stage 1 start) and again
    # at the Stage 1->2 boundary below, so each stage's peak is isolated
    # instead of one number for the whole pipeline. Wall-clock and CPU
    # time are cumulative counters (not resettable) diffed at each
    # boundary instead. empty_cache() calls elsewhere (e.g. cleanup_
    # before_inpainting) don't affect any of this -- they return free
    # cached blocks to the driver but never touch the peak-tracking stat.
    torch.cuda.reset_peak_memory_stats()
    _training_wall_start = time.time()
    _cpu_time_at_stage1_start = _cpu_time_sec()

    # ---- Per-phase timing (new). Populated at the Stage 2/3 call sites
    # below (which each run at most once, at --inpaint_at_iter, then the
    # loop `break`s) and folded into the same training_cost.json written
    # at the true end of training() further down.
    _phase_timings = {}
    _stage1_summary = None    # filled in at the Stage 1->2 boundary below
    _stage23_summary = None   # filled in once Stage 2/3 finish (or Stage 2 alone)
    _stage23_wall_start = None
    _cpu_time_at_stage23_start = None

    first_iter = 1
    tb_writer = prepare_output_and_logger(dataset)
    _trace_enabled = bool(extra_args is not None and getattr(extra_args, "enable_mechanism_trace", False))
    _GaussianModelClass = TrackedGaussianModel if _trace_enabled else GaussianModel
    gaussians = _GaussianModelClass(dataset.sh_degree, opt.optimizer_type)
    gaussians.learnable_logits = bool(getattr(extra_args, 'learnable_logits', False)) if extra_args is not None else False
    scene     = Scene(dataset, gaussians)
    print("source_path:", dataset.source_path)
    print("num train cams:", len(scene.getTrainCameras()))
    print("num test cams:",  len(scene.getTestCameras()))
    print("first train image name:", scene.getTrainCameras()[0].image_name)
    print("first train image shape:", tuple(scene.getTrainCameras()[0].original_image.shape))
    print("white_background:", dataset.white_background)
    print("train_test_exp:",   dataset.train_test_exp)

    cam = scene.getTrainCameras()[0]
    gt = cam.original_image
    print(gt.min().item(), gt.max().item(), gt.mean().item(), gt.std().item())

    backgroundId = scene.getBackgroundObjectID()

    # LABEL-NOISE EXPERIMENT: override frozen initial labels
    # ================================================================
    # Must run before training_setup() / before any densification, so the
    # corrupted labels are exactly what gets frozen at LOGIT_FREEZE_ITER —
    # identical treatment to normal (uncorrupted) COLMAP-init labels.
    override_path = getattr(extra_args, "object_ids_override", None) if extra_args else None
    if override_path:
        with open(override_path, "r") as f:
            override_meta = json.load(f)
        override_ids = torch.tensor(
            override_meta["object_id_array"], dtype=torch.long, device="cuda")
        assert override_ids.shape[0] == gaussians._object_ids.shape[0], (
            f"[LabelNoise] override length {override_ids.shape[0]} != "
            f"init gaussian count {gaussians._object_ids.shape[0]} — "
            f"wrong scene or stale override file: {override_path}")
        n_changed = int((override_ids != gaussians._object_ids).sum().item())
        gaussians._object_ids = override_ids
        if getattr(gaussians, "obj_assignment", None) is not None:
            gaussians.obj_assignment.initialize_from_ids(gaussians._object_ids)
        n_fg = int((gaussians._object_ids != backgroundId).sum().item())
        n_bg = int((gaussians._object_ids == backgroundId).sum().item())
        print(f"\n[LabelNoise] Loaded corrupted labels from {override_path}")
        print(f"[LabelNoise] {n_changed} labels differ from true COLMAP init")
        print(f"[LabelNoise] At init: foreground={n_fg}  background={n_bg}\n")

    # ================================================================
    # MECHANISM TRACE: track surviving-corrupted-Gaussian fate over training
    # ================================================================
    # Needs the TRUE (uncorrupted) label per Gaussian to detect mismatch —
    # separate from --object_ids_override, which is the (possibly corrupted)
    # label actually frozen and trained on.
    trace_writer = trace_file = None
    trace_interval = int(getattr(extra_args, "trace_interval", 100)) if extra_args else 100
    MECHANISM_TRACE_OPACITY_THRESHOLD = 0.005  # same fixed cutoff as densify_and_prune's min_opacity below
    if _trace_enabled:
        true_labels_path = getattr(extra_args, "true_labels_path", None)
        trace_csv_path = getattr(extra_args, "trace_csv", None)
        assert true_labels_path, "--enable_mechanism_trace requires --true_labels_path"
        assert trace_csv_path, "--enable_mechanism_trace requires --trace_csv"
        with open(true_labels_path, "r") as f:
            true_meta = json.load(f)
        true_ids = torch.tensor(true_meta["object_id_array"], dtype=torch.long, device="cuda")
        gaussians.set_true_label_ids(true_ids)

        os.makedirs(os.path.dirname(os.path.abspath(trace_csv_path)) or ".", exist_ok=True)
        _GROUP_STAT_KEYS = ["n", "opacity_mean", "opacity_median", "frac_suppressed",
                            "opacity_p10", "opacity_p25", "opacity_p50", "opacity_p75", "opacity_p90"]
        _TRACE_FIELDS = ["iteration", "n_total", "n_mismatch"]
        for group in ("match", "mismatch", "fg2bg_like", "bg2fg_like"):
            _TRACE_FIELDS += [f"{group}_{k}" for k in _GROUP_STAT_KEYS]
        _TRACE_FIELDS += [
            "frac_mismatch_self_corrected_by_logits",
            "n_cloned", "n_split", "n_pruned",
        ]
        trace_file = open(trace_csv_path, "w", newline="")
        trace_writer = csv.DictWriter(trace_file, fieldnames=_TRACE_FIELDS)
        trace_writer.writeheader()
        print(f"[MechanismTrace] logging every {trace_interval} iters -> {trace_csv_path} "
              f"(fields: {len(_TRACE_FIELDS)}, incl. opacity percentiles per group)")

    def _log_trace_row(iteration, densify_stats=None):
        if trace_writer is None:
            return
        effective_labels = _get_hard_labels_for_mask(gaussians)
        snap = gaussians.get_tracking_snapshot(
            backgroundId, MECHANISM_TRACE_OPACITY_THRESHOLD, effective_labels=effective_labels)
        if snap is None:
            return
        row = {
            "iteration": iteration, "n_total": snap["n_total"], "n_mismatch": snap["n_mismatch"],
            "frac_mismatch_self_corrected_by_logits": snap["frac_mismatch_self_corrected_by_logits"],
        }
        for key in ("match", "mismatch", "fg2bg_like", "bg2fg_like"):
            s = snap[key]
            for stat_key in ("n", "opacity_mean", "opacity_median", "frac_suppressed",
                             "opacity_p10", "opacity_p25", "opacity_p50", "opacity_p75", "opacity_p90"):
                row[f"{key}_{stat_key}"] = s[stat_key]
        d = densify_stats or {}
        row["n_cloned"] = d.get("n_cloned", "")
        row["n_split"] = d.get("n_split", "")
        row["n_pruned"] = d.get("n_pruned", "")
        trace_writer.writerow(row)
        trace_file.flush()

    if _trace_enabled:
        _log_trace_row(0)  # starting state, right after corruption injection

    gaussians.training_setup(opt)

    if checkpoint:
        # weights_only=False: this checkpoint is gaussians.capture() + a plain
        # int (iteration), not just tensors -- PyTorch >=2.6 defaults to
        # weights_only=True and refuses to unpickle it otherwise. Safe here
        # since it's our own locally-generated checkpoint, not third-party.
        model_params, first_iter = torch.load(checkpoint, weights_only=False)

        # Work around a gap in GaussianModel.capture()/restore() (external/
        # gaussian-splatting/scene/gaussian_model.py, unmodified vendor
        # logic): that tuple never includes obj_assignment/_object_logits
        # state at all -- it was bolted on outside it. If logits were
        # already frozen by LOGIT_FREEZE_ITER (default iter 500) before this
        # checkpoint was saved -- true for any checkpoint saved at the
        # Stage 2->3 boundary, e.g. inpaint_at_iter=30000 -- the LIVE
        # optimizer's "object_logits" param group had its 'params' list
        # emptied in place (see the freeze block below) before capture()
        # serialized it, so the SAVED opt_dict has that group present but
        # empty. On restore, gaussians.restore() calls training_setup()
        # internally, which unconditionally re-adds a FRESH, non-empty
        # "object_logits" group (obj_assignment._object_logits still holds
        # whatever Scene() just constructed it as, not None) -- a structural
        # mismatch against the saved state that optimizer.load_state_dict()
        # (also called inside restore(), so not interceptable there) rejects
        # outright. Stage 3 doesn't need live logits anyway (frozen since
        # iter 500 either way) -- it reads gaussians._object_ids, the
        # committed hard labels, which the capture()/restore() tuple DOES
        # correctly include. So: strip the empty group from the saved state
        # and pre-set _object_logits=None so training_setup() skips
        # re-adding it too -- both sides then agree (6 groups, no
        # object_logits), matching what was actually true at save time.
        opt_dict = model_params[10]
        opt_dict["param_groups"] = [
            g for g in opt_dict["param_groups"] if g.get("name") != "object_logits"
        ]
        gaussians.obj_assignment._object_logits = None

        gaussians.restore(model_params, opt)

    # Sanity check: is the learnable-logits path live?
    has_logits = (getattr(gaussians, "obj_assignment", None) is not None
                  and getattr(gaussians.obj_assignment, "_object_logits", None) is not None)
    print(f"\n[Trainer] Learnable object_logits live: {has_logits}")
    if has_logits:
        K = gaussians.obj_assignment._object_logits.shape[1]
        print(f"[Trainer] num_classes K = {K} (background_id={backgroundId})")
    else:
        print("[Trainer] WARNING: obj_assignment._object_logits is None — "
              "mask loss will fall back to frozen labels (no gradient to "
              "labels). Check that obj_assignment.initialize_from_ids(...) "
              "is called during create_from_pcd.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- Resume-aware training-cost tracking: a --start_checkpoint resume
    # is a NEW process, so the reset_peak_memory_stats()/_training_wall_start
    # above only capture THIS session's cost -- if the earlier process that
    # actually produced the checkpoint (e.g. the Stage 1 run that saved
    # chkpnt30000.pth) already wrote its own training_cost.json into this
    # same model_path, that Stage 1 cost would otherwise be silently
    # dropped when this session resumes straight into Stage 2/3 (which
    # does zero real Stage 1 iterations of its own -- see the loop's
    # inpaint_at_iter trigger below). Fold it into _prior_stage1 here;
    # merged with this session's own (near-zero, in the resume case)
    # Stage 1 numbers at the Stage 1->2 boundary below.
    # Same idea applies if this session instead resumes specifically to
    # continue/redo Stage 2 (inpainting) or Stage 3 (refinement) -- e.g.
    # a prior attempt got killed mid-refinement and this run picks back up
    # from chkpnt<inpaint_at_iter>.pth (optionally with --skip_reinpaint to
    # reuse an already-completed Stage 2). _prior_stage23 folds that prior
    # attempt's cost the same additive way as Stage 1 above; phase_timings
    # from it are pre-seeded into _phase_timings so a phase this session
    # skips (e.g. hole_click_sec/inpainting_sec under --skip_reinpaint)
    # still carries its real historical value forward instead of vanishing.
    _prior_cost_path = os.path.join(dataset.model_path, "training_cost.json")
    _prior_stage1 = {}
    _prior_stage23 = {}
    _original_first_iter = first_iter
    if os.path.exists(_prior_cost_path):
        try:
            with open(_prior_cost_path) as f:
                _prior_cost = json.load(f)
            if "stage1_training" in _prior_cost:
                _prior_stage1 = _prior_cost.get("stage1_training") or {}
                _prior_stage23 = _prior_cost.get("stage2_3_refinement") or {}
                _phase_timings.update(_prior_stage23.get("phase_timings_sec", {}))
                _original_first_iter = _prior_cost.get("overall", {}).get(
                    "original_first_iter", first_iter)
            else:
                # Old flat schema (pre-dates the stage1/stage2/stage3
                # breakdown) -- best-effort: the whole thing was written at
                # a Stage-1-only checkpoint boundary, so fold it in as
                # Stage 1's prior cost. (No stage2_3 data to recover from
                # this schema.)
                _prior_stage1 = {
                    "wall_clock_sec": _prior_cost.get("total_wall_clock_sec", 0.0),
                    "peak_gpu_memory_allocated_gb": _prior_cost.get("peak_memory_allocated_gb", 0.0),
                    "peak_gpu_memory_reserved_gb": _prior_cost.get("peak_memory_reserved_gb", 0.0),
                    "cpu_time_sec": 0.0,
                }
                _original_first_iter = _prior_cost.get("original_first_iter",
                                                         _prior_cost.get("first_iter", first_iter))
            print(f"[TRAINING COST] found prior training_cost.json at {_prior_cost_path} "
                  f"(Stage 1: {_prior_stage1.get('wall_clock_sec', 0.0)/60:.2f} min, "
                  f"{_prior_stage1.get('peak_gpu_memory_allocated_gb', 0.0):.2f} GB peak; "
                  f"Stage 2/3: {_prior_stage23.get('wall_clock_sec', 0.0)/60:.2f} min) "
                  f"-- folding into this session's totals.")
        except Exception as e:
            print(f"[TRAINING COST] WARNING: failed to read prior {_prior_cost_path}: {e} "
                  f"-- starting cost tracking fresh for this session.")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = (opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE)
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # ================================================================
    # Label distribution (use LEARNED labels)
    # ================================================================
    labels_for_print = _get_hard_labels_for_mask(gaussians)
    unique_gt, gt_counts = torch.unique(labels_for_print, return_counts=True)
    print(f"\n[Training] Label distribution (current — learned if logits live):")
    for u, c in zip(unique_gt, gt_counts):
        print(f"  Class {int(u)}: {int(c):,} ({100*c/len(labels_for_print):.1f}%)")
    print(f"  Background ID: {backgroundId}")

    # ================================================================
    # Config helper
    # ================================================================
    def _get(name, default):
        if extra_args is not None and hasattr(extra_args, name):
            return getattr(extra_args, name)
        return getattr(training_params, name, default)

    # ================================================================
    # Boundary regularization config
    # ================================================================
    boundary_reg_weight   = _get('boundary_reg_weight',     0.5)
    boundary_reg_interval = _get('boundary_reg_interval',   10)
    boundary_reg_start    = _get('boundary_reg_start',      500)
    boundary_k            = _get('boundary_k',              16)
    boundary_flatten_w    = _get('boundary_flatten_weight', 1.0)
    boundary_shrink_w     = _get('boundary_shrink_weight',  0.5)
    boundary_repulsion_w  = _get('boundary_repulsion_weight', 2.0)
    boundary_repulsion_m  = _get('boundary_repulsion_margin', 0.005)
    boundary_min_disagree = _get('boundary_min_disagreement', 2)
    boundary_sample_size  = _get('boundary_sample_size',     50000)
    ambig_scale_weight    = _get('ambig_scale_weight',       0.0)
    ambig_scale_power     = _get('ambig_scale_power',        1.0)

    print(f"\n[Boundary Reg] weight={boundary_reg_weight}, "
          f"interval={boundary_reg_interval}, start={boundary_reg_start}")
    print(f"[Boundary Reg] flatten={boundary_flatten_w}, shrink={boundary_shrink_w}, "
          f"repulsion={boundary_repulsion_w}, ambig_scale={ambig_scale_weight}")

    # ================================================================
    # Logit freeze config
    # ================================================================
    logit_freeze_iter = _get('logit_freeze_iter', opt.densify_from_iter)
    if logit_freeze_iter is None:
        logit_freeze_iter = opt.densify_from_iter
    LOGIT_FREEZE_ITER = int(logit_freeze_iter)
    print(f"\n[Logit Freeze] Will commit labels and freeze logits at iter {LOGIT_FREEZE_ITER}")

    # ================================================================
    # Cleanup config
    # ================================================================
    cleanup_at_saving                = _get('cleanup_at_saving',                True)
    cleanup_floater_k                = _get('cleanup_floater_k',                8)
    cleanup_floater_distance_ratio   = _get('cleanup_floater_distance_ratio',   4.0)
    cleanup_floater_max_opacity      = _get('cleanup_floater_max_opacity',      0.5)
    cleanup_relabel_k                = _get('cleanup_relabel_k',                16)
    cleanup_isolation_threshold      = _get('cleanup_isolation_threshold',      0.8)
    cleanup_also_flip_bg_to_fg       = _get('cleanup_also_flip_bg_to_fg',       True)
    cleanup_logit_flip_confidence    = _get('cleanup_logit_flip_confidence',    5.0)
    if cleanup_at_saving:
        print(f"[Cleanup] enabled at the final save of each stage "
              f"(per-object floater prune, labels unchanged)")

    # ================================================================
    # Stage 3 refinement config
    # ================================================================
    refine_after_inpaint = _get('refine_after_inpaint',     True)
    refine_iterations    = _get('refine_iterations',        7000)
    skip_reinpaint       = _get('skip_reinpaint',           False)

    if refine_after_inpaint:
        print(f"\n[Stage 3] enabled — {refine_iterations} iterations after inpainting: "
              "hole seeding, then alternating full-scene photometric loss on the "
              "original images and BG-only L1 on the inpainted images, with "
              "densification near the hole.")
    else:
        print(f"\n[Stage 3] disabled (single-pass mode)")

    # ================================================================
    # 2D Mask Loss config
    # ================================================================
    mask_2d_weight    = _get('mask_2d_weight',    0.5)
    mask_2d_start     = _get('mask_2d_start',     500)
    mask_2d_interval  = _get('mask_2d_interval',  1)
    masks_dir         = _get('masks_dir',         '')
    mask_2d_bg_margin = _get('mask_2d_bg_margin', 15)
    mask_2d_per_object = _get('mask_2d_per_object', False)

    mask_loader = None
    if mask_2d_weight > 0 and masks_dir:
        mask_loader = MaskLoader(
            masks_dir=masks_dir,
            object_name_to_id_fn=scene.get_name_to_id_mapping,
            extension="png")
        print(f"\n[2D Mask] weight={mask_2d_weight}, start={mask_2d_start}, "
              f"interval={mask_2d_interval}")
    elif mask_2d_weight > 0:
        print(f"\n[2D Mask] WARNING: weight set but no --masks_dir. Disabling.")
        mask_2d_weight = 0.0

    # ================================================================
    # Inpainting config
    # ================================================================
    inpaint_at_iter   = _get('inpaint_at_iter',   0)
    lama_model_path   = _get('lama_model_path',   '')
    sam2_checkpoint   = _get('sam2_checkpoint',   '')
    sam2_config       = _get('sam2_config',       '')
    inpaint_cache_dir = _get('inpaint_cache_dir', '')
    mask_dilation_pixels = _get('mask_dilation_pixels', 20)
    if not inpaint_cache_dir:
        # Resolved once, here, because Stage 3 reads the hole masks LaMa was
        # given from under this directory.
        inpaint_cache_dir = os.path.join(dataset.model_path, "online_inpainting", "inpaint_cache")

    if inpaint_at_iter > 0:
        if not INPAINTING_AVAILABLE:
            print("\n[Inpaint] WARNING: online_inpainting not found — disabled.")
            inpaint_at_iter = 0
        else:
            print(f"\n[Inpaint] Scheduled at iter {inpaint_at_iter}.")
            if refine_after_inpaint:
                print(f"[Inpaint] After inpainting: Stage 3 ({refine_iterations} iters) + Stage 4 cleanup.")
            else:
                print(f"[Inpaint] After inpainting: trainer will EXIT (no refinement).")

    scene_output = dataset.model_path
    inpainted_output_dir = os.path.join(scene_output, "online_inpainting", "inpainted_images")
    foreground_save_dir = os.path.join(scene_output, "foreground_renders")
    bg_hole_save_dir    = os.path.join(scene_output, "background_hole_renders")
    os.makedirs(inpainted_output_dir, exist_ok=True)
    os.makedirs(foreground_save_dir, exist_ok=True)
    os.makedirs(bg_hole_save_dir, exist_ok=True)

    # Stage 2 / Stage 4 write into their own dedicated dirs, not
    # foreground_save_dir/bg_hole_save_dir directly (those stay iteration-
    # numbered for the regular Stage 1 saving_iterations checkpoints).
    # save_foreground_renders/save_background_hole_renders always nest
    # under iter_{iteration:05d}/, and that iteration number varies between
    # runs (inpaint_at_iter for Stage 2; inpaint_at_iter + refine_iterations
    # for Stage 4, which shifts whenever --refine_iterations changes) -- so
    # without clearing first, re-running with different settings just piles
    # up a new iter_XXXXX/ folder alongside old ones from earlier configs
    # instead of replacing them, and it's easy to end up looking at stale
    # results. _clear_stage_dir wipes the dedicated dir right before each
    # write so only the current run's output survives.
    stage2_foreground_dir = os.path.join(foreground_save_dir, "stage2")
    stage2_bg_hole_dir    = os.path.join(bg_hole_save_dir, "stage2")
    final_foreground_dir  = os.path.join(foreground_save_dir, "final")
    final_bg_hole_dir     = os.path.join(bg_hole_save_dir, "final")

    def _clear_stage_dir(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    all_train_cameras   = scene.getTrainCameras()
    train_cameras       = scene.getTrainCameras().copy()
    viewpoint_indices   = list(range(len(train_cameras)))
    inpainted_cameras   = all_train_cameras.copy()

    ema_photo    = 0.0
    ema_boundary = 0.0
    ema_mask     = 0.0
    ema_ambig    = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training")

    # ================================================================
    # MAIN LOOP
    # ================================================================
    for iteration in range(first_iter, opt.iterations + 1):

        # ============================================================
        # INPAINTING TRIGGER — Stage 2 → Stage 3 → Stage 4
        # ============================================================
        if iteration == inpaint_at_iter and inpaint_at_iter > 0:
            print(f"\n{'='*60}\n[ITER {iteration}] STAGE 2 — INPAINTING TRIGGER\n{'='*60}")

            # ---- Stage 1 -> Stage 2 cost-tracking boundary. Snapshot
            # THIS session's Stage 1 wall-clock/CPU/GPU-peak right here
            # (before any Stage 2/3 work starts), merge with a resumed
            # prior run's Stage 1 cost if any was loaded above, then reset
            # the GPU peak tracker so Stage 2/3's peak is measured in
            # isolation instead of inheriting Stage 1's (typically higher,
            # since Stage 1 holds the full scene) peak.
            _session_stage1_wall_sec = time.time() - _training_wall_start
            _session_stage1_cpu_sec = _cpu_time_sec() - _cpu_time_at_stage1_start
            _stage1_summary = {
                "wall_clock_sec": round(
                    _session_stage1_wall_sec + _prior_stage1.get("wall_clock_sec", 0.0), 1),
                "peak_gpu_memory_allocated_gb": round(max(
                    torch.cuda.max_memory_allocated() / 1024**3,
                    _prior_stage1.get("peak_gpu_memory_allocated_gb", 0.0)), 3),
                "peak_gpu_memory_reserved_gb": round(max(
                    torch.cuda.max_memory_reserved() / 1024**3,
                    _prior_stage1.get("peak_gpu_memory_reserved_gb", 0.0)), 3),
                "cpu_time_sec": round(
                    _session_stage1_cpu_sec + _prior_stage1.get("cpu_time_sec", 0.0), 1),
                "iterations": {"first_iter": _original_first_iter, "last_iter": iteration},
            }
            torch.cuda.reset_peak_memory_stats()
            _stage23_wall_start = time.time()
            _cpu_time_at_stage23_start = _cpu_time_sec()

            # ---- Final Stage 1 cleanup before saving the pre-refine state.
            # This makes the saved Stage 2 ply directly comparable to the
            # final Stage 4 ply (both are cleaned).
            if cleanup_at_saving:
                postprocess_cleanup(
                    gaussians, background_id=backgroundId,
                    floater_k=cleanup_floater_k,
                    floater_distance_ratio=cleanup_floater_distance_ratio,
                    floater_max_opacity=cleanup_floater_max_opacity,
                    relabel_k=cleanup_relabel_k,
                    isolation_threshold=cleanup_isolation_threshold,
                    also_flip_bg_to_fg=cleanup_also_flip_bg_to_fg,
                    logit_flip_confidence=cleanup_logit_flip_confidence,
                    verbose=True,
                )

            # ---- Stage 2 checkpoint: .ply + FG renders + BG hole renders.
            # The BG hole renders here represent "what the hole looks like
            # WITHOUT any refinement on inpainted views" — they're the
            # baseline for evaluating Stage 3's contribution. Writes into
            # the dedicated stage2 dirs (cleared first) so a rerun replaces
            # this stage's output instead of piling up alongside it.
            _clear_stage_dir(stage2_foreground_dir)
            _clear_stage_dir(stage2_bg_hole_dir)
            _save_checkpoint_state(
                gaussians, scene, backgroundId, render, pipe, background,
                stage2_foreground_dir, stage2_bg_hole_dir,
                all_train_cameras, iteration, label="STAGE 2")

            # ---- Resumable checkpoint at the Stage 2->3 boundary, always
            # saved, so Stage 2/3 can be rerun without redoing Stage 1 (the
            # loop's own checkpoint_iterations check is never reached for this
            # iteration: this block ends in `break`). Must happen before
            # cleanup_before_inpainting() below deletes the optimizer and the
            # other training-only tensors gaussians.capture() needs.
            print(f"\n[ITER {iteration}] Saving Checkpoint (pre-Stage-3, automatic)")
            torch.save((gaussians.capture(), iteration),
                       os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

            # ---- Free training state + memory before LaMa/SAM2 heavy work
            inpaint_cache_dir_resolved = (inpaint_cache_dir
                                          if inpaint_cache_dir
                                          else os.path.join(scene_output, "inpaint_cache"))
            cleanup_before_inpainting(scene, gaussians, tb_writer=tb_writer)
            tb_writer = None
            if mask_loader is not None:
                del mask_loader
                mask_loader = None
            progress_bar.close()

            # ---- --skip_reinpaint: reuse the inpainted images already on disk
            # from a previous run (resumed from chkpnt<inpaint_at_iter>.pth)
            # instead of re-running SAM2/LaMa and the hole click. Falls back
            # to inpainting, with a warning, if there are none.
            existing_rgb = (os.path.isdir(inpainted_output_dir)
                             and len(os.listdir(inpainted_output_dir)) > 0)
            reused_existing = False
            if skip_reinpaint and existing_rgb:
                print(f"\n[Inpaint] --skip_reinpaint: reusing existing RGB inpaints "
                      f"({len(os.listdir(inpainted_output_dir))} files) in "
                      f"{inpainted_output_dir} -- not re-running SAM2/LaMa.")
                reused_existing = True
            elif skip_reinpaint:
                print(f"\n[Inpaint] WARNING: --skip_reinpaint given but no inpainted "
                      f"images in {inpainted_output_dir} -- re-inpainting anyway.")

            if not reused_existing:
                # inpainted_output_dir is keyed only by dataset.scene_name (not
                # --model_path) and batch_inpaint_all_cameras nests its output
                # under iter_{iteration:05d}/ inside it -- if a previous run
                # used a different --inpaint_at_iter, that folder would
                # otherwise sit there stale alongside this run's. Clear it
                # first so this run's inpainted images are the only ones
                # present.
                _clear_stage_dir(inpainted_output_dir)
                print("[Inpaint] Initializing OnlineInpainter...")
                inpainter = OnlineInpainter(
                    lama_model_path=lama_model_path,
                    sam_masks_dir=masks_dir if masks_dir else "",
                    cache_dir=inpaint_cache_dir_resolved,
                    save_to_disk=True, save_debug_images=True, device="cuda",
                    use_direct_inference=False,
                    sam2_checkpoint=sam2_checkpoint, sam2_config=sam2_config,
                    use_sam2_propagation=True,
                    mask_dilation_pixels=mask_dilation_pixels,
                )
                batch_inpaint_all_cameras(
                    inpainter=inpainter, gaussians=gaussians,
                    background_id=backgroundId,
                    train_cameras=inpainted_cameras, render_func=render, pipe=pipe,
                    background=background,
                    inpainted_output_dir=inpainted_output_dir,
                    iteration=iteration, hole_threshold=0.25,
                    ghost_clean_method="nlm", clean_ghosts=False,
                )
                _phase_timings.update(get_last_inpainting_timings())
                del inpainter
                torch.cuda.empty_cache()
                print(f"\n[ITER {iteration}] Inpainting complete.")

            # ============================================================
            # STAGE 3 — BG refinement on inpainted images
            # ============================================================
            stage3_stats = None
            if refine_after_inpaint and refine_iterations > 0:
                print(f"\n{'='*60}\n[STAGE 3] BG refinement "
                      f"({refine_iterations} iters)\n{'='*60}")

                # The hole masks LaMa was given, saved by
                # batch_inpaint_all_cameras; Stage 3 seeds the hole from them.
                refinement_mask_dir = os.path.join(
                    inpaint_cache_dir, "temp_input", f"iter_{inpaint_at_iter:05d}")

                _t_refine_start = time.time()
                stage3_stats = run_stage3_refinement(
                    gaussians=gaussians,
                    scene=scene,
                    opt=opt,
                    pipe=pipe,
                    background=background,
                    background_id=backgroundId,
                    inpainted_dir=inpainted_output_dir,
                    refinement_mask_dir=refinement_mask_dir,
                    refine_iterations=refine_iterations,
                    base_iter_offset=iteration,
                )
                _phase_timings['refinement_sec'] = time.time() - _t_refine_start
                print(f"[STAGE 3] Done. Stats: {stage3_stats}")

            # ---- Stage 2/3 cost-tracking summary. Built here (not only
            # inside the `if refine_after_inpaint` block above) so it's
            # still populated for a Stage-2-only run (--no_refine_after_
            # inpaint) -- in that case it just reports the inpainting
            # phase_timings with no refinement_sec key. Folded additively
            # with _prior_stage23 (see the resume-aware loading above) so
            # resuming to continue/redo Stage 2/3 after an interruption
            # accumulates the true total instead of losing the earlier
            # attempt's time.
            _session_stage23_wall_sec = time.time() - _stage23_wall_start
            _session_stage23_cpu_sec = _cpu_time_sec() - _cpu_time_at_stage23_start
            _stage23_summary = {
                "wall_clock_sec": round(
                    _session_stage23_wall_sec + _prior_stage23.get("wall_clock_sec", 0.0), 1),
                "peak_gpu_memory_allocated_gb": round(max(
                    torch.cuda.max_memory_allocated() / 1024**3,
                    _prior_stage23.get("peak_gpu_memory_allocated_gb", 0.0)), 3),
                "peak_gpu_memory_reserved_gb": round(max(
                    torch.cuda.max_memory_reserved() / 1024**3,
                    _prior_stage23.get("peak_gpu_memory_reserved_gb", 0.0)), 3),
                "cpu_time_sec": round(
                    _session_stage23_cpu_sec + _prior_stage23.get("cpu_time_sec", 0.0), 1),
                # hole_click_sec (interactive SAM2 prompt UI), hole_prediction_sec
                # (SAM2 mask propagation across frames), inpainting_sec (LaMa),
                # refinement_sec (Stage 3 training). Any phase that didn't run
                # this session (e.g. all of them under --skip_reinpaint reusing
                # an existing inpaint, or refinement_sec under
                # --no_refine_after_inpaint) falls back to
                # its value from a resumed prior attempt (pre-seeded into
                # _phase_timings above) if there is one, else is simply absent.
                "phase_timings_sec": {k: round(v, 1) for k, v in _phase_timings.items()},
            }

            # ============================================================
            # STAGE 4 — Final cleanup + save
            # ============================================================
            final_iter = iteration + (refine_iterations if refine_after_inpaint else 0)
            print(f"\n{'='*60}\n[STAGE 4] Final cleanup + save at iter {final_iter}\n{'='*60}")

            if cleanup_at_saving:
                postprocess_cleanup(
                    gaussians, background_id=backgroundId,
                    floater_k=cleanup_floater_k,
                    floater_distance_ratio=cleanup_floater_distance_ratio,
                    floater_max_opacity=cleanup_floater_max_opacity,
                    relabel_k=cleanup_relabel_k,
                    isolation_threshold=cleanup_isolation_threshold,
                    also_flip_bg_to_fg=cleanup_also_flip_bg_to_fg,
                    logit_flip_confidence=cleanup_logit_flip_confidence,
                    verbose=True,
                )

            # Final checkpoint: .ply + FG renders + BG hole renders.
            # The BG hole renders here represent "the hole AFTER refinement"
            # — directly comparable to the Stage 2 BG hole renders. final_iter
            # (= inpaint_at_iter + refine_iterations) shifts whenever
            # --refine_iterations changes between runs, which is exactly what
            # you're doing when experimenting with Stage 3 settings -- so
            # this writes into the dedicated "final" dirs (cleared first)
            # instead of foreground_save_dir/bg_hole_save_dir directly, or
            # every rerun would leave its own iter_XXXXX/ folder behind
            # instead of replacing the previous run's result.
            _clear_stage_dir(final_foreground_dir)
            _clear_stage_dir(final_bg_hole_dir)
            _save_checkpoint_state(
                gaussians, scene, backgroundId, render, pipe, background,
                final_foreground_dir, final_bg_hole_dir,
                all_train_cameras, final_iter, label="STAGE 4 / FINAL")

            print(f"\n{'='*60}")
            print(f"[FINAL] Pipeline complete.")
            print(f"  Stage 2 saved at iter {iteration}   (pre-refinement; "
                  f"evaluate hole before Stage 3)")
            print(f"  Stage 4 saved at iter {final_iter}  (post-refinement; "
                  f"evaluate hole after Stage 3)")
            print(f"{'='*60}\n")
            break

        # ------------- Per-iteration bookkeeping ----------------------
        iter_start.record()
        gaussians.update_learning_rate(iteration)


        # ============================================================
        # LOGIT FREEZE — commit labels and detach object_logits
        # ============================================================
        # At LOGIT_FREEZE_ITER (= opt.densify_from_iter by default), commit
        # the current argmax of learned logits into _object_ids, freeze the
        # logits, and detach them from the optimizer. This makes the
        # implicit freeze (caused by _prune_optimizer / cat_tensors_to_optimizer
        # skipping the 'object_logits' group) explicit and deliberate.
        #
        # After this point:
        #   - render_id_mask still uses the frozen logits softmax
        #   - BCE gradient still flows but goes to .grad on a non-leaf-in-optimizer
        #     parameter, so nothing moves the logits
        #   - densification propagates committed (learned) labels via _object_ids,
        #     not COLMAP init
        if iteration == LOGIT_FREEZE_ITER:
            assignment = getattr(gaussians, "obj_assignment", None)
            if (assignment is not None
                    and getattr(assignment, "_object_logits", None) is not None):
                with torch.no_grad():
                    # 1. Commit labels: snapshot argmax of learned logits
                    gaussians._object_ids = assignment.get_object_ids().detach().clone()
                    # 2. Freeze live logits
                    assignment._object_logits.requires_grad_(False)
                    # 3. Detach from optimizer
                    for g in list(gaussians.optimizer.param_groups):
                        if g.get('name') == 'object_logits':
                            g['params'] = []
                # Diagnostic
                if hasattr(assignment, "_object_ids_gt"):
                    n_flipped = (gaussians._object_ids
                                 != assignment._object_ids_gt).sum().item()
                    print(f"[ITER {iteration}] Logits frozen, labels committed "
                          f"(n_flipped from COLMAP init = {n_flipped})")
                else:
                    print(f"[ITER {iteration}] Logits frozen, labels committed.")
            else:
                print(f"[ITER {iteration}] Logit freeze skipped "
                      f"(obj_assignment._object_logits is None)")



        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not train_cameras:
            train_cameras     = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(train_cameras)))

        cam_idx       = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = train_cameras.pop(cam_idx)
        vind          = viewpoint_indices.pop(cam_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = (torch.rand(3, device="cuda") if opt.random_background else background)

        # ============================================================
        # STAGE 1 — full-scene render + losses
        # ============================================================
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)
        image             = render_pkg["render"]
        viewspace_pt      = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii             = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            image = image * viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1      = l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        loss_full = ((1.0 - opt.lambda_dssim) * Ll1
                     + opt.lambda_dssim * (1.0 - ssim_val))

        # ---- 2D Mask BCE -------------------------------------------
        loss_mask_val = 0.0
        if (mask_loader is not None and mask_2d_weight > 0
                and iteration >= mask_2d_start
                and iteration % mask_2d_interval == 0):

            if mask_2d_per_object:
                # Per-object supervision: each class is BCE'd against its OWN
                # mask, so a pixel belonging to one object is a zero target
                # for every other. The composite path below only ever asks
                # "is this pixel some foreground object". With one object
                # the two are the same loss.
                H, W = image.shape[1], image.shape[2]
                gt_by_col, w_by_col = {}, {}
                for obj_name in mask_loader.objects_for_frame(viewpoint_cam.image_name):
                    # returns 0 (== background) for an unknown name, which the
                    # next line drops -- so a stray mask file cannot inject a
                    # bogus class.
                    col = scene.get_name_to_id_mapping(obj_name)
                    if col == backgroundId:
                        continue
                    m = mask_loader.get_mask(viewpoint_cam.image_name, obj_name, H, W)
                    if m is None:
                        continue
                    gt_by_col[col] = m
                    w_by_col[col] = mask_loader.get_bg_eroded_weight(
                        viewpoint_cam.image_name, m, bg_margin_px=mask_2d_bg_margin,
                        object_name=obj_name)

                if gt_by_col:
                    rendered_by_col = render_object_prob_masks(
                        viewpoint_cam, gaussians, pipe, render_fn=render,
                        columns=sorted(gt_by_col))
                    L_mask = mask_loss_per_object(rendered_by_col, gt_by_col, w_by_col)
                    if L_mask is not None:
                        loss_full = loss_full + mask_2d_weight * L_mask
                        loss_mask_val = L_mask.item()
                gt_mask = None          # skip the composite branch below
            else:
                gt_mask = mask_loader.get_composite_mask(
                    viewpoint_cam.image_name, H=image.shape[1], W=image.shape[2])

            if gt_mask is not None:
                rendered_mask = render_id_mask(
                    viewpoint_cam, gaussians, pipe, background,
                    render_fn=render, background_id=backgroundId)

                pixel_weight = mask_loader.get_bg_eroded_weight(
                    viewpoint_cam.image_name, gt_mask,
                    bg_margin_px=mask_2d_bg_margin,
                )
                L_mask = mask_loss_2d_v2(rendered_mask, gt_mask,
                                         pixel_weight, loss_type="bce")
                loss_full = loss_full + mask_2d_weight * L_mask
                loss_mask_val = L_mask.item()

        # ---- 3D Boundary regularizer --------------------------------
        loss_boundary_val = 0.0
        loss_ambig_val    = 0.0
        boundary_stats    = {}
        if (boundary_reg_weight > 0
                and iteration >= boundary_reg_start
                and iteration % boundary_reg_interval == 0):
            L_boundary, boundary_stats = boundary_regularization_loss(
                gaussians,
                k=boundary_k,
                min_disagreement=boundary_min_disagree,
                pair_sample_size=boundary_sample_size,
                flatten_weight=boundary_flatten_w,
                shrink_weight=boundary_shrink_w,
                repulsion_weight=boundary_repulsion_w,
                repulsion_margin=boundary_repulsion_m,
                ambig_scale_weight=ambig_scale_weight,
                ambig_scale_power=ambig_scale_power,
            )
            loss_full         = loss_full + boundary_reg_weight * L_boundary
            loss_boundary_val = float(L_boundary.item())
            loss_ambig_val    = float(boundary_stats.get("L_ambig_scale", 0.0))

        loss_full.backward()
        iter_end.record()

        with torch.no_grad():
            ema_photo    = 0.4 * loss_full.item()       + 0.6 * ema_photo
            ema_boundary = 0.4 * loss_boundary_val      + 0.6 * ema_boundary
            ema_mask     = 0.4 * loss_mask_val          + 0.6 * ema_mask
            ema_ambig    = 0.4 * loss_ambig_val         + 0.6 * ema_ambig

            if iteration % 10 == 0:
                pf = {
                    "L_photo": f"{ema_photo:.4f}",
                    "L_bnd":   f"{ema_boundary:.4f}",
                    "pts":     f"{len(gaussians._xyz)}",
                }
                if mask_2d_weight > 0:
                    pf["L_mask"] = f"{ema_mask:.4f}"
                if ambig_scale_weight > 0:
                    pf["L_ambig"] = f"{ema_ambig:.4f}"
                progress_bar.set_postfix(pf)
                progress_bar.update(10)

            if iteration % 500 == 0:
                labels_now = _get_hard_labels_for_mask(gaussians)
                uids, ucounts = torch.unique(labels_now, return_counts=True)
                id_dist = {int(u): int(c) for u, c in zip(uids, ucounts)}
                print(f"\n[ITER {iteration}] Gaussians per label: {id_dist}")

            if tb_writer is not None:
                training_report(
                    tb_writer, iteration, Ll1, loss_full, 0.0,
                    loss_boundary_val, loss_mask_val,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations, scene, render,
                    (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                    dataset.train_test_exp)
                if ambig_scale_weight > 0:
                    tb_writer.add_scalar('train_loss/ambig_scale',
                                          loss_ambig_val, iteration)

            if iteration in saving_iterations:
                _save_this_iter = True
            else:
                _save_this_iter = False

            # Densification
            _densify_stats_this_iter = None
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_pt, visibility_filter)

                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0):
                    size_threshold = (20 if iteration > opt.opacity_reset_interval else None)
                    _densify_stats_this_iter = gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold, radii)

                if (iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background
                            and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            # Label-noise mechanism trace.
            if _trace_enabled and (iteration % trace_interval == 0
                                    or _densify_stats_this_iter):
                _log_trace_row(iteration, densify_stats=_densify_stats_this_iter)

            # Save + cleanup + FG/BG renders (deferred — same reason)
            if _save_this_iter:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                # Cleanup runs only at a stage's final save, so intermediate
                # checkpoints and the optimisation after them are untouched.
                # With inpainting, Stage 1 ends at the trigger instead.
                if cleanup_at_saving and iteration == opt.iterations:
                    postprocess_cleanup(
                        gaussians,
                        background_id=backgroundId,
                        floater_k=cleanup_floater_k,
                        floater_distance_ratio=cleanup_floater_distance_ratio,
                        floater_max_opacity=cleanup_floater_max_opacity,
                        relabel_k=cleanup_relabel_k,
                        isolation_threshold=cleanup_isolation_threshold,
                        also_flip_bg_to_fg=cleanup_also_flip_bg_to_fg,
                        logit_flip_confidence=cleanup_logit_flip_confidence,
                        verbose=True,
                    )
                _save_checkpoint_state(
                    gaussians, scene, backgroundId, render, pipe, background,
                    foreground_save_dir, bg_hole_save_dir,
                    all_train_cameras, iteration, label=f"ITER {iteration}")

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

    progress_bar.close()
    print("\nTraining loop exited.")

    if trace_file is not None:
        trace_file.close()

    # If the inpaint trigger never fired this session (--inpaint_at_iter 0,
    # or opt.iterations < inpaint_at_iter, or a plain Stage-1-only run) then
    # _stage1_summary is still None -- build it now from the full session,
    # same as the boundary snapshot does, just without a Stage 2/3 to follow.
    if _stage1_summary is None:
        _stage1_summary = {
            "wall_clock_sec": round(
                (time.time() - _training_wall_start) + _prior_stage1.get("wall_clock_sec", 0.0), 1),
            "peak_gpu_memory_allocated_gb": round(max(
                torch.cuda.max_memory_allocated() / 1024**3,
                _prior_stage1.get("peak_gpu_memory_allocated_gb", 0.0)), 3),
            "peak_gpu_memory_reserved_gb": round(max(
                torch.cuda.max_memory_reserved() / 1024**3,
                _prior_stage1.get("peak_gpu_memory_reserved_gb", 0.0)), 3),
            "cpu_time_sec": round(
                (_cpu_time_sec() - _cpu_time_at_stage1_start) + _prior_stage1.get("cpu_time_sec", 0.0), 1),
            "iterations": {"first_iter": _original_first_iter, "last_iter": iteration},
        }

    _stage1_wall_sec = _stage1_summary["wall_clock_sec"]
    _stage23_wall_sec = _stage23_summary["wall_clock_sec"] if _stage23_summary else 0.0
    _total_wall_sec = _stage1_wall_sec + _stage23_wall_sec
    _total_cpu_sec = _stage1_summary["cpu_time_sec"] + (
        _stage23_summary["cpu_time_sec"] if _stage23_summary else 0.0)
    # Peak memory across the whole pipeline -- each stage's own peak was
    # already isolated via the mid-loop reset_peak_memory_stats() call, so
    # the overall figure is just the max of the two (already resume-folded
    # for Stage 1 specifically, at the point each summary was built).
    _peak_alloc_gb = max(
        _stage1_summary["peak_gpu_memory_allocated_gb"],
        _stage23_summary["peak_gpu_memory_allocated_gb"] if _stage23_summary else 0.0)
    _peak_reserved_gb = max(
        _stage1_summary["peak_gpu_memory_reserved_gb"],
        _stage23_summary["peak_gpu_memory_reserved_gb"] if _stage23_summary else 0.0)

    _cost_summary = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_logical_cpus": os.cpu_count(),
        # Stage 1: ordinary 3DGS training, iter first_iter -> inpaint_at_iter
        # (or -> final_iteration if inpainting was never triggered). Folded
        # with a resumed prior run's own Stage 1 cost if one was found (see
        # the resume-aware loading above) -- wall-clock/cpu_time sum across
        # resumes, peak GPU takes the max (a high-water mark, not additive).
        "stage1_training": _stage1_summary,
        # Stage 2 (inpainting) + Stage 3 (BG refinement): hole_click_sec
        # (interactive SAM2 prompt UI), hole_prediction_sec (SAM2 mask
        # propagation across frames), inpainting_sec (LaMa), refinement_sec
        # (Stage 3 training). wall_clock_sec/cpu_time_sec/peak_gpu_* cover this whole
        # window (Stage 2 start -> Stage 3 end), not just the sum of the
        # named phases below it -- there's bookkeeping between them (mask
        # dilation, saving inpainted images to disk, etc.) not worth its
        # own named phase. None if inpainting was never triggered.
        "stage2_3_refinement": _stage23_summary,
        "overall": {
            "total_wall_clock_sec": round(_total_wall_sec, 1),
            "total_wall_clock_min": round(_total_wall_sec / 60.0, 2),
            "total_cpu_time_sec": round(_total_cpu_sec, 1),
            "peak_gpu_memory_allocated_gb": round(_peak_alloc_gb, 3),
            "peak_gpu_memory_reserved_gb": round(_peak_reserved_gb, 3),
            # Peak RSS since process start -- NOT resettable mid-process
            # (unlike the CUDA peak trackers above), so this can only be
            # reported as one whole-process figure, not broken out per stage.
            "peak_cpu_rss_gb": round(_peak_rss_gb(), 3),
            "original_first_iter": _original_first_iter,
            "final_iteration": iteration,
            "resumed_from_prior_cost": bool(_prior_stage1) or bool(_prior_stage23),
        },
    }
    _cost_path = os.path.join(scene.model_path, "training_cost.json")
    with open(_cost_path, "w") as f:
        json.dump(_cost_summary, f, indent=2)
    print(f"[TRAINING COST] total wall_clock={_total_wall_sec/60:.2f} min "
          f"(stage1={_stage1_wall_sec/60:.2f} min, stage2_3={_stage23_wall_sec/60:.2f} min), "
          f"peak_alloc={_peak_alloc_gb:.2f} GB, peak_reserved={_peak_reserved_gb:.2f} GB, "
          f"peak_cpu_rss={_cost_summary['overall']['peak_cpu_rss_gb']:.2f} GB -> {_cost_path}")


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    parser = ArgumentParser(description="VR-RigidGS Unified Trainer (clean)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    tp = TrainingParams(parser)

    parser.add_argument('--ip',             type=str,  default="127.0.0.1")
    parser.add_argument('--port',           type=int,  default=6008)
    parser.add_argument('--debug_from',     type=int,  default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1000,5000, 7_000,  12_000, 15_000,20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[ 1000, 5000, 7_000,  12_000, 15_000,20_000, 30_000])
    parser.add_argument("--quiet",                  action="store_true")
    parser.add_argument('--disable_viewer',         action='store_true')
    parser.add_argument("--checkpoint_iterations",  nargs="+"
                        , type=int, default=[])
    parser.add_argument("--start_checkpoint",       type=str, default=None)
    parser.add_argument('--_images',
                        required=True,
                        help="Folder of the training photos (the same images preprocessing.py used).")
    parser.add_argument("--config", default=None,
                        help="Path to configs/<dataset>/<scene>.json. Sets this scene's paper "
                             "hyperparameters (mask_2d_bg_margin, inpaint_at_iter, "
                             "refine_after_inpaint, refine_iterations, mask_2d_per_object, "
                             "cleanup_at_saving, ...) as defaults; any of those also given "
                             "explicitly on the command line still wins. Does not set "
                             "-s/-m/--_images/--masks_dir -- those still need to be given "
                             "by hand (or use scripts/run_pipeline.py, which derives them too).")

    # Boundary reg
    parser.add_argument("--boundary_reg_weight",      type=float, default=1.0)
    parser.add_argument("--boundary_reg_interval",    type=int,   default=10)
    parser.add_argument("--boundary_reg_start",       type=int,   default=500)
    parser.add_argument("--boundary_k",               type=int,   default=16)
    parser.add_argument("--boundary_min_disagreement", type=int,  default=2)
    parser.add_argument("--boundary_flatten_weight",  type=float, default=1.0)
    parser.add_argument("--boundary_shrink_weight",   type=float, default=1.0)
    parser.add_argument("--boundary_repulsion_weight", type=float, default=1.0)
    parser.add_argument("--boundary_repulsion_margin", type=float, default=0.0005)
    parser.add_argument("--boundary_sample_size",     type=int,   default=50000)
    parser.add_argument("--boundary_prune_opacity",   type=float, default=0.5)

    parser.add_argument("--ambig_scale_weight", type=float, default=0.0)
    parser.add_argument("--ambig_scale_power",  type=float, default=1.0)

    # Cleanup
    # Optional per-object floater prune at the final save of each stage:
    # removes low-opacity foreground Gaussians far from the rest of their own
    # object. Background and labels are never changed. Off by default.
    parser.add_argument("--cleanup_at_saving", action="store_true", default=False)
    parser.add_argument("--no_cleanup_at_saving", action="store_false",
                        dest="cleanup_at_saving")
    parser.add_argument("--cleanup_floater_k", type=int, default=8)
    parser.add_argument("--cleanup_floater_distance_ratio", type=float, default=4.0)
    parser.add_argument("--cleanup_floater_max_opacity", type=float, default=0.5)
    parser.add_argument("--cleanup_relabel_k", type=int, default=16)
    parser.add_argument("--cleanup_isolation_threshold", type=float, default=0.8)
    parser.add_argument("--cleanup_also_flip_bg_to_fg", action="store_true", default=True)
    parser.add_argument("--cleanup_no_flip_bg_to_fg", action="store_false",
                        dest="cleanup_also_flip_bg_to_fg")
    parser.add_argument("--cleanup_logit_flip_confidence", type=float, default=5.0)

    # 2D Mask Loss
    parser.add_argument("--mask_2d_weight",     type=float, default=1.0)
    parser.add_argument("--mask_2d_start",      type=int,   default=500)
    parser.add_argument("--mask_2d_interval",   type=int,   default=1)
    parser.add_argument("--masks_dir",          type=str, default="",
                        help="2D object masks for the mask loss: <scene>/preprocessing/"
                             "refined/refined_masks, or <scene>/preprocessing/segmentation/"
                             "masks for scenes run without mask refinement.")
    parser.add_argument("--mask_2d_bg_margin", type=float, default=1.0,
                        help="v2 silhouette supervision: background pixels within this "
                             "many pixels of the fg silhouette are excluded from BCE. "
                             "Prevents depth-edge artifacts from pushing fg Gaussians "
                             "inward and leaving debris. Default: 15.")
    parser.add_argument("--mask_2d_per_object", action="store_true", default=False,
                        help="BCE each object against its OWN mask instead of the OR-of-all "
                             "composite. Use for scenes with more than one object: the "
                             "composite cannot tell one object from another. Identical to "
                             "the composite loss for single-object scenes.")

    # Inpainting
    parser.add_argument("--inpaint_at_iter", type=int, default=30_000,
                        help="Iteration to trigger inpainting. 0=disabled.")
    parser.add_argument("--bg_hole_weight",  type=float, default=1.0,
                        help="Kept for argument compatibility. Not used.")
    parser.add_argument("--lama_model_path", type=str,
                        default="external/lama/big-lama")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="external/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config",     type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--inpaint_cache_dir", type=str, default="",
                        help="Default: <model_path>/online_inpainting/inpaint_cache.")
    parser.add_argument("--mask_dilation_pixels", type=int, default=20,
                        help="Grow the SAM2 hole mask by this many pixels before handing it "
                             "to LaMa, so no sliver of the removed object's old edge survives "
                             "inpainting.")

    # Stage 3 refinement
    parser.add_argument("--refine_after_inpaint", action="store_true", default=False,
                        help="Run Stage 3 (BG refinement on inpainted images) after "
                             "the inpainter completes, then Stage 4 cleanup + save.")
    parser.add_argument("--no_refine_after_inpaint", action="store_false",
                        dest="refine_after_inpaint",
                        help="Stop after inpainting (no Stage 3).")
    parser.add_argument("--refine_iterations", type=int, default=7000,
                        help="Number of Stage 3 refinement iterations.")
    parser.add_argument("--skip_reinpaint", action="store_true", default=False,
                        help="When resuming from chkpnt<inpaint_at_iter>.pth, reuse the "
                             "inpainted images already on disk instead of re-running "
                             "SAM2/LaMa and the hole click. Falls back to inpainting, "
                             "with a warning, if there are none.")

    parser.add_argument("--object_ids_override", type=str, default=None,
                        help="[Label-noise experiment] Path to an object_metadata.json "
                             "(object_id_array format, index-aligned with points3D.ply) "
                             "whose labels replace the true COLMAP-init labels right "
                             "after Scene() init, before training_setup(). Used to feed "
                             "corrupted labels produced by flip_labels.py.")

    parser.add_argument("--enable_mechanism_trace", action="store_true", default=False,
                        help="[Label-noise experiment] Uses TrackedGaussianModel instead of "
                             "GaussianModel and periodically logs, per Gaussian-lineage, "
                             "whether the frozen label still disagrees with the true "
                             "(uncorrupted) label, that population's opacity / suppression "
                             "fraction, and clone/split/prune counts, to --trace_csv. "
                             "Requires --true_labels_path.")
    parser.add_argument("--true_labels_path", type=str, default=None,
                        help="[Label-noise experiment] export_init_labels.py output for this "
                             "scene (the TRUE uncorrupted object_id_array) — NOT the "
                             "corrupted --object_ids_override file. Required with "
                             "--enable_mechanism_trace.")
    parser.add_argument("--trace_csv", type=str, default=None,
                        help="[Label-noise experiment] Output CSV path for --enable_mechanism_trace.")
    parser.add_argument("--trace_interval", type=int, default=100,
                        help="[Label-noise experiment] Log a trace row every N iterations "
                             "(always also logged on iterations where densify_and_prune ran, "
                             "and once at iteration 0 right after label injection).")

    parser.add_argument("--learnable_logits", action="store_true", default=False,
                        help="[Ablation] Keep the optimizer attached to the per-Gaussian "
                             "object logits through densification, so labels can keep "
                             "changing. Combine with a later --logit_freeze_iter. Off in "
                             "RigidGS: labels are fixed after the first densification.")
    parser.add_argument("--logit_freeze_iter", type=int, default=None,
                        help="Iteration at which to commit labels and freeze "
                             "object_logits. Default: opt.densify_from_iter (500).")

    pre_args, _ = parser.parse_known_args(sys.argv[1:])
    if pre_args.config:
        import config_utils
        cfg = config_utils.load_config(pre_args.config)
        config_utils.apply_config_defaults(parser, cfg, "train")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    model_params = lp.extract(args)
    model_params.images = os.path.abspath(args._images)

    training(
        model_params, op.extract(args), pp.extract(args), tp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, extra_args=args)

    print("\nDone.")
