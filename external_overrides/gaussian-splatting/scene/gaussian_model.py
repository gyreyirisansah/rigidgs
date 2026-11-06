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

# =============================================================================
# RigidGS A/B VARIANT: **FROZEN**
#   - object_logits LR = 0.5 (kept high to match original training, but irrelevant
#     after the first densify event since the optimizer never sees the live param)
#   - prune_points and densification_postfix do NOT re-sync the optimizer's
#     object_logits param group at obj_assignment._object_logits, so optimizer
#     keeps stepping a stale Parameter after the first densify/prune.
#   - Net effect: BCE writes gradient to _object_logits during early iterations
#     (iters 1..~500 before first densify). After that, optimizer.step() is a
#     no-op on the live logits — they freeze at whatever value they had.
#   - Cleanup at save iters can still flip labels via direct .data writes
#     (`relabel_isolated_gaussians`); that's independent of the optimizer.
#
# Pair this file with `gaussian_model_LEARNABLE.py` for A/B comparison.
# =============================================================================

import torch
import numpy as np
from scene.rigidgs_object_assignment import ObjectAssignmentModule,ObjectAssignment2D
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        # False (default, the paper's model): the optimizer is NOT re-pointed
        # at the new object_logits Parameter after prune/densify, so the
        # logits stop moving after the first densification and labels are
        # effectively fixed. True: the optimizer follows the logits through
        # every structural change (train.py --learnable_logits, ablation).
        self.learnable_logits = False
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._object_ids = torch.empty(0) # To handle per object ids

        # Per-Gaussian mask-evidence tracking buffers.
        # These are non-learnable EMA statistics used by RigidGS to diagnose
        # whether each Gaussian is consistently supported by foreground masks,
        # background masks, or mixed evidence. They intentionally do not change
        # the optimizer state and are kept separate from COB-GS-style splitting.
        self.mask_fg_evidence = None
        self.mask_bg_evidence = None
        self.mask_seen_evidence = None

        self.obj_assignment = ObjectAssignment2D(len(set(self._object_ids.tolist())))
        self.setup_functions()
        
        # self.bg_mask = None  # To store background mask

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._object_ids
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._object_ids) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        # self._object_ids = self.load_object_ids(object_ids_path)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure
    


    @property
    def object_ids(self):
        """Current hard IDs (may differ from initial after training)."""
        return self.obj_assignment.get_object_ids()
    
    @property
    def object_logits(self):
        """Learnable logits for boundary loss."""
        return self.obj_assignment.object_logits
    
    @property
    def object_ids_gt(self):
        """Ground truth IDs (frozen, from SAM)."""
        if self.obj_assignment is not None:
            return self.obj_assignment.object_ids_gt
        return self._object_ids  # Fallback
    

    # ------------------------------------------------------------------
    # Per-Gaussian mask evidence / ambiguity tracking
    # ------------------------------------------------------------------
    def initialize_mask_evidence_tracking(self, reset=False):
        """
        Allocate non-learnable per-Gaussian foreground/background evidence buffers.
        These buffers are updated from 2D mask supervision during training and
        are used to detect foreground leakage, missed foreground, and true
        boundary ambiguity.
        """
        n = self.get_xyz.shape[0]
        device = self.get_xyz.device
        if reset or self.mask_fg_evidence is None or len(self.mask_fg_evidence) != n:
            self.mask_fg_evidence = torch.zeros(n, device=device, dtype=torch.float32)
            self.mask_bg_evidence = torch.zeros(n, device=device, dtype=torch.float32)
            self.mask_seen_evidence = torch.zeros(n, device=device, dtype=torch.float32)

    def reset_mask_evidence_tracking(self):
        self.initialize_mask_evidence_tracking(reset=True)

    def _sync_mask_evidence_after_prune(self, valid_points_mask):
        if self.mask_fg_evidence is not None:
            self.mask_fg_evidence = self.mask_fg_evidence[valid_points_mask]
            self.mask_bg_evidence = self.mask_bg_evidence[valid_points_mask]
            self.mask_seen_evidence = self.mask_seen_evidence[valid_points_mask]

    def _sync_mask_evidence_after_densify(self, n_new, parent_indices=None):
        """
        Extend evidence buffers after densification.
        New Gaussians start with zero evidence by default. If parent_indices are
        provided, copy a small fraction of parent evidence to avoid cold starts.
        """
        if self.mask_fg_evidence is None:
            return
        device = self.get_xyz.device
        if parent_indices is not None and len(parent_indices) == n_new:
            parent_indices = parent_indices.to(device=device, dtype=torch.long)
            new_fg = 0.25 * self.mask_fg_evidence[parent_indices].detach()
            new_bg = 0.25 * self.mask_bg_evidence[parent_indices].detach()
            new_seen = 0.25 * self.mask_seen_evidence[parent_indices].detach()
        else:
            new_fg = torch.zeros(n_new, device=device, dtype=torch.float32)
            new_bg = torch.zeros(n_new, device=device, dtype=torch.float32)
            new_seen = torch.zeros(n_new, device=device, dtype=torch.float32)
        self.mask_fg_evidence = torch.cat([self.mask_fg_evidence, new_fg], dim=0)
        self.mask_bg_evidence = torch.cat([self.mask_bg_evidence, new_bg], dim=0)
        self.mask_seen_evidence = torch.cat([self.mask_seen_evidence, new_seen], dim=0)

    @torch.no_grad()
    def update_mask_evidence(self, fg_indices, bg_indices, fg_weights=None, bg_weights=None, decay=0.995):
        """
        EMA update of per-Gaussian mask evidence.
        fg_indices/bg_indices are Gaussian indices whose projections fall into
        foreground/background mask pixels for the current view.
        """
        self.initialize_mask_evidence_tracking(reset=False)
        decay = float(decay)
        self.mask_fg_evidence.mul_(decay)
        self.mask_bg_evidence.mul_(decay)
        self.mask_seen_evidence.mul_(decay)

        if fg_indices is not None and fg_indices.numel() > 0:
            fg_indices = fg_indices.to(device=self.get_xyz.device, dtype=torch.long)
            if fg_weights is None:
                fg_weights = torch.ones_like(fg_indices, dtype=torch.float32, device=self.get_xyz.device)
            else:
                fg_weights = fg_weights.to(device=self.get_xyz.device, dtype=torch.float32)
            self.mask_fg_evidence.index_add_(0, fg_indices, fg_weights)
            self.mask_seen_evidence.index_add_(0, fg_indices, fg_weights)

        if bg_indices is not None and bg_indices.numel() > 0:
            bg_indices = bg_indices.to(device=self.get_xyz.device, dtype=torch.long)
            if bg_weights is None:
                bg_weights = torch.ones_like(bg_indices, dtype=torch.float32, device=self.get_xyz.device)
            else:
                bg_weights = bg_weights.to(device=self.get_xyz.device, dtype=torch.float32)
            self.mask_bg_evidence.index_add_(0, bg_indices, bg_weights)
            self.mask_seen_evidence.index_add_(0, bg_indices, bg_weights)

    def get_mask_evidence_roles(self, background_id=0, min_seen=1.0,
                                leakage_bg_ratio=0.65,
                                missed_fg_ratio=0.65,
                                boundary_low=0.35,
                                boundary_high=0.65,
                                ambiguity_threshold=0.55,
                                eps=1e-6):
        """
        Classify Gaussians using accumulated mask evidence.

        Returns a dict with fg_ratio, bg_ratio, ambiguity, and boolean masks:
          leakage_fg: foreground-labeled Gaussians with dominant background evidence
          missed_fg: background-labeled Gaussians with dominant foreground evidence
          true_boundary: mixed-evidence Gaussians near the FG/BG transition
        """
        self.initialize_mask_evidence_tracking(reset=False)
        fg = self.mask_fg_evidence
        bg = self.mask_bg_evidence
        total = fg + bg + eps
        fg_ratio = fg / total
        bg_ratio = bg / total
        ambiguity = 1.0 - torch.abs(fg - bg) / total
        seen = self.mask_seen_evidence >= float(min_seen)

        is_fg_label = self._object_ids != int(background_id)
        is_bg_label = self._object_ids == int(background_id)

        leakage_fg = seen & is_fg_label & (bg_ratio >= float(leakage_bg_ratio))
        missed_fg = seen & is_bg_label & (fg_ratio >= float(missed_fg_ratio))
        true_boundary = (seen & (ambiguity >= float(ambiguity_threshold)) &
                         (fg_ratio >= float(boundary_low)) & (fg_ratio <= float(boundary_high)))

        return {
            "fg_ratio": fg_ratio,
            "bg_ratio": bg_ratio,
            "ambiguity": ambiguity,
            "seen": seen,
            "leakage_fg": leakage_fg,
            "missed_fg": missed_fg,
            "true_boundary": true_boundary,
        }

    # Add this method to your GaussianModel class

    def get_rotation_matrix(self):
        """
        Convert quaternions to rotation matrices.
        
        Returns:
            rotations: [N, 3, 3] rotation matrices
        """
        q = self.get_rotation  # [N, 4] normalized quaternions
        
        # wxyz convention (w at index 0)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Build rotation matrix
        R00 = 1 - 2*y*y - 2*z*z
        R01 = 2*x*y - 2*w*z
        R02 = 2*x*z + 2*w*y
        R10 = 2*x*y + 2*w*z
        R11 = 1 - 2*x*x - 2*z*z
        R12 = 2*y*z - 2*w*x
        R20 = 2*x*z - 2*w*y
        R21 = 2*y*z + 2*w*x
        R22 = 1 - 2*x*x - 2*y*y
        
        R = torch.stack([
            R00, R01, R02,
            R10, R11, R12,
            R20, R21, R22
        ], dim=-1).reshape(-1, 3, 3)
        
        return R

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float, object_ids: np.ndarray = None):
        """
        Args:
            object_ids: (N,) array of object IDs for each point
        """

       
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"=== INITIALIZING FROM POINT CLOUD ===")
        print(f"Number of points in input PCD: {len(pcd.points)}")
        print(f"fused_point_cloud shape: {fused_point_cloud.shape}")


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        # Store object IDs
        if object_ids is not None:
            self._object_ids = torch.tensor(
                object_ids, dtype=torch.long, device="cuda"
            )

            self.obj_assignment.initialize_from_ids(self._object_ids)
        else:
            # Default: all points are object 0
            self._object_ids = torch.zeros(
                len(pcd.points), dtype=torch.long, device="cuda"
            )
        
        print(f"Initialized with {len(self._object_ids)} object IDs")
        print(f"After initialization:")
        print(f"  _xyz shape: {self._xyz.shape}")
        print(f"  _features_dc shape: {self._features_dc.shape}")
        print(f"  _features_rest shape: {self._features_rest.shape}")
        print(f"  _object_ids shape: {self._object_ids.shape}")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        if self.obj_assignment is not None and self.obj_assignment.object_logits is not None:
            self.optimizer.add_param_group({
                'params': [self.obj_assignment.object_logits],
                # 0.5 is irrelevant when logits are frozen by the optimizer
                # desync; learnable logits use a gentler rate to avoid churn.
                'lr': 0.1 if self.learnable_logits else 0.5,
                'name': 'object_logits'
            })

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        object_ids_path = os.path.join(os.path.dirname(path), "object_metadata.json")

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # Save object IDs
        object_ids = self._object_ids.detach().cpu().numpy()
        meta_dict = {
            "object_id_array": object_ids.tolist(),
        }
        with open(object_ids_path, "w") as f:
            json.dump(meta_dict, f, indent=2)   

    def load_object_ids(self, path):
        if not os.path.exists(path):
            print(f"Warning: Object IDs file not found at {path}. Defaulting to zeros.")
            return torch.zeros((self.get_xyz.shape[0],), dtype=torch.long, device="cuda")
        with open(path, "r") as f:
            meta = json.load(f)
        object_ids = torch.tensor(
            meta["object_id_array"], dtype=torch.long, device="cuda"
        )
        return object_ids

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def reset_opacity_background(self, backgroundId, region_min=None, region_max=None):
        """
        region_min/region_max (optional, both None by default -- fully
        backward compatible with every other caller): an axis-aligned box
        in world space. When given, only background Gaussians inside it
        get reset; every other background Gaussian's opacity is left
        untouched. Without this, every background Gaussian in the WHOLE
        scene gets reset every opacity_reset_interval -- fine for Stage 1
        (that's the point: force periodic re-justification of the entire
        scene), but wrong for a hole-focused refinement pass, where it
        looks indistinguishable from "everywhere just went through an
        opacity reset" because it literally did, repeatedly, for the
        entire run.
        """
        bg_mask = self._object_ids == backgroundId
        if region_min is not None and region_max is not None:
            in_region = ((self._xyz >= region_min) & (self._xyz <= region_max)).all(dim=1)
            bg_mask = bg_mask & in_region
        opacities = self.get_opacity.clone()
        opacities[bg_mask] = torch.min(opacities[bg_mask], torch.ones_like(opacities[bg_mask]) * 0.01)
        opacities_new = self.inverse_opacity_activation(opacities)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        object_ids_path = os.path.join(os.path.dirname(path), "object_metadata.json")
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # Load object IDs
        self._object_ids = self.load_object_ids(object_ids_path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "object_logits":
                continue  # Handled separately by ObjectAssignmentModule
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # =====================================================================
    # LEARNABLE-only: keep optimizer's `object_logits` param group in sync
    # with obj_assignment._object_logits after every prune/densify.
    #
    # obj_assignment.on_prune / on_densification_postfix do
    #   `self._object_logits = nn.Parameter(...)`
    # which creates a NEW Parameter each time. The optimizer was registered
    # with the ORIGINAL Parameter at training_setup; without this helper,
    # the optimizer keeps stepping the original (stale) Parameter while
    # gradient flows into the new one. This helper re-points the param
    # group at the new Parameter and prunes/extends its Adam state.
    # =====================================================================
    def _sync_object_logits_in_optimizer(self, new_param, mask_or_count, mode):
        """
        Args:
            new_param:     the NEW nn.Parameter now in obj_assignment._object_logits.
            mask_or_count: prune  → bool mask of size N_old (True = keep).
                           extend → int n_new (rows to append).
            mode:          'prune' or 'extend'.
        """
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group.get("name") != "object_logits":
                continue
            old_param = group["params"][0]
            stored_state = self.optimizer.state.pop(old_param, None)
            if stored_state is not None:
                if mode == 'prune':
                    valid_mask = mask_or_count
                    try:
                        stored_state["exp_avg"]    = stored_state["exp_avg"][valid_mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                    except (IndexError, RuntimeError):
                        stored_state["exp_avg"]    = torch.zeros_like(new_param)
                        stored_state["exp_avg_sq"] = torch.zeros_like(new_param)
                elif mode == 'extend':
                    n_new = int(mask_or_count)
                    if n_new > 0:
                        K = new_param.shape[1]
                        zeros = torch.zeros((n_new, K),
                                            device=new_param.device,
                                            dtype=stored_state["exp_avg"].dtype)
                        try:
                            stored_state["exp_avg"]    = torch.cat([stored_state["exp_avg"],    zeros], dim=0)
                            stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], zeros], dim=0)
                        except (RuntimeError, TypeError):
                            stored_state["exp_avg"]    = torch.zeros_like(new_param)
                            stored_state["exp_avg_sq"] = torch.zeros_like(new_param)
                else:
                    raise ValueError(f"Unknown mode {mode!r}")
            group["params"][0] = new_param
            if stored_state is not None:
                self.optimizer.state[new_param] = stored_state
            return

    def prune_points(self, mask):
        valid_points_mask = ~mask

        # FROZEN variant: obj_assignment.on_prune replaces _object_logits
        # with a new Parameter, but we do NOT re-point the optimizer at it.
        # Consequence: after the first densify/prune, the optimizer keeps
        # stepping the original (stale) Parameter and the live _object_logits
        # no longer trains via the optimizer. The BCE still writes gradient
        # to the live logits but optimizer.step() never moves them.
        # This is the regime that gives clean FG + fast training.
        if self.obj_assignment is not None:
            self.obj_assignment.on_prune(valid_points_mask)
            if self.learnable_logits:
                new_logits = getattr(self.obj_assignment, "_object_logits", None)
                if new_logits is not None:
                    self._sync_object_logits_in_optimizer(
                        new_logits, valid_points_mask, mode='prune')

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]

        # Prune _object_ids and mask-evidence tracking the same way
        self._object_ids = self._object_ids[valid_points_mask]
        self._sync_mask_evidence_after_prune(valid_points_mask)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom              = self.denom[valid_points_mask]
        self.max_radii2D        = self.max_radii2D[valid_points_mask]
        self.tmp_radii          = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue  # Skip param groups not being extended (e.g., object_logits)
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_object_ids=None, parent_indices=None):
        # Validate shapes before concatenating
        assert new_features_dc.dim() == 3, f"new_features_dc should be 3D, got {new_features_dc.shape}"
        assert new_features_rest.dim() == 3, f"new_features_rest should be 3D, got {new_features_rest.shape}"
        assert new_features_dc.shape[1] == 1, f"DC should have 1 SH coefficient, got {new_features_dc.shape[1]}"
        assert new_features_dc.shape[2] == 3, f"DC should have 3 color channels, got {new_features_dc.shape[2]}"

        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz           = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]

        # ========== Concatenate object IDs ==========
        if new_object_ids is not None:
            self._object_ids = torch.cat([self._object_ids, new_object_ids], dim=0)
        else:
            # Default: assign ID 0 to new Gaussians
            self._object_ids = torch.cat([
                self._object_ids,
                torch.zeros(len(new_xyz), dtype=torch.long, device="cuda")
            ], dim=0)
        # ============================================

        # FROZEN variant: on_densification_postfix replaces _object_logits
        # with a new Parameter (old data + new logits appended), but we do
        # NOT re-point the optimizer's object_logits group at the new
        # Parameter. See the comment in prune_points above for rationale.
        if self.obj_assignment is not None and new_object_ids is not None:
            self.obj_assignment.on_densification_postfix(new_object_ids)
            if self.learnable_logits:
                new_logits = getattr(self.obj_assignment, "_object_logits", None)
                if new_logits is not None:
                    self._sync_object_logits_in_optimizer(
                        new_logits, int(len(new_object_ids)), mode='extend')

        self._sync_mask_evidence_after_densify(len(new_xyz), parent_indices=parent_indices)

        self.tmp_radii          = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom              = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D        = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        # Preserves correct shape: (M, 1, 3) and (M, 15, 3)
        selected_dc = self._features_dc[selected_pts_mask]
        selected_rest = self._features_rest[selected_pts_mask]
        new_features_dc = selected_dc.repeat(N, 1, 1)
        new_features_rest = selected_rest.repeat(N, 1, 1)

        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_object_ids = self._object_ids[selected_pts_mask].repeat(N)

        parent_indices = selected_pts_mask.nonzero(as_tuple=True)[0].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,
                                new_scaling, new_rotation, new_tmp_radii, new_object_ids, parent_indices=parent_indices)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_split_background(self, grads, grad_threshold, scene_extent, N=2, bg_mask=None):
        bg_grads = grads.clone()
        bg_grads[~bg_mask] = 0  # zero-out non-bg grads
        self.densify_and_split(bg_grads, grad_threshold, scene_extent, N)
        del bg_grads


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]    # (M, 1, 3)
        new_features_rest = self._features_rest[selected_pts_mask]  # (M, 15, 3)
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_object_ids = self._object_ids[selected_pts_mask]

        parent_indices = selected_pts_mask.nonzero(as_tuple=True)[0]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,
                                new_scaling, new_rotation, new_tmp_radii, new_object_ids, parent_indices=parent_indices)



    def densify_and_clone_background(self, grads, grad_threshold, scene_extent, bg_mask):
        bg_grads = grads.clone()
        bg_grads[~bg_mask] = 0
        self.densify_and_clone(bg_grads, grad_threshold, scene_extent)
        del bg_grads


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.tmp_radii = None

        torch.cuda.empty_cache()



    def densify_and_prune_background(self, max_grad, min_opacity, extent, max_screen_size, radii, backgroundId, bg_mask,
                                      hole_region_min=None, hole_region_max=None):
        """
        Densify and prune background Gaussians safely.

        hole_region_min/max (optional, both None by default -- fully
        backward compatible with every other caller of this method):
        an axis-aligned box in world space. When given, the final pruning
        step below also requires a point to fall inside this box, on top
        of the existing backgroundId check -- callers that want to leave
        background Gaussians OUTSIDE some region (e.g. far from a removed
        object) completely untouched by pruning, not just by densify/
        split, should pass one. The clone/split calls just above already
        respect `bg_mask` as given by the caller, so if the caller already
        pre-restricted `bg_mask` to that same region, this only needs to
        additionally gate the final prune_mask -- which recomputes its own
        bg_mask fresh (index alignment changes after clone/split), so the
        caller's pre-restriction alone doesn't reach this part.
        """
        # Compute full gradients
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Zero out non-background gradients
        bg_grads = grads.clone()
        bg_grads[~bg_mask] = 0.0

        # `radii` is already full-length and index-aligned with self._xyz --
        # Stage 3 renders the WHOLE Gaussian set with FG opacity temporarily
        # zeroed (not a bg-only slice) specifically to avoid IndexBackward
        # overhead, so every Gaussian (FG included) gets an entry here, not
        # just background ones. Scattering it into bg_mask indices (as if
        # `radii` were bg-only-length) throws a shape mismatch the first
        # time this ever actually ran -- this path was unreachable before
        # since --refine_allow_densify defaulted to False.
        full_tmp_radii = radii.to(torch.float32)
        self.tmp_radii = full_tmp_radii

        # Densify only background Gaussians
        self.densify_and_clone_background(bg_grads, max_grad, extent, bg_mask)
        self.densify_and_split_background(bg_grads, max_grad, extent, bg_mask=bg_mask,)

        # Build pruning mask for background. Recomputed fresh (not reusing
        # the caller's `bg_mask` argument) because clone/split above just
        # changed self._xyz's length -- the old mask is the wrong size now.
        bg_mask = self._object_ids == backgroundId
        if hole_region_min is not None and hole_region_max is not None:
            in_region = ((self._xyz >= hole_region_min) & (self._xyz <= hole_region_max)).all(dim=1)
            bg_mask = bg_mask & in_region
        prune_mask = (self.get_opacity < min_opacity).squeeze() & bg_mask
        if max_screen_size is not None and max_screen_size > 0:
            big_points_vs = (self.max_radii2D > max_screen_size) & bg_mask
            big_points_ws = (self.get_scaling.max(dim=1).values > 0.1 * extent) & bg_mask
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # Apply pruning
        self.prune_points(prune_mask)

        # Cleanup
        self.tmp_radii = None
        del full_tmp_radii
        del grads
        del bg_grads
        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def mask(self, mask_tensor):
        """Return a new GaussianModel containing only masked elements."""
        masked = GaussianModel(self.max_sh_degree, self.optimizer_type)
        masked.active_sh_degree = self.active_sh_degree
        masked._xyz = self._xyz[mask_tensor]
        masked._object_ids = self._object_ids[mask_tensor]
        masked._features_dc = self._features_dc[mask_tensor]
        masked._features_rest = self._features_rest[mask_tensor]
        masked._scaling = self._scaling[mask_tensor]
        masked._rotation = self._rotation[mask_tensor]
        masked._opacity = self._opacity[mask_tensor]
        masked.spatial_lr_scale = self.spatial_lr_scale
        masked.optimizer = None
        return masked
