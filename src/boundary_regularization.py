"""
VR-RigidGS 3D Boundary Regularization  —  CLEAN VERSION

Changes from `boundary_reg_3d3_gpt.py`:

1. Boundary detection now uses LEARNED hard labels from
   `gaussians.obj_assignment.get_object_ids()` (argmax of trainable logits)
   instead of the frozen `gaussians._object_ids`. This means the regularizer
   tracks the mask BCE's corrections, not COLMAP's noisy initialization.

2. NEW loss term: ambiguity-scale pressure. For every Gaussian, multiply
   its uncertainty (1 - max(softmax)) by its max scale. Minimizing this
   shrinks Gaussians whose label is uncertain. This is the COB-GS-style
   boundary-Gaussian shrinking, done continuously (no explicit splitting),
   so 3DGS densification handles the resulting "shrink → high gradient →
   densify" cycle naturally.

3. The flatten/shrink/repulsion losses are unchanged in formulation. They
   just apply to a boundary set defined by *learned* labels.

Notes:
  - When a Gaussian's learned logits have drifted, densify_and_split assigns
    children the parent's GT id (the legacy `_object_ids`), and
    obj_assignment.on_densification_postfix re-initializes their logits to
    fresh GT-aligned ones. This RESETS the parent's learned correction in
    the children. Minor convergence speed issue, not correctness. Fixable
    by inheriting parent's logits in `obj_assignment.on_densification_postfix`,
    deferred for now.
  - The cleanup helpers at the bottom (`prune_boundary_floaters`,
    `cleanup_stray_gaussians`) still work on frozen `_object_ids` because
    they reclassify in place. With learnable logits, training itself
    handles reclassification — these are now mostly redundant. Kept for
    optional post-training cleanup.
"""

import torch
import torch.nn.functional as F
import numpy as np


# =====================================================================
# Helpers
# =====================================================================
def _get_hard_labels(gaussians):
    """Prefer learned argmax labels; fall back to frozen _object_ids."""
    assignment = getattr(gaussians, "obj_assignment", None)
    if (assignment is not None
            and getattr(assignment, "_object_logits", None) is not None):
        return assignment.get_object_ids()
    return gaussians._object_ids


def _get_probs(gaussians):
    """Per-Gaussian softmax probabilities (N, K). None if logits absent."""
    assignment = getattr(gaussians, "obj_assignment", None)
    if (assignment is not None
            and getattr(assignment, "_object_logits", None) is not None):
        return assignment.get_object_probs()
    return None


# =====================================================================
# Boundary detection (unchanged logic, now reads learned IDs)
# =====================================================================
def find_boundary_info(xyz, object_ids, k=16, min_disagreement=2,
                       pair_sample_size=50000):
    """
    Find boundary Gaussians, their boundary normals, and deduplicated
    cross-class pairs. Operates on whatever labels are passed in.

    Returns:
        boundary_mask:    (N,) bool tensor
        boundary_indices: (M,) long indices
        boundary_normals: (M, 3) unit normals pointing toward the other class
        cross_pairs:      (P, 2) deduplicated cross-class pair indices
    """
    # PERFORMANCE NOTE -- measured, scripts/benchmark_boundary_knn.py
    #
    # This function was ~68% Python loops, not kD-tree. On 125k points:
    #     cKDTree build+query      46 ms   26%
    #     disagreement (numpy)     12 ms    7%
    #     normals loop            (87 ms)  50%   <- per boundary point
    #     cross-pairs loop        (28 ms)  16%   <- per (point, neighbour)
    # Vectorising the two loops onto the GPU takes the whole function from
    # 174 ms to 52 ms (3.37x), bit-exact against the loop version.
    #
    # The kD-tree is deliberately KEPT. faiss GpuIndexFlatL2 was tried and
    # is *slower* -- 249 ms -- because a flat index is brute force, O(N^2),
    # which for k=16 in 3 dimensions loses badly to a tree and degrades as
    # the cloud densifies. Keeping scipy also keeps the neighbour lists
    # identical, so there is no tie-ordering drift to reason about.
    from scipy.spatial import cKDTree

    device = xyz.device
    N = xyz.shape[0]
    # Detached throughout: the original derived normals from a numpy copy,
    # so they carry no gradient. Vectorising against the live tensor would
    # quietly make them differentiable and change what the flatten term
    # optimises.
    xyz_d = xyz.detach()
    xyz_cpu = xyz_d.cpu().numpy()

    tree = cKDTree(xyz_cpu)
    _, knn_np = tree.query(xyz_cpu, k=k + 1, workers=-1)
    knn_idx = torch.from_numpy(knn_np[:, 1:].astype(np.int64)).to(device)  # drop self

    ids = object_ids.detach()
    differs = ids[knn_idx] != ids[:, None]                     # (N, k)
    boundary_indices = (differs.sum(dim=1) >= min_disagreement).nonzero(as_tuple=True)[0]

    boundary_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if boundary_indices.numel() == 0:
        return (boundary_mask,
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, 3, dtype=torch.float, device=device),
                torch.zeros(0, 2, dtype=torch.long, device=device))
    boundary_mask[boundary_indices] = True

    bknn = knn_idx[boundary_indices]                           # (M, k)
    bdiff = differs[boundary_indices]                          # (M, k)

    # Boundary normals: mean direction toward cross-class neighbours.
    # Zeroing the agreeing neighbours before summing makes the masked mean
    # a plain sum/count, which avoids a ragged gather.
    dirs = (xyz_d[bknn] - xyz_d[boundary_indices][:, None, :]) * bdiff[..., None]
    avg = dirs.sum(dim=1) / bdiff.sum(dim=1, keepdim=True).clamp(min=1)
    norm = avg.norm(dim=1, keepdim=True)
    boundary_normals = torch.where(
        norm > 1e-8, avg / norm.clamp(min=1e-20),
        torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(avg),
    ).float()

    # Deduplicated cross-class pairs for repulsion. Sorting each pair
    # (lo, hi) before unique is what the min/max into a Python set did.
    rows = boundary_indices[:, None].expand_as(bknn)[bdiff]
    cols = bknn[bdiff]
    cross_pairs = torch.unique(
        torch.stack([torch.minimum(rows, cols), torch.maximum(rows, cols)], dim=1),
        dim=0,
    )
    if len(cross_pairs) > pair_sample_size:
        perm = torch.randperm(len(cross_pairs), device=device)[:pair_sample_size]
        cross_pairs = cross_pairs[perm]

    return boundary_mask, boundary_indices, boundary_normals, cross_pairs


# =====================================================================
# Main regularization loss
# =====================================================================
def boundary_regularization_loss(
    gaussians,
    k=16,
    min_disagreement=2,
    pair_sample_size=50000,
    flatten_weight=1.0,
    shrink_weight=1.0,
    repulsion_weight=1.0,
    repulsion_margin=0.0,
    max_scale_target=None,
    # NEW knobs:
    ambig_scale_weight=0.0,
    ambig_scale_power=1.0,
):
    """
    Boundary regularization for crisp object separation.

    Components:
      1. Oriented Flatten   — minimize Gaussian variance along boundary normal
      2. Shrink             — cap max scale at boundaries
      3. Repulsion          — push cross-class pairs apart by `repulsion_margin`
      4. Ambig-Scale        — shrink Gaussians proportional to their
                              label uncertainty (1 - max(softmax))

    All terms operate on LEARNED hard labels (argmax of logits) so they
    track the mask BCE's corrections of COLMAP's noisy initialization.

    Args:
        gaussians: GaussianModel with obj_assignment._object_logits set.
        k, min_disagreement, pair_sample_size: kNN boundary detection params.
        flatten_weight, shrink_weight, repulsion_weight: existing terms.
        repulsion_margin: target gap distance; auto-calibrates if <=0.
        max_scale_target: target max scale; defaults to 0.5 * median.
        ambig_scale_weight: weight for the ambiguity-scale term.
        ambig_scale_power: raise uncertainty to this power.
    """
    xyz = gaussians.get_xyz                 # (N, 3) with grad
    scales = gaussians.get_scaling          # (N, 3) with grad

    # Learned hard labels for boundary detection
    hard_ids = _get_hard_labels(gaussians)
    probs = _get_probs(gaussians)            # (N, K) with grad, or None

    boundary_mask, boundary_indices, boundary_normals, cross_pairs = \
        find_boundary_info(xyz, hard_ids, k=k,
                           min_disagreement=min_disagreement,
                           pair_sample_size=pair_sample_size)

    n_boundary = len(boundary_indices)

    # ==================================================================
    # 1. ORIENTED FLATTEN  — minimize variance along boundary normal
    #    var_n = sum_j s_j^2 * (R[:,j] . n)^2
    # ==================================================================
    if n_boundary > 0 and flatten_weight > 0:
        boundary_scales = scales[boundary_indices]                  # (M, 3)
        R_all = gaussians.get_rotation_matrix()                     # (N, 3, 3)
        R_boundary = R_all[boundary_indices]                        # (M, 3, 3)
        n = boundary_normals.unsqueeze(2)                           # (M, 3, 1)
        # Dot of each axis with normal
        axis_dots = torch.bmm(R_boundary.transpose(1, 2), n).squeeze(2)  # (M, 3)
        var_along_normal = (boundary_scales ** 2 * axis_dots ** 2).sum(dim=1)  # (M,)
        L_flatten = var_along_normal.mean()
        mean_var_along_normal = float(var_along_normal.mean().item())
    else:
        L_flatten = torch.tensor(0.0, device=xyz.device)
        mean_var_along_normal = 0.0

    # ==================================================================
    # 2. SHRINK  — penalize boundary Gaussians larger than target
    # ==================================================================
    if n_boundary > 0 and shrink_weight > 0:
        boundary_scales = scales[boundary_indices]
        max_scale = boundary_scales.max(dim=1).values
        if max_scale_target is None:
            with torch.no_grad():
                max_scale_target_val = scales.max(dim=1).values.median().item() * 0.5
        else:
            max_scale_target_val = float(max_scale_target)
        L_shrink = (max_scale - max_scale_target_val).clamp(min=0).mean()
        mean_max_scale = float(max_scale.mean().item())
    else:
        L_shrink = torch.tensor(0.0, device=xyz.device)
        max_scale_target_val = 0.0
        mean_max_scale = 0.0

    # ==================================================================
    # 3. REPULSION  — hinge loss pushing cross-class pairs apart
    # ==================================================================
    n_cross_pairs = len(cross_pairs)
    median_dist = mean_dist = min_dist = 0.0
    if n_cross_pairs > 0 and repulsion_weight > 0:
        pos_i = xyz[cross_pairs[:, 0]]
        pos_j = xyz[cross_pairs[:, 1]]
        dists = torch.norm(pos_i - pos_j, dim=1)
        if repulsion_margin is None or repulsion_margin <= 0:
            with torch.no_grad():
                repulsion_margin = dists.median().item() * 2.0
        L_repulsion = (repulsion_margin - dists).clamp(min=0).mean()
        with torch.no_grad():
            median_dist = float(dists.median().item())
            mean_dist = float(dists.mean().item())
            min_dist = float(dists.min().item())
    else:
        L_repulsion = torch.tensor(0.0, device=xyz.device)

    # ==================================================================
    # 4. AMBIG-SCALE  — shrink uncertain Gaussians (NEW)
    #
    #   uncertainty_i = 1 - max(softmax(logits_i))
    #   For binary FG/BG this peaks at 0.5 (label undecided) and is 0 when
    #   the Gaussian is confidently in one class. The loss shrinks
    #   uncertain Gaussians without affecting confident ones.
    #
    #   This is the COB-GS-style boundary-Gaussian shrinking signal, done
    #   continuously so 3DGS densification handles the response naturally
    #   (shrunk Gaussians with high gradients get split by the existing
    #    densification controller).
    # ==================================================================
    if ambig_scale_weight > 0 and probs is not None:
        max_probs = probs.max(dim=1).values         # (N,)
        uncertainty = (1.0 - max_probs)              # 0..(K-1)/K, peaks at 1/K -> 0.5 for K=2
        if ambig_scale_power != 1.0:
            uncertainty = uncertainty.pow(ambig_scale_power)
        max_scale_all = scales.max(dim=1).values     # (N,)
        L_ambig_scale = (uncertainty * max_scale_all).mean()
        mean_uncertainty = float(uncertainty.mean().item())
    else:
        L_ambig_scale = torch.tensor(0.0, device=xyz.device)
        mean_uncertainty = 0.0

    # ==================================================================
    # Combine
    # ==================================================================
    loss = (flatten_weight * L_flatten
            + shrink_weight * L_shrink
            + repulsion_weight * L_repulsion
            + ambig_scale_weight * L_ambig_scale)

    stats = {
        'n_boundary': n_boundary,
        'boundary_ratio': n_boundary / max(1, len(xyz)),
        'n_cross_pairs': n_cross_pairs,
        'L_flatten': float(L_flatten.item()),
        'L_shrink': float(L_shrink.item()),
        'L_repulsion': float(L_repulsion.item()),
        'L_ambig_scale': float(L_ambig_scale.item()),
        'mean_max_scale': mean_max_scale,
        'max_scale_target': max_scale_target_val,
        'mean_var_along_normal': mean_var_along_normal,
        'mean_uncertainty': mean_uncertainty,
        'repulsion_margin': float(repulsion_margin) if isinstance(repulsion_margin, (int, float)) else 0.0,
        'pair_dist_median': median_dist,
        'pair_dist_mean': mean_dist,
        'pair_dist_min': min_dist,
    }

    return loss, stats


# =====================================================================
# Post-training / mid-training cleanup
# =====================================================================
#
# Optional cleanup (--cleanup_at_saving), run at the final save of each
# stage. Order matters: prune floaters FIRST so the kNN-isolation check in
# stage 2 isn't fooled by floaters acting as fake "neighbors" of legitimate
# Gaussians.
#
# Stage 1: prune_3d_floaters
#   Per foreground object, removes low-opacity Gaussians far from the rest
#   of their own object (distance to kth nearest same-object neighbor >>
#   that object's median). Background is never touched; no label changes.
#
# Stage 2: relabel_isolated_gaussians
#   For surviving Gaussians, flips labels of those whose kNN neighbors are
#   mostly the opposite class. Catches FG Gaussians sitting inside a BG
#   region (mislabeled by BCE) and BG Gaussians sitting inside a FG region
#   (missed by BCE). Updates BOTH the legacy _object_ids buffer AND the
#   learnable _object_logits (so the change persists in checkpoints).
#
# postprocess_cleanup orchestrates both stages with consistent stats output.
# =====================================================================
def _knn_to_self(xyz_cpu, k):
    """Build kdtree on xyz_cpu, return (N, k) array of neighbor indices
    (self excluded), and (N, k) array of neighbor distances."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz_cpu)
    dists, idx = tree.query(xyz_cpu, k=k + 1, workers=-1)
    return idx[:, 1:], dists[:, 1:]  # drop self


def prune_3d_floaters(gaussians, background_id, k=8, distance_ratio=4.0,
                      max_opacity=0.5):
    """
    Stage 1: prune foreground Gaussians that drifted away from their object.

    Per object, label-aware: for every foreground object, take each of its
    Gaussians' distance to its kth-nearest neighbour *of the same object*,
    and prune it if that distance is more than distance_ratio times the
    object's median kth-NN distance AND its opacity is below max_opacity.
    Background Gaussians are never touched, and no label is changed -- the
    frozen identities are only ever removed together with their Gaussian.

    Why per object: densification can clone a foreground Gaussian into
    empty space next to background geometry. Measured against all
    Gaussians it looks dense (its neighbours are background), so a
    scene-wide test misses it; measured against its own object it is far
    from everything. The opacity cap keeps solid thin parts (legs, fronds)
    whose spacing is sparse but which the photometric loss keeps opaque.

    The kth-NN distance (not the 1st-NN) is used because two adjacent
    floaters would each see the other as a near neighbour -- the kth-NN
    test looks past that to whether the *cluster* is isolated.

    Args:
        gaussians: GaussianModel (labels read from the frozen _object_ids).
        background_id: integer ID of the background class (never pruned).
        k: which same-object neighbour to measure (default 8).
        distance_ratio: prune if kth-NN dist > this * object median
                        (default 4.0). Lower = more aggressive.
        max_opacity: only Gaussians with opacity below this are pruned
                     (default 0.5).

    Returns:
        stats dict: {'n_total_before', 'n_pruned', 'per_object'}
    """
    with torch.no_grad():
        N = int(gaussians.get_xyz.shape[0])
        xyz_cpu = gaussians.get_xyz.detach().cpu().numpy()
        ids = gaussians._object_ids.detach().cpu().numpy().reshape(-1)
        opacity = gaussians.get_opacity.detach().cpu().numpy().reshape(-1)

        is_floater = np.zeros(N, dtype=bool)
        per_object = {}
        for obj in np.unique(ids):
            if obj == int(background_id):
                continue
            idx = np.where(ids == obj)[0]
            if len(idx) < k + 2:        # too few Gaussians for statistics
                continue
            _, dists = _knn_to_self(xyz_cpu[idx], k=k)
            kth_dist = dists[:, -1]
            threshold = distance_ratio * float(np.median(kth_dist))
            hit = (kth_dist > threshold) & (opacity[idx] < max_opacity)
            is_floater[idx[hit]] = True
            per_object[int(obj)] = int(hit.sum())

        n_floater = int(is_floater.sum())
        if n_floater == 0:
            return {'n_total_before': N, 'n_pruned': 0, 'per_object': per_object}

        # GaussianModel.prune_points expects a mask of points TO REMOVE
        prune_mask = torch.from_numpy(is_floater).to(
            device=gaussians._xyz.device, dtype=torch.bool)

        # tmp_radii needs to exist for prune_points to work
        # if not hasattr(gaussians, "tmp_radii") or gaussians.tmp_radii is None:
        #     gaussians.tmp_radii = torch.zeros(
        #         N, device=gaussians._xyz.device)
        # elif gaussians.tmp_radii.shape[0] != N:
        #     # Stale shape from a previous prune; reset
        #     gaussians.tmp_radii = torch.zeros(
        #         N, device=gaussians._xyz.device)


        # Ensure ALL buffers prune_points touches exist and have shape (N,).
        # After densify_until_iter and other lifecycle events, some of these
        # may have been dropped — recreate as zeros of the right shape.
        device = gaussians._xyz.device
        buffers = {
            "tmp_radii":          (N,),
            "max_radii2D":        (N,),
            "xyz_gradient_accum": (N, 1),
            "denom":              (N, 1),
        }
        for name, shape in buffers.items():
            attr = getattr(gaussians, name, None)
            if attr is None or attr.shape[0] != N:
                setattr(gaussians, name, torch.zeros(shape, device=device))

        gaussians.prune_points(prune_mask)

        return {'n_total_before': N, 'n_pruned': n_floater, 'per_object': per_object}


def relabel_isolated_gaussians(gaussians, background_id, k=16,
                                isolation_threshold=0.8,
                                also_flip_bg_to_fg=True,
                                logit_flip_confidence=5.0):
    """
    Stage 2: flip labels of Gaussians surrounded by the opposite class.

    Two complementary flips:
      (a) FG Gaussians whose neighbors are mostly BG → flip to BG
          (catches BCE-mislabeled specks: foreground in space where there's
          no foreground)
      (b) BG Gaussians whose neighbors are mostly FG → flip to FG
          (catches missed fronds: background in space where the fern is)

    Updates BOTH:
      - The legacy `_object_ids` buffer (so frozen-label code paths see it)
      - The learnable `_object_logits` (so the flip persists in checkpoints
        and survives further training)

    Args:
        gaussians: GaussianModel with obj_assignment._object_logits set.
        background_id: integer ID of the background class.
        k: neighbors to inspect (default 16).
        isolation_threshold: fraction of neighbors that must be the
                             opposite class to trigger flip (default 0.8).
                             Higher = more conservative (less flipping).
        also_flip_bg_to_fg: if True, also do the reverse flip (BG→FG when
                            neighbors are mostly FG). Default True.
        logit_flip_confidence: target logit value for the new class after
                                a flip (default 5.0). The opposite class
                                gets value 0.

    Returns:
        stats dict.
    """
    with torch.no_grad():
        N = int(gaussians.get_xyz.shape[0])
        if N < k + 2:
            return {'n_total': N, 'n_flipped_fg_to_bg': 0,
                    'n_flipped_bg_to_fg': 0, 'skipped': 'too_few_gaussians'}

        xyz_cpu = gaussians.get_xyz.detach().cpu().numpy()

        # Use the CURRENT labels (learned argmax if logits exist, else legacy)
        assignment = getattr(gaussians, "obj_assignment", None)
        has_logits = (assignment is not None
                      and getattr(assignment, "_object_logits", None) is not None)
        if has_logits:
            ids_t = assignment.get_object_ids()
        else:
            ids_t = gaussians._object_ids
        ids_cpu = ids_t.detach().cpu().numpy()

        bg_id_int = int(background_id)
        is_fg = (ids_cpu != bg_id_int)
        is_bg = ~is_fg

        knn_idx, _ = _knn_to_self(xyz_cpu, k=k)
        neighbor_ids = ids_cpu[knn_idx]  # (N, k)

        frac_neighbors_bg = (neighbor_ids == bg_id_int).mean(axis=1)
        frac_neighbors_fg = 1.0 - frac_neighbors_bg

        # Flip (a): FG with mostly BG neighbors → BG
        flip_fg_to_bg = is_fg & (frac_neighbors_bg > isolation_threshold)
        # Flip (b): BG with mostly FG neighbors → FG
        flip_bg_to_fg = (is_bg & (frac_neighbors_fg > isolation_threshold)
                          if also_flip_bg_to_fg
                          else np.zeros_like(flip_fg_to_bg))

        n_flip_fg2bg = int(flip_fg_to_bg.sum())
        n_flip_bg2fg = int(flip_bg_to_fg.sum())

        if n_flip_fg2bg == 0 and n_flip_bg2fg == 0:
            return {'n_total': N, 'n_flipped_fg_to_bg': 0,
                    'n_flipped_bg_to_fg': 0,
                    'isolation_threshold': float(isolation_threshold)}

        device = gaussians._xyz.device

        # ---- Update legacy _object_ids ----
        if n_flip_fg2bg > 0:
            idx = torch.from_numpy(np.where(flip_fg_to_bg)[0]).long().to(device)
            gaussians._object_ids[idx] = bg_id_int

        if n_flip_bg2fg > 0:
            idx_bg2fg = np.where(flip_bg_to_fg)[0]
            # For each flipped BG Gaussian, set its label to the majority
            # FG class among its neighbors (or class 0 if multi-class).
            fg_neighbor_ids = neighbor_ids[idx_bg2fg]  # (M, k)
            new_labels = np.zeros(len(idx_bg2fg), dtype=ids_cpu.dtype)
            for i in range(len(idx_bg2fg)):
                neigh_fg = fg_neighbor_ids[i][fg_neighbor_ids[i] != bg_id_int]
                if len(neigh_fg) > 0:
                    classes, counts = np.unique(neigh_fg, return_counts=True)
                    new_labels[i] = int(classes[counts.argmax()])
                else:
                    new_labels[i] = bg_id_int  # shouldn't happen given the filter
            idx_t = torch.from_numpy(idx_bg2fg).long().to(device)
            new_labels_t = torch.from_numpy(new_labels).long().to(device)
            gaussians._object_ids[idx_t] = new_labels_t

        # ---- Update learnable logits (if present) ----
        if has_logits:
            K = assignment._object_logits.shape[1]
            if n_flip_fg2bg > 0:
                idx = torch.from_numpy(np.where(flip_fg_to_bg)[0]).long().to(device)
                # Zero out, then put `logit_flip_confidence` at BG class
                assignment._object_logits.data[idx] = 0.0
                assignment._object_logits.data[idx, bg_id_int] = float(logit_flip_confidence)
            if n_flip_bg2fg > 0:
                idx_bg2fg = np.where(flip_bg_to_fg)[0]
                fg_neighbor_ids = neighbor_ids[idx_bg2fg]
                new_labels = np.zeros(len(idx_bg2fg), dtype=ids_cpu.dtype)
                for i in range(len(idx_bg2fg)):
                    neigh_fg = fg_neighbor_ids[i][fg_neighbor_ids[i] != bg_id_int]
                    if len(neigh_fg) > 0:
                        classes, counts = np.unique(neigh_fg, return_counts=True)
                        new_labels[i] = int(classes[counts.argmax()])
                    else:
                        new_labels[i] = bg_id_int
                idx_t = torch.from_numpy(idx_bg2fg).long().to(device)
                # Clamp new_labels to valid range [0, K-1]
                new_labels = np.clip(new_labels, 0, K - 1)
                new_labels_t = torch.from_numpy(new_labels).long().to(device)
                assignment._object_logits.data[idx_t] = 0.0
                assignment._object_logits.data[idx_t, new_labels_t] = float(logit_flip_confidence)
            # Keep obj_ids_gt in sync? NO — it's GT, leave it as the COLMAP
            # original. Only the learned logits and the legacy ids change.

        return {
            'n_total': N,
            'n_flipped_fg_to_bg': n_flip_fg2bg,
            'n_flipped_bg_to_fg': n_flip_bg2fg,
            'isolation_threshold': float(isolation_threshold),
        }


def postprocess_cleanup(gaussians, background_id,
                         # floater pruning
                         floater_k=8, floater_distance_ratio=4.0,
                         floater_max_opacity=0.5,
                         # label relabeling
                         relabel_k=16, isolation_threshold=0.8,
                         also_flip_bg_to_fg=True,
                         logit_flip_confidence=5.0,
                         verbose=True):
    """
    Run the two-stage cleanup in the right order. Call this from the
    trainer at every saving_iterations point, before scene.save() and
    before FG/BG renders are produced.

    Returns a combined stats dict.
    """
    if verbose:
        print(f"\n[Cleanup] Starting two-stage cleanup "
              f"(N={int(gaussians.get_xyz.shape[0])}, bg_id={background_id})")

    # Stage 1: prune floaters
    s1 = prune_3d_floaters(
        gaussians,
        background_id=background_id,
        k=floater_k,
        distance_ratio=floater_distance_ratio,
        max_opacity=floater_max_opacity,
    )
    if verbose:
        print(f"[Cleanup] Stage 1 (per-object floater prune): "
              f"{s1['n_pruned']:,} / {s1['n_total_before']:,} pruned, "
              f"per object id {s1['per_object']}")

    # Stage 2: relabel isolated Gaussians
    # ----------------------------------------------------------------
    # DISABLED per A/B experimental decision: we observed that
    # `relabel_isolated_gaussians` was the main source of fern-colored
    # "debris" in the BG-only render. It flips FG Gaussians sitting on
    # sparse frond edges to BG (because most of their kNN neighbors are
    # genuinely BG), and once they're labeled BG with fern-colored SH
    # coefficients, they show up as floating debris in the BG hole render.
    # The function below is kept defined so you can re-enable it by
    # uncommenting the call.
    # ----------------------------------------------------------------
    # s2 = relabel_isolated_gaussians(
    #     gaussians,
    #     background_id=background_id,
    #     k=relabel_k,
    #     isolation_threshold=isolation_threshold,
    #     also_flip_bg_to_fg=also_flip_bg_to_fg,
    #     logit_flip_confidence=logit_flip_confidence,
    # )
    s2 = {'n_total': int(gaussians.get_xyz.shape[0]),
          'n_flipped_fg_to_bg': 0,
          'n_flipped_bg_to_fg': 0,
          'isolation_threshold': float(isolation_threshold),
          'skipped': 'disabled_by_postprocess_cleanup'}
    if verbose:
        if s2.get('skipped'):
            print(f"[Cleanup] Stage 2 (relabel isolated): SKIPPED ({s2['skipped']})")
        else:
            print(f"[Cleanup] Stage 2 (relabel isolated): "
                  f"FG→BG {s2['n_flipped_fg_to_bg']:,}, "
                  f"BG→FG {s2['n_flipped_bg_to_fg']:,}, "
                  f"threshold={s2['isolation_threshold']}")
            print(f"[Cleanup] Final N = {int(gaussians.get_xyz.shape[0]):,}")

    return {'floater_prune': s1, 'relabel_isolated': s2}
