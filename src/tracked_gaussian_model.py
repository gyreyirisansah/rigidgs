"""GaussianModel subclass that tracks the TRUE (uncorrupted) label of every
Gaussian alongside its frozen (possibly corrupted) label, surviving prune and
densify/split exactly like the base class's `_object_ids` and
`mask_fg_evidence`/`mask_bg_evidence`/`mask_seen_evidence` buffers already do.

Why this exists: for the label-noise experiment we need to answer "what
actually happens, over training, to the Gaussians whose initial label was
deliberately corrupted?" — do they get opacity-suppressed, does densification
prune them, does the brief pre-freeze learnable-logit window already
self-correct some of them? Answering this requires knowing, at any iteration,
which currently-alive Gaussians (including splits/clones born after
corruption was injected) still carry a wrong frozen label. Indices alone
can't do this — prune_points() compacts indices and densify_and_split/clone
appends new ones — so this needs the same lineage-propagation treatment the
base class already gives `_object_ids`.

Rather than modify the shared external/gaussian-splatting/scene/gaussian_model.py
(used by every trainer variant), this subclass overrides only the two hook
points that already exist for that exact purpose (`prune_points`,
`densification_postfix`) and adds one more override (`densify_and_prune`) to
report per-step clone/split/prune counts, since the base method doesn't
expose them. All three overrides call `super()` first and only ever *add*
bookkeeping — normal (non-traced) behavior is untouched.

Does not change GaussianModel's public behavior in any way when
set_true_label_ids() is never called (self._true_label_ids stays None and
every hook becomes a no-op past the super() call).
"""
import torch

from scene.gaussian_model import GaussianModel


class TrackedGaussianModel(GaussianModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._true_label_ids = None
        # Populated by densify_and_prune() every time it's called; None before
        # the first call. Read by the trainer right after the call to log it.
        self.last_densify_stats = None

    def set_true_label_ids(self, true_label_ids):
        """true_label_ids: 1D LongTensor, same length/order as self._object_ids
        at the time of the call (i.e. right after create_from_pcd(), before
        any densification). Represents the un-corrupted COLMAP-init label —
        NOT necessarily equal to self._object_ids, which may already hold
        injected corruption via --object_ids_override."""
        true_label_ids = true_label_ids.to(device=self._object_ids.device, dtype=torch.long)
        assert true_label_ids.shape[0] == self._object_ids.shape[0], (
            f"true_label_ids length {true_label_ids.shape[0]} != "
            f"current gaussian count {self._object_ids.shape[0]}")
        self._true_label_ids = true_label_ids

    # ------------------------------------------------------------------
    # Lineage-propagation hooks — mirror how the base class already
    # threads _object_ids / mask_*_evidence through prune and densify.
    # ------------------------------------------------------------------
    def prune_points(self, mask):
        valid_points_mask = ~mask
        super().prune_points(mask)
        if self._true_label_ids is not None:
            self._true_label_ids = self._true_label_ids[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                               new_opacities, new_scaling, new_rotation, new_tmp_radii,
                               new_object_ids=None, parent_indices=None):
        # self._true_label_ids is untouched by super() (it knows nothing
        # about it), so it is still at its pre-densify length here, matching
        # the parent_indices' valid range — same invariant the base class
        # relies on in _sync_mask_evidence_after_densify.
        if self._true_label_ids is not None:
            n_new = new_xyz.shape[0]
            if parent_indices is not None and len(parent_indices) == n_new:
                parent_indices = parent_indices.to(
                    device=self._true_label_ids.device, dtype=torch.long)
                new_true_labels = self._true_label_ids[parent_indices]
            else:
                # No lineage info (shouldn't happen for split/clone, which
                # always pass parent_indices) — fall back to "unknown"
                # by copying the corrupted label so the mismatch check
                # conservatively reads these as "not corrupted".
                new_true_labels = new_object_ids if new_object_ids is not None \
                    else torch.zeros(n_new, dtype=torch.long, device=self._true_label_ids.device)

        super().densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities,
            new_scaling, new_rotation, new_tmp_radii,
            new_object_ids=new_object_ids, parent_indices=parent_indices)

        if self._true_label_ids is not None:
            self._true_label_ids = torch.cat([self._true_label_ids, new_true_labels], dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """Duplicates gaussian_model.py's densify_and_prune body (as of the
        version this was written against — external/gaussian-splatting/
        scene/gaussian_model.py:801-816) purely to capture clone/split/prune
        counts, which the base method doesn't expose. If that method's logic
        ever changes, this override silently goes stale — cross-check against
        the base class if tracked runs start looking inconsistent with
        untracked ones."""
        n_start = self.get_xyz.shape[0]

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        n_after_clone = self.get_xyz.shape[0]
        self.densify_and_split(grads, max_grad, extent)
        n_after_split = self.get_xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        n_end = self.get_xyz.shape[0]
        self.tmp_radii = None

        torch.cuda.empty_cache()

        self.last_densify_stats = {
            "n_start": n_start,
            "n_cloned": n_after_clone - n_start,
            "n_split": n_after_split - n_after_clone,
            "n_pruned": n_after_split - n_end,
            "n_end": n_end,
        }
        return self.last_densify_stats

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_tracking_snapshot(self, background_id, opacity_threshold, effective_labels=None):
        """Returns a dict of scalars summarizing, for the currently-alive
        Gaussians, how many still carry a frozen label that disagrees with
        their true (uncorrupted) label, and what's happened to those Gaussians'
        opacity. If effective_labels is given (e.g. from
        _get_hard_labels_for_mask(), which returns the pre-freeze learnable-
        logit argmax when logits are still live), also reports what fraction
        of still-mislabeled Gaussians have ALREADY self-corrected at the
        rendering level even though their frozen _object_ids hasn't.

        Returns None if set_true_label_ids() was never called (nothing to
        report)."""
        if self._true_label_ids is None:
            return None

        n_total = self._object_ids.shape[0]
        mismatch = self._object_ids != self._true_label_ids
        n_mismatch = int(mismatch.sum().item())
        opacity = self.get_opacity.detach().view(-1)

        _PCTS = (0.10, 0.25, 0.50, 0.75, 0.90)

        def _opacity_stats(sel_mask):
            n = int(sel_mask.sum().item())
            if n == 0:
                stats = {"n": 0, "opacity_mean": float("nan"), "opacity_median": float("nan"),
                         "frac_suppressed": float("nan")}
                for p in _PCTS:
                    stats[f"opacity_p{int(p * 100)}"] = float("nan")
                return stats
            op = opacity[sel_mask]
            stats = {
                "n": n,
                "opacity_mean": float(op.mean().item()),
                "opacity_median": float(op.median().item()),
                "frac_suppressed": float((op < opacity_threshold).float().mean().item()),
            }
            # Finer-grained distribution for the populations under scrutiny
            # (mismatched / fg2bg_like / bg2fg_like) — mean/median alone can
            # hide a bimodal split (some recovering, some crushed).
            qs = torch.quantile(op, torch.tensor(_PCTS, device=op.device, dtype=op.dtype))
            for p, q in zip(_PCTS, qs.tolist()):
                stats[f"opacity_p{int(p * 100)}"] = float(q)
            return stats

        snap = {"n_total": n_total, "n_mismatch": n_mismatch}
        snap["mismatch"] = _opacity_stats(mismatch)
        snap["match"] = _opacity_stats(~mismatch)

        # Split mismatch by direction: true label was foreground but frozen
        # says background (fg2bg-style), vs true label background but frozen
        # says foreground (bg2fg-style). Meaningful even for a symmetric-
        # scheme run, since it's derived per-Gaussian from the true/frozen
        # label pair, not from which flip_labels.py scheme produced the file.
        true_is_fg = self._true_label_ids != int(background_id)
        frozen_is_fg = self._object_ids != int(background_id)
        fg2bg_now = mismatch & true_is_fg & ~frozen_is_fg
        bg2fg_now = mismatch & ~true_is_fg & frozen_is_fg
        snap["fg2bg_like"] = _opacity_stats(fg2bg_now)
        snap["bg2fg_like"] = _opacity_stats(bg2fg_now)

        if effective_labels is not None:
            effective_labels = effective_labels.to(device=self._object_ids.device)
            effective_is_fg = effective_labels != int(background_id)
            # Among currently-mismatched-by-frozen-label Gaussians, how many
            # are already rendered under the TRUE label because the (pre-
            # freeze) learnable logits already disagree with the frozen id?
            already_self_corrected = mismatch & (effective_is_fg == true_is_fg)
            snap["frac_mismatch_self_corrected_by_logits"] = (
                float(already_self_corrected.sum().item()) / n_mismatch
                if n_mismatch > 0 else float("nan"))
        else:
            snap["frac_mismatch_self_corrected_by_logits"] = float("nan")

        return snap
