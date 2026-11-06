import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal
# class ObjectAssignmentModule(nn.Module):
#     """
#     Manages learnable object assignments for Gaussians.
    
#     Usage:
#         # Initialize
#         obj_module = ObjectAssignmentModule(num_objects=5)
        
#         # After SAM projection gives you initial IDs
#         obj_module.initialize_from_ids(initial_object_ids)
        
#         # Add to optimizer
#         optimizer.add_param_group({
#             'params': obj_module.parameters(),
#             'lr': 0.01,
#             'name': 'object_logits'
#         })
        
#         # During training
#         loss, stats = obj_module.compute_boundary_loss(
#             xyz, scales, rotations, ...
#         )
        
#         # Handle densification
#         obj_module.on_densify_split(split_mask, new_xyz)
#         obj_module.on_densify_clone(clone_mask)
#         obj_module.on_prune(prune_mask)
        
#         # Get current IDs
#         current_ids = obj_module.get_object_ids()
#     """
    
#     def __init__(self, num_objects: int, confidence: float = 10.0):
#         super().__init__()
#         self.num_objects = num_objects
#         self.confidence = confidence
        
#         # Will be initialized later
#         self.register_buffer('_object_ids_gt', None)
#         self._object_logits = None
    
#     def initialize_from_ids(self, object_ids: torch.Tensor):
#         """Initialize from SAM-projected object IDs."""
#         device = object_ids.device
#         N = object_ids.shape[0]
#         self.num_objects = int(object_ids.max().item()) + 1
        
#         # Store ground truth (frozen)
#         self._object_ids_gt = object_ids.detach().clone()
        
#         # Create learnable logits
#         logits = self.ids_to_logits(object_ids, self.num_objects, self.confidence)
#         self._object_logits = nn.Parameter(logits)
    
#     @property
#     def object_logits(self) -> torch.Tensor:
#         """Learnable logits [N, K]."""
#         return self._object_logits
    
#     @property
#     def object_ids_gt(self) -> torch.Tensor:
#         """Ground truth IDs [N]."""
#         return self._object_ids_gt
    
#     def get_object_ids(self) -> torch.Tensor:
#         """Current hard IDs from soft logits."""
#         return self._object_logits.argmax(dim=-1)
    
#     def get_object_probs(self) -> torch.Tensor:
#         """Current soft probabilities."""
#         return F.softmax(self._object_logits, dim=-1)
    
#     # def compute_boundary_loss(
#     #     self,
#     #     xyz: torch.Tensor,
#     #     scales: torch.Tensor,
#     #     rotations: torch.Tensor,
#     #     **kwargs,
#     # ) -> Tuple[torch.Tensor, Dict[str, float]]:
#     #     """Compute boundary regularization loss."""
#     #     return boundary_regularization_loss(
#     #         xyz, scales, rotations,
#     #         self._object_logits,
#     #         self._object_ids_gt,
#     #         **kwargs,
#     #     )

#     def ids_to_logits(
#         self,
#         object_ids: torch.Tensor, 
#         num_objects: int, 
#         confidence: float = 10.0,
#     ) -> torch.Tensor:
#         """
#         Convert hard object IDs to soft logits.
        
#         Args:
#             object_ids: [N] integer tensor
#             num_objects: K total objects
#             confidence: how "sure" initial assignment is (higher = more one-hot)
            
#         Returns:
#             logits: [N, K] where softmax(logits) ≈ one_hot(object_ids)
#         """
#         N = object_ids.shape[0]
#         device = object_ids.device
        
#         # Start with small noise (helps optimization)
#         logits = torch.randn(N, num_objects, device=device) * 0.01
        
#         # Set correct class to high value
#         logits.scatter_(1, object_ids.unsqueeze(1), confidence)
        
#         return logits
    
#     # === Densification Callbacks ===
    
#     def on_densify_split(self, split_mask: torch.Tensor, n_splits: int = 2):
#         """
#         Handle split densification.
#         Children inherit parent's logits and ground truth ID.
        
#         Args:
#             split_mask: [N] boolean mask of Gaussians being split
#             n_splits: number of children per parent (usually 2)
#         """
#         # Get values for split Gaussians
#         split_logits = self._object_logits[split_mask]  # [M, K]
#         split_gt = self._object_ids_gt[split_mask]       # [M]
        
#         # Repeat for children
#         new_logits = split_logits.repeat(n_splits, 1)    # [M*n_splits, K]
#         new_gt = split_gt.repeat(n_splits)               # [M*n_splits]
        
#         # Remove parents, add children
#         keep_mask = ~split_mask
#         kept_logits = self._object_logits[keep_mask]
#         kept_gt = self._object_ids_gt[keep_mask]
        
#         # Concatenate
#         self._object_logits = nn.Parameter(torch.cat([kept_logits, new_logits], dim=0))
#         self._object_ids_gt = torch.cat([kept_gt, new_gt], dim=0)
    
#     def on_densify_clone(self, clone_mask: torch.Tensor):
#         """
#         Handle clone densification.
#         Clones inherit parent's logits and ground truth ID.
        
#         Args:
#             clone_mask: [N] boolean mask of Gaussians being cloned
#         """
#         # Get values for cloned Gaussians
#         clone_logits = self._object_logits[clone_mask]
#         clone_gt = self._object_ids_gt[clone_mask]
        
#         # Append clones
#         self._object_logits = nn.Parameter(
#             torch.cat([self._object_logits, clone_logits], dim=0)
#         )
#         self._object_ids_gt = torch.cat([self._object_ids_gt, clone_gt], dim=0)
    
#     def on_prune(self, prune_mask: torch.Tensor):
#         """
#         Handle pruning.
        
#         Args:
#             prune_mask: [N] boolean mask of Gaussians to REMOVE
#         """
#         keep_mask = ~prune_mask
#         self._object_logits = nn.Parameter(self._object_logits[keep_mask])
#         self._object_ids_gt = self._object_ids_gt[keep_mask]
    
#     def on_reset_opacity(self):
#         """Handle opacity reset (no change to IDs)."""
#         pass


class ObjectAssignmentModule(nn.Module):
    """
    Manages learnable object assignments for Gaussians.
    """
    
    def __init__(self, num_objects: int, confidence: float = 3.0):
        super().__init__()
        self.num_objects = num_objects
        self.confidence = confidence
        
        self._object_ids_gt = None
        self._object_logits = None
    
    def initialize_from_ids(self, object_ids: torch.Tensor):
        """Initialize from SAM-projected object IDs."""
        device = object_ids.device
        
        # Store ground truth (frozen)
        self._object_ids_gt = object_ids.detach().clone()
        
        # Create learnable logits
        logits = self._ids_to_logits(object_ids)
        self._object_logits = nn.Parameter(logits)
    
    def _ids_to_logits(self, object_ids: torch.Tensor) -> torch.Tensor:
        """Convert hard IDs to soft logits."""
        N = object_ids.shape[0]
        device = object_ids.device

        self.num_objects = int(object_ids.max().item()) + 1
        
        logits = torch.randn(N, self.num_objects, device=device) * 0.01
        logits.scatter_(1, object_ids.unsqueeze(1), self.confidence)
        
        return logits
    
    @property
    def object_logits(self) -> torch.Tensor:
        return self._object_logits
    
    @property
    def object_ids_gt(self) -> torch.Tensor:
        return self._object_ids_gt
    
    def get_object_ids(self) -> torch.Tensor:
        """Current hard IDs from soft logits."""
        return self._object_logits.argmax(dim=-1)
    
    def get_object_probs(self) -> torch.Tensor:
        """Current soft probabilities."""
        return F.softmax(self._object_logits, dim=-1)
    
    # =========================================================
    # Densification Callbacks - FIXED to match GaussianModel
    # =========================================================
    
    def on_densification_postfix_old(self, new_object_ids: torch.Tensor):
        """
        Called during densification_postfix to ADD new Gaussians.
        Does NOT remove anything - that happens in on_prune.
        
        Args:
            new_object_ids: [M] object IDs for the new Gaussians
        """
        M = new_object_ids.shape[0]
        device = self._object_logits.device
        
        # Create logits for new Gaussians
        new_logits = self._ids_to_logits(new_object_ids)
        
        # Concatenate (don't remove anything!)
        self._object_logits = nn.Parameter(
            torch.cat([self._object_logits.data, new_logits], dim=0)
        )
        self._object_ids_gt = torch.cat([self._object_ids_gt, new_object_ids], dim=0)

    def on_densification_postfix(self, new_object_ids: torch.Tensor):
        """
        Called during densification_postfix to ADD new Gaussians.
        """
        M = new_object_ids.shape[0]
        if M == 0:
            return
        
        device = self._object_logits.device
        
        # Use ACTUAL number of classes from existing logits, not self.num_objects
        num_classes = self._object_logits.shape[1]
        
        # Create logits for new Gaussians
        new_logits = torch.randn(M, num_classes, device=device) * 0.01
        new_logits.scatter_(1, new_object_ids.unsqueeze(1).clamp(0, num_classes - 1), self.confidence)
        
        # Concatenate
        self._object_logits = nn.Parameter(
            torch.cat([self._object_logits.data, new_logits], dim=0)
        )
        self._object_ids_gt = torch.cat([self._object_ids_gt, new_object_ids], dim=0)
    
    def on_prune(self, valid_points_mask: torch.Tensor):
        """
        Called during prune_points to REMOVE Gaussians.
        
        Args:
            valid_points_mask: [N] boolean mask, True = KEEP, False = REMOVE
        """
        self._object_logits = nn.Parameter(self._object_logits.data[valid_points_mask])
        self._object_ids_gt = self._object_ids_gt[valid_points_mask]
    
    def on_reset_opacity(self):
        """Handle opacity reset (no change to IDs)."""
        pass


class ObjectAssignment2D(nn.Module):
    """
    Object assignment module for 2D mask supervision.
    Uses LOW confidence (2.0) for better gradient flow.
    """
    
    def __init__(self, num_objects: int):
        super().__init__()
        self.num_objects = num_objects
        self._object_logits = None
        self._object_ids_gt = None
    
    def initialize_from_ids(self, object_ids: torch.Tensor):
        self.num_objects = int(object_ids.max().item()) + 1
        N = object_ids.shape[0]
        device = object_ids.device
        
        self._object_ids_gt = object_ids.detach().clone()
        
        # LOW confidence = softer softmax = better gradients
        logits = torch.zeros(N, self.num_objects, device=device)
        logits.scatter_(1, object_ids.unsqueeze(1), 2.0)
        
        self._object_logits = nn.Parameter(logits)
    
    @property
    def object_logits(self):
        return self._object_logits
    
    @property
    def object_ids_gt(self):
        return self._object_ids_gt
    
    def get_object_ids(self):
        return self._object_logits.argmax(dim=-1)
    
    def get_object_probs(self):
        return F.softmax(self._object_logits, dim=-1)
    
    def on_densification_postfix(self, new_ids: torch.Tensor):
        if len(new_ids) == 0:
            return

        # Post-freeze checkpoint resume nulls _object_logits entirely (see
        # the trainer's checkpoint-loading workaround for the optimizer
        # restore() shape mismatch on an already-detached, empty
        # 'object_logits' param group). Object identity by that point is
        # fully governed by GaussianModel._object_ids (extended
        # unconditionally in densification_postfix regardless of this
        # module's state), so there's nothing meaningful to extend here.
        if self._object_logits is None:
            return

        device = self._object_logits.device
        K = self._object_logits.shape[1]

        new_logits = torch.zeros(len(new_ids), K, device=device)
        new_logits.scatter_(1, new_ids.unsqueeze(1).clamp(0, K-1), 2.0)

        self._object_logits = nn.Parameter(
            torch.cat([self._object_logits.data, new_logits], dim=0)
        )
        self._object_ids_gt = torch.cat([self._object_ids_gt, new_ids], dim=0)

    def on_prune(self, valid_mask: torch.Tensor):
        # See on_densification_postfix above -- same post-freeze-resume gap.
        if self._object_logits is None:
            return
        self._object_logits = nn.Parameter(self._object_logits.data[valid_mask])
        self._object_ids_gt = self._object_ids_gt[valid_mask]