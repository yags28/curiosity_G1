"""Visual RND curiosity on egocentric depth images + graspability classification.

Intrinsic reward = ||predictor(depth) - target(depth)||^2   (RND, normalised)
                 + grasp_bonus * P(graspable | depth)

Graspability head is trained jointly with the predictor via a depth heuristic:
an object is "graspable" when depth pixels in [NEAR, FAR] metres span a blob
of REACHABLE pixel area — i.e., something arm-sized is within reach.

No joint state used for object detection; all classification comes from the
64×64 depth image rendered by the robot's head camera.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.visual_encoder import DepthEncoder, GraspabilityHead
from src.curiosity.rnd import RunningMeanStd

# Graspability heuristic thresholds (depth in metres, pixel counts at 64×64).
_NEAR_M  = 0.30   # closer = robot's own body parts
_FAR_M   = 0.80   # farther = beyond arm reach
_MIN_PIX = 5      # fewer = sensor noise
_MAX_PIX = 500    # more = floor / wall / large background surface


def _graspability_heuristic(depth: torch.Tensor) -> torch.Tensor:
    """
    (N, H, W) depth → (N,) binary graspability label.
    True when a reachable, appropriately-sized object blob is visible.
    """
    in_range = (depth > _NEAR_M) & (depth < _FAR_M)        # (N, H, W)
    count    = in_range.float().sum(dim=(-1, -2))            # (N,)
    return ((count >= _MIN_PIX) & (count <= _MAX_PIX)).float()


class VisualRNDModule(nn.Module):
    """
    RND curiosity operating purely on depth images.

    target   : frozen random CNN — never updated
    predictor: trained CNN — chases target, high error = novel scene
    grasp_head: binary classifier on predictor embedding, trained via heuristic

    compute_reward(depth) → (N,) intrinsic reward
    predictor_loss(depth) → scalar training loss (RND MSE + graspability BCE)
    """

    def __init__(
        self,
        img_h: int = 64,
        img_w: int = 64,
        out_dim: int = 256,
        grasp_bonus: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        self.device      = device
        self.grasp_bonus = grasp_bonus

        self.target = DepthEncoder(img_h, img_w, out_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor  = DepthEncoder(img_h, img_w, out_dim).to(device)
        self.grasp_head = GraspabilityHead(out_dim).to(device)

        self.reward_rms = RunningMeanStd(1, device)

        # Depth normalisation is baked into DepthEncoder; obs_rms is a no-op.
        class _Noop:
            def update(self, x):
                pass
        self.obs_rms = _Noop()

    def predictor_parameters(self):
        return list(self.predictor.parameters()) + list(self.grasp_head.parameters())

    @torch.no_grad()
    def compute_reward(self, depth: torch.Tensor) -> torch.Tensor:
        """(N,) intrinsic reward = RND novelty + graspability bonus."""
        emb_pred = self.predictor(depth)
        emb_tgt  = self.target(depth)
        raw = ((emb_pred - emb_tgt) ** 2).sum(-1)          # (N,)
        self.reward_rms.update(raw.unsqueeze(-1))
        rnd_rew = raw / (self.reward_rms.var.sqrt().squeeze() + 1e-8)

        grasp_prob = self.grasp_head(emb_pred)              # (N,) in [0, 1]
        return rnd_rew + self.grasp_bonus * grasp_prob

    def predictor_loss(self, depth: torch.Tensor) -> torch.Tensor:
        """RND predictor MSE + graspability BCE on heuristic labels."""
        emb_pred = self.predictor(depth)
        with torch.no_grad():
            emb_tgt = self.target(depth)

        rnd_loss = ((emb_pred - emb_tgt) ** 2).mean()

        grasp_prob   = self.grasp_head(emb_pred)
        grasp_labels = _graspability_heuristic(depth).to(self.device)
        grasp_loss   = F.binary_cross_entropy(grasp_prob, grasp_labels)

        return rnd_loss + grasp_loss
