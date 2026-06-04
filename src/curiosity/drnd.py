"""Distributed/Ensemble RND (DRND) curiosity module.

Uses N independent (target, predictor) pairs.  Intrinsic reward = mean squared
prediction error across the ensemble, normalised by a running std.  Ensemble
diversity provides better state-space coverage than single-pair RND.
"""

import torch
import torch.nn as nn

from src.utils.networks import layer_init
from src.curiosity.rnd import RunningMeanStd


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = [layer_init(nn.Linear(in_dim, hidden_dim)), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()]
    layers.append(layer_init(nn.Linear(hidden_dim, out_dim)))
    return nn.Sequential(*layers)


class DRNDModule(nn.Module):
    """
    DRND curiosity module.

    Intrinsic reward = mean_k ||predictor_k(norm_obs) - target_k(norm_obs)||²,
    then normalised by a running std of past rewards.
    """

    def __init__(
        self,
        obs_dim: int,
        n_ensemble: int = 5,
        output_dim: int = 64,
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.n_ensemble = n_ensemble

        # Frozen random target networks (one per ensemble member)
        self.targets = nn.ModuleList([
            _mlp(obs_dim, hidden_dim, output_dim, depth=1) for _ in range(n_ensemble)
        ])
        for t in self.targets:
            for p in t.parameters():
                p.requires_grad = False

        # Trained predictor networks (deeper than targets, like RND)
        self.predictors = nn.ModuleList([
            _mlp(obs_dim, hidden_dim, output_dim, depth=2) for _ in range(n_ensemble)
        ])

        self.targets.to(device)
        self.predictors.to(device)

        self.obs_rms    = RunningMeanStd(obs_dim, device)
        self.reward_rms = RunningMeanStd(1, device)

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.obs_rms.mean) / (self.obs_rms.var.sqrt() + 1e-8)).clamp(-5.0, 5.0)

    def predictor_parameters(self):
        return self.predictors.parameters()

    @torch.no_grad()
    def compute_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """(N,) normalised intrinsic reward. Caller is responsible for updating obs_rms."""
        obs_n = self._norm_obs(obs)
        errors = torch.stack([
            ((pred(obs_n) - tgt(obs_n)) ** 2).sum(-1)
            for pred, tgt in zip(self.predictors, self.targets)
        ])  # (n_ensemble, N)
        raw = errors.mean(0)  # (N,)
        self.reward_rms.update(raw.unsqueeze(-1))
        return raw / (self.reward_rms.var.sqrt().squeeze() + 1e-8)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        obs_n = self._norm_obs(obs).detach()
        total = torch.tensor(0.0, device=self.device)
        for pred, tgt in zip(self.predictors, self.targets):
            with torch.no_grad():
                feat = tgt(obs_n)
            total = total + ((pred(obs_n) - feat) ** 2).mean()
        return total / self.n_ensemble
