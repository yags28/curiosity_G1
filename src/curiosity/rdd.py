"""Random Distillation Disagreement (RDD) curiosity module.

Uses a single fixed random target and an ensemble of predictors.  All
predictors are trained to predict the same target; the intrinsic reward is the
*disagreement* (variance across predictors) rather than the raw prediction
error.  Disagreement captures epistemic uncertainty: high in novel states where
predictors diverge, low in well-explored states where they converge.

sigma: bandwidth — rewards are scaled by 1/sigma².  Larger sigma suppresses
small-disagreement states, focusing exploration on regions of high uncertainty.
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


class RDDModule(nn.Module):
    """
    RDD curiosity module.

    Intrinsic reward = var_k[predictor_k(norm_obs)].sum(-1) / sigma²,
    then normalised by a running std of past rewards.
    """

    def __init__(
        self,
        obs_dim: int,
        n_ensemble: int = 5,
        sigma: float = 1.0,
        output_dim: int = 64,
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.n_ensemble = n_ensemble
        self.sigma2 = sigma ** 2

        # Single frozen target — all predictors learn to predict this
        self.target = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, output_dim)),
        ).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        # Ensemble of trained predictors (independently initialised)
        self.predictors = nn.ModuleList([
            _mlp(obs_dim, hidden_dim, output_dim, depth=2) for _ in range(n_ensemble)
        ]).to(device)

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
        preds = torch.stack([pred(obs_n) for pred in self.predictors])  # (K, N, D)
        # Variance across ensemble members, summed over feature dim, scaled by bandwidth
        raw = preds.var(dim=0).sum(-1) / self.sigma2  # (N,)
        self.reward_rms.update(raw.unsqueeze(-1))
        return raw / (self.reward_rms.var.sqrt().squeeze() + 1e-8)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        obs_n = self._norm_obs(obs).detach()
        with torch.no_grad():
            feat = self.target(obs_n)
        total = sum(((pred(obs_n) - feat) ** 2).mean() for pred in self.predictors)
        return total / self.n_ensemble
