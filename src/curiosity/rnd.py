"""Random Network Distillation (Burda et al. 2019) curiosity module."""

import torch
import torch.nn as nn

from src.utils.networks import layer_init


class RunningMeanStd:
    """Online Welford running mean/variance — used for obs and reward normalisation."""

    def __init__(self, shape: int | tuple, device: str):
        self.mean  = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var   = torch.ones(shape,  device=device, dtype=torch.float32)
        self.count = torch.tensor(1e-4, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x           = x.float().reshape(-1, *self.mean.shape)
        batch_mean  = x.mean(0)
        batch_var   = x.var(0, unbiased=False) if x.shape[0] > 1 else torch.zeros_like(batch_mean)
        batch_count = x.shape[0]
        delta       = batch_mean - self.mean
        tot         = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot
        m_a         = self.var * self.count
        m_b         = batch_var * batch_count
        m2          = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.var    = m2 / tot
        self.count  = tot


class RNDModule(nn.Module):
    """
    RND curiosity module.

    Intrinsic reward = ||predictor(norm_obs) - target(norm_obs)||²,
    then normalised by a running std of past rewards.
    """

    def __init__(self, obs_dim: int, output_dim: int = 64, hidden_dim: int = 256, device: str = "cuda"):
        super().__init__()
        self.device = device

        self.target = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, output_dim)),
        ).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, output_dim)),
        ).to(device)

        self.obs_rms    = RunningMeanStd(obs_dim, device)
        self.reward_rms = RunningMeanStd(1, device)

    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.obs_rms.mean) / (self.obs_rms.var.sqrt() + 1e-8)).clamp(-5.0, 5.0)

    @torch.no_grad()
    def compute_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """(N,) normalised intrinsic reward. Does NOT update obs_rms — caller does that."""
        obs_n      = self._norm_obs(obs)
        raw        = ((self.predictor(obs_n) - self.target(obs_n)) ** 2).sum(-1)  # (N,)
        self.reward_rms.update(raw.unsqueeze(-1))
        return raw / (self.reward_rms.var.sqrt().squeeze() + 1e-8)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        obs_n = self._norm_obs(obs).detach()
        with torch.no_grad():
            target_feat = self.target(obs_n)
        return ((self.predictor(obs_n) - target_feat) ** 2).mean()

    def predictor_parameters(self):
        return self.predictor.parameters()
