import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, out_std: float = 1.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [layer_init(nn.Linear(prev, h)), nn.ELU()]
            prev = h
        layers.append(layer_init(nn.Linear(prev, out_dim), std=out_std))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Asymmetric actor-critic for PPO + curiosity.

    Actor  : policy obs (no privilege) → continuous Gaussian action
    Critics: privileged critic obs → V_ext and V_int (separate heads)
    """

    def __init__(self, obs_dim: int, critic_obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor_mean   = MLP(obs_dim,        [hidden_dim, hidden_dim], action_dim, out_std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic_ext   = MLP(critic_obs_dim, [hidden_dim, hidden_dim], 1, out_std=1.0)
        self.critic_int   = MLP(critic_obs_dim, [hidden_dim, hidden_dim], 1, out_std=1.0)

    def get_value(self, critic_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.critic_ext(critic_obs).squeeze(-1),
            self.critic_int(critic_obs).squeeze(-1),
        )

    def get_action_and_value(
        self,
        policy_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean    = self.actor_mean(policy_obs)
        logstd  = self.actor_logstd.expand_as(mean)
        dist    = torch.distributions.Normal(mean, logstd.exp())
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        v_ext   = self.critic_ext(critic_obs).squeeze(-1)
        v_int   = self.critic_int(critic_obs).squeeze(-1)
        return action, logprob, entropy, v_ext, v_int
