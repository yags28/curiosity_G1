"""Depth image encoder and graspability classifier for visual curiosity."""

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """64×64 depth image → out_dim embedding via 3-layer CNN."""

    def __init__(self, img_h: int = 64, img_w: int = 64, out_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # → (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → (64,  8,  8)
            nn.ReLU(),
            nn.Flatten(),
        )
        flat = 64 * (img_h // 8) * (img_w // 8)
        self.proj = nn.Linear(flat, out_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: (N, H, W) or (N, 1, H, W) → (N, out_dim)."""
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        x = depth.float().clamp(0.0, 5.0) / 5.0   # → [0, 1]
        return self.proj(self.cnn(x))


class GraspabilityHead(nn.Module):
    """embedding → graspable probability in [0, 1]."""

    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Returns (N,) graspability probability."""
        return torch.sigmoid(self.net(emb)).squeeze(-1)
