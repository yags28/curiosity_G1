"""Custom MDP termination terms for CDL tool-use environments."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv


def robot_fell(
    env: DirectRLEnv,
    asset_cfg_name: str = "robot",
    min_pelvis_height: float = 0.5,
) -> torch.Tensor:
    """True for envs where the pelvis dropped below min_pelvis_height. Shape: (N,)."""
    robot: Articulation = env.scene[asset_cfg_name]
    pelvis_z = robot.data.root_pos_w[:, 2]  # (N,)
    return pelvis_z < min_pelvis_height


def task_succeeded(env: DirectRLEnv, success_buf: torch.Tensor) -> torch.Tensor:
    """True for envs that achieved task success — triggers episode end. Shape: (N,)."""
    return success_buf.bool()
