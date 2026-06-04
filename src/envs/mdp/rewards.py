"""Custom MDP reward terms for CDL tool-use environments."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor


def time_penalty(env: DirectRLEnv, penalty: float = -0.001) -> torch.Tensor:
    """Per-timestep penalty to encourage efficiency. Shape: (N,)."""
    return torch.full((env.num_envs,), penalty, device=env.device)


def task_success(env: DirectRLEnv, success_buf: torch.Tensor) -> torch.Tensor:
    """Binary +1 reward on the step task success is first detected. Shape: (N,)."""
    return success_buf.float()


def target_contact_success(
    env: DirectRLEnv,
    sensor_cfg_name: str,
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Returns 1.0 for envs where contact force on the target sensor exceeds
    force_threshold (Newtons). Used for DistantTarget and OccludedRetrieval.
    Shape: (N,).
    """
    sensor: ContactSensor = env.scene[sensor_cfg_name]
    net_forces = sensor.data.net_forces_w  # (N, num_bodies, 3)
    max_force = torch.norm(net_forces, dim=-1).max(dim=-1).values  # (N,)
    return (max_force > force_threshold).float()


def object_displacement_success(
    env: DirectRLEnv,
    object_cfg_name: str,
    origin: torch.Tensor,
    min_displacement: float = 0.5,
) -> torch.Tensor:
    """
    Returns 1.0 for envs where the object has moved ≥ min_displacement metres
    from its spawn origin. Used for OccludedRetrieval.
    Shape: (N,).
    """
    obj: RigidObject = env.scene[object_cfg_name]
    displacement = torch.norm(obj.data.root_pos_w - origin, dim=-1)  # (N,)
    return (displacement >= min_displacement).float()


def button_press_success(
    env: DirectRLEnv,
    sensor_cfg_name: str,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """
    Returns 1.0 when button contact force exceeds threshold.
    Used for ElevatedButton and Composite.
    Shape: (N,).
    """
    sensor: ContactSensor = env.scene[sensor_cfg_name]
    net_forces = sensor.data.net_forces_w  # (N, num_bodies, 3)
    max_force = torch.norm(net_forces, dim=-1).max(dim=-1).values  # (N,)
    return (max_force > force_threshold).float()
