"""Custom MDP observation terms for CDL tool-use environments."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor


def joint_pos_rel(env: DirectRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """Joint positions relative to default pose. Shape: (N, n_joints)."""
    robot: Articulation = env.scene[asset_cfg_name]
    return robot.data.joint_pos - robot.data.default_joint_pos


def joint_vel(env: DirectRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """Joint velocities. Shape: (N, n_joints)."""
    robot: Articulation = env.scene[asset_cfg_name]
    return robot.data.joint_vel


def base_orientation_quat(env: DirectRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """Root link orientation as wxyz quaternion in world frame. Shape: (N, 4)."""
    robot: Articulation = env.scene[asset_cfg_name]
    return robot.data.root_quat_w


def base_ang_vel_body(env: DirectRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """Root angular velocity in the robot body frame. Shape: (N, 3)."""
    robot: Articulation = env.scene[asset_cfg_name]
    return robot.data.root_ang_vel_b


def foot_contact_binary(
    env: DirectRLEnv,
    sensor_cfg_name: str = "foot_contact",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot-contact signal for 4 ankle contact points. Shape: (N, 4)."""
    sensor: ContactSensor = env.scene[sensor_cfg_name]
    net_forces = sensor.data.net_forces_w  # (N, num_bodies, 3)
    force_norms = torch.norm(net_forces, dim=-1)  # (N, num_bodies)
    return (force_norms > threshold).float()


def object_position_relative(
    env: DirectRLEnv,
    object_names: list[str],
    robot_cfg_name: str = "robot",
) -> torch.Tensor:
    """
    Positions of scene objects relative to the robot root (pelvis).
    Returns (N, len(object_names) * 3).
    """
    robot: Articulation = env.scene[robot_cfg_name]
    pelvis_pos = robot.data.root_pos_w  # (N, 3)
    parts = []
    for name in object_names:
        obj: RigidObject = env.scene[name]
        rel = obj.data.root_pos_w - pelvis_pos  # (N, 3)
        parts.append(rel)
    return torch.cat(parts, dim=-1)


def object_velocity(
    env: DirectRLEnv,
    object_names: list[str],
) -> torch.Tensor:
    """
    Linear + angular velocities of scene objects.
    Returns (N, len(object_names) * 6).
    """
    parts = []
    for name in object_names:
        obj: RigidObject = env.scene[name]
        lin_vel = obj.data.root_lin_vel_w  # (N, 3)
        ang_vel = obj.data.root_ang_vel_w  # (N, 3)
        parts.append(torch.cat([lin_vel, ang_vel], dim=-1))
    return torch.cat(parts, dim=-1)


def object_mass(env: DirectRLEnv, object_names: list[str]) -> torch.Tensor:
    """
    Total mass of each object (privileged info). Returns (N, len(object_names)).
    Note: mass is constant per env — uses body_mass sum over all links.
    """
    parts = []
    for name in object_names:
        obj: RigidObject = env.scene[name]
        # body_masses: (N, num_links) — sum over links for total mass
        mass = obj.data.body_mass.sum(dim=-1, keepdim=True)  # (N, 1)
        parts.append(mass)
    return torch.cat(parts, dim=-1)
