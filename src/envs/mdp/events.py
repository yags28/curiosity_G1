"""Custom MDP event terms for CDL tool-use environments (object randomisation)."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv


def randomise_object_pose(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    object_cfg_name: str,
    nominal_pos: tuple[float, float, float],
    pos_range: float = 0.30,
    yaw_randomise: bool = True,
) -> None:
    """
    Reset object position to nominal ± pos_range fraction, random yaw.
    Implements §3.4: ±30% position randomisation, full SO(2) yaw randomisation.
    """
    obj: RigidObject = env.scene[object_cfg_name]
    n = len(env_ids)
    device = env.device

    nominal = torch.tensor(nominal_pos, device=device).unsqueeze(0).expand(n, -1)
    delta = (torch.rand(n, 3, device=device) * 2 - 1) * pos_range * nominal.abs()
    pos = nominal + delta
    pos[:, 2] = nominal[:, 2]  # keep height fixed

    # add env origins
    pos = pos + env.scene.env_origins[env_ids]

    if yaw_randomise:
        yaw = torch.rand(n, device=device) * 2 * torch.pi  # full SO(2)
        half = yaw * 0.5
        quat = torch.stack([
            torch.cos(half),          # w
            torch.zeros(n, device=device),  # x
            torch.zeros(n, device=device),  # y
            torch.sin(half),          # z
        ], dim=-1)  # wxyz
    else:
        quat = torch.zeros(n, 4, device=device)
        quat[:, 0] = 1.0  # identity

    root_state = torch.cat([pos, quat, torch.zeros(n, 6, device=device)], dim=-1)
    obj.write_root_state_to_sim(root_state, env_ids)


def randomise_object_mass(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    object_cfg_name: str,
    nominal_mass: float,
    mass_range: float = 0.30,
) -> None:
    """
    Randomise object mass ± mass_range fraction of nominal.
    Implements §3.4: ±30% mass randomisation.
    """
    obj: RigidObject = env.scene[object_cfg_name]
    n = len(env_ids)
    scale = 1.0 + (torch.rand(n, device=env.device) * 2 - 1) * mass_range
    new_mass = nominal_mass * scale  # (n,)
    # Set mass via simulation — one body per object assumed
    obj.root_physx_view.set_masses(new_mass.unsqueeze(-1), env_ids)


def reset_robot_default(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    asset_cfg_name: str = "robot",
) -> None:
    """Reset robot to its default state (pose + zero velocity) for given envs."""
    robot: Articulation = env.scene[asset_cfg_name]
    robot.reset(env_ids)

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
    robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
