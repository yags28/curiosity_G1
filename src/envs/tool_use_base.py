"""
ToolUseEnv — base DirectRLEnv for all 5 CDL tool-use tasks.

Handles:
  - G1 EDU robot setup (43 DOF, Dex3-1 hands)
  - Proprioceptive observations (actor) + privileged state (critic)
  - Asymmetric actor-critic observation dict
  - Time penalty reward
  - Fall and timeout terminations
  - Robot reset logic

Subclasses must implement:
  - _setup_objects()
  - _get_task_obs()   → extra privileged obs rows (N, task_obs_dim)
  - _compute_success() → bool tensor (N,)
  - _reset_objects(env_ids)
"""

from __future__ import annotations

from abc import abstractmethod

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from .g1_cfg import G1_EDU_CFG


# ── Programmatic ground plane (no Nucleus / cloud-asset dependency) ───────────

class _LocalGroundTerrainImporter(TerrainImporter):
    """Overrides import_ground_plane to build a USD physics plane from scratch.

    Isaac Lab's default TerrainImporter loads `default_environment.usd` from
    NVIDIA's Nucleus/S3 server to get the ground plane geometry. That fails
    when Nucleus is not configured (ISAAC_NUCLEUS_DIR is None on this machine).

    This subclass replaces that with an in-memory UsdGeom.Plane prim that PhysX
    treats as an infinite static collision plane — no external asset required.
    """

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        from pxr import UsdGeom, UsdPhysics
        from isaaclab.sim.utils import bind_physics_material, get_current_stage

        prim_path = self.cfg.prim_path + f"/{name}"
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. "
                f"Existing terrains: {', '.join(self.terrain_prim_paths)}"
            )
        self.terrain_prim_paths.append(prim_path)

        stage = get_current_stage()

        # Parent Xform container
        UsdGeom.Xform.Define(stage, prim_path)

        # Infinite collision plane (Z-up = horizontal)
        plane_usd_path = prim_path + "/CollisionPlane"
        plane = UsdGeom.Plane.Define(stage, plane_usd_path)
        plane.CreateAxisAttr("Z")

        # Apply physics collision
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

        # Apply physics material if configured
        if self.cfg.physics_material is not None:
            mat_path = prim_path + "/physicsMaterial"
            self.cfg.physics_material.func(mat_path, self.cfg.physics_material)
            bind_physics_material(plane_usd_path, mat_path, stage=stage)

# ── Observation dimensions ────────────────────────────────────────────────────
# Actor (no privilege): joint_pos(43) + joint_vel(43) + base_quat(4)
#                       + base_ang_vel(3) + foot_contact(4) + wrist_ft(12)
ACTOR_OBS_DIM = 109

# Base privileged extra: base_lin_vel(3) — not observable without state estimator
BASE_PRIV_EXTRA = 3
BASE_CRITIC_DIM = ACTOR_OBS_DIM + BASE_PRIV_EXTRA  # 112


# ── Env config ────────────────────────────────────────────────────────────────

@configclass
class ToolUseEnvCfg(DirectRLEnvCfg):
    """Base config for all CDL tool-use environments."""

    # Simulation: 200 Hz physics, 50 Hz policy (decimation=4)
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    decimation: int = 4
    episode_length_s: float = 40.0

    # Minimal scene — all entities are spawned manually in _setup_scene()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # Flat ground plane — uses _LocalGroundTerrainImporter so no Nucleus USD is needed
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        class_type=_LocalGroundTerrainImporter,
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot (43 DOF G1 EDU with Dex3-1 hands)
    robot: ArticulationCfg = G1_EDU_CFG

    # Foot contact sensor — all robot bodies; ankle IDs resolved via find_bodies()
    foot_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=1,
        update_period=0.0,
        track_air_time=False,
    )

    # Observation / action spaces — subclasses set correct critic dim
    observation_space: int = ACTOR_OBS_DIM
    state_space: int = BASE_CRITIC_DIM  # overridden by subclasses adding objects
    action_space: int = 43

    # Task-specific
    fall_height: float = 0.5        # pelvis below this → fall termination
    success_reward: float = 1.0
    time_penalty: float = -0.001


# ── Base environment ──────────────────────────────────────────────────────────

class ToolUseEnv(DirectRLEnv):
    """
    Base class for CDL tool-use environments. Inherit from this and implement
    the four abstract methods to add a new task.
    """

    cfg: ToolUseEnvCfg

    def __init__(self, cfg: ToolUseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint action scaling: maps [-1, 1] → [lower, upper] soft limits
        lowers = self.robot.data.soft_joint_pos_limits[0, :, 0]
        uppers = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self._action_offset = 0.5 * (uppers + lowers)   # (43,)
        self._action_scale  = 0.5 * (uppers - lowers)   # (43,)

        # Ankle body IDs for foot contact detection
        self._foot_ids, _ = self.foot_contact_sensor.find_bodies(".*ankle.*")

        # Per-env success flag — set by _compute_success(), cleared on reset
        self._success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self):
        # 1. Robot
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # 2. Foot contact sensor (all robot bodies; ankle IDs resolved in __init__)
        self.foot_contact_sensor = ContactSensor(self.cfg.foot_contact)
        self.scene.sensors["foot_contact"] = self.foot_contact_sensor

        # 3. Flat terrain (procedural plane — no nucleus USD)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 4. Task-specific objects (subclass spawns and registers before cloning)
        self._setup_objects()

        # 5. Clone all environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 6. Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @abstractmethod
    def _setup_objects(self) -> None:
        """Spawn task-specific rigid objects and sensors; register with self.scene before cloning."""

    # ── Action processing ─────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self._action_offset + self._action_scale * self.actions
        self.robot.set_joint_position_target(targets)

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict[str, torch.Tensor]:
        actor_obs = self._build_actor_obs()
        critic_obs = torch.cat([actor_obs, self._build_base_privileged(), self._get_task_obs()], dim=-1)
        return {"policy": actor_obs, "critic": critic_obs}

    def _build_actor_obs(self) -> torch.Tensor:
        """(N, 109 + actor_task) proprioceptive obs (+ optional object pose)."""
        joint_pos  = self.robot.data.joint_pos                  # (N, 43)
        joint_vel  = self.robot.data.joint_vel                  # (N, 43)
        base_quat  = self.robot.data.root_quat_w                # (N, 4)  wxyz
        ang_vel    = self.robot.data.root_ang_vel_b             # (N, 3)
        foot_cont  = self._get_foot_contact()                   # (N, 4)
        wrist_ft   = torch.zeros(self.num_envs, 12, device=self.device)  # TODO: wire sensors §3.2
        return torch.cat([joint_pos, joint_vel, base_quat, ang_vel, foot_cont,
                          wrist_ft, self._get_actor_task_obs()], dim=-1)

    def _get_actor_task_obs(self) -> torch.Tensor:
        """Optional task-specific obs exposed to the ACTOR (closed-loop control).

        Default: none (proprioception-only). Tasks that need the policy to see
        objects (e.g. for sim-to-sim/sim-to-real transfer) override this.
        """
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _build_base_privileged(self) -> torch.Tensor:
        """(N, 3) privileged info available to critic: base linear velocity."""
        return self.robot.data.root_lin_vel_b  # (N, 3) in body frame

    @abstractmethod
    def _get_task_obs(self) -> torch.Tensor:
        """
        Task-specific privileged observations added to the critic input.
        Returns (N, task_obs_dim).
        """

    def _get_foot_contact(self) -> torch.Tensor:
        """(N, 4) binary foot-contact for the 4 ankle links."""
        forces = self.foot_contact_sensor.data.net_forces_w[:, self._foot_ids, :]  # (N, 4, 3)
        norms  = torch.norm(forces, dim=-1)                                          # (N, 4)
        return (norms > 1.0).float()

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        self._success_buf |= self._compute_success()
        reward = torch.full((self.num_envs,), self.cfg.time_penalty, device=self.device)
        reward += self._success_buf.float() * self.cfg.success_reward
        return reward

    @abstractmethod
    def _compute_success(self) -> torch.Tensor:
        """Returns (N,) bool tensor — True for envs that achieved the goal this step."""

    # ── Terminations ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fell     = self.robot.data.root_pos_w[:, 2] < self.cfg.fall_height
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = fell | self._success_buf
        return terminated, timed_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self._success_buf[env_ids] = False

        # Reset robot
        self.robot.reset(env_ids)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset task objects
        self._reset_objects(env_ids)

        super()._reset_idx(env_ids)

    @abstractmethod
    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        """Randomise and reset all task-specific objects for the given envs."""
