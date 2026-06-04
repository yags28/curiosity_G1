"""
Task 3 — Occluded Retrieval (Medium, Curriculum Stage 3).

Goal:   Retrieve a target object placed behind a rigid barrier (15 cm slot gap).
Latent: Use a stick/rod to push the target through the slot toward the robot.

Success criterion: Target XY displacement ≥ 0.5 m from its per-env spawn position.

Scene layout (env-local, robot at origin facing +X):
  Barrier      : two kinematic wall pieces at X=1.0 m; slot from Z=0.35 to Z=0.50 m
  Target object: 10×10×10 cm box behind barrier at ~X=1.4 m, centered in slot (Z=0.425 m)
  Stick        : 90 cm capsule at (0.5, 0.4, 0.5) — beside robot, within arm reach

Observation extras (critic only, same 10-per-object layout as Tasks 1-2):
  stick_pos_rel  (3)  — stick position relative to pelvis
  stick_vel      (6)  — stick linear + angular velocity
  stick_mass     (1)  — privileged mass (varies ±30%)
  target_pos_rel (3)  — target position relative to pelvis
  target_vel     (6)  — target linear + angular velocity
  target_mass    (1)  — privileged mass (varies ±30%)
  Total extra    (20)

Critic obs dim = BASE_CRITIC_DIM (112) + 20 = 132
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from ..tool_use_base import BASE_CRITIC_DIM, ToolUseEnv, ToolUseEnvCfg

# ── Task constants ────────────────────────────────────────────────────────────

N_OBJECTS      = 2
TASK_OBS_DIM   = N_OBJECTS * (3 + 6 + 1)          # pos_rel + vel + mass = 20
CRITIC_OBS_DIM = BASE_CRITIC_DIM + TASK_OBS_DIM   # 132

SUCCESS_DISPLACEMENT = 0.5   # metres — XY distance target must travel from spawn

# Barrier geometry (env-local X)
BARRIER_X         = 1.00   # barrier centre X
BARRIER_THICKNESS = 0.08   # wall thickness
BARRIER_WIDTH_Y   = 1.20   # wall width in Y — wide enough target can't go around
BARRIER_BOTTOM_H  = 0.35   # height of lower wall piece
SLOT_H            = 0.15   # 15 cm gap
BARRIER_TOP_H     = 0.80   # height of upper wall piece (wall total = 1.30 m)

# Slot Z span (env-local)
SLOT_Z_BOT = BARRIER_BOTTOM_H                  # 0.35 m
SLOT_Z_TOP = BARRIER_BOTTOM_H + SLOT_H         # 0.50 m
SLOT_Z_MID = (SLOT_Z_BOT + SLOT_Z_TOP) / 2.0  # 0.425 m

# Barrier piece centres (env-local, spawned as init_state.pos)
BARRIER_BOTTOM_POS = (BARRIER_X, 0.0, BARRIER_BOTTOM_H / 2.0)
BARRIER_TOP_POS    = (BARRIER_X, 0.0, BARRIER_BOTTOM_H + SLOT_H + BARRIER_TOP_H / 2.0)

# Nominal object positions (env-local)
STICK_NOMINAL_POS  = (0.5,  0.4, 0.5)
TARGET_NOMINAL_POS = (1.4,  0.0, SLOT_Z_MID)   # behind barrier, in slot height

# Randomisation
TARGET_X_RANGE  = 0.20   # ±0.20 m in X — target stays behind barrier
TARGET_Y_RANGE  = 0.15   # ±0.15 m in Y — within barrier width ±0.60 m
STICK_POS_RANGE = 0.30   # ±30% of nominal XY for stick
MASS_RANGE      = 0.30   # ±30% mass
FRICTION_RANGE  = 0.50   # ±50% friction

STICK_NOMINAL_MASS             = 0.30   # kg — PVC rod ~90 cm
TARGET_NOMINAL_MASS            = 0.50   # kg — small object
STICK_NOMINAL_STATIC_FRICTION  = 0.80
STICK_NOMINAL_DYNAMIC_FRICTION = 0.60


# ── Env config ────────────────────────────────────────────────────────────────

@configclass
class OccludedRetrievalEnvCfg(ToolUseEnvCfg):
    """Config for Task 3 — Occluded Retrieval."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    observation_space: int = 109
    state_space: int = CRITIC_OBS_DIM   # 132
    episode_length_s: float = 40.0

    # ── Stick (dynamic tool — robot picks up and pushes through slot) ─────────
    stick: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Stick",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=STICK_NOMINAL_POS,
            rot=(0.7071, 0.0, 0.7071, 0.0),   # horizontal (wxyz)
        ),
        spawn=sim_utils.CapsuleCfg(
            radius=0.02,
            height=0.90,   # 90 cm rod
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=STICK_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=STICK_NOMINAL_STATIC_FRICTION,
                dynamic_friction=STICK_NOMINAL_DYNAMIC_FRICTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
    )

    # ── Target object (dynamic — robot must push it through the slot) ─────────
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TARGET_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10),   # 10×10×10 cm box — fits through 15 cm slot
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TARGET_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.4,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.1)),
        ),
    )

    # ── Barrier lower piece (kinematic — blocks access below slot) ────────────
    barrier_bottom: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BarrierBottom",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BARRIER_BOTTOM_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BARRIER_THICKNESS, BARRIER_WIDTH_Y, BARRIER_BOTTOM_H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.45)),
        ),
    )

    # ── Barrier upper piece (kinematic — blocks access above slot) ────────────
    barrier_top: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BarrierTop",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BARRIER_TOP_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BARRIER_THICKNESS, BARRIER_WIDTH_Y, BARRIER_TOP_H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.45)),
        ),
    )

    # ── Contact sensor on target — logs tool_contact_events ───────────────────
    target_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Target",
        history_length=1,
        update_period=0.0,
    )


# ── Environment ───────────────────────────────────────────────────────────────

class OccludedRetrievalEnv(ToolUseEnv):
    """
    Curriculum Stage 3 — Occluded Retrieval task.

    Unit-test checklist (§3.3):
      ✓ reset 100 times, no NaN/Inf, objects above floor, target behind barrier
      ✓ random-agent success rate < 0.1%
      ✓ tool_contact_events logged
    """

    cfg: OccludedRetrievalEnvCfg

    def __init__(self, cfg: OccludedRetrievalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._stick_nominal  = torch.tensor(STICK_NOMINAL_POS,  device=self.device)
        self._target_nominal = torch.tensor(TARGET_NOMINAL_POS, device=self.device)

        # Per-env stick mass and friction buffers (CPU for PhysX API)
        self._stick_mass = torch.full((self.num_envs,), STICK_NOMINAL_MASS, device=self.device)
        self._stick_friction = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._stick_friction[:, 0, 0] = STICK_NOMINAL_STATIC_FRICTION
        self._stick_friction[:, 0, 1] = STICK_NOMINAL_DYNAMIC_FRICTION

        # Per-env target mass and spawn-position buffers
        self._target_mass = torch.full((self.num_envs,), TARGET_NOMINAL_MASS, device=self.device)
        # World-frame XY spawn position; used by _compute_success for displacement check
        self._target_spawn_xy = torch.zeros(self.num_envs, 2, device=self.device)

    # ── Object setup ──────────────────────────────────────────────────────────

    def _setup_objects(self) -> None:
        self.stick          = RigidObject(self.cfg.stick)
        self.target         = RigidObject(self.cfg.target)
        self.barrier_bottom = RigidObject(self.cfg.barrier_bottom)
        self.barrier_top    = RigidObject(self.cfg.barrier_top)
        self.target_contact = ContactSensor(self.cfg.target_contact)

        self.scene.rigid_objects["stick"]          = self.stick
        self.scene.rigid_objects["target"]         = self.target
        self.scene.rigid_objects["barrier_bottom"] = self.barrier_bottom
        self.scene.rigid_objects["barrier_top"]    = self.barrier_top
        self.scene.sensors["target_contact"]       = self.target_contact

    # ── Task observations (critic only) ───────────────────────────────────────

    def _get_task_obs(self) -> torch.Tensor:
        pelvis_pos = self.robot.data.root_pos_w   # (N, 3)

        stick_pos_rel  = self.stick.data.root_pos_w  - pelvis_pos   # (N, 3)
        stick_lin_vel  = self.stick.data.root_lin_vel_w              # (N, 3)
        stick_ang_vel  = self.stick.data.root_ang_vel_w              # (N, 3)
        stick_mass     = self._stick_mass.unsqueeze(-1)               # (N, 1)

        target_pos_rel = self.target.data.root_pos_w - pelvis_pos    # (N, 3)
        target_lin_vel = self.target.data.root_lin_vel_w             # (N, 3)
        target_ang_vel = self.target.data.root_ang_vel_w             # (N, 3)
        target_mass    = self._target_mass.unsqueeze(-1)             # (N, 1)

        return torch.cat([
            stick_pos_rel,  stick_lin_vel,  stick_ang_vel,  stick_mass,
            target_pos_rel, target_lin_vel, target_ang_vel, target_mass,
        ], dim=-1)   # (N, 20)

    # ── Success detection ─────────────────────────────────────────────────────

    def _compute_success(self) -> torch.Tensor:
        target_xy = self.target.data.root_pos_w[:, :2]   # (N, 2)
        displacement = torch.norm(target_xy - self._target_spawn_xy, dim=-1)  # (N,)
        success = displacement >= SUCCESS_DISPLACEMENT

        # Log contact events diagnostic
        net_forces = self.target_contact.data.net_forces_w          # (N, 1, 3)
        contact    = torch.norm(net_forces, dim=-1).squeeze(-1) > 0.1
        self.extras["tool_contact_events"] = contact.float().mean().item()

        return success

    # ── Object reset ──────────────────────────────────────────────────────────

    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        n       = len(env_ids)
        origins = self.scene.env_origins[env_ids]   # (n, 3)
        all_ids = torch.arange(self.num_envs, device=self.device)
        unit_quat = torch.zeros(n, 4, device=self.device)
        unit_quat[:, 0] = 1.0

        # ── Stick: XY jitter ±30% of nominal, random yaw ─────────────────────
        stick_nom = self._stick_nominal.unsqueeze(0).expand(n, -1).clone()
        delta_xy  = (torch.rand(n, 2, device=self.device) * 2 - 1) * STICK_POS_RANGE * stick_nom[:, :2].abs()
        stick_pos = stick_nom.clone()
        stick_pos[:, :2] += delta_xy
        stick_pos = stick_pos + origins

        yaw  = torch.rand(n, device=self.device) * 2 * torch.pi
        half = yaw * 0.5
        stick_quat = torch.stack([
            torch.cos(half),
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.sin(half),
        ], dim=-1)

        stick_state = torch.cat([stick_pos, stick_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.stick.write_root_state_to_sim(stick_state, env_ids)

        # ── Stick mass + friction randomisation ───────────────────────────────
        mass_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * MASS_RANGE
        self._stick_mass[env_ids] = STICK_NOMINAL_MASS * mass_scale
        self.stick.root_physx_view.set_masses(
            self._stick_mass.unsqueeze(-1).cpu(), all_ids.cpu()
        )

        fric_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * FRICTION_RANGE
        self._stick_friction[env_ids, 0, 0] = STICK_NOMINAL_STATIC_FRICTION  * fric_scale
        self._stick_friction[env_ids, 0, 1] = STICK_NOMINAL_DYNAMIC_FRICTION * fric_scale
        self.stick.root_physx_view.set_material_properties(
            self._stick_friction.cpu(), all_ids.cpu()
        )

        # ── Target: absolute X/Y jitter, fixed Z centred in slot ─────────────
        # X: nominal ± TARGET_X_RANGE (always behind barrier back face)
        # Y: ± TARGET_Y_RANGE (within barrier width)
        # Z: fixed at SLOT_Z_MID (object fits inside the 15 cm slot)
        target_nom = self._target_nominal.unsqueeze(0).expand(n, -1).clone()
        rand_xy = (torch.rand(n, 2, device=self.device) * 2 - 1)
        target_pos = target_nom.clone()
        target_pos[:, 0] += rand_xy[:, 0] * TARGET_X_RANGE
        target_pos[:, 1] += rand_xy[:, 1] * TARGET_Y_RANGE
        target_pos = target_pos + origins

        target_state = torch.cat([target_pos, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.target.write_root_state_to_sim(target_state, env_ids)

        # Record world-frame XY spawn position for displacement check
        self._target_spawn_xy[env_ids] = target_pos[:, :2]

        # ── Target mass randomisation ─────────────────────────────────────────
        t_mass_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * MASS_RANGE
        self._target_mass[env_ids] = TARGET_NOMINAL_MASS * t_mass_scale
        self.target.root_physx_view.set_masses(
            self._target_mass.unsqueeze(-1).cpu(), all_ids.cpu()
        )

        # ── Barrier pieces: kinematic, fixed nominal positions ─────────────────
        b_bot_nom = torch.tensor(BARRIER_BOTTOM_POS, device=self.device).unsqueeze(0).expand(n, -1)
        b_top_nom = torch.tensor(BARRIER_TOP_POS,    device=self.device).unsqueeze(0).expand(n, -1)

        bot_pos = b_bot_nom + origins
        top_pos = b_top_nom + origins

        self.barrier_bottom.write_root_state_to_sim(
            torch.cat([bot_pos, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1), env_ids
        )
        self.barrier_top.write_root_state_to_sim(
            torch.cat([top_pos, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1), env_ids
        )
