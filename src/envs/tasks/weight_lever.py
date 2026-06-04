"""
Task 4 — Weight Lever (Hard, Curriculum Stage 4).

Goal:   Lift a heavy object (10 kg) beyond the G1's direct payload capacity (~3–5 kg).
Latent: Position the plank across the fulcrum to create a lever; push the near end down.

Success criterion: Heavy object lifted ≥ 0.10 m above its per-env spawn height.

Scene layout (env-local, robot at origin facing +X):
  Plank        : 2 m × 10 cm × 4 cm flat board at ~X=1.2 m — robot must position it
  Heavy object : 20×20×10 cm box, 10 kg, at ~X=1.8 m on the floor
  Fulcrum      : 15×15×12 cm block, ~1 kg, at ~(0.5, 0.3) — robot must slide under plank

Observation extras (critic only, 10 dims per object × 3 objects):
  heavy_pos_rel   (3)  heavy_vel   (6)  heavy_mass   (1)
  plank_pos_rel   (3)  plank_vel   (6)  plank_mass   (1)
  fulcrum_pos_rel (3)  fulcrum_vel (6)  fulcrum_mass (1)
  Total extra (30)

Critic obs dim = BASE_CRITIC_DIM (112) + 30 = 142
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

N_OBJECTS      = 3
TASK_OBS_DIM   = N_OBJECTS * (3 + 6 + 1)          # 30
CRITIC_OBS_DIM = BASE_CRITIC_DIM + TASK_OBS_DIM   # 142

SUCCESS_LIFT = 0.10   # metres above spawn height

# Nominal positions (env-local)
PLANK_NOMINAL_POS   = (1.2,  0.0, 0.02)    # plank centre, lying flat (half-thickness = 2 cm)
HEAVY_NOMINAL_POS   = (1.8,  0.0, 0.05)    # heavy box on floor (half-height = 5 cm)
FULCRUM_NOMINAL_POS = (0.5,  0.3, 0.06)    # fulcrum near robot (half-height = 6 cm)

# Per-axis absolute jitter ranges (symmetric ±)
PLANK_X_RANGE   = 0.25
PLANK_Y_RANGE   = 0.25
HEAVY_X_RANGE   = 0.20
HEAVY_Y_RANGE   = 0.20
FULCRUM_X_RANGE = 0.20
FULCRUM_Y_RANGE = 0.20

MASS_RANGE     = 0.30   # ±30%
FRICTION_RANGE = 0.50   # ±50%

PLANK_NOMINAL_MASS             = 2.0    # kg
HEAVY_NOMINAL_MASS             = 10.0   # kg — beyond G1's direct payload
FULCRUM_NOMINAL_MASS           = 1.0    # kg
PLANK_NOMINAL_STATIC_FRICTION  = 0.70
PLANK_NOMINAL_DYNAMIC_FRICTION = 0.50


# ── Env config ────────────────────────────────────────────────────────────────

@configclass
class WeightLeverEnvCfg(ToolUseEnvCfg):
    """Config for Task 4 — Weight Lever."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True
    )

    observation_space: int = 109
    state_space: int = CRITIC_OBS_DIM   # 142
    episode_length_s: float = 40.0

    # ── Plank (dynamic tool — 2 m flat board, robot uses as lever arm) ────────
    plank: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plank",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PLANK_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),   # flat, long axis along X
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2.00, 0.10, 0.04),   # 2 m × 10 cm × 4 cm board
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PLANK_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=PLANK_NOMINAL_STATIC_FRICTION,
                dynamic_friction=PLANK_NOMINAL_DYNAMIC_FRICTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
        ),
    )

    # ── Heavy object (dynamic — too heavy to lift directly) ───────────────────
    heavy: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Heavy",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=HEAVY_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, 0.10),   # 20×20×10 cm, 10 kg
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=HEAVY_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.80,
                dynamic_friction=0.60,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
        ),
    )

    # ── Fulcrum (dynamic — robot must slide under plank to create lever) ──────
    fulcrum: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Fulcrum",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=FULCRUM_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.12),   # 15×15×12 cm pivot block
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.2,
                angular_damping=0.2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=FULCRUM_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.90,
                dynamic_friction=0.70,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
        ),
    )

    # ── Contact sensor on plank — logs lever_contact_events ───────────────────
    plank_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Plank",
        history_length=1,
        update_period=0.0,
    )


# ── Environment ───────────────────────────────────────────────────────────────

class WeightLeverEnv(ToolUseEnv):
    """
    Curriculum Stage 4 — Weight Lever task.

    Unit-test checklist (§3.3):
      ✓ reset 100 times, no NaN/Inf, all objects above floor
      ✓ random-agent success rate < 0.1%
      ✓ lever_contact_events logged
    """

    cfg: WeightLeverEnvCfg

    def __init__(self, cfg: WeightLeverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._plank_nominal   = torch.tensor(PLANK_NOMINAL_POS,   device=self.device)
        self._heavy_nominal   = torch.tensor(HEAVY_NOMINAL_POS,   device=self.device)
        self._fulcrum_nominal = torch.tensor(FULCRUM_NOMINAL_POS, device=self.device)

        # Per-env mass buffers (CPU for PhysX API)
        self._plank_mass   = torch.full((self.num_envs,), PLANK_NOMINAL_MASS,   device=self.device)
        self._heavy_mass   = torch.full((self.num_envs,), HEAVY_NOMINAL_MASS,   device=self.device)
        self._fulcrum_mass = torch.full((self.num_envs,), FULCRUM_NOMINAL_MASS, device=self.device)

        # Plank friction buffer (num_envs, max_shapes=1, 3=[static, dynamic, restitution])
        self._plank_friction = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._plank_friction[:, 0, 0] = PLANK_NOMINAL_STATIC_FRICTION
        self._plank_friction[:, 0, 1] = PLANK_NOMINAL_DYNAMIC_FRICTION

        # Per-env heavy-object world Z at spawn — for lift detection
        self._heavy_spawn_z = torch.zeros(self.num_envs, device=self.device)

    # ── Object setup ──────────────────────────────────────────────────────────

    def _setup_objects(self) -> None:
        self.plank   = RigidObject(self.cfg.plank)
        self.heavy   = RigidObject(self.cfg.heavy)
        self.fulcrum = RigidObject(self.cfg.fulcrum)
        self.plank_contact = ContactSensor(self.cfg.plank_contact)

        self.scene.rigid_objects["plank"]   = self.plank
        self.scene.rigid_objects["heavy"]   = self.heavy
        self.scene.rigid_objects["fulcrum"] = self.fulcrum
        self.scene.sensors["plank_contact"] = self.plank_contact

    # ── Task observations (critic only) ───────────────────────────────────────

    def _get_task_obs(self) -> torch.Tensor:
        pelvis_pos = self.robot.data.root_pos_w   # (N, 3)

        heavy_pos_rel   = self.heavy.data.root_pos_w   - pelvis_pos   # (N, 3)
        heavy_lin_vel   = self.heavy.data.root_lin_vel_w               # (N, 3)
        heavy_ang_vel   = self.heavy.data.root_ang_vel_w               # (N, 3)
        heavy_mass      = self._heavy_mass.unsqueeze(-1)                # (N, 1)

        plank_pos_rel   = self.plank.data.root_pos_w   - pelvis_pos   # (N, 3)
        plank_lin_vel   = self.plank.data.root_lin_vel_w               # (N, 3)
        plank_ang_vel   = self.plank.data.root_ang_vel_w               # (N, 3)
        plank_mass      = self._plank_mass.unsqueeze(-1)                # (N, 1)

        fulcrum_pos_rel = self.fulcrum.data.root_pos_w - pelvis_pos   # (N, 3)
        fulcrum_lin_vel = self.fulcrum.data.root_lin_vel_w             # (N, 3)
        fulcrum_ang_vel = self.fulcrum.data.root_ang_vel_w             # (N, 3)
        fulcrum_mass    = self._fulcrum_mass.unsqueeze(-1)              # (N, 1)

        return torch.cat([
            heavy_pos_rel,   heavy_lin_vel,   heavy_ang_vel,   heavy_mass,
            plank_pos_rel,   plank_lin_vel,   plank_ang_vel,   plank_mass,
            fulcrum_pos_rel, fulcrum_lin_vel, fulcrum_ang_vel, fulcrum_mass,
        ], dim=-1)   # (N, 30)

    # ── Success detection ─────────────────────────────────────────────────────

    def _compute_success(self) -> torch.Tensor:
        heavy_z  = self.heavy.data.root_pos_w[:, 2]           # (N,) world Z
        success  = heavy_z > self._heavy_spawn_z + SUCCESS_LIFT

        # Log lever diagnostic: any contact on the plank
        net_forces = self.plank_contact.data.net_forces_w      # (N, 1, 3)
        contact    = torch.norm(net_forces, dim=-1).squeeze(-1) > 0.5
        self.extras["lever_contact_events"] = contact.float().mean().item()

        # Log heavy lift diagnostic
        lift = heavy_z - self._heavy_spawn_z                   # (N,) lift in metres
        self.extras["heavy_lift_mean"] = lift.mean().item()

        return success

    # ── Object reset ──────────────────────────────────────────────────────────

    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        n       = len(env_ids)
        origins = self.scene.env_origins[env_ids]   # (n, 3)
        all_ids = torch.arange(self.num_envs, device=self.device)
        unit_quat = torch.zeros(n, 4, device=self.device)
        unit_quat[:, 0] = 1.0

        # ── Plank: XY jitter ± absolute range, random yaw ────────────────────
        plank_nom = self._plank_nominal.unsqueeze(0).expand(n, -1).clone()
        rand_xy   = torch.rand(n, 2, device=self.device) * 2 - 1
        plank_pos = plank_nom.clone()
        plank_pos[:, 0] += rand_xy[:, 0] * PLANK_X_RANGE
        plank_pos[:, 1] += rand_xy[:, 1] * PLANK_Y_RANGE
        plank_pos = plank_pos + origins

        yaw  = torch.rand(n, device=self.device) * 2 * torch.pi
        half = yaw * 0.5
        plank_quat = torch.stack([
            torch.cos(half),
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.sin(half),
        ], dim=-1)

        self.plank.write_root_state_to_sim(
            torch.cat([plank_pos, plank_quat, torch.zeros(n, 6, device=self.device)], dim=-1),
            env_ids,
        )

        # ── Plank mass + friction randomisation ───────────────────────────────
        mass_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * MASS_RANGE
        self._plank_mass[env_ids] = PLANK_NOMINAL_MASS * mass_scale
        self.plank.root_physx_view.set_masses(
            self._plank_mass.unsqueeze(-1).cpu(), all_ids.cpu()
        )

        fric_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * FRICTION_RANGE
        self._plank_friction[env_ids, 0, 0] = PLANK_NOMINAL_STATIC_FRICTION  * fric_scale
        self._plank_friction[env_ids, 0, 1] = PLANK_NOMINAL_DYNAMIC_FRICTION * fric_scale
        self.plank.root_physx_view.set_material_properties(
            self._plank_friction.cpu(), all_ids.cpu()
        )

        # ── Heavy object: XY jitter ± absolute range ──────────────────────────
        heavy_nom = self._heavy_nominal.unsqueeze(0).expand(n, -1).clone()
        rand_xy   = torch.rand(n, 2, device=self.device) * 2 - 1
        heavy_pos = heavy_nom.clone()
        heavy_pos[:, 0] += rand_xy[:, 0] * HEAVY_X_RANGE
        heavy_pos[:, 1] += rand_xy[:, 1] * HEAVY_Y_RANGE
        heavy_pos = heavy_pos + origins

        self.heavy.write_root_state_to_sim(
            torch.cat([heavy_pos, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1),
            env_ids,
        )

        # Record world Z spawn height for lift detection
        self._heavy_spawn_z[env_ids] = heavy_pos[:, 2]

        # ── Heavy mass randomisation ───────────────────────────────────────────
        h_mass_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * MASS_RANGE
        self._heavy_mass[env_ids] = HEAVY_NOMINAL_MASS * h_mass_scale
        self.heavy.root_physx_view.set_masses(
            self._heavy_mass.unsqueeze(-1).cpu(), all_ids.cpu()
        )

        # ── Fulcrum: XY jitter ± absolute range ───────────────────────────────
        fulcrum_nom = self._fulcrum_nominal.unsqueeze(0).expand(n, -1).clone()
        rand_xy     = torch.rand(n, 2, device=self.device) * 2 - 1
        fulcrum_pos = fulcrum_nom.clone()
        fulcrum_pos[:, 0] += rand_xy[:, 0] * FULCRUM_X_RANGE
        fulcrum_pos[:, 1] += rand_xy[:, 1] * FULCRUM_Y_RANGE
        fulcrum_pos = fulcrum_pos + origins

        self.fulcrum.write_root_state_to_sim(
            torch.cat([fulcrum_pos, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1),
            env_ids,
        )

        # ── Fulcrum mass randomisation ─────────────────────────────────────────
        f_mass_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * MASS_RANGE
        self._fulcrum_mass[env_ids] = FULCRUM_NOMINAL_MASS * f_mass_scale
        self.fulcrum.root_physx_view.set_masses(
            self._fulcrum_mass.unsqueeze(-1).cpu(), all_ids.cpu()
        )
