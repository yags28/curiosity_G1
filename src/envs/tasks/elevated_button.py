"""
Task 2 — Elevated Button (Medium, Curriculum Stage 2).

Goal:   Press a button pad mounted 1.8 m above the floor — beyond the robot's
        standing reach of ~1.0 m from pelvis.
Latent: Push a 30×30×30 cm box under the button, step onto it, then press.

Success criterion: Contact force on the button pad ≥ FORCE_THRESHOLD Newtons.

Observation extras (critic only, same 10-per-object layout as Task 1):
  box_pos_rel  (3)  — box position relative to pelvis
  box_vel      (6)  — box linear + angular velocity
  box_mass     (1)  — privileged mass (varies ±30%)
  btn_pos_rel  (3)  — button position relative to pelvis
  btn_vel      (6)  — button velocity (zero, kinematic)
  btn_mass     (1)  — privileged mass (nominal)
  Total extra  (20)

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

N_OBJECTS    = 2
TASK_OBS_DIM = N_OBJECTS * (3 + 6 + 1)          # pos_rel + vel + mass = 20
CRITIC_OBS_DIM = BASE_CRITIC_DIM + TASK_OBS_DIM  # 132

FORCE_THRESHOLD = 2.0  # Newtons — stronger press required than Task 1

# Nominal positions (env-local, env origin subtracted)
BOX_NOMINAL_POS    = (0.8,  0.0, 0.15)   # directly in front of robot, on floor
BUTTON_NOMINAL_POS = (1.5,  0.0, 1.80)   # 1.5 m forward, 1.8 m height

# Randomisation ranges (§3.4)
POS_RANGE       = 0.30   # ±30% box XY
BUTTON_XY_RANGE = 0.20   # ±20% button XY (prevents memorising fixed location)
MASS_RANGE      = 0.30   # ±30% box mass
FRICTION_RANGE  = 0.50   # ±50% box friction

BOX_NOMINAL_MASS             = 20.0   # kg — heavy enough for G1 (~35 kg) to stand on
BUTTON_NOMINAL_MASS          = 0.10   # kg
BOX_NOMINAL_STATIC_FRICTION  = 0.80
BOX_NOMINAL_DYNAMIC_FRICTION = 0.60


# ── Env config ────────────────────────────────────────────────────────────────

@configclass
class ElevatedButtonEnvCfg(ToolUseEnvCfg):
    """Config for Task 2 — Elevated Button."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    observation_space: int = 109
    state_space: int = CRITIC_OBS_DIM   # 132
    episode_length_s: float = 40.0

    # ── Box (dynamic tool — robot pushes / climbs on it) ──────────────────────
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BOX_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.30, 0.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=BOX_NOMINAL_STATIC_FRICTION,
                dynamic_friction=BOX_NOMINAL_DYNAMIC_FRICTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.15)),
        ),
    )

    # ── Button post (kinematic — visual support column, collision disabled) ─────
    # Post centre at half the button height; button pad sits on top.
    button_post: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ButtonPost",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BUTTON_NOMINAL_POS[0], BUTTON_NOMINAL_POS[1],
                 BUTTON_NOMINAL_POS[2] / 2.0),   # half-height = 0.90 m
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, BUTTON_NOMINAL_POS[2]),  # 8×8 cm post, full height
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
    )

    # ── Button pad (kinematic — on top of post, robot must climb to press) ────
    button: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Button",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BUTTON_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, 0.05),   # flat 20×20 cm pad
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=BUTTON_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
    )

    # ── Contact sensor on button pad ──────────────────────────────────────────
    button_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Button",
        history_length=1,
        update_period=0.0,
    )

    # ── Contact sensor on box top — logs when robot climbs (§3.3 stub) ───────
    # Tracks all robot bodies contacting the box to detect climbing events.
    box_top_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Box",
        history_length=1,
        update_period=0.0,
    )

    # ── Distractor ────────────────────────────────────────────────────────────
    distractor_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Distractor0",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, 0.5, 0.10)),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.8)),
        ),
    )


# ── Environment ───────────────────────────────────────────────────────────────

class ElevatedButtonEnv(ToolUseEnv):
    """
    Curriculum Stage 2 — Elevated Button task.

    Unit-test checklist (§3.3):
      ✓ reset 100 times, no interpenetration / NaN
      ✓ random-agent success rate < 0.1%
      ✓ button_contact_events logged
    """

    cfg: ElevatedButtonEnvCfg

    def __init__(self, cfg: ElevatedButtonEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._box_nominal    = torch.tensor(BOX_NOMINAL_POS,    device=self.device)
        self._button_nominal = torch.tensor(BUTTON_NOMINAL_POS, device=self.device)
        # Per-env box mass and friction buffers (CPU for PhysX API)
        self._box_mass     = torch.full((self.num_envs,), BOX_NOMINAL_MASS, device=self.device)
        self._box_friction = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._box_friction[:, 0, 0] = BOX_NOMINAL_STATIC_FRICTION
        self._box_friction[:, 0, 1] = BOX_NOMINAL_DYNAMIC_FRICTION

    # ── Object setup ──────────────────────────────────────────────────────────

    def _setup_objects(self) -> None:
        self.box          = RigidObject(self.cfg.box)
        self.button_post  = RigidObject(self.cfg.button_post)
        self.button       = RigidObject(self.cfg.button)
        self.distractor_0 = RigidObject(self.cfg.distractor_0)
        self.button_contact   = ContactSensor(self.cfg.button_contact)
        self.box_top_contact  = ContactSensor(self.cfg.box_top_contact)

        self.scene.rigid_objects["box"]          = self.box
        self.scene.rigid_objects["button_post"]  = self.button_post
        self.scene.rigid_objects["button"]       = self.button
        self.scene.rigid_objects["distractor_0"] = self.distractor_0
        self.scene.sensors["button_contact"]     = self.button_contact
        self.scene.sensors["box_top_contact"]    = self.box_top_contact

    # ── Task observations (critic only) ───────────────────────────────────────

    def _get_task_obs(self) -> torch.Tensor:
        pelvis_pos = self.robot.data.root_pos_w   # (N, 3)

        box_pos_rel  = self.box.data.root_pos_w  - pelvis_pos   # (N, 3)
        box_lin_vel  = self.box.data.root_lin_vel_w              # (N, 3)
        box_ang_vel  = self.box.data.root_ang_vel_w              # (N, 3)
        box_mass     = self._box_mass.unsqueeze(-1)               # (N, 1)

        btn_pos_rel  = self.button.data.root_pos_w - pelvis_pos  # (N, 3)
        btn_vel      = torch.zeros(self.num_envs, 6, device=self.device)
        btn_mass     = torch.full((self.num_envs, 1), BUTTON_NOMINAL_MASS, device=self.device)

        return torch.cat([
            box_pos_rel, box_lin_vel, box_ang_vel, box_mass,
            btn_pos_rel, btn_vel, btn_mass,
        ], dim=-1)  # (N, 20)

    # ── Success detection ─────────────────────────────────────────────────────

    def _compute_success(self) -> torch.Tensor:
        net_forces = self.button_contact.data.net_forces_w   # (N, 1, 3)
        force_norm = torch.norm(net_forces, dim=-1).squeeze(-1)  # (N,)
        success = force_norm > FORCE_THRESHOLD
        self.extras["button_contact_events"] = success.float().mean().item()

        # Log climbing diagnostic: any robot body contacting the box
        box_forces = self.box_top_contact.data.net_forces_w  # (N, 1, 3)
        climbing = torch.norm(box_forces, dim=-1).squeeze(-1) > 5.0  # >5 N = significant load
        self.extras["robot_on_box"] = climbing.float().mean().item()

        return success

    # ── Object reset ──────────────────────────────────────────────────────────

    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        n       = len(env_ids)
        origins = self.scene.env_origins[env_ids]  # (n, 3)
        all_ids = torch.arange(self.num_envs, device=self.device)

        # ── Box: randomise XY ±30%, random yaw ───────────────────────────────
        box_nom   = self._box_nominal.unsqueeze(0).expand(n, -1).clone()
        delta_xy  = (torch.rand(n, 2, device=self.device) * 2 - 1) * POS_RANGE * box_nom[:, :2].abs()
        box_pos   = box_nom.clone()
        box_pos[:, :2] += delta_xy
        box_pos   = box_pos + origins

        yaw  = torch.rand(n, device=self.device) * 2 * torch.pi
        half = yaw * 0.5
        box_quat = torch.stack([
            torch.cos(half),
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.sin(half),
        ], dim=-1)   # wxyz

        box_state = torch.cat([box_pos, box_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.box.write_root_state_to_sim(box_state, env_ids)

        # ── Box mass + friction: ±30% / ±50% ─────────────────────────────────
        mass_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * MASS_RANGE
        self._box_mass[env_ids] = BOX_NOMINAL_MASS * mass_scale
        self.box.root_physx_view.set_masses(
            self._box_mass.unsqueeze(-1).cpu(), all_ids.cpu()
        )

        fric_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * FRICTION_RANGE
        self._box_friction[env_ids, 0, 0] = BOX_NOMINAL_STATIC_FRICTION  * fric_scale
        self._box_friction[env_ids, 0, 1] = BOX_NOMINAL_DYNAMIC_FRICTION * fric_scale
        self.box.root_physx_view.set_material_properties(
            self._box_friction.cpu(), all_ids.cpu()
        )

        # ── Button + post: same XY jitter ±20% (post follows button) ─────────
        btn_nom = self._button_nominal.unsqueeze(0).expand(n, -1).clone()
        btn_xy_jitter = (torch.rand(n, 2, device=self.device) * 2 - 1) * BUTTON_XY_RANGE * btn_nom[:, :2].abs()
        btn_pos = btn_nom.clone()
        btn_pos[:, :2] += btn_xy_jitter
        btn_pos = btn_pos + origins
        btn_quat = torch.zeros(n, 4, device=self.device)
        btn_quat[:, 0] = 1.0
        btn_state = torch.cat([btn_pos, btn_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.button.write_root_state_to_sim(btn_state, env_ids)

        # Post shares XY with button, Z at half button height
        post_pos = btn_pos.clone()
        post_pos[:, 2] = origins[:, 2] + BUTTON_NOMINAL_POS[2] / 2.0
        post_state = torch.cat([post_pos, btn_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.button_post.write_root_state_to_sim(post_state, env_ids)

        # ── Distractor ────────────────────────────────────────────────────────
        nom = torch.tensor((-0.4, 0.5, 0.10), device=self.device).unsqueeze(0).expand(n, -1)
        jitter = (torch.rand(n, 3, device=self.device) * 2 - 1) * 0.4 * nom.abs()
        jitter[:, 2] = 0.0
        dist_pos = nom + jitter + origins
        dist_quat = torch.zeros(n, 4, device=self.device); dist_quat[:, 0] = 1.0
        dist_state = torch.cat([dist_pos, dist_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.distractor_0.write_root_state_to_sim(dist_state, env_ids)
