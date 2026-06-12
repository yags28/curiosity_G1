"""
Task 1 — Distant Target (Easy, Curriculum Stage 1).

Goal:   Touch a force-sensitive pad mounted 1.2 m in front of the robot
        (beyond direct arm reach of ≈0.7 m from standing position).
Latent: Pick up an 80–120 cm PVC rod and use it to bridge the distance gap.

Success criterion: Contact force on the target pad ≥ FORCE_THRESHOLD Newtons.

Observation extras (critic only):
  stick_pos_rel  (3)  — stick position relative to pelvis
  stick_vel      (6)  — stick linear + angular velocity
  stick_mass     (1)  — privileged mass (varies ±30%)
  target_pos_rel (3)  — target pad position relative to pelvis
  target_vel     (6)  — target velocity (zero, it is kinematic)
  target_mass    (1)  — privileged mass
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

N_OBJECTS = 2          # stick + target (distractors not included in obs)
TASK_OBS_DIM = N_OBJECTS * (3 + 6 + 1)   # pos_rel + vel + mass = 10 × 2 = 20
# Closed-loop: actor also sees stick + target relative position (6) so the
# policy can aim the stick rather than acting open-loop (sim-to-sim transfer).
ACTOR_TASK_OBS_DIM = N_OBJECTS * 3       # stick_pos_rel + target_pos_rel = 6
ACTOR_OBS_DIM_T1 = 109 + ACTOR_TASK_OBS_DIM            # 115
CRITIC_OBS_DIM = BASE_CRITIC_DIM + ACTOR_TASK_OBS_DIM + TASK_OBS_DIM  # 112+6+20=138

FORCE_THRESHOLD = 0.5  # Newtons — contact force required for success

# Nominal object positions (in env-local frame, env origin subtracted)
STICK_NOMINAL_POS  = (0.5,  0.4, 0.5)   # beside robot, within arm's reach
TARGET_NOMINAL_POS = (1.2,  0.0, 0.8)   # beyond arm's reach, at waist height

# Randomisation ranges (§3.4)
POS_RANGE     = 0.30   # ±30% of nominal XY
MASS_RANGE    = 0.30   # ±30% of nominal mass
FRICTION_RANGE = 0.50  # ±50% of nominal friction coefficients

STICK_NOMINAL_MASS            = 0.30   # kg — PVC rod ~90 cm
TARGET_NOMINAL_MASS           = 0.50   # kg
STICK_NOMINAL_STATIC_FRICTION  = 0.80
STICK_NOMINAL_DYNAMIC_FRICTION = 0.60


# ── Env config ────────────────────────────────────────────────────────────────

@configclass
class DistantTargetEnvCfg(ToolUseEnvCfg):
    """Config for Task 1 — Distant Target."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    observation_space: int = ACTOR_OBS_DIM_T1   # 115 (proprio 109 + stick/target rel 6)
    state_space: int = CRITIC_OBS_DIM           # 138
    episode_length_s: float = 40.0

    # ── Stick (rigid, dynamic) ────────────────────────────────────────────────
    stick: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Stick",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=STICK_NOMINAL_POS,
            rot=(0.7071, 0.0, 0.7071, 0.0),  # horizontal (wxyz)
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
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
    )

    # ── Target pad (kinematic — fixed to ground post) ─────────────────────────
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TARGET_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, 0.02),   # 20×20 cm pad
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=TARGET_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
    )

    # ── Contact sensor on target pad ──────────────────────────────────────────
    target_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Target",
        history_length=1,
        update_period=0.0,
    )

    # ── Distractor 1 ──────────────────────────────────────────────────────────
    distractor_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Distractor0",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.4, 0.15)),
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.30),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
    )

    # ── Distractor 2 ──────────────────────────────────────────────────────────
    distractor_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Distractor1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, 0.5, 0.15)),
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.25, 0.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
        ),
    )


# ── Environment ───────────────────────────────────────────────────────────────

class DistantTargetEnv(ToolUseEnv):
    """
    Curriculum Stage 1 — Distant Target task.

    Unit-test checklist (§3.3):
      ✓ reset 100 times, no interpenetration / NaN
      ✓ random-agent success rate < 0.1%
      ✓ tool_contact_events logged
    """

    cfg: DistantTargetEnvCfg

    def __init__(self, cfg: DistantTargetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Cached nominal positions (env-local, without env origins)
        self._stick_nominal  = torch.tensor(STICK_NOMINAL_POS, device=self.device)
        self._target_nominal = torch.tensor(TARGET_NOMINAL_POS, device=self.device)
        # Per-env mass (set during reset, used for privileged obs)
        self._stick_mass = torch.full((self.num_envs,), STICK_NOMINAL_MASS, device=self.device)
        # Per-env friction (num_envs, max_shapes=1, 3=[static, dynamic, restitution])
        self._stick_friction = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._stick_friction[:, 0, 0] = STICK_NOMINAL_STATIC_FRICTION
        self._stick_friction[:, 0, 1] = STICK_NOMINAL_DYNAMIC_FRICTION

    # ── Object setup ──────────────────────────────────────────────────────────

    def _setup_objects(self) -> None:
        self.stick  = RigidObject(self.cfg.stick)
        self.target = RigidObject(self.cfg.target)
        self.distractor_0 = RigidObject(self.cfg.distractor_0)
        self.distractor_1 = RigidObject(self.cfg.distractor_1)
        self.target_contact = ContactSensor(self.cfg.target_contact)

        self.scene.rigid_objects["stick"]        = self.stick
        self.scene.rigid_objects["target"]       = self.target
        self.scene.rigid_objects["distractor_0"] = self.distractor_0
        self.scene.rigid_objects["distractor_1"] = self.distractor_1
        self.scene.sensors["target_contact"]     = self.target_contact

    # ── Actor object obs (closed-loop) ────────────────────────────────────────

    def _get_actor_task_obs(self) -> torch.Tensor:
        """Stick + target position relative to pelvis (N, 6) — actor-visible."""
        pelvis_pos = self.robot.data.root_pos_w
        stick_rel  = self.stick.data.root_pos_w  - pelvis_pos
        target_rel = self.target.data.root_pos_w - pelvis_pos
        return torch.cat([stick_rel, target_rel], dim=-1)

    # ── Task observations (critic only) ───────────────────────────────────────

    def _get_task_obs(self) -> torch.Tensor:
        pelvis_pos = self.robot.data.root_pos_w  # (N, 3)

        stick_pos_rel = self.stick.data.root_pos_w - pelvis_pos        # (N, 3)
        stick_lin_vel = self.stick.data.root_lin_vel_w                 # (N, 3)
        stick_ang_vel = self.stick.data.root_ang_vel_w                 # (N, 3)
        stick_mass    = self._stick_mass.unsqueeze(-1)                  # (N, 1)

        target_pos_rel = self.target.data.root_pos_w - pelvis_pos      # (N, 3)
        target_vel     = torch.zeros(self.num_envs, 6, device=self.device)  # kinematic
        target_mass    = torch.full((self.num_envs, 1), TARGET_NOMINAL_MASS, device=self.device)

        return torch.cat([
            stick_pos_rel, stick_lin_vel, stick_ang_vel, stick_mass,
            target_pos_rel, target_vel, target_mass,
        ], dim=-1)  # (N, 20)

    # ── Success detection ─────────────────────────────────────────────────────

    def _compute_success(self) -> torch.Tensor:
        net_forces = self.target_contact.data.net_forces_w   # (N, 1, 3)
        force_norm = torch.norm(net_forces, dim=-1).squeeze(-1)  # (N,)
        success = force_norm > FORCE_THRESHOLD
        # Log tool_contact_events diagnostic (§3.3 checklist)
        self.extras["tool_contact_events"] = success.float().mean().item()
        return success

    # ── Object reset ──────────────────────────────────────────────────────────

    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]   # (n, 3)

        # ── Stick: randomise XY position ±30%, random yaw ────────────────────
        stick_nom = self._stick_nominal.unsqueeze(0).expand(n, -1).clone()
        delta_xy = (torch.rand(n, 2, device=self.device) * 2 - 1) * POS_RANGE * stick_nom[:, :2].abs()
        stick_pos = stick_nom.clone()
        stick_pos[:, :2] += delta_xy
        stick_pos = stick_pos + origins

        yaw = torch.rand(n, device=self.device) * 2 * torch.pi
        half = yaw * 0.5
        stick_quat = torch.stack([
            torch.cos(half),
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.sin(half),
        ], dim=-1)  # wxyz

        stick_state = torch.cat([stick_pos, stick_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.stick.write_root_state_to_sim(stick_state, env_ids)

        # ── Stick mass + friction: ±30% / ±50% randomisation (§3.4) ─────────
        # Both physx_view calls require full-env tensor + explicit all-env indices.
        all_ids = torch.arange(self.num_envs, device=self.device)

        mass_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * MASS_RANGE
        self._stick_mass[env_ids] = STICK_NOMINAL_MASS * mass_scale
        self.stick.root_physx_view.set_masses(
            self._stick_mass.unsqueeze(-1).cpu(), all_ids.cpu()
        )

        fric_scale = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * FRICTION_RANGE
        self._stick_friction[env_ids, 0, 0] = STICK_NOMINAL_STATIC_FRICTION  * fric_scale
        self._stick_friction[env_ids, 0, 1] = STICK_NOMINAL_DYNAMIC_FRICTION * fric_scale
        # Domain randomisation: restitution (contact bounce) ∈ [0, 0.4] — the
        # contact property that differs most across physics engines, so the
        # policy learns a stick interaction robust to the sim-to-sim gap.
        self._stick_friction[env_ids, 0, 2] = torch.rand(n, device=self.device) * 0.4
        self.stick.root_physx_view.set_material_properties(
            self._stick_friction.cpu(), all_ids.cpu()
        )

        # ── Target: fixed (kinematic), no randomisation needed ───────────────
        target_pos = self._target_nominal.unsqueeze(0).expand(n, -1) + origins
        target_quat = torch.zeros(n, 4, device=self.device)
        target_quat[:, 0] = 1.0
        target_state = torch.cat([target_pos, target_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.target.write_root_state_to_sim(target_state, env_ids)

        # ── Distractors: randomise within ±50% of nominal ────────────────────
        for dist_obj, dist_nom_pos in [
            (self.distractor_0, (0.3, -0.4, 0.15)),
            (self.distractor_1, (-0.3, 0.5, 0.15)),
        ]:
            nom = torch.tensor(dist_nom_pos, device=self.device).unsqueeze(0).expand(n, -1)
            jitter = (torch.rand(n, 3, device=self.device) * 2 - 1) * 0.4 * nom.abs()
            jitter[:, 2] = 0.0  # keep height
            pos = nom + jitter + origins
            quat = torch.zeros(n, 4, device=self.device); quat[:, 0] = 1.0
            state = torch.cat([pos, quat, torch.zeros(n, 6, device=self.device)], dim=-1)
            dist_obj.write_root_state_to_sim(state, env_ids)
