"""
Task 5 — Composite (Hard, Curriculum Stage 5).

Goal:   Navigate through a wall gap, then press an elevated button (1.8 m) using a box.
Latent: Walk through gap → push box under button → climb on box → press button.

Combines ElevatedButton (Task 2) with a locomotion obstacle: a kinematic wall at
X=1.5 m with a 0.8 m central gap the robot must navigate through first. Box and
button are placed 3 m ahead of spawn, requiring full traversal before interaction.

Success criterion: Contact force on the button pad ≥ FORCE_THRESHOLD Newtons.

Scene layout (env-local, robot at origin facing +X):
  Wall        : kinematic, X=1.5 m; two pieces flanking 0.8 m central gap
  Box         : 30×30×30 cm, 20 kg, at ~(3.0, 0.0, 0.15) behind wall
  Button post : kinematic 8×8 cm post, full height 1.8 m
  Button pad  : 20×20 cm kinematic pad at Z=1.8 m

Observation extras (critic only, 10-per-object × 4 objects = 40 dims):
  box_pos_rel        (3)  box_vel       (6)  box_mass    (1)
  button_pos_rel     (3)  button_vel    (6)  button_mass (1)
  wall_left_pos_rel  (3)  wall_l_vel    (6)  wall_l_mass (1)
  wall_right_pos_rel (3)  wall_r_vel    (6)  wall_r_mass (1)
  Total extra (40)

Critic obs dim = BASE_CRITIC_DIM (112) + 40 = 152
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

N_OBJECTS      = 4
TASK_OBS_DIM   = N_OBJECTS * (3 + 6 + 1)          # 40
CRITIC_OBS_DIM = BASE_CRITIC_DIM + TASK_OBS_DIM   # 152

FORCE_THRESHOLD = 2.0   # Newtons — same press requirement as Task 2

# ── Obstacle wall geometry ────────────────────────────────────────────────────
WALL_X           = 1.5    # metres ahead of robot spawn
WALL_THICKNESS   = 0.10   # depth along X
WALL_HEIGHT      = 2.00   # tall enough to be impassable
WALL_GAP_HALF    = 0.40   # ±0.4 m gap centred at Y=0 → 0.8 m walkable gap
WALL_PIECE_W_Y   = 0.80   # each wall piece spans 0.8 m in Y

# Piece centres (env-local):  left at Y=-0.8, right at Y=+0.8
WALL_LEFT_POS  = (WALL_X, -(WALL_GAP_HALF + WALL_PIECE_W_Y / 2), WALL_HEIGHT / 2)
WALL_RIGHT_POS = (WALL_X,  (WALL_GAP_HALF + WALL_PIECE_W_Y / 2), WALL_HEIGHT / 2)

# ── Box and button behind wall ────────────────────────────────────────────────
BOX_NOMINAL_POS    = (3.0, 0.0, 0.15)   # 30×30×30 cm cube, half-height = 0.15
BUTTON_NOMINAL_POS = (3.5, 0.0, 1.80)   # pad centre

# Randomisation
BOX_POS_RANGE    = 0.25   # ±0.25 m absolute XY jitter on box
BUTTON_XY_RANGE  = 0.15   # ±0.15 m absolute XY jitter on button
MASS_RANGE       = 0.30   # ±30% mass
FRICTION_RANGE   = 0.50   # ±50% friction

BOX_NOMINAL_MASS             = 20.0
BUTTON_NOMINAL_MASS          = 0.10
WALL_NOMINAL_MASS            = 100.0   # kinematic but needs a value
BOX_NOMINAL_STATIC_FRICTION  = 0.80
BOX_NOMINAL_DYNAMIC_FRICTION = 0.60

# Subtask thresholds
BOX_UNDER_BTN_XY_THRESH = 0.35   # m — box XY must be within this of button XY
ROBOT_ON_BOX_PELVIS_Z   = 0.80   # m above env-origin — pelvis height when on box
# (robot pelvis on flat floor ≈ 0.65 m; on 0.30 m box ≈ 0.95 m → 0.80 is the midpoint)


# ── Env config ────────────────────────────────────────────────────────────────

@configclass
class CompositeEnvCfg(ToolUseEnvCfg):
    """Config for Task 5 — Composite."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=6.0, replicate_physics=True
    )

    observation_space: int = 109
    state_space: int = CRITIC_OBS_DIM   # 152
    episode_length_s: float = 60.0      # 3000 steps @ 50 Hz

    # ── Box (dynamic — robot pushes it under button then climbs on it) ────────
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CompBox",
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

    # ── Button post (kinematic — support column, visual only) ─────────────────
    button_post: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CompButtonPost",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BUTTON_NOMINAL_POS[0], BUTTON_NOMINAL_POS[1], BUTTON_NOMINAL_POS[2] / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, BUTTON_NOMINAL_POS[2]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
    )

    # ── Button pad (kinematic — robot must climb to press) ────────────────────
    button: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CompButton",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BUTTON_NOMINAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, 0.05),
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

    # ── Obstacle wall — left piece (kinematic) ────────────────────────────────
    wall_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CompWallLeft",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=WALL_LEFT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, WALL_PIECE_W_Y, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=WALL_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.55)),
        ),
    )

    # ── Obstacle wall — right piece (kinematic) ───────────────────────────────
    wall_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CompWallRight",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=WALL_RIGHT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, WALL_PIECE_W_Y, WALL_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=WALL_NOMINAL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.55)),
        ),
    )

    # ── Contact sensor on button pad ──────────────────────────────────────────
    button_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/CompButton",
        history_length=1,
        update_period=0.0,
    )

    # ── Contact sensor on box — detects climbing ──────────────────────────────
    box_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/CompBox",
        history_length=1,
        update_period=0.0,
    )


# ── Environment ───────────────────────────────────────────────────────────────

class CompositeEnv(ToolUseEnv):
    """
    Curriculum Stage 5 — Composite task.

    Unit-test checklist (§3.3):
      ✓ reset 100 times, no NaN/Inf, all objects above floor
      ✓ box and button spawn behind wall (X > WALL_X)
      ✓ random-agent success rate < 0.1%
      ✓ subtask diagnostics logged: box_placed_rate, robot_on_box_rate, button_contact_events
    """

    cfg: CompositeEnvCfg

    def __init__(self, cfg: CompositeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._box_nominal    = torch.tensor(BOX_NOMINAL_POS,    device=self.device)
        self._button_nominal = torch.tensor(BUTTON_NOMINAL_POS, device=self.device)

        # Per-env box mass and friction buffers (CPU for PhysX API)
        self._box_mass = torch.full((self.num_envs,), BOX_NOMINAL_MASS, device=self.device)
        self._box_friction = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._box_friction[:, 0, 0] = BOX_NOMINAL_STATIC_FRICTION
        self._box_friction[:, 0, 1] = BOX_NOMINAL_DYNAMIC_FRICTION

        # Per-env subtask progress — latched True once achieved, cleared on reset
        self._box_placed_buf   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._robot_on_box_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ── Object setup ──────────────────────────────────────────────────────────

    def _setup_objects(self) -> None:
        self.box          = RigidObject(self.cfg.box)
        self.button_post  = RigidObject(self.cfg.button_post)
        self.button       = RigidObject(self.cfg.button)
        self.wall_left    = RigidObject(self.cfg.wall_left)
        self.wall_right   = RigidObject(self.cfg.wall_right)
        self.button_contact = ContactSensor(self.cfg.button_contact)
        self.box_contact    = ContactSensor(self.cfg.box_contact)

        self.scene.rigid_objects["comp_box"]          = self.box
        self.scene.rigid_objects["comp_button_post"]  = self.button_post
        self.scene.rigid_objects["comp_button"]       = self.button
        self.scene.rigid_objects["comp_wall_left"]    = self.wall_left
        self.scene.rigid_objects["comp_wall_right"]   = self.wall_right
        self.scene.sensors["comp_button_contact"]     = self.button_contact
        self.scene.sensors["comp_box_contact"]        = self.box_contact

    # ── Task observations (critic only) ───────────────────────────────────────

    def _get_task_obs(self) -> torch.Tensor:
        pelvis_pos = self.robot.data.root_pos_w   # (N, 3)
        zeros6 = torch.zeros(self.num_envs, 6, device=self.device)

        box_pos_rel = self.box.data.root_pos_w   - pelvis_pos   # (N, 3)
        box_lin_vel = self.box.data.root_lin_vel_w               # (N, 3)
        box_ang_vel = self.box.data.root_ang_vel_w               # (N, 3)
        box_mass    = self._box_mass.unsqueeze(-1)                # (N, 1)

        btn_pos_rel = self.button.data.root_pos_w - pelvis_pos   # (N, 3)
        btn_mass    = torch.full((self.num_envs, 1), BUTTON_NOMINAL_MASS, device=self.device)

        wl_pos_rel  = self.wall_left.data.root_pos_w  - pelvis_pos   # (N, 3)
        wl_mass     = torch.full((self.num_envs, 1), WALL_NOMINAL_MASS, device=self.device)

        wr_pos_rel  = self.wall_right.data.root_pos_w - pelvis_pos   # (N, 3)
        wr_mass     = torch.full((self.num_envs, 1), WALL_NOMINAL_MASS, device=self.device)

        return torch.cat([
            box_pos_rel, box_lin_vel,  box_ang_vel,  box_mass,
            btn_pos_rel, zeros6,       btn_mass,
            wl_pos_rel,  zeros6,       wl_mass,
            wr_pos_rel,  zeros6,       wr_mass,
        ], dim=-1)   # (N, 40)

    # ── Success detection + subtask diagnostics ───────────────────────────────

    def _compute_success(self) -> torch.Tensor:
        # Subtask 1 — box positioned under button XY
        box_xy = self.box.data.root_pos_w[:, :2]       # (N, 2)
        btn_xy = self.button.data.root_pos_w[:, :2]    # (N, 2)
        self._box_placed_buf |= torch.norm(box_xy - btn_xy, dim=-1) < BOX_UNDER_BTN_XY_THRESH

        # Subtask 2 — robot elevated above floor level (proxy for standing on box)
        # Pelvis on floor ≈ env_origin_z + 0.65 m; on box ≈ env_origin_z + 0.95 m
        pelvis_z    = self.robot.data.root_pos_w[:, 2]       # (N,)
        env_orig_z  = self.scene.env_origins[:, 2]           # (N,)
        self._robot_on_box_buf |= pelvis_z > env_orig_z + ROBOT_ON_BOX_PELVIS_Z

        # Success — button pressed
        net_forces = self.button_contact.data.net_forces_w     # (N, 1, 3)
        force_norm = torch.norm(net_forces, dim=-1).squeeze(-1)  # (N,)
        success    = force_norm > FORCE_THRESHOLD

        # Diagnostics
        box_forces   = self.box_contact.data.net_forces_w      # (N, 1, 3)
        box_pressed  = torch.norm(box_forces, dim=-1).squeeze(-1) > 5.0
        self.extras["button_contact_events"] = success.float().mean().item()
        self.extras["box_placed_rate"]       = self._box_placed_buf.float().mean().item()
        self.extras["robot_on_box_rate"]     = self._robot_on_box_buf.float().mean().item()
        self.extras["box_load_now"]          = box_pressed.float().mean().item()

        return success

    # ── Object reset ──────────────────────────────────────────────────────────

    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        n       = len(env_ids)
        origins = self.scene.env_origins[env_ids]   # (n, 3)
        all_ids = torch.arange(self.num_envs, device=self.device)
        unit_quat = torch.zeros(n, 4, device=self.device)
        unit_quat[:, 0] = 1.0

        # Clear subtask latches
        self._box_placed_buf[env_ids]   = False
        self._robot_on_box_buf[env_ids] = False

        # ── Box: XY jitter ± absolute range, random yaw ──────────────────────
        box_nom = self._box_nominal.unsqueeze(0).expand(n, -1).clone()
        rand_xy = (torch.rand(n, 2, device=self.device) * 2 - 1) * BOX_POS_RANGE
        box_pos = box_nom.clone()
        box_pos[:, 0] += rand_xy[:, 0]
        box_pos[:, 1] += rand_xy[:, 1]
        box_pos = box_pos + origins

        yaw  = torch.rand(n, device=self.device) * 2 * torch.pi
        half = yaw * 0.5
        box_quat = torch.stack([
            torch.cos(half),
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.sin(half),
        ], dim=-1)

        self.box.write_root_state_to_sim(
            torch.cat([box_pos, box_quat, torch.zeros(n, 6, device=self.device)], dim=-1),
            env_ids,
        )

        # ── Box mass + friction randomisation ─────────────────────────────────
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

        # ── Button + post: same XY jitter ─────────────────────────────────────
        btn_nom = self._button_nominal.unsqueeze(0).expand(n, -1).clone()
        btn_jitter = (torch.rand(n, 2, device=self.device) * 2 - 1) * BUTTON_XY_RANGE
        btn_pos = btn_nom.clone()
        btn_pos[:, 0] += btn_jitter[:, 0]
        btn_pos[:, 1] += btn_jitter[:, 1]
        btn_pos = btn_pos + origins

        btn_state = torch.cat([btn_pos, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.button.write_root_state_to_sim(btn_state, env_ids)

        post_pos = btn_pos.clone()
        post_pos[:, 2] = origins[:, 2] + BUTTON_NOMINAL_POS[2] / 2.0
        post_state = torch.cat([post_pos, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.button_post.write_root_state_to_sim(post_state, env_ids)

        # ── Obstacle walls: kinematic, fixed nominal positions ─────────────────
        wl_nom = torch.tensor(WALL_LEFT_POS,  device=self.device).unsqueeze(0).expand(n, -1)
        wr_nom = torch.tensor(WALL_RIGHT_POS, device=self.device).unsqueeze(0).expand(n, -1)

        self.wall_left.write_root_state_to_sim(
            torch.cat([wl_nom + origins, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1),
            env_ids,
        )
        self.wall_right.write_root_state_to_sim(
            torch.cat([wr_nom + origins, unit_quat, torch.zeros(n, 6, device=self.device)], dim=-1),
            env_ids,
        )
