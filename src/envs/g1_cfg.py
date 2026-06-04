"""
Isaac Lab ArticulationCfg for the Unitree G1 EDU (43 DOF, Dex3-1 hands).

Extends the upstream G1_29DOF_CFG with:
  - Dex3-1 hand actuators (14 DOF total, 7 per hand)
  - IMU sensor on pelvis link
  - Wrist force-torque sensors (left + right)
  - Egocentric depth camera on head link (64×64 @ 50 Hz)
  - Contact sensors on feet and hands

Usage in an Isaac Lab environment:
    from src.envs.g1_cfg import G1_EDU_CFG, G1_EDU_SENSOR_CFG
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import (
    CameraCfg,
    ContactSensorCfg,
    ImuCfg,
    FrameTransformerCfg,
)
from isaaclab.sensors.frame_transformer import OffsetCfg

# ── URDF / USD paths ──────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent.parent
_URDF_PATH = str(
    _REPO_ROOT
    / "deps/unitree_rl_gym/resources/robots/g1_description"
    / "g1_29dof_with_hand_rev_1_0.urdf"
)

# Isaac Lab can auto-convert URDF → USD on first load.
# Once converted the USD is cached under /tmp/isaaclab_urdf_cache/.
_SPAWN_CFG = sim_utils.UrdfFileCfg(
    asset_path=_URDF_PATH,
    activate_contact_sensors=True,
    fix_base=False,
    # joint_drive=None: skip URDF drive setup — ArticulationCfg.actuators handles all PD gains
    joint_drive=None,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        fix_root_link=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    ),
)

# ── Robot articulation config ─────────────────────────────────────────────────

G1_EDU_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(1.0, 0.0, 0.0, 0.0),  # wxyz — upright
        joint_pos={
            # Slight crouch for stable stand
            ".*_hip_pitch_joint":   -0.10,
            ".*_knee_joint":         0.30,
            ".*_ankle_pitch_joint": -0.20,
            # Arms hanging at sides
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint":  0.0,
            ".*_elbow_joint":          0.0,
            # Hands open
            ".*_thumb_.*":   0.0,
            ".*_index_.*":   0.0,
            ".*_middle_.*":  0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ── Legs ──────────────────────────────────────────────────────────────
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint":   88.0,
                ".*_hip_roll_joint":  88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint":     139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint":   32.0,
                ".*_hip_roll_joint":  32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint":      20.0,
            },
            stiffness={
                ".*_hip_yaw_joint":   100.0,
                ".*_hip_roll_joint":  100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint":      200.0,
            },
            damping={
                ".*_hip_yaw_joint":   2.5,
                ".*_hip_roll_joint":  2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint":      5.0,
            },
            armature={".*_hip_.*": 0.03, ".*_knee_joint": 0.03},
            saturation_effort=180.0,
        ),
        # ── Feet ──────────────────────────────────────────────────────────────
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint":  50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint":  37.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 80.0,   # raised from 20 — ankles must hold body weight
                ".*_ankle_roll_joint":  40.0,   # raised from 20 — lateral stability
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,    # raised from 0.2
                ".*_ankle_roll_joint":  1.0,    # raised from 0.1
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        # ── Waist (3 DOF) ──────────────────────────────────────────────────────
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit={
                "waist_yaw_joint":   88.0,
                "waist_roll_joint":  50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint":   32.0,
                "waist_roll_joint":  37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness=5000.0,
            damping=5.0,
            armature=0.001,
        ),
        # ── Arms (14 DOF, 7 per arm) ───────────────────────────────────────────
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=300.0,
            velocity_limit=100.0,
            stiffness=3000.0,
            damping=10.0,
            armature=0.001,
        ),
        # ── Dex3-1 hands (14 DOF, 7 per hand) ────────────────────────────────
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_thumb_0_joint",
                ".*_hand_thumb_1_joint",
                ".*_hand_thumb_2_joint",
                ".*_hand_index_0_joint",
                ".*_hand_index_1_joint",
                ".*_hand_middle_0_joint",
                ".*_hand_middle_1_joint",
            ],
            effort_limit=5.0,
            velocity_limit=10.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.001,
        ),
    },
)
"""G1 EDU ArticulationCfg with Dex3-1 hands (43 DOF total).

Joint groups:
  legs  : 12 DOF — hip yaw/roll/pitch, knee, ankle pitch/roll (×2)
  waist :  3 DOF — yaw, roll, pitch
  arms  : 14 DOF — shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw (×2)
  hands : 14 DOF — thumb 0/1/2, index 0/1, middle 0/1 (×2, Dex3-1)
"""


# ── Sensor configs ────────────────────────────────────────────────────────────

def make_imu_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot/pelvis") -> ImuCfg:
    """IMU on pelvis link — orientation + angular velocity."""
    return ImuCfg(
        prim_path=prim_path,
        update_period=0.005,  # 200 Hz — matches physics_dt
        gravity_bias=(0.0, 0.0, 0.0),
    )


def make_wrist_ft_cfg(side: str) -> FrameTransformerCfg:
    """
    Pseudo force-torque sensor via FrameTransformer on the wrist yaw link.
    Isaac Lab does not expose a dedicated F/T sensor; we track wrist-frame
    transforms and compute wrench in post-processing from contact forces.
    """
    assert side in ("left", "right")
    return FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_wrist_yaw_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_wrist_yaw_link",
                name=f"{side}_wrist_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )


def make_head_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/head_link",
    width: int = 64,
    height: int = 64,
    update_period: float = 0.02,  # 50 Hz
) -> CameraCfg:
    """Egocentric depth camera mounted on the head link."""
    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=height,
        width=width,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.05),   # slightly in front of and above head link origin
            rot=(0.5, -0.5, 0.5, -0.5),  # wxyz — looking forward
            convention="ros",
        ),
    )


def make_foot_contact_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    history_length: int = 3,
) -> ContactSensorCfg:
    """Contact sensors on both feet (used for fall detection and locomotion reward)."""
    return ContactSensorCfg(
        prim_path=prim_path,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
            "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        ],
        history_length=history_length,
        update_period=0.0,  # every physics step
        track_air_time=True,
    )


def make_hand_contact_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    history_length: int = 1,
) -> ContactSensorCfg:
    """Contact sensors on fingertips — used to detect tool_contact_events."""
    return ContactSensorCfg(
        prim_path=prim_path,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/left_hand_index_1_link",
            "{ENV_REGEX_NS}/Robot/left_hand_middle_1_link",
            "{ENV_REGEX_NS}/Robot/left_hand_thumb_2_link",
            "{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
            "{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
            "{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
        ],
        history_length=history_length,
        update_period=0.0,
    )


# ── Convenience bundle ────────────────────────────────────────────────────────

class G1_EDU_SENSOR_CFG:
    """All sensor configs for G1 EDU — instantiate and attach to your SceneCfg."""
    imu              = make_imu_cfg()
    left_wrist_ft    = make_wrist_ft_cfg("left")
    right_wrist_ft   = make_wrist_ft_cfg("right")
    head_camera      = make_head_camera_cfg()
    foot_contact     = make_foot_contact_cfg()
    hand_contact     = make_hand_contact_cfg()
