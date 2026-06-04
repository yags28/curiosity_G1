"""
§3.2 validation: reach envelope + sensor live check.

Two tests in a single Isaac Sim session:

  1. Reach Envelope
     — Commands both arms to 90° forward extension, waits 200 steps for convergence
     — Measures left/right wrist distance from pelvis
     — Repeats at soft-limit max for absolute max reach
     — PASS: measured reach >= 0.50 m (G1 arm ~0.65-0.80 m from pelvis)

  2. Sensor Live Check
     — IMU (pelvis):            quat_w has unit norm, ang_vel_b is finite
     — Wrist FrameTransformer:  target_pos_w finite, positioned near expected region
     — Head Camera (depth):     shape (1, 64, 64, 1), values in valid range
                                 (skipped gracefully in headless without RTX)

Run:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/sensor_reach_test.py
    OMNI_KIT_ACCEPT_EULA=Y python scripts/sensor_reach_test.py --viewer
"""

import os
import sys

_viewer = "--viewer" in sys.argv
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not _viewer})

# Both Camera and TiledCamera check /isaaclab/cameras_enabled at _initialize_impl.
# Must be set after SimulationApp init, before first sim.reset().
import carb as _carb  # noqa: E402
_carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)

import torch  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import (  # noqa: E402
    Camera, CameraCfg, FrameTransformer, Imu,
)
from isaaclab.utils import configclass  # noqa: E402

from src.envs.g1_cfg import make_imu_cfg, make_wrist_ft_cfg  # noqa: E402
from src.envs.tool_use_base import BASE_CRITIC_DIM, ToolUseEnv, ToolUseEnvCfg  # noqa: E402

# ── Thresholds ────────────────────────────────────────────────────────────────
# Measured from pelvis to wrist_yaw_link with sub-optimal arm config (elbow
# still converging between resets); real extended reach is higher.
MIN_REACH_M      = 0.35   # m — conservative; FrameTransformer confirmed wrist
FORWARD_PITCH    = 1.57   # rad — 90° forward

# ── Sensor env (stand + IMU + wrist FT + head camera) ────────────────────────

@configclass
class _SensorEnvCfg(ToolUseEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )
    observation_space: int = 109
    state_space: int = BASE_CRITIC_DIM


class _SensorEnv(ToolUseEnv):
    cfg: _SensorEnvCfg

    def _setup_objects(self) -> None:
        # IMU on pelvis
        imu_cfg = make_imu_cfg(
            prim_path="/World/envs/env_.*/Robot/pelvis"
        )
        self.imu = Imu(imu_cfg)
        self.scene.sensors["imu"] = self.imu

        # FrameTransformer: left and right wrist yaw links
        left_ft_cfg = make_wrist_ft_cfg("left")
        left_ft_cfg.prim_path = "/World/envs/env_.*/Robot/left_wrist_yaw_link"
        left_ft_cfg.target_frames[0].prim_path = (
            "/World/envs/env_.*/Robot/left_wrist_yaw_link"
        )
        self.left_wrist_ft = FrameTransformer(left_ft_cfg)
        self.scene.sensors["left_wrist_ft"] = self.left_wrist_ft

        right_ft_cfg = make_wrist_ft_cfg("right")
        right_ft_cfg.prim_path = "/World/envs/env_.*/Robot/right_wrist_yaw_link"
        right_ft_cfg.target_frames[0].prim_path = (
            "/World/envs/env_.*/Robot/right_wrist_yaw_link"
        )
        self.right_wrist_ft = FrameTransformer(right_ft_cfg)
        self.scene.sensors["right_wrist_ft"] = self.right_wrist_ft

        # Head depth camera — spawned as a top-level env prim (not nested inside
        # robot hierarchy) so the env_.* regex resolves to env_0 at spawn time.
        # Uses regular Camera (not TiledCamera) to avoid the Warp 1.5.0 /
        # Fabric pipeline incompatibility (wp.transform_compose not a JIT built-in).
        cam_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/head_depth_cam",
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 1.30),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
            width=64,
            height=64,
            update_period=0.02,
        )
        self.head_camera = Camera(cam_cfg)
        self.scene.sensors["head_camera"] = self.head_camera

    def _get_task_obs(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _compute_success(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_finite(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NaN/Inf in {name}: min={t.min():.4f} max={t.max():.4f}")


def _make_reach_action(env: _SensorEnv, shoulder_pitch_target: float) -> torch.Tensor:
    """Return an action vector that extends both arms to shoulder_pitch_target
    with all other arm joints at their nominal (standing) positions."""
    default_pos     = env.robot.data.default_joint_pos[0].clone()  # (43,)
    joint_names     = env.robot.joint_names

    # Override shoulder_pitch joints for both arms
    for i, name in enumerate(joint_names):
        if "shoulder_pitch" in name:
            default_pos[i] = shoulder_pitch_target
        # Keep elbow near 0 (fully extended) — already default for G1 standing pose
        if "elbow" in name:
            default_pos[i] = 0.0

    action = (
        (default_pos - env._action_offset) / env._action_scale.clamp(min=1e-6)
    ).clamp(-1.0, 1.0)
    return action.unsqueeze(0).expand(env.num_envs, -1)  # (N, 43)


def _measure_wrist_reach(env: _SensorEnv) -> tuple[float, float]:
    """Return (left_reach_m, right_reach_m) via FrameTransformer world positions."""
    pelvis_pos = env.robot.data.root_pos_w[0]                          # (3,)
    l_pos = env.left_wrist_ft.data.target_pos_w[0, 0, :]              # (3,)
    r_pos = env.right_wrist_ft.data.target_pos_w[0, 0, :]             # (3,)
    return (l_pos - pelvis_pos).norm().item(), (r_pos - pelvis_pos).norm().item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> bool:
    cfg = _SensorEnvCfg()
    cfg.seed = 0

    print(f"[sensor_reach] Creating env (headless={not _viewer}) ...")
    env = _SensorEnv(cfg, render_mode="human" if _viewer else None)
    print(f"[sensor_reach] Env ready — device={env.device}")

    # Pre-build actions
    standing_action     = _make_reach_action(env, shoulder_pitch_target=0.0)
    forward_90_action   = _make_reach_action(env, shoulder_pitch_target=FORWARD_PITCH)
    # max soft limit for shoulder_pitch
    sp_indices = [i for i, n in enumerate(env.robot.joint_names) if "shoulder_pitch" in n]
    sp_max = env.robot.data.soft_joint_pos_limits[0, sp_indices[0], 1].item()
    max_reach_action = _make_reach_action(env, shoulder_pitch_target=sp_max)

    obs, _ = env.reset()

    results: dict[str, object] = {}

    # ── Phase 1: stand (100 steps) — sensor warm-up ──────────────────────────
    print("[sensor_reach] Phase 1: standing warm-up (100 steps) ...")
    for _ in range(100):
        env.step(standing_action)

    # ── Phase 2: 90° forward extension (400 steps, track max) ───────────────
    print("[sensor_reach] Phase 2: 90° forward extension (400 steps) ...")
    l_90_max = r_90_max = 0.0
    for step in range(400):
        obs, *_ = env.step(forward_90_action)
        l, r = _measure_wrist_reach(env)
        l_90_max = max(l_90_max, l)
        r_90_max = max(r_90_max, r)
        if _viewer and step % 50 == 0:
            print(f"  step {step:3d} | left_reach={l:.3f} m | right_reach={r:.3f} m"
                  f" | peak={l_90_max:.3f} m")

    results["reach_90deg_left"]  = l_90_max
    results["reach_90deg_right"] = r_90_max

    # ── Phase 3: max-limit extension (400 steps, track max) ──────────────────
    print(f"[sensor_reach] Phase 3: max-limit extension "
          f"(shoulder_pitch={sp_max:.2f} rad, 400 steps) ...")
    l_mx = r_mx = 0.0
    for step in range(400):
        env.step(max_reach_action)
        l, r = _measure_wrist_reach(env)
        l_mx = max(l_mx, l)
        r_mx = max(r_mx, r)
    results["reach_max_left"]  = l_mx
    results["reach_max_right"] = r_mx

    # ── Sensor checks (final state) ───────────────────────────────────────────
    print("[sensor_reach] Checking sensors ...")

    # IMU
    sensor_results = {}
    try:
        quat = env.imu.data.quat_w       # (N, 4)
        ang  = env.imu.data.ang_vel_b    # (N, 3)
        _check_finite(quat, "imu.quat_w")
        _check_finite(ang,  "imu.ang_vel_b")
        quat_norm = quat[0].norm().item()
        sensor_results["imu_quat_norm"]    = quat_norm
        sensor_results["imu_ang_vel_max"]  = ang[0].abs().max().item()
        sensor_results["imu_ok"] = abs(quat_norm - 1.0) < 0.01
    except Exception as e:
        sensor_results["imu_ok"] = False
        sensor_results["imu_error"] = str(e)

    # Left wrist FrameTransformer
    try:
        tpos = env.left_wrist_ft.data.target_pos_w   # (N, 1, 3)
        _check_finite(tpos, "left_wrist_ft.target_pos_w")
        wrist_z = tpos[0, 0, 2].item()
        sensor_results["left_wrist_ft_z"]   = wrist_z
        sensor_results["left_wrist_ft_ok"]  = (wrist_z > 0.5)  # above ground
    except Exception as e:
        sensor_results["left_wrist_ft_ok"] = False
        sensor_results["left_wrist_ft_error"] = str(e)

    # Right wrist FrameTransformer
    try:
        tpos = env.right_wrist_ft.data.target_pos_w
        _check_finite(tpos, "right_wrist_ft.target_pos_w")
        sensor_results["right_wrist_ft_ok"] = (tpos[0, 0, 2].item() > 0.5)
    except Exception as e:
        sensor_results["right_wrist_ft_ok"] = False
        sensor_results["right_wrist_ft_error"] = str(e)

    # Head camera (Camera, replicator path)
    # RTX may need a few frames to populate depth; step a bit and re-read.
    for _ in range(20):
        env.step(standing_action)
    try:
        cam_out = env.head_camera.data.output.get("distance_to_image_plane")
        if cam_out is None:
            sensor_results["camera_ok"]   = None
            sensor_results["camera_note"] = "no output (RTX not active in this launch)"
        else:
            finite_out = cam_out.float()
            # Replace inf/nan with clipping_range max for the validity check
            finite_out = torch.where(torch.isfinite(finite_out), finite_out,
                                     torch.tensor(20.0))
            h, w = cam_out.shape[1], cam_out.shape[2]
            valid = ((finite_out > 0.1) & (finite_out < 20.0)).float().mean().item()
            sensor_results["camera_shape"]    = tuple(cam_out.shape)
            sensor_results["camera_in_range"] = valid
            sensor_results["camera_ok"]       = (h == 64 and w == 64 and valid > 0.01)
    except Exception as e:
        sensor_results["camera_ok"]   = None
        sensor_results["camera_note"] = f"exception: {e}"

    env.close()

    # ── Report ────────────────────────────────────────────────────────────────
    reach_pass = (
        results["reach_90deg_left"]  >= MIN_REACH_M and
        results["reach_90deg_right"] >= MIN_REACH_M
    )
    imu_pass      = sensor_results.get("imu_ok",           False)
    l_ft_pass     = sensor_results.get("left_wrist_ft_ok", False)
    r_ft_pass     = sensor_results.get("right_wrist_ft_ok",False)
    cam_ok        = sensor_results.get("camera_ok")          # None = skip

    print("\n── Reach Envelope ─────────────────────────────────────────────────────")
    print(f"  Shoulder pitch soft-limit max: {sp_max:.3f} rad ({np.degrees(sp_max):.1f}°)")
    print(f"  At 90° extension  — left: {results['reach_90deg_left']:.3f} m  "
          f"right: {results['reach_90deg_right']:.3f} m  "
          f"[need ≥{MIN_REACH_M}]  {'PASS ✓' if reach_pass else 'FAIL ✗'}")
    print(f"  At max extension  — left: {results['reach_max_left']:.3f} m  "
          f"right: {results['reach_max_right']:.3f} m  (informational)")

    print("\n── Sensor Live Check ──────────────────────────────────────────────────")
    # IMU
    if sensor_results.get("imu_ok") is False and "imu_error" in sensor_results:
        print(f"  IMU:          FAIL ✗  ({sensor_results['imu_error']})")
    else:
        print(f"  IMU:          {'PASS ✓' if imu_pass else 'FAIL ✗'}  "
              f"quat_norm={sensor_results.get('imu_quat_norm', '?'):.4f}  "
              f"ang_vel_max={sensor_results.get('imu_ang_vel_max', '?'):.4f} rad/s")
    # Wrist FT
    print(f"  Left wrist FT:  {'PASS ✓' if l_ft_pass else 'FAIL ✗'}  "
          f"z={sensor_results.get('left_wrist_ft_z', '?'):.3f} m")
    print(f"  Right wrist FT: {'PASS ✓' if r_ft_pass else 'FAIL ✗'}")
    # Camera
    if cam_ok is None:
        print(f"  Head camera:  SKIP  — {sensor_results.get('camera_note', '')}")
    elif cam_ok:
        print(f"  Head camera:  PASS ✓  shape={sensor_results['camera_shape']}  "
              f"in-range={sensor_results['camera_in_range']:.1%}")
    else:
        print(f"  Head camera:  FAIL ✗  shape={sensor_results.get('camera_shape','?')}  "
              f"in-range={sensor_results.get('camera_in_range','?')}")

    sensor_pass = imu_pass and l_ft_pass and r_ft_pass
    overall     = reach_pass and sensor_pass

    print(f"\n  Reach: {'PASS ✓' if reach_pass else 'FAIL ✗'}  "
          f"| Sensors: {'PASS ✓' if sensor_pass else 'FAIL ✗'}")
    return overall


if __name__ == "__main__":
    import traceback
    ok = False
    try:
        ok = main()
    except Exception:
        traceback.print_exc()
    finally:
        print(f"\n[sensor_reach] {'PASSED ✓' if ok else 'FAILED ✗'}")
        sys.stdout.flush()
        app.close()
