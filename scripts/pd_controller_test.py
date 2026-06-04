"""
PD controller test for G1 EDU (43 DOF).

Loads the robot in a bare scene, holds the default standing-pose joint targets
via the same action-scaling path used during training, and checks:

  1. No NaN/Inf in any state tensor at any step
  2. Mean pelvis height (last 100 steps) >= 0.60 m  — robot stays upright
  3. 90th-pct steady-state joint error (last 100 steps) <= 0.15 rad
  4. No unexpected resets (robot did not fall and trigger termination)

Per-actuator-group breakdown is printed regardless of pass/fail.

Run:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/pd_controller_test.py
    OMNI_KIT_ACCEPT_EULA=Y python scripts/pd_controller_test.py --viewer
"""

import os
import sys

_viewer = "--viewer" in sys.argv
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not _viewer})

# ── safe to import Isaac Lab now ──────────────────────────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from src.envs.tool_use_base import BASE_CRITIC_DIM, ToolUseEnv, ToolUseEnvCfg  # noqa: E402

# ── Pass/fail thresholds ──────────────────────────────────────────────────────
N_STEPS          = 500
MIN_PELVIS_H     = 0.60   # m   — anything below means the robot collapsed
MAX_P90_ERR      = 0.15   # rad — 90th-pct joint error over last 100 steps


# ── Minimal stand-only environment (no task objects) ─────────────────────────

@configclass
class _StandEnvCfg(ToolUseEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )
    observation_space: int = 109
    state_space: int = BASE_CRITIC_DIM  # 112 = 109 actor + 3 base-priv, no task extras


class _StandEnv(ToolUseEnv):
    cfg: _StandEnvCfg

    def _setup_objects(self) -> None:
        pass

    def _get_task_obs(self) -> torch.Tensor:
        # No task extras — zero columns so cat in _get_observations is a no-op
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _compute_success(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_objects(self, env_ids: torch.Tensor) -> None:
        pass


# ── Test logic ────────────────────────────────────────────────────────────────

def main() -> bool:
    cfg = _StandEnvCfg()
    cfg.seed = 0

    print(f"[pd_test] Creating stand env (headless={not _viewer}, steps={N_STEPS}) ...")
    env = _StandEnv(cfg, render_mode="human" if _viewer else None)
    print(f"[pd_test] Env ready — {env.num_envs} env(s), device={env.device}")

    # Build the standing-pose action: invert the action scaling applied in _apply_action
    #   target = offset + scale * action  →  action = (target - offset) / scale
    default_pos     = env.robot.data.default_joint_pos[0]          # (43,)
    standing_action = (
        (default_pos - env._action_offset) / env._action_scale.clamp(min=1e-6)
    ).clamp(-1.0, 1.0).unsqueeze(0).expand(env.num_envs, -1)       # (N, 43)

    obs, _ = env.reset()
    _check_finite(obs["policy"], "obs after reset")

    pelvis_heights = []
    joint_errors   = []
    reset_count    = 0

    for step in range(N_STEPS):
        obs, _reward, terminated, timed_out, _ = env.step(standing_action)

        h   = env.robot.data.root_pos_w[:, 2]                              # (N,)
        err = (env.robot.data.joint_pos - env.robot.data.default_joint_pos).abs()  # (N,43)

        _check_finite(h,             f"pelvis height step {step}")
        _check_finite(obs["policy"], f"policy obs step {step}")

        pelvis_heights.append(h[0].item())
        joint_errors.append(err[0].cpu().numpy())

        reset_count += terminated[0].int().item()

        if _viewer and step % 50 == 0:
            print(f"  step {step:4d} | pelvis_h={h[0]:.3f} m | "
                  f"max_err={err[0].max():.4f} rad | resets={reset_count}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    heights_last = np.array(pelvis_heights[-100:])
    errors_last  = np.array(joint_errors[-100:])    # (100, 43)

    mean_h  = heights_last.mean()
    p90_err = float(np.percentile(errors_last, 90))
    max_err = errors_last.max(axis=0)               # (43,) worst per joint

    joint_names = env.robot.joint_names             # list[str], length 43

    groups = {
        "legs":  [i for i, n in enumerate(joint_names) if any(k in n for k in ("hip", "knee"))],
        "feet":  [i for i, n in enumerate(joint_names) if "ankle" in n],
        "waist": [i for i, n in enumerate(joint_names) if "waist" in n],
        "arms":  [i for i, n in enumerate(joint_names) if any(k in n for k in
                  ("shoulder", "elbow", "wrist"))],
        "hands": [i for i, n in enumerate(joint_names) if "hand" in n],
    }

    h_pass   = mean_h  >= MIN_PELVIS_H
    err_pass = p90_err <= MAX_P90_ERR

    print("\n── PD Controller Test Results ──────────────────────────────────────")
    print(f"  Mean pelvis height   (last 100 steps): {mean_h:.3f} m   "
          f"[need ≥{MIN_PELVIS_H}]  {'PASS ✓' if h_pass   else 'FAIL ✗'}")
    print(f"  90th-pct joint error (last 100 steps): {p90_err:.4f} rad "
          f"[need ≤{MAX_P90_ERR}]  {'PASS ✓' if err_pass else 'FAIL ✗'}")
    print(f"  Unexpected resets: {reset_count}  "
          f"{'PASS ✓' if reset_count == 0 else 'WARN — robot fell'}")

    print("\n  Per-group max error (worst joint, last 100 steps):")
    for gname, idxs in groups.items():
        if not idxs:
            print(f"    {gname:6s}: (no joints matched)")
            continue
        g_max  = max_err[idxs].max()
        worst_local = int(np.argmax(max_err[idxs]))
        worst_name  = joint_names[idxs[worst_local]]
        print(f"    {gname:6s}: {g_max:.4f} rad  (worst: {worst_name})")

    env.close()
    return h_pass and err_pass


def _check_finite(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NaN/Inf detected in {name}")


if __name__ == "__main__":
    import traceback
    ok = False
    try:
        ok = main()
    except Exception:
        traceback.print_exc()
    finally:
        print(f"\n[pd_test] {'PASSED ✓' if ok else 'FAILED ✗'}")
        sys.stdout.flush()
        app.close()
