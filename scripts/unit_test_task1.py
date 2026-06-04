"""
Task 1 (DistantTargetEnv) unit tests — §3.3 / §3.4 checklist.

Test A — 100-reset integrity check:
  • No NaN/Inf in observations or rewards after any reset
  • No object below floor (z > 0.05 m) — proxy for interpenetration
  • Stick not inside robot at spawn (pelvis-to-stick distance > 0.25 m)
  • Stick and target not overlapping at spawn (distance > 0.15 m)

Test B — Random-agent baseline:
  • 200 episodes × 200 steps with uniform-random actions
  • Success rate must be < 0.1% (random agent shouldn't accidentally hit target)
  • Confirms task difficulty is non-trivial

Run:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task1.py
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task1.py --viewer
"""

import os
import sys

_viewer = "--viewer" in sys.argv
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not _viewer})

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from src.envs.tasks.distant_target import DistantTargetEnv, DistantTargetEnvCfg  # noqa: E402

# ── Test parameters ───────────────────────────────────────────────────────────
N_RESET_CHECKS   = 100    # Test A: number of resets to validate
N_BASELINE_EPS   = 200    # Test B: episodes for random-agent baseline
N_BASELINE_STEPS = 200    # Test B: max steps per episode
MAX_SUCCESS_RATE = 0.001  # Test B: < 0.1% success required

# Interpenetration proxy thresholds
MIN_STICK_Z          = 0.05   # m — stick center above floor
MIN_STICK_ROBOT_DIST = 0.25   # m — stick not inside robot torso
MIN_STICK_TARGET_DIST = 0.15  # m — stick and target not overlapping at spawn


# ── Env factory ───────────────────────────────────────────────────────────────

def make_env(num_envs: int = 1) -> DistantTargetEnv:
    cfg = DistantTargetEnvCfg()
    cfg.scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)
    cfg.seed = 42
    return DistantTargetEnv(cfg, render_mode="human" if _viewer else None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_finite(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise AssertionError(f"NaN/Inf in {name}")


def _check_above_floor(z: torch.Tensor, name: str) -> None:
    if (z < MIN_STICK_Z).any():
        bad = z[z < MIN_STICK_Z].tolist()
        raise AssertionError(f"{name} below floor threshold ({MIN_STICK_Z} m): {bad}")


# ── Test A ────────────────────────────────────────────────────────────────────

def test_a_reset_integrity(env: DistantTargetEnv) -> tuple[bool, str]:
    """100 resets — NaN check + interpenetration proxy."""
    failures = []

    for i in range(N_RESET_CHECKS):
        obs, _ = env.reset()

        try:
            _check_finite(obs["policy"], f"obs[policy] reset {i}")
            _check_finite(obs["critic"], f"obs[critic] reset {i}")
        except AssertionError as e:
            failures.append(str(e))
            continue

        stick_pos  = env.stick.data.root_pos_w[:, :3]    # (N, 3)
        target_pos = env.target.data.root_pos_w[:, :3]   # (N, 3)
        robot_pos  = env.robot.data.root_pos_w[:, :3]    # (N, 3)

        # No object below floor
        try:
            _check_above_floor(stick_pos[:, 2],  "stick z")
            _check_above_floor(target_pos[:, 2], "target z")
        except AssertionError as e:
            failures.append(f"reset {i}: {e}")

        # Stick not inside robot torso
        stick_robot_dist = (stick_pos - robot_pos).norm(dim=-1)
        if (stick_robot_dist < MIN_STICK_ROBOT_DIST).any():
            failures.append(
                f"reset {i}: stick too close to robot "
                f"({stick_robot_dist.min().item():.3f} m < {MIN_STICK_ROBOT_DIST})"
            )

        # Stick and target not overlapping
        stick_target_dist = (stick_pos - target_pos).norm(dim=-1)
        if (stick_target_dist < MIN_STICK_TARGET_DIST).any():
            failures.append(
                f"reset {i}: stick overlaps target "
                f"({stick_target_dist.min().item():.3f} m < {MIN_STICK_TARGET_DIST})"
            )

        if _viewer and i % 20 == 0:
            print(f"  [A] reset {i:3d} | stick_z={stick_pos[0,2]:.3f} | "
                  f"stick↔robot={stick_robot_dist[0]:.3f} | "
                  f"stick↔target={stick_target_dist[0]:.3f}")

    if failures:
        return False, f"{len(failures)} failures:\n  " + "\n  ".join(failures[:5])
    return True, f"{N_RESET_CHECKS} resets — all checks passed"


# ── Test B ────────────────────────────────────────────────────────────────────

def test_b_random_baseline(env: DistantTargetEnv) -> tuple[bool, str]:
    """200 episodes of random actions — success rate must be < 0.1%."""
    successes = 0
    total_eps = 0

    obs, _ = env.reset()

    for ep in range(N_BASELINE_EPS):
        ep_success = False
        for step in range(N_BASELINE_STEPS):
            actions = torch.rand(env.num_envs, env.cfg.action_space,
                                 device=env.device) * 2 - 1
            obs, reward, terminated, timed_out, extras = env.step(actions)

            contact = extras.get("tool_contact_events", 0.0)
            if contact > 0.0:
                ep_success = True

            if terminated[0] or timed_out[0]:
                break

        if ep_success:
            successes += 1
        total_eps += 1

        if _viewer and ep % 50 == 0:
            rate = successes / total_eps if total_eps else 0.0
            print(f"  [B] ep {ep:3d}/{N_BASELINE_EPS} | "
                  f"successes={successes} | rate={rate:.4%}")

    rate = successes / total_eps if total_eps else 0.0
    passed = rate < MAX_SUCCESS_RATE
    msg = (f"{total_eps} episodes | successes={successes} | "
           f"rate={rate:.4%} [need < {MAX_SUCCESS_RATE:.1%}]")
    return passed, msg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> bool:
    print("[unit_test_task1] Creating DistantTargetEnv (1 env) ...")
    env = make_env(num_envs=1)
    print(f"[unit_test_task1] Env ready — device={env.device}")

    print(f"\n[unit_test_task1] Test A: {N_RESET_CHECKS}-reset integrity check ...")
    a_pass, a_msg = test_a_reset_integrity(env)
    print(f"  Result: {'PASS ✓' if a_pass else 'FAIL ✗'}  {a_msg}")

    print(f"\n[unit_test_task1] Test B: random-agent baseline "
          f"({N_BASELINE_EPS} eps × {N_BASELINE_STEPS} steps) ...")
    b_pass, b_msg = test_b_random_baseline(env)
    print(f"  Result: {'PASS ✓' if b_pass else 'FAIL ✗'}  {b_msg}")

    env.close()

    print("\n── Summary ────────────────────────────────────────────────────────")
    print(f"  Test A (reset integrity): {'PASS ✓' if a_pass else 'FAIL ✗'}")
    print(f"  Test B (random baseline): {'PASS ✓' if b_pass else 'FAIL ✗'}")

    return a_pass and b_pass


if __name__ == "__main__":
    import traceback
    ok = False
    try:
        ok = main()
    except Exception:
        traceback.print_exc()
    finally:
        print(f"\n[unit_test_task1] {'PASSED ✓' if ok else 'FAILED ✗'}")
        sys.stdout.flush()
        app.close()
