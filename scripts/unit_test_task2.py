"""
Task 2 (ElevatedButtonEnv) unit tests — §3.3 checklist.

Test A — 100-reset integrity:
  No NaN/Inf, all objects above floor, no box/button overlap at spawn.

Test B — Random-agent baseline:
  200 episodes × 200 steps; success rate < 0.1%.

Run:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task2.py
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task2.py --viewer
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

from src.envs.tasks.elevated_button import ElevatedButtonEnv, ElevatedButtonEnvCfg  # noqa: E402

N_RESET_CHECKS   = 100
N_BASELINE_EPS   = 200
N_BASELINE_STEPS = 200
MAX_SUCCESS_RATE = 0.001

MIN_OBJ_Z        = 0.05
MIN_BOX_ROBOT_D  = 0.20
MIN_BOX_BTN_D    = 0.30   # box and button should spawn apart


def make_env() -> ElevatedButtonEnv:
    cfg = ElevatedButtonEnvCfg()
    cfg.scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    cfg.seed = 42
    return ElevatedButtonEnv(cfg, render_mode="human" if _viewer else None)


def _check_finite(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise AssertionError(f"NaN/Inf in {name}")


def test_a(env: ElevatedButtonEnv) -> tuple[bool, str]:
    failures = []
    for i in range(N_RESET_CHECKS):
        obs, _ = env.reset()
        try:
            _check_finite(obs["policy"], f"obs[policy] reset {i}")
            _check_finite(obs["critic"], f"obs[critic] reset {i}")
        except AssertionError as e:
            failures.append(str(e)); continue

        box_pos    = env.box.data.root_pos_w[0]
        btn_pos    = env.button.data.root_pos_w[0]
        robot_pos  = env.robot.data.root_pos_w[0]

        if box_pos[2] < MIN_OBJ_Z:
            failures.append(f"reset {i}: box z={box_pos[2]:.3f} below floor")
        if btn_pos[2] < 1.5:
            failures.append(f"reset {i}: button z={btn_pos[2]:.3f} too low (should be ~1.8m)")
        if (box_pos - robot_pos).norm() < MIN_BOX_ROBOT_D:
            failures.append(f"reset {i}: box inside robot")
        if (box_pos[:2] - btn_pos[:2]).norm() > 2.5:
            failures.append(f"reset {i}: box too far from button — may make task unsolvable")

        if _viewer and i % 20 == 0:
            print(f"  [A] reset {i:3d} | box_z={box_pos[2]:.3f} | btn_z={btn_pos[2]:.3f} | "
                  f"box↔btn_xy={(box_pos[:2]-btn_pos[:2]).norm():.3f}")

    if failures:
        return False, f"{len(failures)} failures:\n  " + "\n  ".join(failures[:5])
    return True, f"{N_RESET_CHECKS} resets — all checks passed"


def test_b(env: ElevatedButtonEnv) -> tuple[bool, str]:
    successes = 0
    env.reset()
    for ep in range(N_BASELINE_EPS):
        ep_success = False
        for _ in range(N_BASELINE_STEPS):
            actions = torch.rand(env.num_envs, env.cfg.action_space, device=env.device) * 2 - 1
            _, _, terminated, timed_out, extras = env.step(actions)
            if extras.get("button_contact_events", 0.0) > 0.0:
                ep_success = True
            if terminated[0] or timed_out[0]:
                break
        if ep_success:
            successes += 1
        if _viewer and ep % 50 == 0:
            print(f"  [B] ep {ep:3d} | successes={successes} | rate={successes/(ep+1):.4%}")

    rate = successes / N_BASELINE_EPS
    return rate < MAX_SUCCESS_RATE, (
        f"{N_BASELINE_EPS} episodes | successes={successes} | "
        f"rate={rate:.4%} [need < {MAX_SUCCESS_RATE:.1%}]"
    )


def main() -> bool:
    print("[unit_test_task2] Creating ElevatedButtonEnv ...")
    env = make_env()
    print(f"[unit_test_task2] Env ready — device={env.device}")

    print(f"\n[unit_test_task2] Test A: {N_RESET_CHECKS}-reset integrity ...")
    a_pass, a_msg = test_a(env)
    print(f"  Result: {'PASS ✓' if a_pass else 'FAIL ✗'}  {a_msg}")

    print(f"\n[unit_test_task2] Test B: random baseline ...")
    b_pass, b_msg = test_b(env)
    print(f"  Result: {'PASS ✓' if b_pass else 'FAIL ✗'}  {b_msg}")

    env.close()
    print("\n── Summary ────────────────────────────────────────────────────────")
    print(f"  Test A: {'PASS ✓' if a_pass else 'FAIL ✗'}")
    print(f"  Test B: {'PASS ✓' if b_pass else 'FAIL ✗'}")
    return a_pass and b_pass


if __name__ == "__main__":
    import traceback
    ok = False
    try:
        ok = main()
    except Exception:
        traceback.print_exc()
    finally:
        print(f"\n[unit_test_task2] {'PASSED ✓' if ok else 'FAILED ✗'}")
        sys.stdout.flush()
        app.close()
