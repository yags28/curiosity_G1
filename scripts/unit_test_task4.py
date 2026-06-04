"""
Task 4 (WeightLeverEnv) unit tests — §3.3 checklist.

Test A — 100-reset integrity:
  No NaN/Inf, all objects above floor, heavy object not inside robot,
  plank and fulcrum within expected bounds.

Test B — Random-agent baseline:
  200 episodes × 200 steps; success rate < 0.1%.

Run:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task4.py
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task4.py --viewer
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

from src.envs.tasks.weight_lever import (  # noqa: E402
    HEAVY_NOMINAL_MASS,
    SUCCESS_LIFT,
    WeightLeverEnv,
    WeightLeverEnvCfg,
)

N_RESET_CHECKS   = 100
N_BASELINE_EPS   = 200
N_BASELINE_STEPS = 200
MAX_SUCCESS_RATE = 0.001

MIN_OBJ_Z        = 0.02   # objects must start above floor
MIN_HEAVY_ROBOT_D = 0.30  # heavy box must not start inside robot


def make_env() -> WeightLeverEnv:
    cfg = WeightLeverEnvCfg()
    cfg.scene = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)
    cfg.seed = 42
    return WeightLeverEnv(cfg, render_mode="human" if _viewer else None)


def _check_finite(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise AssertionError(f"NaN/Inf in {name}")


def test_a(env: WeightLeverEnv) -> tuple[bool, str]:
    failures = []
    for i in range(N_RESET_CHECKS):
        obs, _ = env.reset()
        try:
            _check_finite(obs["policy"], f"obs[policy] reset {i}")
            _check_finite(obs["critic"], f"obs[critic] reset {i}")
        except AssertionError as e:
            failures.append(str(e))
            continue

        plank_pos   = env.plank.data.root_pos_w[0]
        heavy_pos   = env.heavy.data.root_pos_w[0]
        fulcrum_pos = env.fulcrum.data.root_pos_w[0]
        robot_pos   = env.robot.data.root_pos_w[0]

        for name, pos in [("plank", plank_pos), ("heavy", heavy_pos), ("fulcrum", fulcrum_pos)]:
            if pos[2] < MIN_OBJ_Z:
                failures.append(f"reset {i}: {name} z={pos[2]:.3f} below floor")

        if (heavy_pos - robot_pos).norm() < MIN_HEAVY_ROBOT_D:
            failures.append(f"reset {i}: heavy box spawned inside robot")

        # Heavy object mass should be near nominal (±30%)
        heavy_mass = env._heavy_mass[0].item()
        if not (HEAVY_NOMINAL_MASS * 0.65 <= heavy_mass <= HEAVY_NOMINAL_MASS * 1.35):
            failures.append(f"reset {i}: heavy mass={heavy_mass:.2f} outside ±30% of {HEAVY_NOMINAL_MASS}")

        if _viewer and i % 20 == 0:
            lift_needed = env._heavy_spawn_z[0].item() + SUCCESS_LIFT
            print(
                f"  [A] reset {i:3d} | heavy_z={heavy_pos[2]:.3f} "
                f"| plank_z={plank_pos[2]:.3f} | fulcrum_z={fulcrum_pos[2]:.3f} "
                f"| need_z={lift_needed:.3f}"
            )

    if failures:
        return False, f"{len(failures)} failures:\n  " + "\n  ".join(failures[:5])
    return True, f"{N_RESET_CHECKS} resets — all checks passed"


def test_b(env: WeightLeverEnv) -> tuple[bool, str]:
    successes = 0
    env.reset()
    for ep in range(N_BASELINE_EPS):
        for _ in range(N_BASELINE_STEPS):
            actions = torch.rand(env.num_envs, env.cfg.action_space, device=env.device) * 2 - 1
            _, _, terminated, timed_out, _ = env.step(actions)
            if terminated[0] or timed_out[0]:
                break
        if env._success_buf[0]:
            successes += 1
        if _viewer and ep % 50 == 0:
            print(f"  [B] ep {ep:3d} | successes={successes} | rate={successes/(ep+1):.4%}")

    rate = successes / N_BASELINE_EPS
    return rate < MAX_SUCCESS_RATE, (
        f"{N_BASELINE_EPS} episodes | successes={successes} | "
        f"rate={rate:.4%} [need < {MAX_SUCCESS_RATE:.1%}]"
    )


def main() -> bool:
    print("[unit_test_task4] Creating WeightLeverEnv ...")
    env = make_env()
    print(f"[unit_test_task4] Env ready — device={env.device}")

    print(f"\n[unit_test_task4] Test A: {N_RESET_CHECKS}-reset integrity ...")
    a_pass, a_msg = test_a(env)
    print(f"  Result: {'PASS ✓' if a_pass else 'FAIL ✗'}  {a_msg}")

    print(f"\n[unit_test_task4] Test B: random baseline ...")
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
        print(f"\n[unit_test_task4] {'PASSED ✓' if ok else 'FAILED ✗'}")
        sys.stdout.flush()
        app.close()
