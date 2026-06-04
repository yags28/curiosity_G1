"""
Task 5 (CompositeEnv) unit tests — §3.3 checklist.

Test A — 100-reset integrity:
  No NaN/Inf, all objects above floor, box and button spawn behind wall (X > WALL_X),
  wall pieces at correct X, box mass within ±30% of nominal.

Test B — Random-agent baseline:
  200 episodes × 300 steps; success rate < 0.1%.

Run:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task5.py
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task5.py --viewer
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

from src.envs.tasks.composite import (  # noqa: E402
    BOX_NOMINAL_MASS,
    BUTTON_NOMINAL_POS,
    WALL_X,
    CompositeEnv,
    CompositeEnvCfg,
)

N_RESET_CHECKS   = 100
N_BASELINE_EPS   = 200
N_BASELINE_STEPS = 300   # longer episode horizon
MAX_SUCCESS_RATE = 0.001

MIN_OBJ_Z        = 0.02   # all objects must start above floor


def make_env() -> CompositeEnv:
    cfg = CompositeEnvCfg()
    cfg.scene = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True)
    cfg.seed = 42
    return CompositeEnv(cfg, render_mode="human" if _viewer else None)


def _check_finite(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise AssertionError(f"NaN/Inf in {name}")


def test_a(env: CompositeEnv) -> tuple[bool, str]:
    failures = []
    for i in range(N_RESET_CHECKS):
        obs, _ = env.reset()
        try:
            _check_finite(obs["policy"], f"obs[policy] reset {i}")
            _check_finite(obs["critic"], f"obs[critic] reset {i}")
        except AssertionError as e:
            failures.append(str(e))
            continue

        box_pos   = env.box.data.root_pos_w[0]
        btn_pos   = env.button.data.root_pos_w[0]
        wl_pos    = env.wall_left.data.root_pos_w[0]
        wr_pos    = env.wall_right.data.root_pos_w[0]
        robot_pos = env.robot.data.root_pos_w[0]
        env_orig  = env.scene.env_origins[0]

        # All objects above floor
        for name, pos in [("box", box_pos), ("button", btn_pos),
                          ("wall_left", wl_pos), ("wall_right", wr_pos)]:
            if pos[2] < env_orig[2] + MIN_OBJ_Z:
                failures.append(f"reset {i}: {name} z={pos[2]:.3f} below floor")

        # Box and button spawn behind wall in env-local X
        box_local_x = (box_pos[0] - env_orig[0]).item()
        btn_local_x = (btn_pos[0] - env_orig[0]).item()
        if box_local_x <= WALL_X:
            failures.append(f"reset {i}: box local_x={box_local_x:.3f} not behind wall (>{WALL_X})")
        if btn_local_x <= WALL_X:
            failures.append(f"reset {i}: button local_x={btn_local_x:.3f} not behind wall (>{WALL_X})")

        # Wall pieces at correct X
        wl_local_x = (wl_pos[0] - env_orig[0]).item()
        wr_local_x = (wr_pos[0] - env_orig[0]).item()
        for side, lx in [("wall_left", wl_local_x), ("wall_right", wr_local_x)]:
            if abs(lx - WALL_X) > 0.05:
                failures.append(f"reset {i}: {side} local_x={lx:.3f} != {WALL_X}")

        # Box mass within ±30% of nominal
        box_mass = env._box_mass[0].item()
        if not (BOX_NOMINAL_MASS * 0.65 <= box_mass <= BOX_NOMINAL_MASS * 1.35):
            failures.append(f"reset {i}: box mass={box_mass:.2f} outside ±30% of {BOX_NOMINAL_MASS}")

        # Critic obs dimension check
        expected_critic_dim = 152
        actual_dim = obs["critic"].shape[-1]
        if actual_dim != expected_critic_dim:
            failures.append(f"reset {i}: critic dim={actual_dim} != {expected_critic_dim}")

        if _viewer and i % 20 == 0:
            print(
                f"  [A] reset {i:3d} | box_x={box_local_x:.2f} "
                f"| btn_h={btn_pos[2]:.2f} | box_mass={box_mass:.1f} kg"
            )

    if failures:
        return False, f"{len(failures)} failures:\n  " + "\n  ".join(failures[:5])
    return True, f"{N_RESET_CHECKS} resets — all checks passed"


def test_b(env: CompositeEnv) -> tuple[bool, str]:
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
    print("[unit_test_task5] Creating CompositeEnv ...")
    env = make_env()
    print(f"[unit_test_task5] Env ready — device={env.device}")
    print(f"  actor_obs={env.cfg.observation_space}  critic_obs={env.cfg.state_space}  "
          f"actions={env.cfg.action_space}  episode={env.cfg.episode_length_s}s")

    print(f"\n[unit_test_task5] Test A: {N_RESET_CHECKS}-reset integrity ...")
    a_pass, a_msg = test_a(env)
    print(f"  Result: {'PASS ✓' if a_pass else 'FAIL ✗'}  {a_msg}")

    print(f"\n[unit_test_task5] Test B: random baseline ...")
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
        print(f"\n[unit_test_task5] {'PASSED ✓' if ok else 'FAILED ✗'}")
        sys.stdout.flush()
        app.close()
