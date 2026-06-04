"""
Task 3 (OccludedRetrievalEnv) unit tests — §3.3 checklist.

Test A — 100-reset integrity:
  No NaN/Inf, stick above floor, target behind barrier and within slot height,
  no target/robot overlap.

Test B — Random-agent baseline:
  200 episodes × 200 steps; success rate < 0.1%.

Run:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task3.py
    OMNI_KIT_ACCEPT_EULA=Y python scripts/unit_test_task3.py --viewer
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

from src.envs.tasks.occluded_retrieval import (  # noqa: E402
    BARRIER_X,
    BARRIER_THICKNESS,
    SLOT_Z_BOT,
    SLOT_Z_TOP,
    OccludedRetrievalEnv,
    OccludedRetrievalEnvCfg,
)

N_RESET_CHECKS   = 100
N_BASELINE_EPS   = 200
N_BASELINE_STEPS = 200
MAX_SUCCESS_RATE = 0.001

MIN_OBJ_Z       = 0.05   # objects must be above floor
MIN_TGT_ROBOT_D = 0.20   # target must not start inside robot

# Target must be behind the barrier's back face
MIN_TARGET_X = BARRIER_X + BARRIER_THICKNESS / 2.0 + 0.01


def make_env() -> OccludedRetrievalEnv:
    cfg = OccludedRetrievalEnvCfg()
    cfg.scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    cfg.seed = 42
    return OccludedRetrievalEnv(cfg, render_mode="human" if _viewer else None)


def _check_finite(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise AssertionError(f"NaN/Inf in {name}")


def test_a(env: OccludedRetrievalEnv) -> tuple[bool, str]:
    failures = []
    for i in range(N_RESET_CHECKS):
        obs, _ = env.reset()
        try:
            _check_finite(obs["policy"], f"obs[policy] reset {i}")
            _check_finite(obs["critic"], f"obs[critic] reset {i}")
        except AssertionError as e:
            failures.append(str(e))
            continue

        stick_pos  = env.stick.data.root_pos_w[0]
        target_pos = env.target.data.root_pos_w[0]
        robot_pos  = env.robot.data.root_pos_w[0]

        if stick_pos[2] < MIN_OBJ_Z:
            failures.append(f"reset {i}: stick z={stick_pos[2]:.3f} below floor")

        if target_pos[2] < MIN_OBJ_Z:
            failures.append(f"reset {i}: target z={target_pos[2]:.3f} below floor")

        # Target must be behind the barrier
        if target_pos[0] < MIN_TARGET_X:
            failures.append(
                f"reset {i}: target x={target_pos[0]:.3f} not behind barrier "
                f"(need > {MIN_TARGET_X:.3f})"
            )

        # Target centre must be within slot height (env-local Z ≈ world Z since origins Z~0)
        env_z = env.scene.env_origins[0, 2].item()
        target_z_local = target_pos[2].item() - env_z
        if not (SLOT_Z_BOT - 0.01 <= target_z_local <= SLOT_Z_TOP + 0.01):
            failures.append(
                f"reset {i}: target local z={target_z_local:.3f} outside slot "
                f"[{SLOT_Z_BOT:.2f}, {SLOT_Z_TOP:.2f}]"
            )

        if (target_pos - robot_pos).norm() < MIN_TGT_ROBOT_D:
            failures.append(f"reset {i}: target spawned inside robot")

        if _viewer and i % 20 == 0:
            print(
                f"  [A] reset {i:3d} | stick_z={stick_pos[2]:.3f} | "
                f"target_x={target_pos[0]:.3f} | target_z={target_pos[2]:.3f}"
            )

    if failures:
        return False, f"{len(failures)} failures:\n  " + "\n  ".join(failures[:5])
    return True, f"{N_RESET_CHECKS} resets — all checks passed"


def test_b(env: OccludedRetrievalEnv) -> tuple[bool, str]:
    successes = 0
    env.reset()
    for ep in range(N_BASELINE_EPS):
        ep_success = False
        for _ in range(N_BASELINE_STEPS):
            actions = torch.rand(env.num_envs, env.cfg.action_space, device=env.device) * 2 - 1
            _, _, terminated, timed_out, extras = env.step(actions)
            if extras.get("tool_contact_events", 0.0) > 0.0:
                ep_success = True   # contact logged — diagnostic only, not success gate
            if terminated[0] or timed_out[0]:
                break
        # Check actual success via env success buffer
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
    print("[unit_test_task3] Creating OccludedRetrievalEnv ...")
    env = make_env()
    print(f"[unit_test_task3] Env ready — device={env.device}")

    print(f"\n[unit_test_task3] Test A: {N_RESET_CHECKS}-reset integrity ...")
    a_pass, a_msg = test_a(env)
    print(f"  Result: {'PASS ✓' if a_pass else 'FAIL ✗'}  {a_msg}")

    print(f"\n[unit_test_task3] Test B: random baseline ...")
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
        print(f"\n[unit_test_task3] {'PASSED ✓' if ok else 'FAILED ✗'}")
        sys.stdout.flush()
        app.close()
