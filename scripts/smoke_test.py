"""
Isaac Lab smoke test for DistantTargetEnv.

Run with:
    OMNI_KIT_ACCEPT_EULA=Y python scripts/smoke_test.py            # headless, 5 steps
    OMNI_KIT_ACCEPT_EULA=Y python scripts/smoke_test.py --viewer   # with GUI, 2000 steps
    OMNI_KIT_ACCEPT_EULA=Y python scripts/smoke_test.py --viewer --steps 500

Pass to run:
  1. Env initialises without errors
  2. obs/state shapes match config
  3. N random-action steps — no NaN/Inf in obs or rewards
  4. tool_contact_events key present in extras
"""

import os
import sys

# ── Parse --viewer / --steps BEFORE SimulationApp (Isaac Sim requirement) ────
_viewer  = "--viewer" in sys.argv
_steps   = 5
for _i, _a in enumerate(sys.argv):
    if _a == "--steps" and _i + 1 < len(sys.argv):
        _steps = int(sys.argv[_i + 1])
if _viewer and _steps == 5:
    _steps = 2000   # default to 2000 steps when GUI is open

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not _viewer})

# ── Now safe to import Isaac Lab & project code ───────────────────────────────
import torch  # noqa: E402

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.tasks.distant_target import DistantTargetEnv, DistantTargetEnvCfg  # noqa: E402


def main():
    cfg = DistantTargetEnvCfg()
    cfg.scene.num_envs = 1       # single env for fast smoke test / viewing
    cfg.seed = 42

    render_mode = "human" if _viewer else None
    print(f"[smoke_test] Creating DistantTargetEnv (headless={not _viewer}, steps={_steps}) ...")
    env = DistantTargetEnv(cfg, render_mode=render_mode)
    print(f"[smoke_test] Env created — num_envs={env.num_envs}, device={env.device}")

    # Check observation / state dims (shape may include batch dim, check last)
    assert env.observation_space.shape[-1] == cfg.observation_space, \
        f"actor obs shape mismatch: {env.observation_space.shape}"
    assert env.state_space.shape[-1] == cfg.state_space, \
        f"critic state shape mismatch: {env.state_space.shape}"
    print(f"[smoke_test] Spaces OK — actor={cfg.observation_space}, critic={cfg.state_space}")

    # Reset
    obs, _ = env.reset()
    _check_no_nan(obs["policy"], "obs[policy] after reset")
    _check_no_nan(obs["critic"], "obs[critic] after reset")
    print("[smoke_test] Reset OK")

    # Run for _steps steps
    for step in range(_steps):
        actions = torch.rand(env.num_envs, env.cfg.action_space, device=env.device) * 2 - 1
        obs, reward, terminated, timed_out, extras = env.step(actions)
        _check_no_nan(obs["policy"], f"obs[policy] step {step}")
        _check_no_nan(reward, f"reward step {step}")
        assert "tool_contact_events" in extras, "tool_contact_events not logged in extras"
        if _viewer and step % 100 == 0:
            print(f"[smoke_test] step {step}/{_steps} | reward={reward.item():.4f}")

    print(f"[smoke_test] {_steps} steps OK — final reward:", reward.tolist())
    print("[smoke_test] tool_contact_events:", extras["tool_contact_events"])

    env.close()
    print("[smoke_test] PASSED ✓")


def _check_no_nan(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise RuntimeError(f"NaN/Inf detected in {name}: {tensor}")


if __name__ == "__main__":
    import traceback
    _ok = False
    try:
        main()
        _ok = True
    except Exception:
        traceback.print_exc()
    finally:
        # app.close() calls sys.exit(0) which hides exceptions; print result first
        print(f"\n[smoke_test] {'PASSED ✓' if _ok else 'FAILED ✗'}")
        import sys; sys.stdout.flush()
        app.close()
