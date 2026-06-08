"""
DAgger distillation launcher for CDL.

Usage:
  python distill.py --config configs/distill_local.yaml
  python distill.py --config configs/distill_local.yaml --viewer
  python distill.py --config configs/distill_local.yaml --teacher checkpoints/distant_target__drnd__seed42/step_9900288.pt

IMPORTANT: SimulationApp must be created before any Isaac Lab imports.
"""

import argparse
import os
import sys

import yaml

# ── Parse args ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CDL DAgger distillation launcher")
parser.add_argument("--config",   type=str, required=True)
parser.add_argument("--teacher",  type=str, default=None, help="Override teacher checkpoint path")
parser.add_argument("--viewer",   action="store_true",    help="Open Isaac Lab GUI")
parser.add_argument("--seed",     type=int, default=None)
parser.add_argument("--num-envs", type=int, default=None)
args = parser.parse_args()

# ── Load config ───────────────────────────────────────────────────────────────

with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.seed     is not None: cfg["seed"]            = args.seed
if args.num_envs is not None: cfg["env"]["num_envs"] = args.num_envs
if args.teacher  is not None: cfg["teacher_checkpoint"] = args.teacher

# ── Launch Isaac Sim ──────────────────────────────────────────────────────────

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
from isaacsim import SimulationApp  # noqa: E402

headless = not args.viewer and cfg.get("env", {}).get("headless", True)
app = SimulationApp({"headless": headless})

# ── All Isaac Lab / src imports AFTER SimulationApp ───────────────────────────

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs import make_env                   # noqa: E402
from src.distill.dagger import DAggerDistiller  # noqa: E402


def main():
    task   = cfg["task"]
    seed   = cfg["seed"]
    device = cfg["env"].get("device", "cuda")

    torch.manual_seed(seed)

    teacher_ckpt = cfg["teacher_checkpoint"]
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

    print(f"[distill] task={task} | seed={seed} | teacher={teacher_ckpt}")
    print("[distill] creating environment ...")
    env = make_env(cfg)
    print(
        f"[distill] env ready — obs={env.cfg.observation_space} | "
        f"act={env.cfg.action_space} | num_envs={env.num_envs}"
    )

    distiller = DAggerDistiller(
        cfg          = cfg,
        teacher_ckpt = teacher_ckpt,
        obs_dim      = env.cfg.observation_space,
        action_dim   = env.cfg.action_space,
        device       = device,
    )
    distiller.run(env)

    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        app.close()
