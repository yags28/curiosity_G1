"""
Experiment launcher for CDL-HumanoidToolUse.

Usage:
  python launch.py --config configs/local.yaml
  python launch.py --config configs/local.yaml --task distant_target --curiosity rnd --seed 1
  python launch.py --config configs/local.yaml --viewer   # open Isaac Lab GUI
  python launch.py --config configs/local.yaml --no-wandb  # skip wandb for quick tests

  Hyperparameter sweep (after `wandb sweep configs/sweep_rnd.yaml`):
    wandb agent ydawanka-purdue-university/CDL-HumanoidToolUse/<SWEEP_ID>

IMPORTANT: SimulationApp must be created before any Isaac Lab imports.
All src.envs / src.agents imports happen AFTER app creation below.
"""

import argparse
import os
import sys

import yaml

# ── Parse args (before SimulationApp — needed for headless flag) ──────────────

parser = argparse.ArgumentParser(description="CDL experiment launcher")
parser.add_argument("--config",       type=str, required=True)
parser.add_argument("--task",         type=str, default=None,
                    choices=["distant_target", "elevated_button",
                             "occluded_retrieval", "weight_lever", "composite"])
parser.add_argument("--curiosity",    type=str, default=None,
                    choices=["rnd", "drnd", "rdd", "icm", "qflex"])
parser.add_argument("--seed",         type=int, default=None)
parser.add_argument("--num-envs",     type=int, default=None)
parser.add_argument("--total-steps",  type=int, default=None)
parser.add_argument("--viewer",       action="store_true", help="Open Isaac Lab GUI")
parser.add_argument("--no-wandb",     action="store_true", help="Disable wandb logging")
parser.add_argument("--resume",       type=str, default=None, help="Checkpoint path to resume")
parser.add_argument("--run-name",     type=str, default=None)
parser.add_argument("--tags",         type=str, nargs="*", default=[])
args = parser.parse_args()

# ── Load and patch config ─────────────────────────────────────────────────────

with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.seed        is not None: cfg["seed"]                        = args.seed
if args.task        is not None: cfg["task"]                        = args.task
if args.curiosity   is not None: cfg["curiosity"]["method"]         = args.curiosity
if args.num_envs    is not None: cfg["env"]["num_envs"]             = args.num_envs
if args.total_steps is not None: cfg["training"]["total_timesteps"] = args.total_steps
if args.resume      is not None: cfg["training"]["resume_checkpoint"] = args.resume

# ── Launch Isaac Sim ──────────────────────────────────────────────────────────

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
from isaacsim import SimulationApp  # noqa: E402

headless = not args.viewer and cfg.get("env", {}).get("headless", True)
app = SimulationApp({"headless": headless})

# ── All Isaac Lab / src imports AFTER SimulationApp ───────────────────────────

import torch           # noqa: E402
import wandb           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs import make_env        # noqa: E402
from src.agents.ppo import PPOAgent  # noqa: E402


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    task   = cfg["task"]
    method = cfg["curiosity"]["method"]
    seed   = cfg["seed"]
    device = cfg["env"].get("device", "cuda")

    torch.manual_seed(seed)

    run_name = args.run_name or f"{task}__{method}__seed{seed}"
    tags     = [task, method] + args.tags

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project   = cfg["wandb"]["project"],
            entity    = cfg["wandb"]["entity"],
            name      = run_name,
            config    = cfg,
            tags      = tags,
            save_code = True,
        )
        # Re-read cfg in case a sweep agent injected different values
        cfg.update(dict(wandb.config))

        wandb.define_metric("global_step")
        for metric, summary in {
            "episode_reward":    "max",
            "intrinsic_reward":  "mean",
            "extrinsic_reward":  "max",
            "success_rate":      "max",
            "tool_contact_count":"max",
            "alpha_annealing":   "last",
            "policy_loss":       "last",
            "value_loss":        "last",
            "entropy":           "last",
            "approx_kl":         "last",
            "curiosity_loss":    "last",
            "reward_int_mean":   "mean",
            "reward_int_std":    "last",
        }.items():
            wandb.define_metric(metric, step_metric="global_step", summary=summary)

        print(f"[launch] wandb run : {run.name}")
        print(f"[launch]        url: {run.url}")

    print(
        f"[launch] task={task} | curiosity={method} | seed={seed} | "
        f"num_envs={cfg['env']['num_envs']} | steps={cfg['training']['total_timesteps']:,}"
    )

    print("[launch] creating environment ...")
    env = make_env(cfg)
    print(
        f"[launch] env ready — device={env.device} | "
        f"obs={env.cfg.observation_space} | critic={env.cfg.state_space} | "
        f"act={env.cfg.action_space} | num_envs={env.num_envs}"
    )

    agent = PPOAgent(
        cfg           = cfg,
        obs_dim       = env.cfg.observation_space,
        critic_obs_dim= env.cfg.state_space,
        action_dim    = env.cfg.action_space,
        device        = device,
    )

    resume = cfg["training"].get("resume_checkpoint")
    if resume:
        agent.load(resume)

    agent.train(env, run=run)

    env.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        app.close()
