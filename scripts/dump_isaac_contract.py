"""
Dump the authoritative Isaac Lab joint/observation contract for sim-to-sim.

Isaac Lab orders the articulation DOF array by its own parse, which is NOT
the URDF/MJCF tree order. The distilled policy's 43-dim obs/action vectors are
in Isaac's order, so MuJoCo cross-validation must replicate that exact order.

Outputs JSON to logs/isaac_contract_<task>.json:
  joint_names            (43)      — Isaac DOF order (authoritative)
  soft_joint_pos_limits  (43, 2)   — [lower, upper]; drives action scaling
  default_joint_pos      (43)      — reset pose
  default_root_state     (13)      — pos(3)+quat(4,wxyz)+linvel(3)+angvel(3)

Usage:
  /home/kevin/isaacsim/python.sh scripts/dump_isaac_contract.py --task distant_target
"""

import argparse
import json
import os

argp = argparse.ArgumentParser()
argp.add_argument("--task", default="distant_target")
args = argp.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import torch  # noqa: E402
import yaml    # noqa: E402
import sys     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import make_env  # noqa: E402


def main():
    cfg = {
        "task": args.task,
        "seed": 42,
        "env": {"num_envs": 1, "episode_length": 1000, "device": "cuda"},
    }
    env = make_env(cfg)
    env.reset()

    robot = env.robot
    joint_names = list(robot.data.joint_names)
    soft_limits = robot.data.soft_joint_pos_limits[0].cpu().tolist()   # (43, 2)
    default_qpos = robot.data.default_joint_pos[0].cpu().tolist()      # (43,)
    default_root = robot.data.default_root_state[0].cpu().tolist()     # (13,)

    contract = {
        "task": args.task,
        "num_dof": len(joint_names),
        "joint_names": joint_names,
        "soft_joint_pos_limits": soft_limits,
        "default_joint_pos": default_qpos,
        "default_root_state": default_root,
    }

    out = f"logs/isaac_contract_{args.task}.json"
    os.makedirs("logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(contract, f, indent=2)

    print(f"[dump] {len(joint_names)} DOF written to {out}")
    print("[dump] Isaac joint order:")
    for i, n in enumerate(joint_names):
        lo, hi = soft_limits[i]
        print(f"  {i:2d}  {n:32s}  [{lo:+.3f}, {hi:+.3f}]  q0={default_qpos[i]:+.3f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
