"""
Ground-truth check: run the distilled student IN ISAAC and record obs/actions.

Verifies the cross-val claims at the source:
  • Does the student actually succeed in Isaac?
  • Are the action means really bang-bang (saturated)?
  • What does the policy obs look like step-by-step (to compare against the
    MuJoCo reconstruction)?

Dumps logs/isaac_rollout_distant_target.json with the first episode's
obs[0], action[0], and per-step pelvis/success trace.

Usage:
  /home/kevin/isaacsim/python.sh scripts/dump_isaac_rollout.py
"""

import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import make_env                  # noqa: E402
from src.distill.dagger import StudentPolicy   # noqa: E402

_CKPT = "checkpoints/dagger__distant_target__seed42/iter_020.pt"


def main():
    cfg = {
        "task": "distant_target",
        "seed": 42,
        "env": {"num_envs": 1, "episode_length": 1000, "device": "cuda"},
    }
    env = make_env(cfg)
    dev = env.device

    student = StudentPolicy(109, 43).to(dev)
    student.load_state_dict(torch.load(_CKPT, map_location=dev)["student"])
    student.eval()

    n_episodes = 10
    successes = 0
    first_trace = None

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        pol = obs_dict["policy"]
        trace = {"obs0": None, "act0": None, "steps": []}
        succeeded = False
        for t in range(300):
            with torch.no_grad():
                action = student(pol)
            if t == 0:
                trace["obs0"] = pol[0].cpu().numpy().tolist()
                trace["act0"] = action[0].cpu().numpy().tolist()
            obs_dict, rew, term, timo, extras = env.step(action.clamp(-1, 1))
            pol = obs_dict["policy"]
            hit = extras.get("tool_contact_events", 0.0)
            pelvis = float(env.robot.data.root_pos_w[0, 2].cpu())
            trace["steps"].append({"t": t, "pelvis": round(pelvis, 3),
                                   "hit": float(hit)})
            if hit > 0:
                succeeded = True
            if bool(term[0]) or bool(timo[0]):
                break
        successes += int(succeeded)
        if first_trace is None:
            first_trace = trace
        print(f"[roll] ep {ep+1}/{n_episodes}: success={succeeded} steps={t+1}")

    a0 = np.array(first_trace["act0"])
    print(f"\n[roll] Isaac success {successes}/{n_episodes}")
    print(f"[roll] action[0] min/max/mean = {a0.min():.2f}/{a0.max():.2f}/{a0.mean():.2f}"
          f"  saturated(|a|>0.99)={int((np.abs(a0) > 0.99).sum())}/43")
    print(f"[roll] ep0 length = {len(first_trace['steps'])} steps")

    out = "logs/isaac_rollout_distant_target.json"
    with open(out, "w") as f:
        json.dump(first_trace, f)
    print(f"[roll] trace → {out}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
