"""
Diagnose HOW Task 1 actually succeeds in Isaac: is it a stick grasp, or a
forward lunge where a hand/body hits the target pad directly?

Records, for successful episodes, the positions of stick, both wrists, pelvis,
and target — then reports what is closest to the target at the success step.
This decides what the sim-to-sim fix must reproduce.

Usage:
  /home/kevin/isaacsim/python.sh scripts/diagnose_task1_success.py
"""

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
    cfg = {"task": "distant_target", "seed": 42,
           "env": {"num_envs": 1, "episode_length": 1000, "device": "cuda"}}
    env = make_env(cfg)
    dev = env.device

    ck = torch.load(_CKPT, map_location=dev)
    student = StudentPolicy(109, 43, tanh=ck.get("tanh", False)).to(dev)
    student.load_state_dict(ck["student"])
    student.eval()

    names = list(env.robot.data.body_names)
    lw = names.index("left_wrist_yaw_link")
    rw = names.index("right_wrist_yaw_link")
    print(f"[diag] wrist body idx: left={lw} right={rw}")

    for ep in range(8):
        obs_dict, _ = env.reset()
        pol = obs_dict["policy"]
        hit_step = None
        rows = []
        for t in range(300):
            with torch.no_grad():
                action = student(pol)
            obs_dict, _, term, timo, extras = env.step(action.clamp(-1, 1))
            pol = obs_dict["policy"]
            pel = env.robot.data.root_pos_w[0].cpu().numpy()
            stick = env.stick.data.root_pos_w[0].cpu().numpy()
            tgt = env.target.data.root_pos_w[0].cpu().numpy()
            lwp = env.robot.data.body_pos_w[0, lw].cpu().numpy()
            rwp = env.robot.data.body_pos_w[0, rw].cpu().numpy()
            rows.append((t, pel, stick, tgt, lwp, rwp))
            if extras.get("tool_contact_events", 0.0) > 0 and hit_step is None:
                hit_step = t
            if bool(term[0]) or bool(timo[0]):
                break

        if hit_step is None:
            print(f"[diag] ep{ep}: no success ({len(rows)} steps)")
            continue
        t, pel, stick, tgt, lwp, rwp = rows[hit_step]
        d_stick = np.linalg.norm(stick - tgt)
        d_lw = np.linalg.norm(lwp - tgt)
        d_rw = np.linalg.norm(rwp - tgt)
        d_pel = np.linalg.norm(pel - tgt)
        closest = min([("stick", d_stick), ("L_wrist", d_lw),
                       ("R_wrist", d_rw), ("pelvis", d_pel)], key=lambda x: x[1])
        print(f"[diag] ep{ep}: HIT@t={hit_step} | target={np.round(tgt,2)} | "
              f"dist→target: stick={d_stick:.2f} Lw={d_lw:.2f} Rw={d_rw:.2f} "
              f"pelvis={d_pel:.2f} | CLOSEST={closest[0]}({closest[1]:.2f}) | "
              f"pelvis_xz=({pel[0]:.2f},{pel[2]:.2f}) stick_xy=({stick[0]:.2f},{stick[1]:.2f})")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
