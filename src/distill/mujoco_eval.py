"""
Sim-to-sim cross-validation: run an Isaac-trained policy in MuJoCo.

The distilled student was trained in Isaac Sim (PhysX). This harness replays it
in MuJoCo to test whether it overfit to PhysX-specific dynamics — a proxy for
sim-to-real robustness.

The hard part is faithfully reproducing Isaac's contract in MuJoCo:
  • Joint order differs (Isaac interleaves L/R by kinematic depth; MJCF is
    depth-first per limb) → we permute via joint names against the dumped
    logs/isaac_contract_<task>.json.
  • Action = offset + scale * a, applied as a PD torque law matching the
    per-joint Kp/Kd from src/envs/g1_cfg.py.
  • Observation = 109-dim [joint_pos(43), joint_vel(43), base_quat(4,wxyz),
    base_ang_vel_body(3), foot_contact(4), wrist_ft(12=zeros)] in Isaac order.

Modes:
  --standing-test : hold default pose under PD (no policy/tools). Validates the
                    joint map + gains: pelvis should stay near spawn height.
  (default)       : load student, run N episodes of Task 1, report success.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mujoco

_ROBOT_XML = (
    "deps/unitree_rl_gym/resources/robots/g1_description/"
    "g1_29dof_with_hand_rev_1_0.xml"
)

# Leg-stiffness multiplier. Isaac's implicit leg actuators track targets far
# more stiffly than the same nominal Kp does as explicit MuJoCo torque, so at
# 1× the legs sag and the robot collapses. At ~20× MuJoCo reproduces Isaac's
# gross body motion (rises to ~0.85 m vs Isaac's 0.93 m). Set for fidelity to
# the source dynamics so future (robust) policies get a fair test.
_LEG_KP_SCALE = 20.0

# Per-joint PD gains, mirrored from src/envs/g1_cfg.py actuator groups.
# (stiffness, damping) keyed by a substring matched against the joint name.
_GAIN_RULES: list[tuple[str, float, float]] = [
    ("hip_yaw",        100.0, 2.5),
    ("hip_roll",       100.0, 2.5),
    ("hip_pitch",      100.0, 2.5),
    ("knee",           200.0, 5.0),
    ("ankle_pitch",     80.0, 2.0),
    ("ankle_roll",      40.0, 1.0),
    ("waist_yaw",     5000.0, 5.0),
    ("waist_roll",    5000.0, 5.0),
    ("waist_pitch",   5000.0, 5.0),
    ("shoulder",      3000.0, 10.0),
    ("elbow",         3000.0, 10.0),
    ("wrist",         3000.0, 10.0),
    ("hand",            20.0, 2.0),
]

# Effort (torque) limits by substring, from g1_cfg effort_limit.
_EFFORT_RULES: list[tuple[str, float]] = [
    ("hip_yaw",    88.0), ("hip_roll", 88.0), ("hip_pitch", 88.0),
    ("knee",      139.0), ("ankle",    50.0),
    ("waist_yaw",  88.0), ("waist_roll", 50.0), ("waist_pitch", 50.0),
    ("shoulder",  300.0), ("elbow",   300.0), ("wrist",      300.0),
    ("hand",        5.0),
]


def _match(name: str, rules):
    for key, *vals in rules:
        if key in name:
            return vals if len(vals) > 1 else vals[0]
    raise KeyError(f"no rule for joint {name!r}")


class IsaacContract:
    """Authoritative Isaac DOF order + limits + reset pose (from the JSON dump)."""

    def __init__(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.joint_names = d["joint_names"]                       # (43,) Isaac order
        self.n = d["num_dof"]
        lim = np.array(d["soft_joint_pos_limits"])                # (43, 2)
        self.lower, self.upper = lim[:, 0], lim[:, 1]
        self.offset = 0.5 * (self.upper + self.lower)             # action → target
        self.scale  = 0.5 * (self.upper - self.lower)
        self.default_qpos = np.array(d["default_joint_pos"])      # (43,)
        self.default_root = np.array(d["default_root_state"])     # (13,) pos,quat(wxyz),lin,ang
        self.kp = np.array([_match(n, _GAIN_RULES)[0] for n in self.joint_names])
        self.kd = np.array([_match(n, _GAIN_RULES)[1] for n in self.joint_names])
        self.tau_max = np.array([_match(n, _EFFORT_RULES) for n in self.joint_names])


# Task 1 object constants (mirror src/envs/tasks/distant_target.py).
_STICK_POS  = (0.5, 0.4, 0.5)
_TARGET_POS = (1.2, 0.0, 0.8)
_POS_RANGE  = 0.30
_FORCE_THRESHOLD = 0.5


def _build_task1_model(timestep: float) -> mujoco.MjModel:
    """Robot XML + Task 1 tools (stick capsule + kinematic target pad)."""
    spec = mujoco.MjSpec.from_file(_ROBOT_XML)
    world = spec.worldbody

    # Stick: 0.9 m capsule, horizontal (90° about y), free joint, 0.3 kg.
    stick = world.add_body(name="cdl_stick", pos=list(_STICK_POS),
                           quat=[0.7071, 0.0, 0.7071, 0.0])
    stick.add_freejoint()
    sg = stick.add_geom()
    sg.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    sg.size = [0.02, 0.45, 0.0]
    sg.mass = 0.3
    sg.friction = [0.7, 0.005, 0.0001]   # Isaac stick μ≈0.6–0.8

    # Target pad: 20×20×2 cm box, fixed (no joint = kinematic).
    tgt = world.add_body(name="cdl_target", pos=list(_TARGET_POS))
    tg = tgt.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = [0.10, 0.10, 0.01]
    tg.friction = [1.0, 0.005, 0.0001]   # Isaac target μ=1.0

    m = spec.compile()
    m.opt.timestep = timestep
    return m


class MujocoG1:
    """MuJoCo G1 wrapper exposing obs/action in Isaac's joint order."""

    def __init__(self, contract: IsaacContract, timestep: float = 0.005,
                 with_tools: bool = False):
        self.c = contract
        if with_tools:
            self.model = _build_task1_model(timestep)
        else:
            self.model = mujoco.MjModel.from_xml_path(_ROBOT_XML)
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        m = self.model
        # Implicit integration matches Isaac's ImplicitActuatorCfg (waist/arms/
        # hands) and stabilises stiff PD on tiny-inertia finger joints.
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # Map each Isaac joint → MuJoCo qpos addr, dof (qvel) addr, actuator id.
        self.qpos_adr = np.zeros(self.c.n, dtype=int)
        self.dof_adr  = np.zeros(self.c.n, dtype=int)
        self.act_id   = np.zeros(self.c.n, dtype=int)
        for i, name in enumerate(self.c.joint_names):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint {name} not in MJCF"
            self.qpos_adr[i] = m.jnt_qposadr[jid]
            self.dof_adr[i]  = m.jnt_dofadr[jid]
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"actuator {name} not in MJCF"
            self.act_id[i] = aid
            # Match Isaac armature (legs/knee 0.03, all others 0.001) — damps
            # high-frequency oscillation on light joints.
            arm = 0.03 if ("hip" in name or "knee" in name) else 0.001
            m.dof_armature[self.dof_adr[i]] = arm

            # All joints → position servos (force = Kp*(ctrl-q) - Kd*qd applied
            # implicitly under implicitfast). This matches Isaac's tight target
            # tracking. Isaac's nominal leg Kp (≤200) statically equilibrates to
            # a deep squat under explicit MuJoCo torque; Isaac avoids this via
            # implicit integration + ~0.24 s episodes. We stiffen the legs by
            # _LEG_KP_SCALE so MuJoCo tracks the policy's target sequence as
            # tightly as Isaac, isolating policy transfer from gain mismatch.
            is_leg = any(k in name for k in ("hip", "knee", "ankle"))
            kp = self.c.kp[i] * (_LEG_KP_SCALE if is_leg else 1.0)
            m.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[aid, :3] = [kp, 0.0, 0.0]
            m.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_biasprm[aid, :3] = [0.0, -kp, 0.0]
            m.dof_damping[self.dof_adr[i]] = self.c.kd[i]   # implicit damping

        # Free-joint root address (floating_base_joint is joint 0, qpos[0:7]).
        self.root_qadr = m.jnt_qposadr[0]
        self.root_vadr = m.jnt_dofadr[0]

        # Ankle bodies for foot contact, in Isaac find_bodies order
        # (interleaved L/R by depth: pitch pair then roll pair).
        self.foot_bodies = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_ankle_pitch_link", "right_ankle_pitch_link",
                      "left_ankle_roll_link",  "right_ankle_roll_link")
        ]

        # Tool addresses (only present when built with tools).
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cdl_stick")
        self.stick_qadr = int(m.jnt_qposadr[m.body_jntadr[sid]]) if sid >= 0 else -1
        self.target_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cdl_target")

    # ── reset ──────────────────────────────────────────────────────────────
    def reset(self, rng: np.random.Generator | None = None):
        mujoco.mj_resetData(self.model, self.data)
        d, c = self.data, self.c
        d.qpos[self.root_qadr : self.root_qadr + 3] = c.default_root[0:3]
        d.qpos[self.root_qadr + 3 : self.root_qadr + 7] = c.default_root[3:7]  # wxyz
        for i in range(c.n):
            d.qpos[self.qpos_adr[i]] = c.default_qpos[i]
        if self.stick_qadr >= 0:
            self._place_stick(rng)
        mujoco.mj_forward(self.model, self.data)

    def _place_stick(self, rng):
        """Stick at nominal XY ±30% with random yaw, horizontal (matches Isaac)."""
        rng = rng or np.random.default_rng()
        pos = np.array(_STICK_POS, dtype=float)
        pos[:2] += (rng.random(2) * 2 - 1) * _POS_RANGE * np.abs(pos[:2])
        yaw = rng.random() * 2 * np.pi
        q_y90 = np.array([0.7071, 0.0, 0.7071, 0.0])
        q_z   = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, q_z, q_y90)
        a = self.stick_qadr
        self.data.qpos[a : a + 3] = pos
        self.data.qpos[a + 3 : a + 7] = quat

    def settle(self, steps: int = 10):
        """Absorb the startup ground-contact transient before measuring."""
        for _ in range(steps):
            self.hold_default()
            self.step()

    # ── observation (109-dim, Isaac order) ───────────────────────────────────
    def joint_pos(self):
        return self.data.qpos[self.qpos_adr]

    def joint_vel(self):
        return self.data.qvel[self.dof_adr]

    def base_quat(self):  # wxyz
        return self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7].copy()

    def base_ang_vel_body(self):
        # Free-joint qvel angular part (qvel[3:6]) is in the body local frame,
        # matching Isaac's root_ang_vel_b.
        return self.data.qvel[self.root_vadr + 3 : self.root_vadr + 6].copy()

    def foot_contact(self):
        # Isaac reports contact on BOTH ankle bodies (pitch+roll) of a grounded
        # foot, but the MJCF foot collision geom lives only on ankle_roll. So we
        # compute a per-leg contact flag (either ankle body of that leg touching)
        # and assign it to both the pitch and roll obs slots.
        # foot_bodies order = [L_pitch, R_pitch, L_roll, R_roll].
        leg = {0: False, 1: False}  # 0=left, 1=right
        f = np.zeros(6)
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            bodies = (self.model.geom_bodyid[con.geom1],
                      self.model.geom_bodyid[con.geom2])
            for side, (bp, br) in enumerate(((self.foot_bodies[0], self.foot_bodies[2]),
                                             (self.foot_bodies[1], self.foot_bodies[3]))):
                if bp in bodies or br in bodies:
                    mujoco.mj_contactForce(self.model, self.data, ci, f)
                    if np.linalg.norm(f[:3]) > 1.0:
                        leg[side] = True
        return np.array([leg[0], leg[1], leg[0], leg[1]], dtype=float)

    def observation(self, closed_loop: bool = False):
        parts = [self.joint_pos(), self.joint_vel(), self.base_quat(),
                 self.base_ang_vel_body(), self.foot_contact(), np.zeros(12)]
        if closed_loop:
            # Match env._get_actor_task_obs: stick & target pos relative to
            # pelvis, world frame (109 → 115).
            parts += [self.stick_pos_rel(), self.target_pos_rel()]
        return np.concatenate(parts).astype(np.float32)

    def pelvis_pos(self):
        return self.data.qpos[self.root_qadr : self.root_qadr + 3].copy()

    def stick_pos_rel(self):
        return self.data.qpos[self.stick_qadr : self.stick_qadr + 3] - self.pelvis_pos()

    def target_pos_rel(self):
        return self.data.xpos[self.target_bid] - self.pelvis_pos()

    def pelvis_height(self):
        return float(self.data.qpos[self.root_qadr + 2])

    # ── action → ctrl (all position servos, Isaac order) ─────────────────────
    # ctrl = target joint angle; solver applies Kp*(ctrl-q) - Kd*qd implicitly.
    def apply_action(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)
        self.data.ctrl[self.act_id] = self.c.offset + self.c.scale * a

    def hold_default(self):
        self.data.ctrl[self.act_id] = self.c.default_qpos

    def step(self, n_substeps: int = 4):
        mujoco.mj_step(self.model, self.data, nstep=n_substeps)

    def target_contact_force(self) -> float:
        """Max contact-force magnitude on the target pad this step."""
        if self.target_bid < 0:
            return 0.0
        best = 0.0
        f = np.zeros(6)
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if self.target_bid in (b1, b2):
                mujoco.mj_contactForce(self.model, self.data, ci, f)
                best = max(best, float(np.linalg.norm(f[:3])))
        return best


def standing_test(task: str, seconds: float = 3.0):
    c = IsaacContract(f"logs/isaac_contract_{task}.json")
    g = MujocoG1(c)
    g.reset()
    print(f"[stand] start pelvis height = {g.pelvis_height():.3f} m")
    steps = int(seconds / (g.model.opt.timestep * 4))
    for k in range(steps):
        g.hold_default()
        g.step()
        if k % 25 == 0:
            print(f"[stand] t={k*0.02:4.2f}s  pelvis={g.pelvis_height():.3f}m  "
                  f"feet={g.foot_contact().astype(int)}")
    h = g.pelvis_height()
    ok = h > 0.5
    print(f"[stand] final pelvis = {h:.3f} m → {'STANDING ✓' if ok else 'FELL ✗'}")
    return ok


def evaluate(ckpt_path: str, n_episodes: int = 50, max_steps: int = 200,
             seed: int = 0):
    """Run the distilled Task 1 student in MuJoCo; report success rate."""
    import torch
    from src.distill.dagger import StudentPolicy

    c = IsaacContract("logs/isaac_contract_distant_target.json")
    g = MujocoG1(c, with_tools=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    obs_dim = ckpt["student"]["mean_net.net.0.weight"].shape[1]
    closed_loop = obs_dim >= 115   # 115 = proprio 109 + stick/target rel 6
    student = StudentPolicy(obs_dim, 43, tanh=ckpt.get("tanh", False))
    student.load_state_dict(ckpt["student"])
    student.eval()
    print(f"[mujoco-eval] obs_dim={obs_dim} closed_loop={closed_loop} "
          f"tanh={ckpt.get('tanh', False)}")

    rng = np.random.default_rng(seed)
    successes, lengths = 0, []
    for ep in range(n_episodes):
        g.reset(rng)
        g.settle(10)
        hit, fell, steps = False, False, max_steps
        for t in range(max_steps):
            with torch.no_grad():
                obs = torch.from_numpy(g.observation(closed_loop)).unsqueeze(0)
                action = student(obs).squeeze(0).numpy()
            g.apply_action(action)
            g.step()
            if g.target_contact_force() > _FORCE_THRESHOLD:
                hit, steps = True, t + 1
                break
            if g.pelvis_height() < 0.5:
                fell, steps = True, t + 1
                break
        successes += int(hit)
        lengths.append(steps)
        tag = "HIT" if hit else ("FELL" if fell else "timeout")
        print(f"  ep {ep+1:2d}/{n_episodes}: {tag:7s} steps={steps}")

    rate = successes / n_episodes
    print(f"\n[mujoco-eval] success {successes}/{n_episodes} = {rate:.1%} | "
          f"mean steps={np.mean(lengths):.1f}")
    return rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="distant_target")
    ap.add_argument("--standing-test", action="store_true")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--ckpt", default="checkpoints/dagger__distant_target__seed42/iter_020.pt")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    if args.standing_test:
        standing_test(args.task, args.seconds)
    else:
        evaluate(args.ckpt, args.episodes, args.max_steps)
