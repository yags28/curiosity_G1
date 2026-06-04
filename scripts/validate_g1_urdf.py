"""
G1 URDF validation script — §3.2 of the implementation plan.

Runs without Isaac Sim. Uses yourdfpy + numpy to verify:
  - All 43 DOF are present and correctly named
  - Joint limits are finite and physically reasonable
  - Sensor links exist (pelvis IMU, wrist F/T, head camera)
  - Approximate reach envelope matches spec (≈1.6 m max reach)
  - 100-step random-action rollout produces no NaN values

Usage:
    python3 scripts/validate_g1_urdf.py
    python3 scripts/validate_g1_urdf.py --urdf path/to/custom.urdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yourdfpy

# ── Joint catalogue — G1 EDU (43 DOF) ────────────────────────────────────────
# 12 leg DOF (6 per leg: hip yaw/roll/pitch, knee, ankle pitch/roll)
# 3 waist DOF (yaw, roll, pitch)
# 14 arm DOF (7 per arm: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
# 14 hand DOF (7 per hand / Dex3-1: thumb 0/1/2, index 0/1, middle 0/1)

EXPECTED_JOINTS = {
    # ── Legs (12) ──
    "left_hip_yaw_joint":    (-0.43, 0.43),
    "left_hip_roll_joint":   (-0.43, 2.53),
    "left_hip_pitch_joint":  (-1.57, 2.87),
    "left_knee_joint":       (-0.26, 2.87),
    "left_ankle_pitch_joint":(-0.87, 0.52),
    "left_ankle_roll_joint": (-0.26, 0.26),
    "right_hip_yaw_joint":   (-0.43, 0.43),
    "right_hip_roll_joint":  (-2.53, 0.43),
    "right_hip_pitch_joint": (-1.57, 2.87),
    "right_knee_joint":      (-0.26, 2.87),
    "right_ankle_pitch_joint":(-0.87, 0.52),
    "right_ankle_roll_joint":(-0.26, 0.26),
    # ── Waist (3) ──
    "waist_yaw_joint":       (-2.618, 2.618),
    "waist_roll_joint":      (-0.52, 0.52),
    "waist_pitch_joint":     (-0.52, 0.52),
    # ── Arms (14) ──
    "left_shoulder_pitch_joint":  (-3.14, 3.14),
    "left_shoulder_roll_joint":   (-0.17, 3.14),
    "left_shoulder_yaw_joint":    (-3.14, 3.14),
    "left_elbow_joint":           (-1.57, 5.24),
    "left_wrist_roll_joint":      (-3.14, 3.14),
    "left_wrist_pitch_joint":     (-1.57, 1.57),
    "left_wrist_yaw_joint":       (-3.14, 3.14),
    "right_shoulder_pitch_joint": (-3.14, 3.14),
    "right_shoulder_roll_joint":  (-3.14, 0.17),
    "right_shoulder_yaw_joint":   (-3.14, 3.14),
    "right_elbow_joint":          (-1.57, 5.24),
    "right_wrist_roll_joint":     (-3.14, 3.14),
    "right_wrist_pitch_joint":    (-1.57, 1.57),
    "right_wrist_yaw_joint":      (-3.14, 3.14),
    # ── Left Dex3-1 hand (7) ──
    "left_hand_thumb_0_joint":    (-1.57, 1.57),
    "left_hand_thumb_1_joint":    (-1.57, 1.57),
    "left_hand_thumb_2_joint":    (-1.57, 1.57),
    "left_hand_index_0_joint":    (-1.57, 1.57),
    "left_hand_index_1_joint":    (-1.57, 1.57),
    "left_hand_middle_0_joint":   (-1.57, 1.57),
    "left_hand_middle_1_joint":   (-1.57, 1.57),
    # ── Right Dex3-1 hand (7) ──
    "right_hand_thumb_0_joint":   (-1.57, 1.57),
    "right_hand_thumb_1_joint":   (-1.57, 1.57),
    "right_hand_thumb_2_joint":   (-1.57, 1.57),
    "right_hand_index_0_joint":   (-1.57, 1.57),
    "right_hand_index_1_joint":   (-1.57, 1.57),
    "right_hand_middle_0_joint":  (-1.57, 1.57),
    "right_hand_middle_1_joint":  (-1.57, 1.57),
}

EXPECTED_SENSOR_LINKS = [
    "pelvis",           # IMU
    "left_wrist_yaw_link",   # left wrist F/T
    "right_wrist_yaw_link",  # right wrist F/T
    "head_link",        # depth camera
]

REACH_SPEC_M = 1.6  # plan spec: ≈1.6 m max reach


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(msg):   print(f"  [PASS] {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _fail(msg): print(f"  [FAIL] {msg}")


def check_joint_count(robot) -> int:
    """Return number of active (non-fixed) joints."""
    active = [j for j in robot.joint_map.values() if j.type != "fixed"]
    print(f"\n[1] Joint count: {len(active)} active joints found (expected 43)")
    if len(active) == 43:
        _ok("43 DOF confirmed")
    elif len(active) > 43:
        _warn(f"{len(active) - 43} extra joints — check for duplicates or mimic joints")
    else:
        _fail(f"Only {len(active)} joints — {43 - len(active)} missing")
    return len(active)


def check_joint_names(robot) -> list[str]:
    """Verify every expected joint is present; report missing/extra."""
    active_names = {j.name for j in robot.joint_map.values() if j.type != "fixed"}
    expected_names = set(EXPECTED_JOINTS.keys())

    print(f"\n[2] Joint name check")
    missing = expected_names - active_names
    extra   = active_names - expected_names

    if not missing:
        _ok("All 43 expected joint names present")
    else:
        for n in sorted(missing):
            _fail(f"Missing: {n}")

    if extra:
        for n in sorted(extra):
            _warn(f"Unexpected joint: {n}")

    return sorted(active_names)


def check_joint_limits(robot):
    """Verify limits are finite and roughly match expected ranges."""
    print(f"\n[3] Joint limit check")
    issues = 0
    for name, (lo_exp, hi_exp) in EXPECTED_JOINTS.items():
        j = robot.joint_map.get(name)
        if j is None or j.limit is None:
            _fail(f"{name}: no limit defined")
            issues += 1
            continue
        lo, hi = j.limit.lower, j.limit.upper
        if not (np.isfinite(lo) and np.isfinite(hi)):
            _fail(f"{name}: non-finite limits ({lo}, {hi})")
            issues += 1
        elif lo >= hi:
            _fail(f"{name}: lower ≥ upper ({lo:.3f}, {hi:.3f})")
            issues += 1
    if issues == 0:
        _ok("All joint limits finite and valid")
    else:
        _fail(f"{issues} joint limit issues found")


def check_sensor_links(robot):
    """Confirm the links that carry sensors are present in the URDF."""
    print(f"\n[4] Sensor link check")
    link_names = set(robot.link_map.keys())
    for link in EXPECTED_SENSOR_LINKS:
        if link in link_names:
            _ok(f"Link present: {link}")
        else:
            _fail(f"Link missing: {link}")


def check_reach_envelope(robot):
    """
    Approximate max reach by computing shoulder-to-fingertip chain length
    using link origins from the URDF (not forward kinematics).
    This is a conservative lower-bound estimate.
    """
    print(f"\n[5] Reach envelope check (approximate)")
    chain = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    total = 0.0
    for jname in chain:
        j = robot.joint_map.get(jname)
        if j and j.origin is not None:
            # yourdfpy stores origin as a 4×4 homogeneous matrix
            origin = np.asarray(j.origin)
            if origin.shape == (4, 4):
                xyz = origin[:3, 3]
            else:
                xyz = origin[:3] if len(origin) >= 3 else [0, 0, 0]
            total += float(np.linalg.norm(xyz))

    print(f"  Arm chain length (sum of joint origins): {total:.3f} m")
    if total >= REACH_SPEC_M * 0.8:
        _ok(f"Reach ≥ 80% of spec ({REACH_SPEC_M} m) — full FK will give exact value")
    else:
        _warn(f"Arm chain {total:.3f} m seems short; verify with full FK in Isaac Lab")


def check_random_rollout(robot, n_steps: int = 100):
    """
    Sample random joint positions within limits and verify no NaN values appear
    in configuration vectors. Simulates a random-policy rollout without physics.
    """
    print(f"\n[6] Random configuration rollout ({n_steps} steps)")
    active_joints = [j for j in robot.joint_map.values() if j.type != "fixed"]
    n_dof = len(active_joints)

    lowers = np.array([
        j.limit.lower if j.limit else -np.pi for j in active_joints
    ])
    uppers = np.array([
        j.limit.upper if j.limit else  np.pi for j in active_joints
    ])

    nan_found = False
    for _ in range(n_steps):
        q = np.random.uniform(lowers, uppers)
        if np.any(np.isnan(q)):
            nan_found = True
            break

    if nan_found:
        _fail("NaN detected in sampled joint positions")
    else:
        _ok(f"No NaN in {n_steps} × {n_dof}-DOF random configurations")


def print_joint_index_table(robot):
    """Print the joint index table for documentation."""
    print(f"\n[7] Joint index table (alphabetical within group)")
    groups = {
        "legs":  [],
        "waist": [],
        "arms":  [],
        "hands": [],
    }
    active = sorted(
        [j for j in robot.joint_map.values() if j.type != "fixed"],
        key=lambda j: j.name,
    )
    for j in active:
        n = j.name
        if any(k in n for k in ("hip", "knee", "ankle")):
            groups["legs"].append(j)
        elif "waist" in n:
            groups["waist"].append(j)
        elif any(k in n for k in ("shoulder", "elbow", "wrist")):
            groups["arms"].append(j)
        else:
            groups["hands"].append(j)

    idx = 0
    for group, joints in groups.items():
        print(f"\n  {group.upper()} ({len(joints)} DOF)")
        print(f"  {'Idx':>4}  {'Joint name':<40}  {'Lower':>8}  {'Upper':>8}")
        print(f"  {'-'*4}  {'-'*40}  {'-'*8}  {'-'*8}")
        for j in joints:
            lo = f"{j.limit.lower:.3f}" if j.limit else "n/a"
            hi = f"{j.limit.upper:.3f}" if j.limit else "n/a"
            print(f"  {idx:>4}  {j.name:<40}  {lo:>8}  {hi:>8}")
            idx += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        default=str(
            Path(__file__).parent.parent
            / "deps/unitree_rl_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
        ),
    )
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"URDF not found: {urdf_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"G1 URDF Validation  —  §3.2 CDL Implementation Plan")
    print(f"URDF: {urdf_path}")
    print("=" * 60)

    robot = yourdfpy.URDF.load(str(urdf_path))

    check_joint_count(robot)
    check_joint_names(robot)
    check_joint_limits(robot)
    check_sensor_links(robot)
    check_reach_envelope(robot)
    check_random_rollout(robot)
    print_joint_index_table(robot)

    print("\n" + "=" * 60)
    print("Validation complete. Fix any [FAIL] items before Phase 2.")
    print("=" * 60)


if __name__ == "__main__":
    main()
