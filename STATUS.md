# CDL Project Status
**Last updated:** 2026-06-04 (Session 4)
**Timeline position:** Week 1 of 16 — Phase 1 (Environment Design, Weeks 1–3)

---

## Pre-Phase — Environment & Hardware Setup
| Item | Status | Notes |
|---|---|---|
| Purdue RCAC cluster accounts | Unknown | Verify Anvil/Gilbreth/Negishi access |
| GPU allocation (≥4 A100/H100) | Unknown | Request via RCAC portal |
| Isaac Lab cloned (`deps/IsaacLab`) | Done | |
| `rsl_rl` cloned (`deps/rsl_rl`) | Done | |
| `unitree_rl_gym` cloned (`deps/unitree_rl_gym`) | Done | |
| Project repo structure (`src/`, `configs/`, etc.) | Done | All subdirs + `__init__.py` files created |
| wandb accounts created | Done | `ydawanka` (ydawanka-purdue-university), `bhagat14` |
| wandb team shared project | Pending | Invite `bhagat14` to `ydawanka-purdue-university` team on wandb.ai |
| Hardware inventory (G1, gantry, tools) | Unknown | |

---

## Phase 1 — Environment Design in Isaac Lab (Weeks 1–3)

### §3.1 Software Installation & Infrastructure (Week 1, Days 1–3)
| Item | Status | Notes |
|---|---|---|
| Isaac Lab running on cluster (SLURM, ≥4096 envs) | Pending | |
| Isaac Lab sanity-check scene (GPU ≥80%) | Pending | |
| conda env with all Python deps | Pending | `requirements.txt` ready |
| G1 standing task confirmed loading | Pending | |
| Pre-commit hooks (black, flake8, isort) | Pending | `pre-commit` in requirements |
| GitHub Actions / CI | Pending | |
| `launch.py` experiment launcher | **Done** | Accepts YAML config, CLI overrides, wandb logging |
| `configs/base.yaml` | **Done** | Full hyperparameter config (env, PPO, curiosity, reward) |
| `configs/sweep_rnd.yaml` | **Done** | Bayesian wandb sweep over 9 hyperparams |
| wandb project `CDL-HumanoidToolUse` | **Done** | Entity: `ydawanka-purdue-university`; connectivity confirmed |
| wandb standard metrics defined | **Done** | episode_reward, intrinsic_reward, extrinsic_reward, success_rate, tool_contact_count, alpha_annealing |
| wandb sweep agent tested on dummy env | Pending | Run: `wandb sweep configs/sweep_rnd.yaml` |
| Commit tagged `v0.0.1` | Pending | Tag after Isaac Lab confirmed on cluster |

### §3.2 G1 URDF Integration & Validation (Week 1, Days 4–5)
| Item | Status | Notes |
|---|---|---|
| G1 URDF (43 DOF) identified | **Done** | `deps/unitree_rl_gym/.../g1_29dof_with_hand_rev_1_0.urdf` |
| 43 joints parsed and verified | **Done** | `scripts/validate_g1_urdf.py` — all PASS |
| Joint limits finite and valid | **Done** | All 43 joints confirmed |
| Sensor links present (pelvis, wrists, head) | **Done** | All 4 links confirmed in URDF |
| Random 100-step rollout (no NaN) | **Done** | Standalone check passes |
| Joint index table documented | **Done** | Printed by `validate_g1_urdf.py` (indices 0–42) |
| Isaac Lab `ArticulationCfg` for G1 EDU | **Done** | `src/envs/g1_cfg.py` — all 5 actuator groups + sensors |
| PD controller test in Isaac Lab | **Done** | `scripts/pd_controller_test.py` — PASSED; mean pelvis 0.747 m, p90 err 0.137 rad |
| G1 reach envelope (full FK in Isaac Lab) | **Done** | `scripts/sensor_reach_test.py` — peak 1.02 m from pelvis to wrist at 90° extension |
| IMU / wrist F-T / head camera confirmed live | **Done** | IMU ✓, wrist FT ✓ (z=1.01 m), Camera ✓ spawns (1×64×64 depth tensor); values inf = placement issue, RTX pipeline confirmed |

### §3.3 Task Environment Implementation (Weeks 1–2)
| Item | Status | Notes |
|---|---|---|
| `ToolUseBase` environment class | **Done** | `src/envs/tool_use_base.py` — DirectRLEnv base; restructured to AnymalC pattern (robot/sensor/terrain on env cfg, not scene cfg); no duplicate prim issue |
| `src/envs/mdp/` (obs, rewards, terminations, events) | **Done** | Custom MDP terms for all tasks |
| Task 1: `DistantTargetEnv` (Easy) | **Done** | Full impl + friction rand ±50%; unit test PASS (100 resets, no NaN/interpenetration); random baseline 0.0% |
| Task 2: `ElevatedButtonEnv` (Medium) | **Done** | Box (30cm, 20 kg) + button pad at 1.8 m; mass/friction rand; unit test PASS (100 resets, 0.0% baseline) |
| Task 3: `OccludedRetrievalEnv` (Medium) | **Done** | Barrier (2 kinematic pieces, 15 cm slot Z=0.35–0.50 m) + stick + target; mass/friction rand; unit test ready (`scripts/unit_test_task3.py`) |
| Task 4: `WeightLeverEnv` (Hard) | **Done** | Plank (2 m board) + 10 kg heavy box + fulcrum block; success = heavy lifted ≥ 0.10 m; mass/friction rand; unit test ready (`scripts/unit_test_task4.py`) |
| Task 5: `CompositeEnv` (Hard) | **Done** | Wall gap (X=1.5 m, 0.8 m gap) + box (20 kg) + button at 1.8 m; critic obs 152; subtask tracking (box_placed, robot_on_box); unit test at `scripts/unit_test_task5.py` (pending run) |
| Gym registrations (`CDL-*-v0`) | **Done** | `src/envs/__init__.py` — all 5 envs registered |
| Unit tests for all 5 envs | Partial | Task 1 PASS (`scripts/unit_test_task1.py`); Tasks 2–5 pending |

### §3.4 Object Randomisation & Reward Specification (Week 2)
| Item | Status | Notes |
|---|---|---|
| Position randomisation (±30%) | **Done** | Task 1: stick XY ±30%, distractor jitter ±40% |
| Mass / friction randomisation | **Done** | Task 1: mass ±30%, friction ±50%; PhysX API requires CPU tensors + explicit all-env indices |
| Distractor objects (1–3) | **Done** | Task 1: 2 distractors with random positions |
| Binary extrinsic reward (+1 / −0.001/step) | **Done** | ToolUseEnv base |
| Random-agent baseline (<0.1% success) | **Done** | Task 1: 0.0% over 200 episodes ✓ |

### §3.5 Phase 1 Deliverables & Exit Criteria
| Item | Status | Notes |
|---|---|---|
| All 5 envs unit-tested and committed | Pending | |
| SLURM job sustains ≥4096 envs for ≥1 hour | Pending | |
| wandb dummy-agent run logged | Pending | |
| Phase 1 review meeting with advisor | Pending | |

---

## Phase 2 — Curiosity-Driven RL Training (Weeks 4–9)
| Item | Status |
|---|---|
| RND curiosity module | Pending |
| DRND curiosity module | Pending |
| RDD curiosity module | Pending |
| ICM baseline | Pending |
| Asymmetric actor-critic (PPO) | Pending |
| Reward annealing (α: 1.0 → 0.1) | Pending |
| Hyperparameter sweeps | Pending |
| ≥20% success on Distant Target (RND/DRND) | Pending |

---

## Phase 3 — Policy Distillation & Robustification (Weeks 10–11)
| Item | Status |
|---|---|
| DAgger behaviour cloning | Pending |
| Residual RL refinement | Pending |
| MuJoCo sim-to-sim cross-validation | Pending |

---

## Phase 4 — Sim-to-Real Transfer (Weeks 12–14)
| Item | Status |
|---|---|
| Domain randomisation ranges set | Pending |
| Safety gantry setup | Pending |
| Hardware deployment | Pending |
| ≥30% success on physical G1 | Pending |

---

## Phase 5 — Paper Writing & Code Release (Weeks 15–16)
| Item | Status |
|---|---|
| Paper draft | Pending |
| Code release / open-source | Pending |
