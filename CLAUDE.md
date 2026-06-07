## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

# CDL Project — Curiosity-Driven Learning for Unitree G1

## Project Overview
Curiosity-driven RL for emergent tool use on the Unitree G1 humanoid robot (Purdue).  
Curiosity methods: RND, DRND, RDD. Goal: sim-to-real transfer over 16 weeks, 5 tasks.

## Current Status
- **Phase 2 STARTED** — PPO+RND training on Task 1 running locally (32 envs, 10M steps)
- **Running locally** on RTX 3050 (8 GB) — use `configs/local.yaml` (32 envs, 10M steps) instead of `base.yaml` (4096 envs, 200M steps)
- **Viewer on by default** — `configs/local.yaml` has `headless: false`; add `--headless` manually if you need a blind training run
- Last updated: 2026-06-04

## Recent Progress
- Task 5 (`CompositeEnv`): kinematic wall (X=1.5 m, 0.8 m gap) + box (20 kg) + button at 1.8 m; success = button contact ≥ 2 N; 4-object critic obs (40 dims, total=152); subtask tracking: box_placed_rate, robot_on_box_rate; unit test PASSED (2026-06-04)
- Task 4 (`WeightLeverEnv`): plank (2 m board, 2 kg) + heavy box (10 kg) + fulcrum block (15 cm); success = heavy lifted ≥ 0.10 m above spawn Z; 3-object critic obs (30 dims, total=142); unit test PASSED (2026-06-04)
- Task 3 (`OccludedRetrievalEnv`): barrier (2 kinematic walls, 15 cm slot Z=0.35-0.50 m) + 90 cm stick + 10 cm target box; success = target XY displaced >= 0.5 m; mass/friction rand; unit test PASSED (2026-06-04)
- Task 2 (`ElevatedButtonEnv`): box (30×30×30 cm, 20 kg) + button pad at 1.8 m; unit test PASSED (re-confirmed 2026-06-04 on Isaac Sim 4.5 local GPU)
- Task 1: friction rand ±50%, unit test PASSED (100 resets, 0.0% baseline)
- PhysX set_masses/set_material_properties: CPU tensors + all-env indices required
- IsaacLab URDF converter fix: `set_merge_fixed_ignore_inertia` absent in Isaac Sim 4.5 — guarded with `hasattr()` at `/home/kevin/IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py:143`

## Architecture / Key Files
- `src/utils/networks.py` — `ActorCritic`: asymmetric actor (109 policy obs) + dual ext/int critics (132 privileged obs), 43-dim Gaussian
- `src/curiosity/rnd.py` — RND: fixed target + trained predictor, Welford running normalizers
- `src/curiosity/drnd.py` — DRND: N independent (target, predictor) pairs; reward = mean MSE across ensemble
- `src/curiosity/rdd.py` — RDD: 1 fixed target + N predictors; reward = inter-predictor variance (disagreement); sigma = bandwidth
- `src/curiosity/__init__.py` — `make_curiosity(cfg, obs_dim, device)` factory: rnd | drnd | rdd
- `src/agents/ppo.py` — PPO loop with generic `self.curiosity` interface; writes `logs/{run_name}/metrics.csv` every log interval
- `plot_runs.py` — compare RND/DRND/RDD from CSVs: `python3 plot_runs.py [--smooth N] [--out file.png]`
- `src/envs/__init__.py` — `make_env(cfg)` factory maps task name → env class + config
- `launch.py` — rewritten: SimulationApp created before Isaac Lab imports; `--no-wandb`, `--viewer`, `--resume` flags

## GitHub / Version Control
- Repo: `git@github.com:yags28/curiosity_G1.git` (branch: `main`)
- SSH key: `~/.ssh/id_ed25519_github` — public key must be added to GitHub account
- `.gitignore` excludes: `deps/` (525 MB), `checkpoints/`, `wandb/`, `logs/`, `videos/`, `.claude/settings.local.json`
- Auto-push: cron runs `scripts/auto_push.sh` daily at 5 pm Eastern; log → `~/.local/share/cdl_push.log`
- First push requires: (1) add SSH pub key to GitHub, (2) create empty `curiosity_G1` repo, (3) `git push -u origin main`

## Shell Aliases
- `task1`–`task5` defined in `~/.bash_aliases` — run each unit test with Isaac Lab viewer (`--viewer`, headless off) via `/home/kevin/isaacsim/python.sh`

## Experiment Results (Task 1 — distant_target)
- **RND** (10M steps, no metrics): policy logstd=6.58 → likely 0% success; no CSV/wandb data
- **DRND** (COMPLETE, 10M steps): phase transition at ~4.8M → 44% → **100% success** at end; ep_reward=+1.246, ep_len=12 steps; CSV at `logs/distant_target__drnd__seed42/`
- **RDD** (COMPLETE, 10M steps): **0% success throughout**; ep_len collapsed to 5.5 steps; disagreement signal decayed too fast; CSV at `logs/distant_target__rdd__seed42/`

## Next Steps
- DRND T3+T5 RUNNING (background ID: bujyfvmyw) via `scripts/run_drnd_shaped.sh`; T3 at 100% success already
- DRND T2 QUEUED (background ID: b78ixvset) via `scripts/run_task2_after_chain.sh`; starts after T5; box mass 10 kg + dense shaping
- Phase 3 after all 5 tasks: DAgger policy distillation + residual RL (`src/distill/`)
- Cluster: RCAC account/GPU allocation (when available) — needed for Phase 4 sim-to-real
