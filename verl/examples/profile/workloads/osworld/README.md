# OSWorld / ARPO GUI offline-replay profiling

This workload targets **long-horizon GUI agentic RL** style rollouts (dvlab
[ARPO](https://github.com/dvlab-research/ARPO) + [OSWorld](https://github.com/xlang-ai/OSWorld)):

- multi-turn: each turn is a new screenshot observation
- long decode: Thought + Action (configurable hundreds–thousands of tokens)
- growing context: history text + up to N screenshots (10K+ context possible)
- default ``VERL_PROFILE_ROLLOUT_ONLY=1`` so training does not dominate wall time

We do **not** invent a new GUI task set. Tasks come from OSWorld; trajectories
are either:

1. **OSWorld task-backed synthetic** (default, no Docker) — real task JSON
   instructions/domains with controllable rendered screenshots
2. **Converted from real ARPO/OSWorld result dumps** (`traj.jsonl` + PNGs)
3. **Single-instruction synthetic** (`USE_OSWORLD_TASKS=0`) — smoke tests only

Online Docker env rollouts stay in the ARPO repo; GraftRL consumes **offline
replay** of those dumps for profiling and CacheBlend experiments.

## Prepare data

```bash
cd verl
export PATH=/data/conda_envs/verl_vision/bin:$PATH

# A) OSWorld task-backed synthetic 256/64, 8 turns, long Thought strings
bash examples/profile/workloads/osworld/prepare_data.sh

# B) convert real OSWorld/ARPO results
RESULTS_ROOT=/path/to/OSWorld/results \
DATA_ROOT=data/osworld_gui_real \
FORCE=1 \
bash examples/profile/workloads/osworld/prepare_data.sh

# C) pure synthetic smoke data
USE_OSWORLD_TASKS=0 DATA_ROOT=data/osworld_gui_synth FORCE=1 \
bash examples/profile/workloads/osworld/prepare_data.sh
```

ARPO OSWorld submodule (task JSONs) lives at:

```text
/workspace/repo/ARPO/OSWorld/evaluation_examples/
```

## Run profiling (64×4)

```bash
CUDA_VISIBLE_DEVICES=0,1,4,7 \
DATA_ROOT=data/osworld_gui_tasksynth \
TRAIN_BATCH_SIZE=64 ROLLOUT_N=4 \
OSWORLD_RUNTIME_TURNS=8 \
MAX_PROMPT_LENGTH=32768 \
MAX_RESPONSE_LENGTH=2048 \
bash examples/profile/workloads/osworld/run_osworld_profile.sh
```

CacheBlend across turns:

```bash
SGLANG_VLM_CACHEBLEND=1 \
SGLANG_VLM_CACHEBLEND_SELECT=kvdev \
SGLANG_VLM_CACHEBLEND_TARGET_TURNS=all \
bash examples/profile/workloads/osworld/run_osworld_profile.sh
```

## Row schema

| field | meaning |
|-------|---------|
| `prompt` | UI-TARS style instruction + first `<image>` |
| `images` | first screenshot only (dataset placeholder count) |
| `extra_info.screenshots` | all frames for turn-by-turn snowball |
| `extra_info.step_predictions` | recorded Thought+Action (optional labels) |
| `reward_model.ground_truth` | OSWorld score / synthetic `1.0` |

Agent: `osworld_gui_agent` — model generates each assistant turn; next screenshot
is appended as a new user message. Trainer sees **final-turn** response only.

## Relation to online ARPO

| | ARPO online | This GraftRL workload |
|--|-------------|------------------------|
| Env | Docker OSWorld VM | none (offline screenshots) |
| New obs | `env.step` → real GUI | next dumped / synthetic frame |
| Purpose | train GUI policy | rollout / CacheBlend profiling |

After you have real dumps, point `RESULTS_ROOT` at them and keep the same run script.
