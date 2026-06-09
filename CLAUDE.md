# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build / Run / Test

```bash
# Install dependencies (Python 3.9–3.11)
pip install -r requirements.txt

# === DDQN ===
# Train with defaults (gamma=0.95, replay=100K, 1000 episodes)
python train.py

# Train with custom hyperparameters (all overridable from CLI)
python train.py --episodes 500 --action-set complex --device cuda
python train.py --lr 5e-5 --gamma 0.99 --replay-size 200000 --batch-size 64

# Resume from a checkpoint
python train.py --checkpoint weights/ddqn/checkpoint_662.pth

# Full CLI reference
python train.py --help

# === PPO ===
# Train with defaults (world 1-1, simple actions)
python train_ppo.py

# Train specific level
python train_ppo.py --world 1 --stage 2 --lr 1e-4 --device cuda

# === A3C+LSTM (cross-level generalization) ===
# Download pretrained weights, convert to .pth, export ONNX
python scripts/convert_a3c.py

# Evaluate on levels 1-1 through 1-4
python scripts/eval_a3c.py

# Evaluate with rendering
python scripts/eval_a3c.py --levels 1-1 --render

# Multi-level visual demo (default 1-1 through 1-4)
python demo.py

# Demo specific levels
python demo.py --levels 1-1 1-2

# Demo with slow motion
python demo.py --slow
```

There is no test suite or linter configured. `env_test.py` is the only automated check — it runs random actions through the wrapped environment to confirm the Gym pipeline works.

## Remote training (Win-SCP server)

A remote GPU server is available for training at `10.122.242.98:20002` (account `stu_519`).

### Quick commands

```bash
# Check remote GPU utilization
python remote_train.py gpus

# Check training sessions and recent logs
python remote_train.py status

# Sync code to remote + launch training (foreground)
python remote_train.py full --episodes 1000 --action-set simple

# Sync code to remote + launch training (background via screen)
python remote_train.py full --episodes 1000 --action-set simple --background

# Sync only (no training)
python remote_train.py sync

# Run training without re-syncing
python remote_train.py train --episodes 500 --background

# Download checkpoints from remote to local weights/
python remote_train.py download

# Stop remote training sessions
python remote_train.py kill                    # kill all mario_train sessions
python remote_train.py kill --session <name>   # kill a specific session
```

The server has 2× RTX 2080 Ti (22GB each), PyTorch 2.5.1+cu121, gym 0.21.0, gym-super-mario-bros 7.4.0. Background training uses `screen` (not nohup) for reliability — use `status` to check progress and `kill` to stop.

### Remote server details

- **Host**: 10.122.242.98:20002
- **Account**: stu_519 / 519123
- **Remote project dir**: `/home/stu_519/mario_rl/`
- **Python**: python3 (system Python 3.10 with packages in `~/.local`)
- **GPU**: 2× NVIDIA GeForce RTX 2080 Ti (22,528 MiB each)
- **CUDA**: 12.2 driver, PyTorch compiled with CUDA 12.1
- **Remote packages**: `pip3 install --user <package>` to add new deps

### Shared server considerations

Other students also train on this server. Before launching a long training run, check GPU availability with `python remote_train.py gpus`. Both GPUs may be in use — the training script uses `cuda` (GPU 0) by default. To use GPU 1, pass `--device cuda:1`.

### Compatibility patches (first-time remote setup)

Python 3.10 has numpy overflow issues with `nes_py` and `gym_super_mario_bros` due to uint8 arithmetic. If you get `OverflowError: Python integer ... out of bounds for uint8`, apply these patches on the remote:

```bash
# Fix nes_py _rom.py (2 locations)
sed -i 's/return self.prg_rom_start + self.prg_rom_size \* 2\*\*10/return int(self.prg_rom_start) + int(self.prg_rom_size) * 2**10/' ~/.local/lib/python3.10/site-packages/nes_py/_rom.py
sed -i 's/return self.chr_rom_start + self.chr_rom_size \* 2\*\*10/return int(self.chr_rom_start) + int(self.chr_rom_size) * 2**10/' ~/.local/lib/python3.10/site-packages/nes_py/_rom.py

# Fix gym_super_mario_bros smb_env.py (2 locations)
sed -i 's/return self.ram\[0x6d\] \* 0x100 + self.ram\[0x86\]/return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])/' ~/.local/lib/python3.10/site-packages/gym_super_mario_bros/smb_env.py
sed -i 's/return self.ram\[0x075f\] \* 4 + self.ram\[0x075c\]/return int(self.ram[0x075f]) * 4 + int(self.ram[0x075c])/' ~/.local/lib/python3.10/site-packages/gym_super_mario_bros/smb_env.py
```

These patches have already been applied on the server as of 2026-05-31.

## Architecture

This codebase implements **three RL algorithms** for `gym-super-mario-bros`, organized by model type under `mario_rl/`.

### Directory layout

```
mario_rl/
├── __init__.py          # Top-level API exports
├── actions.py           # Shared action set mappings
├── utils.py             # Shared: device, seed, Gym API compat
├── metrics.py           # Shared: training metrics logging
├── ddqn/                # Double DQN
│   ├── agent.py         #   MarioAgent: ε-greedy + deque replay
│   ├── model.py         #   DoubleDQNNetwork: online/target conv nets
│   ├── config.py        #   EnvConfig / AgentConfig / TrainingConfig
│   ├── env.py           #   build_env(): SkipFrame + GrayScale + Resize + FrameStack
│   └── trainer.py       #   Trainer: episode loop + checkpoint
├── ppo/                 # PPO (Proximal Policy Optimization)
│   ├── model.py         #   PPONetwork: 4-conv Actor-Critic
│   ├── env.py           #   build_ppo_env(): CustomReward + CustomSkipFrame
│   └── trainer.py       #   PPOAgent + train_ppo(): GAE + Clipped Surrogate
└── a3c/                 # A3C+LSTM (MarioNET, cross-level generalization)
    ├── model.py         #   MarioNET: ResBlock + LSTMCell + Actor/Critic
    └── env.py           #   build_a3c_env(): 80×80 crop + RunningMeanStd norm
```

### Data flow (DDQN training loop)

```
train.py                    — CLI → assemble configs, env, agent, trainer
  └─ mario_rl/ddqn/config.py   — EnvConfig / AgentConfig / TrainingConfig dataclasses
  └─ mario_rl/ddqn/env.py      — build_env() wraps gym-super-mario-bros through a stack:
       JoypadSpace → StepCompatWrapper → SkipFrame(4) → GrayScaleObservation → ResizeObservation(84×84) → FrameStack(4)
  └─ mario_rl/ddqn/agent.py    — MarioAgent: ε-greedy act(), remember() to deque replay buffer, learn() via DDQN
  └─ mario_rl/ddqn/model.py    — DoubleDQNNetwork: identical online/target conv nets (3 conv → flatten → 2 FC)
  └─ mario_rl/ddqn/trainer.py  — Episode loop: run_episode() then checkpoint on best moving-average reward
```

### Data flow (PPO training loop)

```
train_ppo.py                     — CLI → build env, agent, launch train_ppo()
  └─ mario_rl/ppo/env.py        — build_ppo_env(): JoypadSpace → CustomReward → CustomSkipFrame(4)
  └─ mario_rl/ppo/model.py      — PPONetwork: 4×Conv2d(3×3,stride2) → FC(1152→512) → Actor/Critic heads
  └─ mario_rl/ppo/trainer.py    — PPOAgent: GAE + Clipped Surrogate + Entropy bonus; train_ppo(): single-env loop
```

### Data flow (A3C+LSTM evaluation)

```
scripts/eval_a3c.py              — Load MarioNET, evaluate across levels
  └─ mario_rl/a3c/model.py      — MarioNET: 3×ResBlock → FC(3200→512) → LSTMCell(512→512) → Actor(10)/Critic(1)
  └─ mario_rl/a3c/env.py        — build_a3c_env(): JoypadSpace → A3CSkipFrame(4) → A3CFrameStack(4) → A3CNormalize
```

### Key design decisions

**DDQN:**
- Online network selects best action; target network evaluates it — decouples selection from estimation
- Target sync: hard copy every `sync_steps` (10) steps (no Polyak by default)
- ε decay: multiplied by 0.999995 per action (very slow exponential)
- Replay: plain `deque(maxlen=100,000)`, no prioritization, no n-step returns
- Checkpoint: `{"model": <state_dict>, "exploration_rate": <float>}`

**PPO:**
- Single-process implementation (not multi-process like uvipen/vietnh1009)
- Actor-Critic with shared CNN backbone
- GAE advantage estimation (γ=0.9, τ=1.0)
- Clipped surrogate objective (ε=0.2)
- Critic loss weighted by 0.5

**A3C+LSTM (MarioNET):**
- LSTM hidden/cell states track temporal context — this is WHY it generalizes across levels
- ResBlock residual connections for deeper gradient flow
- RunningMeanStd normalization (obs_rms.pkl) — must match training exactly
- 10-action CUSTOM_MOVEMENT space (vs 7 for DDQN/PPO simple set)

**Shared:**
- Gym API compat: `step_env()` / `reset_env()` handle both 4-tuple and 5-tuple
- All checkpoints use `{"model": state_dict}` wrapper format

### Where to change specific things

| Concern | File |
|---|---|
| DDQN hyperparameters (γ, η, batch size, ε decay, sync freq) | `mario_rl/ddqn/config.py` → `AgentConfig` |
| DDQN training loop (episodes, checkpoint period, log window) | `mario_rl/ddqn/config.py` → `TrainingConfig` |
| DDQN environment settings (env id, action set, frame skip, resolution) | `mario_rl/ddqn/config.py` → `EnvConfig` |
| DDQN neural network architecture | `mario_rl/ddqn/model.py` → `DoubleDQNNetwork` |
| DDQN exploration strategy or replay buffer | `mario_rl/ddqn/agent.py` → `act()` / `remember()` / `learn()` |
| DDQN frame preprocessing pipeline | `mario_rl/ddqn/env.py` → `build_env()` + wrapper classes |
| PPO network architecture | `mario_rl/ppo/model.py` → `PPONetwork` |
| PPO training loop + GAE | `mario_rl/ppo/trainer.py` → `train_ppo()` / `PPOAgent.learn()` |
| PPO environment + reward shaping | `mario_rl/ppo/env.py` → `CustomReward` / `build_ppo_env()` |
| A3C+LSTM network | `mario_rl/a3c/model.py` → `MarioNET` / `ResBlock` |
| A3C+LSTM environment preprocessing | `mario_rl/a3c/env.py` → `build_a3c_env()` / `process_frame_a3c()` |
| Action space definitions | `mario_rl/actions.py` → `ACTION_SETS` |
