# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build / Run / Test

```bash
# Install dependencies (Python 3.9–3.11)
pip install -r requirements.txt

# Train with defaults (DDQN, gamma=0.95, replay=100K, 1000 episodes)
python train.py

# Train with custom hyperparameters (all overridable from CLI)
python train.py --episodes 500 --action-set complex --device cuda
python train.py --lr 5e-5 --gamma 0.99 --replay-size 200000 --batch-size 64

# Resume from a checkpoint
python train.py --checkpoint weights/checkpoint_662.pth

# Full CLI reference
python train.py --help

# Smoke-test the wrapped environment (no training)
python env_test.py --steps 200
python env_test.py --steps 200 --render
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

This is a **Double DQN** (DDQN) agent for `gym-super-mario-bros`. The codebase was refactored from a single-file script into modules by responsibility.

### Data flow (training loop)

```
train.py                    — CLI → assemble configs, env, agent, trainer
  └─ mario_rl/config.py    — EnvConfig / AgentConfig / TrainingConfig dataclasses
  └─ mario_rl/env.py       — build_env() wraps gym-super-mario-bros through a stack:
       JoypadSpace → SkipFrame(4) → GrayScaleObservation → ResizeObservation(84×84) → FrameStack(4)
  └─ mario_rl/agent.py     — MarioAgent: ε-greedy act(), remember() to deque replay buffer, learn() via DDQN
  └─ mario_rl/model.py     — DoubleDQNNetwork: identical online/target conv nets (3 conv → flatten → 2 FC)
  └─ mario_rl/trainer.py   — Episode loop: run_episode() then checkpoint on best moving-average reward
  └─ mario_rl/utils.py     — resolve_device(), seed_everything(), + compat wrappers for Gym 0.21 vs 0.26 step/reset APIs
```

### Key design decisions

- **Double DQN**: The online network selects the best action for the next state; the target network evaluates it. This decouples action selection from value estimation to reduce overestimation bias.
- **Target network sync**: Every `sync_steps` (default 10) learning steps, the target network's weights are copied from the online network (hard update, no Polyak averaging).
- **Exploration decay**: ε is multiplied by `exploration_rate_decay` (0.999995) **per action**, not per episode. This is a very slow exponential decay.
- **Gamma**: Default 0.95 (increased from original 0.78) — high enough for the agent to receive meaningful reward signal from the flag pole 300-500 steps away.
- **Replay memory**: A plain `collections.deque` with `maxlen=100,000` (increased from 10K). Not prioritized, no n-step returns.
- **Checkpoint format**: A dict `{"model": <state_dict>, "exploration_rate": <float>}` saved as `.pth`. Loading restores both the network weights and the current exploration rate.
- **Dynamic conv dimension**: The flattened feature size after conv layers is computed from the input shape, not hardcoded. Changing `resize_shape` or `stack_frames` now works without manual dimension updates.
- **Gym API compat**: `step_env()` and `reset_env()` in `utils.py` handle both the old 4-tuple `(obs, reward, done, info)` and new 5-tuple `(obs, reward, terminated, truncated, info)` return formats.

### Where to change specific things

| Concern | File |
|---|---|
| DDQN hyperparameters (γ, η, batch size, ε decay, sync freq) | `mario_rl/config.py` → `AgentConfig` |
| Training loop (episodes, checkpoint period, log window) | `mario_rl/config.py` → `TrainingConfig` |
| Environment settings (env id, action set, frame skip, resolution) | `mario_rl/config.py` → `EnvConfig` |
| Neural network architecture | `mario_rl/model.py` → `DoubleDQNNetwork` |
| Exploration strategy or replay buffer | `mario_rl/agent.py` → `MarioAgent.act()` / `remember()` / `learn()` |
| Frame preprocessing pipeline | `mario_rl/env.py` → `build_env()` + wrapper classes |
| Checkpoint save/load logic | `mario_rl/agent.py` → `save_checkpoint()` / `load()` |
| Action space definitions | `mario_rl/actions.py` |

### Checkpoint model structure

The network expects input shape `(batch, 4, 84, 84)` — 4 stacked grayscale frames resized to 84×84. The conv backbone produces a 3136-dim flattened feature vector before the FC head. If you change the input resolution or stack size, you must also update `resize_shape` / `stack_frames` in `EnvConfig` and the linear layer input dimension (3136) in `model.py`.

### Saved weights

`weights/checkpoint_662.pth` ships with the repo and is the known-good checkpoint referenced in the README. It is compatible with the current model structure.
