from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EnvConfig:
    env_id: str = "SuperMarioBros-1-1-v3"
    action_set: str = "simple"
    skip_frames: int = 4
    resize_shape: int = 84
    stack_frames: int = 4
    seed: int = 42


@dataclass(slots=True)
class AgentConfig:
    batch_size: int = 32
    gamma: float = 0.95
    learning_rate: float = 1e-4
    replay_buffer_size: int = 100_000
    exploration_rate: float = 0.75
    exploration_rate_decay: float = 0.999995
    exploration_rate_min: float = 0.01
    sync_steps: int = 10
    # Soft target update (tau > 0 enables Polyak averaging; tau = 0 uses hard sync)
    tau: float = 0.0
    # Gradient clipping (None = disabled, e.g. 10.0)
    grad_clip: float | None = None
    # Dueling DQN architecture
    dueling: bool = False


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = 1000
    checkpoint_period: int = 20
    checkpoint_path: str | None = None
    save_dir: Path = Path("weights")
    device: str = "cpu"
    render: bool = False
    max_steps_per_episode: int | None = None
    log_window: int = 10
    # Evaluation
    eval_every: int | None = None  # evaluate every N episodes (None = disabled)
    eval_episodes: int = 5  # number of episodes per eval run
    # Early stopping
    early_stop_patience: int = 0  # 0 = disabled
