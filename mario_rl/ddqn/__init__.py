"""DDQN (Double DQN) 子包。

提供 DDQN 智能体训练和评估的完整工具链。
"""

from mario_rl.ddqn.agent import MarioAgent
from mario_rl.ddqn.config import AgentConfig, EnvConfig, TrainingConfig
from mario_rl.ddqn.env import build_env
from mario_rl.ddqn.model import DoubleDQNNetwork
from mario_rl.ddqn.trainer import Trainer

__all__ = [
    "MarioAgent",
    "AgentConfig",
    "EnvConfig",
    "TrainingConfig",
    "build_env",
    "DoubleDQNNetwork",
    "Trainer",
]
