"""A3C+LSTM (MarioNET) 子包。

提供跨关卡泛化的 A3C 智能体模型和环境。
基于 dgriff777/SuperMarioRL 的预训练权重。
"""

from mario_rl.a3c.env import CUSTOM_MOVEMENT, NUM_ACTIONS, build_a3c_env
from mario_rl.a3c.model import MarioNET, ResBlock

__all__ = [
    "MarioNET",
    "ResBlock",
    "CUSTOM_MOVEMENT",
    "NUM_ACTIONS",
    "build_a3c_env",
]
