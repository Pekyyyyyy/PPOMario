"""PPO 环境包装器

- CustomReward: 基于分数的奖励 + 通关/死亡终极奖励
- CustomSkipFrame: 跳帧 + 最大池化
- 参考 uvipen/Super-mario-bros-PPO-pytorch
"""

from __future__ import annotations

import cv2
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace


def process_frame(frame: np.ndarray) -> np.ndarray:
    """灰度化 + 缩放 84x84 + 归一化。"""
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.0
        return frame.astype(np.float32)
    else:
        return np.zeros((1, 84, 84), dtype=np.float32)


class CustomReward(Wrapper):
    """自定义奖励：
    - 每帧: score 增量
    - 通关: +50
    - 死亡: -50
    - 世界 4-4/7-4/8-4 的迷宫惩罚（可选）
    """

    def __init__(self, env, world: int = 1, stage: int = 1):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.float32)
        self.curr_score = 0
        self.world = world
        self.stage = stage

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            state, reward, done, info = result
        state = process_frame(state)
        # 分数增量奖励（除以 40 归一化）
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]
        if done:
            if info.get("flag_get", False):
                reward += 50
            else:
                reward -= 50
        return state, reward / 10.0, done, info

    def reset(self, **kwargs):
        self.curr_score = 0
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            result = result[0]  # gymnasium returns (obs, info)
        return process_frame(result)


class CustomSkipFrame(Wrapper):
    """跳帧包装器：每 4 帧采样一次，对后 2 帧做最大池化。"""

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84), dtype=np.float32)
        self.skip = skip
        self._frames = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_frames = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_frames.append(state)
            if done:
                self.reset()
                return self._frames[None, :, :, :].astype(np.float32), total_reward, done, info
        # 最大池化：取其最强特征
        max_state = np.max(np.concatenate(last_frames, 0), 0)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = max_state
        return self._frames[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            result = result[0]
        self._frames = np.concatenate([result for _ in range(self.skip)], 0)
        return self._frames[None, :, :, :].astype(np.float32)


def build_ppo_env(world: int = 1, stage: int = 1, action_type: str = "simple"):
    """构建 PPO 训练环境。"""
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

    action_map = {"right_only": RIGHT_ONLY, "simple": SIMPLE_MOVEMENT, "complex": COMPLEX_MOVEMENT}
    actions = action_map.get(action_type, SIMPLE_MOVEMENT)

    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v3")
    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage)
    env = CustomSkipFrame(env, skip=4)
    return env
