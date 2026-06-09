"""A3C 评估环境包装器 —— 预处理流程。

预处理管线：
  NES 原始帧 → 裁剪 [15:-25, 43:-13]
  → 双阶段缩放 (100×100 → 80×80)
  → BT.601 灰度化 → 跳帧最大池化 (skip=4)
  → 4 帧堆叠 → RunningMeanStd 归一化（[-10,10]）

输出观测: (4, 80, 80) float32, 值域 [-10, 10]
动作空间: CUSTOM_MOVEMENT (10 个离散动作)
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace


# ── 自定义动作空间（10 个动作） ─────────────────────────────────────

CUSTOM_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left", "A"],
    ["left", "B"],
    ["left", "A", "B"],
    ["down"],
]
NUM_ACTIONS: int = len(CUSTOM_MOVEMENT)


# ── 帧预处理 ────────────────────────────────────────────────────────

def process_frame_a3c(frame: np.ndarray | None) -> np.ndarray:
    """原始 NES 帧 (240, 256, 3) → (80, 80) float32 灰度图。

    1. 裁剪 [15:-25, 43:-13] 去 UI 和天空/地面边框
    2. INTER_AREA 缩放至 100×100
    3. INTER_AREA 缩放至 80×80
    4. BT.601 亮度权重灰度化
    """
    if frame is None:
        return np.zeros((80, 80), dtype=np.float32)

    # 裁剪：去天空(上15px)、地面(下25px)、左右 UI 边框
    cropped = frame[15:-25, 43:-13]
    # 双阶段缩放（INTER_AREA 适合下采样）
    resized = cv2.resize(cropped, (100, 100), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(resized, (80, 80), interpolation=cv2.INTER_AREA)
    # BT.601 亮度灰度化
    gray = (
        0.2989 * resized[:, :, 0]
        + 0.587 * resized[:, :, 1]
        + 0.114 * resized[:, :, 2]
    )
    return gray.astype(np.float32)


# ── 跳帧 + 最大池化 ──────────────────────────────────────────────────

class A3CSkipFrame(Wrapper):
    """跳帧包装器：每 skip 帧采样一次，只保留最后一帧。

    - deque(maxlen=1)：只保留最后一帧（不做最大池化）
    - 帧预处理（process_frame_a3c）在此阶段完成
    - 与原始代码一致：skip 帧 → 取最后一帧 → 预处理
    """

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self._obs_buffer: deque = deque(maxlen=1)
        self.observation_space = Box(
            low=0.0, high=255.0, shape=(80, 80), dtype=np.float32
        )

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            total_reward += reward
            self._obs_buffer.append(process_frame_a3c(obs))
            if done:
                break
        # 只取最后一帧（maxlen=1 保证 buffer 只有一帧）
        max_frame = np.max(np.stack(list(self._obs_buffer)), axis=0)
        return max_frame, total_reward, done, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, _ = result
        else:
            obs = result
        processed = process_frame_a3c(obs)
        self._obs_buffer.clear()
        self._obs_buffer.append(processed)
        return processed


# ── 帧堆叠 ──────────────────────────────────────────────────────────

class A3CFrameStack(Wrapper):
    """帧堆叠包装器：堆叠最近 4 帧。输出 shape: (4, 80, 80)。"""

    def __init__(self, env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: deque = deque(maxlen=num_stack)
        self.observation_space = Box(
            low=0.0, high=255.0, shape=(num_stack, 80, 80), dtype=np.float32
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        stacked = np.stack(list(self.frames), axis=0)
        return stacked, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple) and len(obs) == 2:
            obs = obs[0]
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        stacked = np.stack(list(self.frames), axis=0)
        return stacked


# ── RunningMeanStd 归一化 ────────────────────────────────────────────

class RunningMeanStd:
    """运行均值/方差跟踪器

    训练时更新统计量；评估时使用固定的 mean/var 进行归一化。
    输出被裁剪到 [-10, 10]。
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean = np.float32(0.0)
        self.var = np.float32(1.0)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, dtype=np.float32)
        batch_var = np.var(arr, dtype=np.float32)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: float, batch_var: float, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / tot_count
        )
        new_var = m_2 / tot_count
        self.count = tot_count
        self.mean = np.float32(new_mean)
        self.var = np.float32(new_var)


class A3CNormalize(Wrapper):
    """RunningMeanStd 归一化包装器。

    从 obs_rms.pkl 加载统计量（如果可用），否则回退到 [0,1] 简易归一化。
    评估时不更新统计量（training_off 模式）。
    输出范围：[-10, 10]。
    """

    def __init__(self, env, rms_path: str | None = None):
        super().__init__(env)
        self.obs_rms = RunningMeanStd()
        self.is_training = False

        # 尝试加载预训练的 RunningMeanStd 统计量
        if rms_path is None:
            rms_path = "weights/a3c/obs_rms.pkl"
        self._rms_loaded = False
        if Path(rms_path).exists():
            try:
                self.load(rms_path)
                self._rms_loaded = True
                print(f"[A3CNormalize] 已加载归一化统计量: {rms_path}")
            except Exception as e:
                print(f"[A3CNormalize] 加载统计量失败 ({e})，回退到 [0,1] 归一化")

        self.observation_space = Box(
            low=-10.0, high=10.0, shape=(4, 80, 80), dtype=np.float32
        )

    def set_training_on(self) -> None:
        self.is_training = True
        self.observation_space = Box(
            low=0.0, high=255.0, shape=(4, 80, 80), dtype=np.float32
        )

    def set_training_off(self) -> None:
        self.is_training = False
        self.observation_space = Box(
            low=-10.0, high=10.0, shape=(4, 80, 80), dtype=np.float32
        )

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._rms_loaded:
            # 使用 RunningMeanStd 归一化（与训练时一致）
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
                -10.0,
                10.0,
            )
        else:
            # 回退：简单 [0,1] 归一化
            obs = obs / 255.0
        return obs.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_training:
            self.obs_rms.update(obs)
        return self._normalize(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple) and len(obs) == 2:
            obs = obs[0]
        if self.is_training:
            self.obs_rms.update(obs)
        return self._normalize(obs)

    def load(self, path: str) -> None:
        """加载预训练的 RunningMeanStd 统计量 (pickle)。
        """
        import pickle
        import sys as _sys

        # 临时注册模块别名，使 pickle 能正确反序列化
        _saved = _sys.modules.get("environment")
        import mario_rl.a3c.env as _self_module
        _sys.modules["environment"] = _self_module
        try:
            with open(path, "rb") as fh:
                loaded = pickle.load(fh)
            # 如果 pickle 返回的是 dict（stateless）则手动赋值
            if isinstance(loaded, dict):
                self.obs_rms.mean = loaded.get("mean", self.obs_rms.mean)
                self.obs_rms.var = loaded.get("var", self.obs_rms.var)
                self.obs_rms.count = loaded.get("count", self.obs_rms.count)
            else:
                self.obs_rms = loaded
        finally:
            if _saved is not None:
                _sys.modules["environment"] = _saved
            elif "environment" in _sys.modules:
                del _sys.modules["environment"]

    def save(self, path: str) -> None:
        """保存 RunningMeanStd 统计量。"""
        with open(path, "wb") as fh:
            import pickle
            pickle.dump(self.obs_rms, fh)


# ── 环境构建入口 ────────────────────────────────────────────────────

def build_a3c_env(world: int = 1, stage: int = 1, render: bool = False):
    """构建 A3C 评估环境。

    管线: gym_super_mario_bros.make → JoypadSpace(CUSTOM_MOVEMENT)
          → A3CSkipFrame(4) → A3CFrameStack(4) → A3CNormalize

    Args:
        world: 世界编号 (1-8)
        stage: 关卡编号 (1-4)
        render: 是否启用 rgb_array 渲染

    Returns:
        gym.Env，观测 shape (4, 80, 80)，float32
    """
    import logging
    logging.getLogger("gymnasium").setLevel(logging.ERROR)
    import gym_super_mario_bros

    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v0",
        render_mode="rgb_array" if render else None,
    )
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    env = A3CSkipFrame(env, skip=4)
    env = A3CFrameStack(env, num_stack=4)
    env = A3CNormalize(env)
    return env
