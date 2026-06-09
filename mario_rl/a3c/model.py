"""A3C+LSTM 网络（MarioNET）

架构：3 个 ResBlock → FC(3200→512) → LSTMCell(512→512) → Actor/Critic 双头
输入：(B, 4, 80, 80) float32  输出：(critic, logits, hx, cx)

关键：LSTM 记忆单元是模型能跨关卡泛化的核心 —— 它学到了关卡无关的导航策略。
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 权重初始化辅助函数 ──────────────────────────────────────────────

def _weights_init(m: nn.Module) -> None:
    """Xavier-uniform 风格的 Conv / Linear 初始化。"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


def _norm_col_init(weights: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """按列归一化的权重初始化。"""
    x = torch.randn(weights.shape)
    x *= std / (x**2).sum(dim=0, keepdim=True).sqrt()
    return x


# ── ResBlock ────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """残差卷积块：5 层 Conv2d + 2 个残差连接 + MaxPool 下采样。

    前向流程：
      conv1 → MaxPool2d(3,2,1) → [conv2→conv3]+残差 → [conv4→conv5]+残差
    所有卷积层 kernel_size=3, stride=1, padding=1（保持分辨率）。
    MaxPool stride=2 将空间尺寸减半。
    ReLU 在 conv2-conv5 之前应用（预激活风格）。
    """

    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1)
        res_input = x
        # 原始代码: conv2(F.relu(x)) → conv3(F.relu(conv2_output))
        # 即 conv2 和 conv3 之间有 ReLU
        x = self.conv3(F.relu(self.conv2(F.relu(x))))
        x = x + res_input  # 第一残差连接

        res_input = x
        # 同理: conv4 和 conv5 之间也有 ReLU
        x = self.conv5(F.relu(self.conv4(F.relu(x))))
        x = x + res_input  # 第二残差连接
        return x


# ── MarioNET ────────────────────────────────────────────────────────

class MarioNET(nn.Module):
    """A3C + LSTM 网络，用于 Super Mario Bros 全关卡通关。

    输入：  (B, 4, 80, 80) float32
    输出：  (critic, actor_logits, hx, cx)

    架构：
      3× ResBlock (通道 16→32→32，空间 80→40→20→10)
      → Flatten (3200) → FC(3200→512) → ReLU
      → LSTMCell(512→512) → Actor(512→10) + Critic(512→1)
    """

    def __init__(
        self,
        num_inputs: int,
        action_space,           # gym.spaces.Discrete 或 int
        hidden_size: int = 512,
    ) -> None:
        super().__init__()

        # 兼容 argparse.Namespace（有 .hidden_size 属性）和直接传 int
        if hasattr(hidden_size, "hidden_size"):
            self.hidden_size: int = hidden_size.hidden_size
        else:
            self.hidden_size: int = int(hidden_size)

        # 兼容 gym.spaces.Discrete（有 .n 属性）和直接传 int
        if hasattr(action_space, "n"):
            self.num_actions: int = action_space.n
        else:
            self.num_actions: int = int(action_space)

        # ── 残差块：通道递进 [16, 32, 32] ──
        channel_progression = [16, 32, 32]
        inplanes = num_inputs
        self.resnet_blocks = nn.ModuleList()
        for planes in channel_progression:
            self.resnet_blocks.append(ResBlock(inplanes, planes))
            inplanes = planes

        # ── 全连接 + LSTM ──
        # 3 次 stride-2 下采样: 80→40→20→10, 32 通道 → 3200 特征
        self.fc = nn.Linear(3200, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # ── Actor / Critic 双头 ──
        self.actor_linear = nn.Linear(self.hidden_size, self.num_actions)
        self.critic_linear = nn.Linear(self.hidden_size, 1)

        self._init_weights()
        self.train()

    def _init_weights(self) -> None:
        """初始化所有权重（load_state_dict 后会完全覆盖）。"""
        self.apply(_weights_init)

        # FC 层：正交初始化 × ReLU gain
        nn.init.orthogonal_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        self.fc.bias.data.fill_(0)

        # LSTM：均匀初始化，遗忘门偏置 = 1
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                param.data.uniform_(-stdv, stdv)
            elif "bias" in name:
                n = param.size(0)
                param.data.fill_(0)
                param.data[(n // 4) : (n // 2)].fill_(1)  # 遗忘门偏置

        # Actor 头：小标准差
        self.actor_linear.weight.data = _norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        # Critic 头：大标准差
        self.critic_linear.weight.data = _norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

    def forward(
        self,
        inputs: torch.Tensor,
        hx: torch.Tensor,
        cx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步前向传播。

        Args:
            inputs: (B, 4, 80, 80) 观测
            hx:     (B, hidden_size) LSTM 隐藏状态
            cx:     (B, hidden_size) LSTM 细胞状态

        Returns:
            (critic, actor_logits, hn, cn)
        """
        x: torch.Tensor = inputs
        for block in self.resnet_blocks:
            x = block(x)
        x = F.relu(x)
        x = x.view(x.size(0), 3200)  # 动态 batch size（兼容 ONNX）
        x = F.relu(self.fc(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), hx, cx
