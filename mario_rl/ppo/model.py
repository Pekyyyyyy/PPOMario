"""PPO Actor-Critic 网络

基于 uvipen/Super-mario-bros-PPO-pytorch 架构。
4 层卷积 → 共享特征 → Actor 头 + Critic 头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPONetwork(nn.Module):
    """Actor-Critic 网络：共享 CNN 编码器 + 独立的 Actor/Critic 线性头。"""

    def __init__(self, input_channels: int, num_actions: int):
        super().__init__()
        # 共享卷积编码器
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # 计算 conv 输出维度 (84 → 6 after 4× stride-2)
        self._feature_size = 32 * 6 * 6  # 1152

        self.linear = nn.Linear(self._feature_size, 512)
        self.actor_linear = nn.Linear(512, num_actions)
        self.critic_linear = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x)
