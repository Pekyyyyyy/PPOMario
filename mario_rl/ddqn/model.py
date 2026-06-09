import copy

import torch
import torch.nn as nn


class DuelingHead(nn.Module):
    """Dueling DQN head: separates state-value V(s) and advantage A(s,a).

    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """

    def __init__(self, in_features: int, hidden: int, output_dim: int) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value(x)          # (batch, 1)
        a = self.advantage(x)      # (batch, n_actions)
        return v + a - a.mean(dim=1, keepdim=True)


class DoubleDQNNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        input_shape: tuple = (4, 84, 84),
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling

        conv_layers = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ]
        self.online = nn.Sequential(*conv_layers)
        conv_out_size = self._compute_conv_output(input_channels, *input_shape[1:])

        if dueling:
            # Shared feature layer → Dueling head
            self.online.append(nn.Linear(conv_out_size, 512))
            self.online.append(nn.ReLU())
            self.dueling_head = DuelingHead(512, 256, output_dim)
            self.target_dueling_head = copy.deepcopy(self.dueling_head)
            for p in self.target_dueling_head.parameters():
                p.requires_grad = False
        else:
            self.online.append(nn.Linear(conv_out_size, 512))
            self.online.append(nn.ReLU())
            self.online.append(nn.Linear(512, output_dim))
            self.dueling_head = None

        self.target = copy.deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _compute_conv_output(in_channels: int, h: int, w: int) -> int:
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        h = (h - 4) // 2 + 1
        w = (w - 4) // 2 + 1
        h = (h - 3) // 1 + 1
        w = (w - 3) // 1 + 1
        return h * w * 64

    def forward(self, inputs: torch.Tensor, branch: str = "online") -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.float() / 255.0

        if branch == "online":
            features = self.online(inputs)
            if self.dueling_head is not None:
                return self.dueling_head(features)
            return features
        else:
            features = self.target(inputs)
            if self.dueling_head is not None:
                target_head = self.target_dueling_head if hasattr(self, 'target_dueling_head') else self.dueling_head
                return target_head(features)
            return features

    def sync_target_head(self) -> None:
        """Sync dueling target head separately (called during target sync)."""
        if self.dueling_head is not None and hasattr(self, 'target_dueling_head'):
            self.target_dueling_head.load_state_dict(self.dueling_head.state_dict())
