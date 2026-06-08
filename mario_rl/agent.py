import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mario_rl.config import AgentConfig
from mario_rl.model import DoubleDQNNetwork
from mario_rl.utils import ensure_dir


class MarioAgent:
    def __init__(
        self,
        action_dim: int,
        config: AgentConfig,
        device: str,
        save_dir: str | Path,
        input_channels: int = 4,
    ) -> None:
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.save_dir = ensure_dir(save_dir)

        self.net = DoubleDQNNetwork(
            input_channels=input_channels,
            output_dim=action_dim,
            dueling=config.dueling,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=self.config.replay_buffer_size)
        self.exploration_rate = self.config.exploration_rate
        self.curr_step = 0

    def act(self, state) -> int:
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = self._state_tensor(state).unsqueeze(0).to(self.device)
                action_values = self.net(state_tensor, branch="online")
                action = torch.argmax(action_values, dim=1).item()

        self.exploration_rate *= self.config.exploration_rate_decay
        self.exploration_rate = max(self.config.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action

    def eval_act(self, state) -> int:
        """Select action greedily (no exploration, no decay, no learning)."""
        with torch.no_grad():
            state_tensor = self._state_tensor(state).unsqueeze(0).to(self.device)
            action_values = self.net(state_tensor, branch="online")
            return torch.argmax(action_values, dim=1).item()

    def remember(self, state, next_state, action: int, reward: float, done: bool) -> None:
        self.memory.append(
            (
                self._state_tensor(state),
                self._state_tensor(next_state),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(done, dtype=torch.bool),
            )
        )

    def learn(self) -> float | None:
        # Target network update
        if self.config.tau > 0:
            self._soft_update()
        elif self.curr_step % self.config.sync_steps == 0:
            self.sync_target()

        if len(self.memory) < self.config.batch_size:
            return None

        state, next_state, action, reward, done = self.recall()
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        q_values = self.net(state, branch="online").gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_action = self.net(next_state, branch="online").argmax(dim=1, keepdim=True)
            next_q_values = self.net(next_state, branch="target").gather(1, best_action).squeeze(1)
            q_target = reward + (1 - done.float()) * self.config.gamma * next_q_values

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), max_norm=self.config.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def recall(self):
        batch = random.sample(self.memory, self.config.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done

    def sync_target(self) -> None:
        self.net.target.load_state_dict(self.net.online.state_dict())
        if self.net.dueling_head is not None:
            self.net.sync_target_head()

    def _soft_update(self) -> None:
        """Polyak averaging: target = tau * online + (1 - tau) * target."""
        for target_param, online_param in zip(self.net.target.parameters(), self.net.online.parameters()):
            target_param.data.copy_(self.config.tau * online_param.data + (1 - self.config.tau) * target_param.data)
        if self.net.dueling_head is not None:
            for tp, op in zip(self.net.target_dueling_head.parameters(), self.net.dueling_head.parameters()):
                tp.data.copy_(self.config.tau * op.data + (1 - self.config.tau) * tp.data)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint.get("exploration_rate", self.exploration_rate)

    def save_checkpoint(self, episode: int, tag: str = "") -> Path:
        stem = f"checkpoint_{episode}" if not tag else f"checkpoint_{episode}_{tag}"
        path = self.save_dir / f"{stem}.pth"
        torch.save(
            {
                "model": self.net.state_dict(),
                "exploration_rate": self.exploration_rate,
            },
            path,
        )
        return path

    @staticmethod
    def _state_tensor(state) -> torch.Tensor:
        return torch.as_tensor(np.asarray(state), dtype=torch.uint8)
