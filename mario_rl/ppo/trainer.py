"""PPO Agent 和 Trainer

单进程实现（非多进程并行），适合单 GPU 环境。
使用 GAE 优势估计 + Clipped Surrogate Objective。
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from mario_rl.ppo.env import build_ppo_env
from mario_rl.ppo.model import PPONetwork


class PPOAgent:
    """PPO 智能体：Actor-Critic + 经验收集。"""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        device: str = "cuda",
        lr: float = 1e-4,
        gamma: float = 0.9,
        tau: float = 1.0,
        epsilon: float = 0.2,
        beta: float = 0.01,
        num_epochs: int = 10,
        batch_size: int = 16,
        grad_clip: float = 0.5,
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.beta = beta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.model = PPONetwork(state_dim, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 经验缓冲
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        """采样一个动作。返回 (action, log_prob, value)。"""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits, value = self.model(state_t)
        policy = F.softmax(logits, dim=1)
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach(), value.detach()

    def eval_act(self, state: np.ndarray) -> int:
        """贪婪动作（评估用）。"""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(state_t)
        return torch.argmax(logits, dim=1).item()

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state.clone())
        self.actions.append(action.clone())
        self.log_probs.append(log_prob.clone())
        self.values.append(value.clone())
        self.rewards.append(reward.clone())
        self.dones.append(done.clone())

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def learn(self):
        """PPO 更新：GAE + Clipped Surrogate + Value Loss + Entropy Bonus。"""
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze(-1)

        # GAE 计算
        _, next_value = self.model(states[-1:])
        next_value = next_value.squeeze().detach()
        gae = torch.tensor(0.0, device=self.device)
        returns = []
        for i in reversed(range(len(self.rewards))):
            gae = gae * self.gamma * self.tau
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - values[i]
            gae = gae + delta
            next_value = values[i]
            returns.insert(0, gae + values[i])
        returns = torch.cat(returns).detach()
        advantages = returns - values

        # 多 epoch 更新
        total_losses = []
        data_size = len(states)
        for _ in range(self.num_epochs):
            indices = torch.randperm(data_size, device=self.device)
            batch_size = max(1, data_size // self.batch_size)
            for j in range(0, data_size, batch_size):
                batch_idx = indices[j: j + batch_size]
                s = states[batch_idx]
                a = actions[batch_idx]
                old_lp = old_log_probs[batch_idx]
                adv = advantages[batch_idx]
                ret = returns[batch_idx]

                logits, value_est = self.model(s)
                policy = F.softmax(logits, dim=1)
                dist = Categorical(policy)
                new_log_prob = dist.log_prob(a)

                ratio = torch.exp(new_log_prob - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
                actor_loss = -torch.mean(torch.min(surr1, surr2))

                critic_loss = F.smooth_l1_loss(ret, value_est.squeeze())
                entropy_loss = torch.mean(dist.entropy())

                total_loss = actor_loss + 0.5 * critic_loss - self.beta * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_losses.append(total_loss.item())

        self.clear_memory()
        return float(np.mean(total_losses)) if total_losses else 0.0

    def save(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])


def train_ppo(
    world: int = 1,
    stage: int = 1,
    action_type: str = "simple",
    lr: float = 1e-4,
    gamma: float = 0.9,
    tau: float = 1.0,
    beta: float = 0.01,
    epsilon: float = 0.2,
    num_epochs: int = 10,
    batch_size: int = 16,
    num_steps: int = 512,
    total_steps: int = 5_000_000,
    max_actions: int = 200,
    eval_every: int = 10000,
    eval_episodes: int = 5,
    save_dir: str | Path = "weights/ppo",
    device: str = "cuda",
) -> None:
    """PPO 训练主函数。单环境采样 + 批量更新。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = build_ppo_env(world, stage, action_type)
    state_dim = env.observation_space.shape[0]  # 4
    num_actions = len(env.unwrapped._actions) if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_actions') else env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
        lr=lr,
        gamma=gamma,
        tau=tau,
        epsilon=epsilon,
        beta=beta,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # 指标记录
    training_records: list[dict] = []
    eval_records: list[dict] = []
    reward_window = deque(maxlen=20)
    start_time = time.time()

    state = env.reset()
    state = torch.as_tensor(state, dtype=torch.float32, device=device)
    total_reward = 0.0
    episode = 0
    episode_steps = 0
    best_eval_cr = -1.0
    global_step = 0

    print(f"PPO Training | World {world}-{stage} | {action_type} actions | device={device}")
    print(f"Target steps: {total_steps} | LR: {lr} | Gamma: {gamma}")

    try:
        while global_step < total_steps:
            # 采样 num_steps 步
            for _ in range(num_steps):
                action, log_prob, value = agent.act(state.cpu().numpy())
                next_state, reward, done, info = env.step(action)

                agent.store(
                    state,
                    torch.tensor([action], device=device),
                    log_prob,
                    value,
                    torch.tensor([reward], device=device, dtype=torch.float32),
                    torch.tensor([float(done)], device=device, dtype=torch.float32),
                )

                state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
                total_reward += reward
                episode_steps += 1
                global_step += 1

                if done:
                    episode += 1
                    reward_window.append(total_reward)
                    moving_avg = float(np.mean(reward_window)) if reward_window else total_reward

                    record = {
                        "episode": episode,
                        "steps": global_step,
                        "episode_steps": episode_steps,
                        "total_reward": round(total_reward, 1),
                        "x_pos": info.get("x_pos", 0),
                        "flag_get": int(info.get("flag_get", False)),
                        "mva": round(moving_avg, 1),
                    }
                    training_records.append(record)

                    if episode % 10 == 0 or done:
                        print(
                            f"Ep {episode:4d} | Step {global_step:7d} | "
                            f"Reward {total_reward:7.1f} | MVA {moving_avg:7.1f} | "
                            f"X={info.get('x_pos', '?')} | {'FLAG!' if info.get('flag_get') else ''}"
                        )

                    total_reward = 0.0
                    episode_steps = 0
                    state = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)

            # PPO 更新
            loss = agent.learn()

            # 评估
            if global_step >= eval_every and global_step % eval_every < num_steps:
                eval_metrics = evaluate_ppo(agent, world, stage, action_type, eval_episodes, device)
                eval_metrics["episode"] = episode
                eval_metrics["global_step"] = global_step
                eval_records.append(eval_metrics)

                cr = eval_metrics["completion_rate"]
                print(f"  Eval@{global_step}: CR={cr:.1%} avgX={eval_metrics['avg_x_pos']:.0f}")

                if cr >= best_eval_cr:
                    best_eval_cr = cr
                    agent.save(save_dir / "ppo_best.pth")
                    print(f"  -> Best checkpoint saved (CR={cr:.1%})")

                agent.save(save_dir / f"ppo_step_{global_step}.pth")

                # Save metrics
                metrics = {
                    "training": training_records,
                    "evaluation": eval_records,
                    "summary": {
                        "total_steps": global_step,
                        "total_episodes": episode,
                        "best_eval_cr": best_eval_cr,
                        "time_seconds": round(time.time() - start_time, 1),
                    },
                }
                with open(save_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()
        agent.save(save_dir / "ppo_final.pth")
        metrics = {
            "training": training_records,
            "evaluation": eval_records,
            "summary": {
                "total_steps": global_step,
                "total_episodes": episode,
                "best_eval_cr": best_eval_cr,
                "time_seconds": round(time.time() - start_time, 1),
            },
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nTraining complete. {episode} episodes, {global_step} steps.")
        if eval_records:
            print(f"Best eval CR: {best_eval_cr:.1%}")


def evaluate_ppo(
    agent: PPOAgent,
    world: int,
    stage: int,
    action_type: str,
    num_episodes: int,
    device: str,
) -> dict:
    """评估 PPO 智能体。"""
    env = build_ppo_env(world, stage, action_type)
    rewards = []
    x_positions = []
    completions = 0
    time_to_flag = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            action = agent.eval_act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward

        rewards.append(ep_reward)
        x_positions.append(info.get("x_pos", 0))
        if info.get("flag_get", False):
            completions += 1

    env.close()

    return {
        "avg_reward": round(float(np.mean(rewards)), 2),
        "avg_x_pos": round(float(np.mean(x_positions)), 1),
        "max_x_pos": int(np.max(x_positions)),
        "completion_rate": round(completions / num_episodes, 4),
        "completions": completions,
        "num_episodes": num_episodes,
        "avg_time_to_flag": round(float(np.mean(time_to_flag)), 1) if time_to_flag else None,
    }
