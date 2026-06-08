from collections import deque

import numpy as np

from mario_rl.agent import MarioAgent
from mario_rl.config import TrainingConfig
from mario_rl.metrics import MetricsLogger
from mario_rl.utils import reset_env, step_env


class Trainer:
    def __init__(self, env, agent: MarioAgent, config: TrainingConfig) -> None:
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = MetricsLogger(save_dir=config.save_dir, log_window=config.log_window)
        self._best_eval_cr: float = -1.0
        self._best_eval_x: float = 0.0

    # ── Main training loop ────────────────────────────────────────────

    def train(self) -> None:
        print(f"Training on device: {self.config.device}")
        try:
            for episode in range(1, self.config.episodes + 1):
                reward, loss_list, steps, info = self.run_episode()

                self.logger.log_episode(
                    episode=episode,
                    total_reward=reward,
                    loss_list=loss_list,
                    epsilon=self.agent.exploration_rate,
                    steps=steps,
                    info=info,
                )

                moving_avg = self.logger.moving_average

                print(
                    "Episode={episode} Step={step} Epsilon={epsilon:.4f} Reward={reward:.2f} "
                    "MovingAvg({window})={moving_avg:.2f} XPos={x_pos}".format(
                        episode=episode,
                        step=self.agent.curr_step,
                        epsilon=self.agent.exploration_rate,
                        reward=reward,
                        window=len(self.logger._reward_window),
                        moving_avg=moving_avg,
                        x_pos=info.get("x_pos", "?"),
                    )
                )

                # Best checkpoint: prefer eval_completion_rate over moving_avg_reward
                if self._best_eval_cr >= 0:
                    # Already have eval data — eval-based best already handled in eval block
                    pass
                else:
                    # Fallback: use moving average until eval data is available
                    moving_avg = self.logger.moving_average
                    if moving_avg > self.logger.best_moving_average and episode > self.config.log_window:
                        path = self.agent.save_checkpoint(episode, tag="best")
                        print(f"Saved new best checkpoint (reward): {path}")

                # Periodic checkpoint
                if episode % self.config.checkpoint_period == 0:
                    path = self.agent.save_checkpoint(episode)
                    print(f"Saved periodic checkpoint: {path}")

                # Periodic evaluation
                if self.config.eval_every and episode % self.config.eval_every == 0:
                    eval_metrics = self.evaluate(self.config.eval_episodes)
                    self.logger.log_eval(episode, eval_metrics)
                    print(
                        "Eval@{}: avg_reward={:.2f} completion_rate={:.2%} avg_x_pos={:.0f}".format(
                            episode,
                            eval_metrics["avg_reward"],
                            eval_metrics["completion_rate"],
                            eval_metrics.get("avg_x_pos", 0),
                        )
                    )
                    # Checkpoint selection: completion_rate primary, avg_x_pos tiebreaker
                    cr = eval_metrics["completion_rate"]
                    ax = eval_metrics.get("avg_x_pos", 0)
                    is_better = (cr > self._best_eval_cr) or \
                                (cr == self._best_eval_cr and ax > self._best_eval_x)
                    if is_better:
                        self._best_eval_cr = cr
                        self._best_eval_x = ax
                        path = self.agent.save_checkpoint(episode, tag="best_eval")
                        print(f"Saved new best checkpoint (eval CR={cr:.2%}): {path}")
                    self.logger.save()  # persist after each eval

                # Early stopping
                if self.config.early_stop_patience > 0:
                    if self._should_early_stop():
                        print(f"Early stopping triggered at episode {episode}")
                        break
        finally:
            self.env.close()
            self.logger.save()
            summary = self.logger.get_summary()
            print(f"Training complete. Summary: {summary}")

    # ── Single episode ────────────────────────────────────────────────

    def run_episode(self) -> tuple[float, list[float], int, dict]:
        """Run one training episode. Returns (total_reward, loss_list, steps, final_info)."""
        state = reset_env(self.env)
        done = False
        episode_reward = 0.0
        loss_list: list[float] = []
        final_info: dict = {}
        step = 0

        while not done:
            if self.config.render:
                self.env.render()

            action = self.agent.act(state)
            next_state, reward, done, info = step_env(self.env, action)
            self.agent.remember(state, next_state, action, reward, done)
            loss = self.agent.learn()

            if loss is not None:
                loss_list.append(loss)

            state = next_state
            episode_reward += reward
            step += 1
            final_info = info or {}

            if self.config.max_steps_per_episode and step >= self.config.max_steps_per_episode:
                break

        return episode_reward, loss_list, step, final_info

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, num_episodes: int = 10) -> dict:
        """Run N evaluation episodes with epsilon=0 (greedy policy)."""
        rewards: list[float] = []
        steps_list: list[int] = []
        x_positions: list[int] = []
        completions = 0
        time_to_flag_list: list[int] = []

        for _ in range(num_episodes):
            state = reset_env(self.env)
            done = False
            episode_reward = 0.0
            step = 0
            final_info: dict = {}

            while not done:
                action = self.agent.eval_act(state)
                next_state, reward, done, info = step_env(self.env, action)
                state = next_state
                episode_reward += reward
                step += 1
                final_info = info or {}

            rewards.append(episode_reward)
            steps_list.append(step)
            x_positions.append(final_info.get("x_pos", 0))
            if final_info.get("flag_get", False):
                completions += 1
                time_to_flag_list.append(step)

        result = {
            "avg_reward": round(float(np.mean(rewards)), 2),
            "max_reward": int(np.max(rewards)),
            "min_reward": int(np.min(rewards)),
            "std_reward": round(float(np.std(rewards)), 2),
            "avg_steps": round(float(np.mean(steps_list)), 1),
            "avg_x_pos": round(float(np.mean(x_positions)), 1),
            "completion_rate": round(completions / num_episodes, 4),
            "completions": completions,
            "num_episodes": num_episodes,
        }
        if time_to_flag_list:
            result["avg_time_to_flag"] = round(float(np.mean(time_to_flag_list)), 1)
        return result

    def eval_checkpoint(self, checkpoint_path: str, num_episodes: int = 20) -> dict:
        """Load a checkpoint and evaluate it."""
        self.agent.load(checkpoint_path)
        return self.evaluate(num_episodes)

    # ── Early stopping ────────────────────────────────────────────────

    def _should_early_stop(self) -> bool:
        """Check if training should stop early based on moving average plateau."""
        patience = self.config.early_stop_patience
        records = self.logger.records
        if len(records) < patience:
            return False
        recent = [r["moving_avg_reward"] for r in records[-patience:]]
        best = max(recent)
        # Stop if no improvement in the last `patience` episodes
        return best <= records[-patience - 1]["moving_avg_reward"]
