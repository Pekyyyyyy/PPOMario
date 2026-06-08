"""Training metrics collection and persistence.

Provides MetricsLogger: collects per-episode statistics during training,
saves to CSV and JSON, and computes summary statistics.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Collect and persist per-episode training metrics.

    Usage:
        logger = MetricsLogger(save_dir="weights/experiment1")
        for episode in range(episodes):
            reward, loss_list, steps, info = run_episode()
            logger.log_episode(episode, reward, loss_list, epsilon, steps, info)
        logger.save()  # writes metrics.csv and metrics.json
    """

    def __init__(self, save_dir: str | Path, log_window: int = 10) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_window = log_window
        self.start_time = time.time()

        self._records: list[dict[str, Any]] = []
        self._reward_window: deque[float] = deque(maxlen=log_window)
        self._eval_records: list[dict[str, Any]] = []

    # ── Core logging ──────────────────────────────────────────────────

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        loss_list: list[float] | None,
        epsilon: float,
        steps: int,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Record one training episode."""
        info = info or {}
        avg_loss = float(sum(loss_list) / len(loss_list)) if loss_list else 0.0

        self._reward_window.append(total_reward)
        moving_avg = float(sum(self._reward_window) / len(self._reward_window)) if self._reward_window else total_reward

        record: dict[str, Any] = {
            "episode": episode,
            "total_reward": total_reward,
            "avg_loss": round(avg_loss, 6),
            "steps": steps,
            "epsilon": round(epsilon, 6),
            "moving_avg_reward": round(moving_avg, 2),
            "x_pos": info.get("x_pos", 0),
            "flag_get": int(info.get("flag_get", False)),
            "coins": info.get("coins", 0),
            "status": info.get("status", ""),
            "time_elapsed": round(time.time() - self.start_time, 1),
        }
        self._records.append(record)

    def log_eval(self, episode: int, metrics: dict[str, Any]) -> None:
        """Record a periodic evaluation run."""
        metrics["episode"] = episode
        self._eval_records.append(metrics)

    # ── Persistence ───────────────────────────────────────────────────

    def to_csv(self, path: str | Path | None = None) -> Path:
        """Write training records to CSV. Default: save_dir/metrics.csv"""
        path = Path(path) if path else self.save_dir / "metrics.csv"
        if not self._records:
            return path
        fieldnames = list(self._records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        return path

    def to_json(self, path: str | Path | None = None) -> Path:
        """Write all records (train + eval) to JSON. Default: save_dir/metrics.json"""
        path = Path(path) if path else self.save_dir / "metrics.json"
        data = {
            "training": self._records,
            "evaluation": self._eval_records,
            "summary": self.get_summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save(self) -> tuple[Path, Path]:
        """Write both CSV and JSON. Returns (csv_path, json_path)."""
        csv_path = self.to_csv()
        json_path = self.to_json()
        return csv_path, json_path

    # ── Summary ───────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Return summary statistics over all training episodes."""
        if not self._records:
            return {}
        rewards = [r["total_reward"] for r in self._records]
        steps = [r["steps"] for r in self._records]
        completions = sum(1 for r in self._records if r["flag_get"])
        return {
            "episodes": len(self._records),
            "avg_reward": round(sum(rewards) / len(rewards), 2),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "avg_steps": round(sum(steps) / len(steps), 1),
            "completion_rate": round(completions / len(self._records), 4),
            "completions": completions,
            "final_moving_avg": self._records[-1]["moving_avg_reward"] if self._records else 0,
            "total_time_seconds": round(time.time() - self.start_time, 1),
        }

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    @property
    def moving_average(self) -> float:
        return float(sum(self._reward_window) / len(self._reward_window)) if self._reward_window else 0.0

    @property
    def best_moving_average(self) -> float:
        return max((r["moving_avg_reward"] for r in self._records), default=float("-inf"))

    @property
    def best_eval_completion_rate(self) -> float:
        """Best eval completion rate across all evaluations (-1 if no evals)."""
        if not self._eval_records:
            return -1.0
        return max((e.get("completion_rate", -1) for e in self._eval_records), default=-1.0)

    @property
    def has_eval_completion(self) -> bool:
        return self.best_eval_completion_rate >= 0
