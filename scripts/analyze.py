"""分析单个实验的训练结果：生成训练曲线图和快速摘要。

用法：
    python scripts/analyze.py weights/Round01_Baseline          # 生成 analysis.png
    python scripts/analyze.py weights/Round01_Baseline --summary  # 打印一行摘要
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(exp_dir: Path) -> dict:
    """加载实验目录中的 metrics.json。"""
    json_path = exp_dir / "metrics.json"
    if not json_path.exists():
        print(f"错误：在 {exp_dir} 中未找到 metrics.json。训练是否已完成？")
        sys.exit(1)
    with open(json_path) as f:
        return json.load(f)


def plot_curves(data: dict, output_path: Path) -> None:
    """生成四面板训练曲线图。"""
    training = data.get("training", [])
    if not training:
        print("训练数据为空，跳过绘图。")
        return

    episodes = [r["episode"] for r in training]
    rewards = [r["total_reward"] for r in training]
    moving_avg = [r.get("moving_avg_reward", 0) for r in training]
    losses = [r.get("avg_loss", 0) for r in training]
    x_positions = [r.get("x_pos", 0) for r in training]
    epsilons = [r.get("epsilon", 0) for r in training]
    flags = [r.get("flag_get", 0) for r in training]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(output_path.parent.name, fontsize=14, fontweight="bold")

    # (a) 奖励曲线
    ax = axes[0][0]
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    ax.plot(episodes, moving_avg, color="darkorange", linewidth=2, label=f"Moving Avg (window)")
    flag_eps = [e for e, f in zip(episodes, flags) if f]
    flag_vals = [r for e, r, f in zip(episodes, rewards, flags) if f]
    if flag_eps:
        ax.scatter(flag_eps, flag_vals, marker="*", color="gold", s=80, zorder=5, label="Flag Get")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) 损失曲线
    ax = axes[0][1]
    ax.plot(episodes, losses, color="crimson", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Loss")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)

    # (c) X 位置
    ax = axes[1][0]
    ax.plot(episodes, x_positions, color="forestgreen", alpha=0.7, linewidth=0.8)
    ax.axhline(y=3168, color="gray", linestyle="--", alpha=0.5, label="Flag (~3168)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Max X Position")
    ax.set_title("X Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Epsilon 衰减
    ax = axes[1][1]
    ax.plot(episodes, epsilons, color="mediumpurple", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (ε)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已保存到 {output_path}")


def find_best_checkpoint(exp_dir: Path, data: dict) -> str:
    """根据评估通关率找到最佳检查点。"""
    evaluation = data.get("evaluation", [])
    if evaluation:
        best = max(evaluation, key=lambda e: (e.get("completion_rate", 0), e.get("avg_x_pos", 0)))
        best_ep = best["episode"]
        ckpt = exp_dir / f"checkpoint_{best_ep}.pth"
        if ckpt.exists():
            return str(ckpt)

    # 回退：找最新的检查点
    checkpoints = sorted(exp_dir.glob("checkpoint_*.pth"))
    if checkpoints:
        return str(checkpoints[-1])
    return "无"


def print_summary(exp_dir: Path, data: dict) -> None:
    """打印一行摘要。"""
    evaluation = data.get("evaluation", [])
    training = data.get("training", [])
    summary = data.get("summary", {})

    if evaluation:
        best_eval = max(evaluation, key=lambda e: (e.get("completion_rate", 0), e.get("avg_x_pos", 0)))
        cr = best_eval.get("completion_rate", 0)
        completions = best_eval.get("completions", 0)
        num_ep = best_eval.get("num_episodes", 0)
        avg_x = best_eval.get("avg_x_pos", 0)
        max_x_eval = max((e.get("avg_x_pos", 0) for e in evaluation), default=0)
        best_ep = best_eval["episode"]
    else:
        cr = summary.get("completion_rate", 0)
        completions = summary.get("completions", 0)
        num_ep = summary.get("episodes", 0)
        avg_x = max((r.get("x_pos", 0) for r in training[-10:]), default=0) if training else 0
        max_x_eval = max((r.get("x_pos", 0) for r in training), default=0) if training else 0
        best_ep = "?"

    best_ckpt = find_best_checkpoint(exp_dir, data)
    avg_time = best_eval.get("avg_time_to_flag", "N/A") if evaluation else "N/A"

    print(
        f"通关率: {cr:.1%} ({completions}/{num_ep}) | "
        f"平均 X: {avg_x:.0f} | "
        f"最大 X: {max_x_eval:.0f} | "
        f"平均通关步数: {avg_time} | "
        f"最佳检查点: {best_ckpt}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="分析单次实验的训练结果")
    parser.add_argument("exp_dir", type=Path, help="实验目录路径")
    parser.add_argument("--summary", action="store_true", help="仅打印一行摘要")
    parser.add_argument("--output", "-o", type=Path, default=None, help="图表输出路径（默认: <exp_dir>/analysis.png）")
    args = parser.parse_args()

    data = load_metrics(args.exp_dir)

    if args.summary:
        print_summary(args.exp_dir, data)
    else:
        output_path = args.output or (args.exp_dir / "analysis.png")
        plot_curves(data, output_path)
        print_summary(args.exp_dir, data)


if __name__ == "__main__":
    main()
