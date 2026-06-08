"""评估单个检查点：加载权重，运行 N 轮贪婪评估，打印逐轮明细和汇总统计。

用法：
    python scripts/eval_checkpoint.py weights/checkpoint_100.pth           # 默认 20 轮
    python scripts/eval_checkpoint.py checkpoint.pth --episodes 50         # 自定义轮数
    python scripts/eval_checkpoint.py checkpoint.pth --phase 2             # 与阶段门槛对比
    python scripts/eval_checkpoint.py checkpoint.pth --phase 2 --render    # 渲染模式
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 本地路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario_rl.config import AgentConfig, EnvConfig
from mario_rl.env import build_env
from mario_rl.agent import MarioAgent
from mario_rl.utils import reset_env, step_env

# 阶段门槛定义
PHASE_GATES = {
    0: {"completion_rate": 0.0, "avg_x_pos": 300, "description": "avg_x_pos > 300"},
    1: {"completion_rate": 0.20, "avg_x_pos": 0, "description": "通关率 ≥ 20%"},
    2: {"completion_rate": 0.50, "avg_x_pos": 0, "description": "通关率 ≥ 50%"},
    3: {"completion_rate": 0.80, "avg_x_pos": 0, "description": "通关率 ≥ 80%"},
}


def evaluate_checkpoint(
    checkpoint_path: str,
    num_episodes: int = 20,
    action_set: str = "simple",
    phase: int | None = None,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """加载检查点并运行评估。返回汇总 dict。"""
    env = build_env(EnvConfig(action_set=action_set))
    agent = MarioAgent(
        action_dim=env.action_space.n,
        config=AgentConfig(),
        device=device,
        save_dir="/tmp",
    )
    agent.load(checkpoint_path)

    results = []
    for ep in range(1, num_episodes + 1):
        state = reset_env(env)
        done = False
        ep_reward = 0.0
        steps = 0
        info = {}
        while not done:
            if render:
                env.render()
            action = agent.eval_act(state)
            next_state, reward, done, info = step_env(env, action)
            state = next_state
            ep_reward += reward
            steps += 1
        flag = info.get("flag_get", False)
        x_pos = info.get("x_pos", 0)
        status = "通关！" if flag else f"x_pos={x_pos}"
        results.append({
            "episode": ep,
            "reward": round(ep_reward, 1),
            "steps": steps,
            "x_pos": x_pos,
            "flag_get": flag,
        })
        if render:
            print(f"Ep {ep}: 奖励={ep_reward:.0f} 步数={steps} {status}")
        else:
            print(f"Ep {ep:2d}: 奖励={ep_reward:7.0f}  步数={steps:4d}  x={x_pos:5d}  {status}")

    env.close()

    clears = sum(1 for r in results if r["flag_get"])
    time_to_flag = [r["steps"] for r in results if r["flag_get"]]
    total = len(results)

    summary = {
        "checkpoint": checkpoint_path,
        "num_episodes": total,
        "completions": clears,
        "completion_rate": round(clears / total, 4),
        "avg_reward": round(float(np.mean([r["reward"] for r in results])), 1),
        "avg_x_pos": round(float(np.mean([r["x_pos"] for r in results])), 1),
        "max_x_pos": max(r["x_pos"] for r in results),
        "avg_steps": round(float(np.mean([r["steps"] for r in results])), 1),
        "avg_time_to_flag": round(float(np.mean(time_to_flag)), 1) if time_to_flag else None,
    }

    # Print summary
    print(f"\n{'='*50}")
    print(f"评估汇总")
    print(f"{'='*50}")
    print(f"检查点:    {checkpoint_path}")
    print(f"通关率:    {summary['completion_rate']:.1%} ({clears}/{total})")
    print(f"平均奖励:  {summary['avg_reward']:.1f}")
    print(f"平均 X:    {summary['avg_x_pos']:.0f}")
    print(f"最大 X:    {summary['max_x_pos']}")
    print(f"平均步数:  {summary['avg_steps']:.0f}")
    if summary["avg_time_to_flag"] is not None:
        print(f"平均通关步数: {summary['avg_time_to_flag']:.0f}（仅统计通关回合）")

    # Phase gate check
    if phase is not None and phase in PHASE_GATES:
        gate = PHASE_GATES[phase]
        print(f"\n阶段 {phase} 门槛检查: {gate['description']}")
        cr_pass = summary["completion_rate"] >= gate["completion_rate"]
        x_pass = summary["avg_x_pos"] > gate["avg_x_pos"]
        overall_pass = cr_pass and x_pass
        status_text = "通过" if overall_pass else "未通过"
        print(f"通关率: {summary['completion_rate']:.1%} (需要 ≥ {gate['completion_rate']:.0%}) -> {'✓' if cr_pass else '✗'}")
        if gate["avg_x_pos"] > 0:
            print(f"平均 X:  {summary['avg_x_pos']:.0f} (需要 > {gate['avg_x_pos']}) -> {'✓' if x_pass else '✗'}")
        print(f"阶段 {phase} 门槛: [{status_text}]")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="评估单个检查点的表现")
    parser.add_argument("checkpoint", type=str, help="检查点 .pth 文件路径")
    parser.add_argument("--episodes", type=int, default=20, help="评估 episode 数量（默认 20）")
    parser.add_argument("--action-set", default="simple", choices=["simple", "complex", "right_only"])
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3], help="与指定阶段门槛对比")
    parser.add_argument("--render", action="store_true", help="渲染游戏画面")
    parser.add_argument("--device", default="cpu", help="cpu / cuda")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        action_set=args.action_set,
        phase=args.phase,
        render=args.render,
        device=args.device,
    )


if __name__ == "__main__":
    main()
