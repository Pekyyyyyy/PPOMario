"""用转换后的 MarioNET 模型评估 Super Mario Bros 关卡。"""

import logging
logging.getLogger("gymnasium").setLevel(logging.ERROR)

# Windows 控制台 UTF-8 兼容
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario_rl.a3c.model import MarioNET
from mario_rl.a3c.env import build_a3c_env, NUM_ACTIONS

# ── 常量 ────────────────────────────────────────────────────────────

HIDDEN_SIZE = 512
MAX_STEPS = 3000       # 单回合最大步数
MAX_REPEAT = 200        # 重复动作检测阈值

DEFAULT_LEVELS = [(1, 1), (1, 2), (1, 3), (1, 4)]


# ── 模型加载 ────────────────────────────────────────────────────────

def load_model(model_path: str, device: str = "cpu") -> MarioNET:
    """加载转换后的 .pth 模型。

    兼容两种格式：
    - {"model": state_dict}（本项目的包装格式）
    - 原始 state_dict（直接 torch.save 的格式）
    """
    model = MarioNET(4, NUM_ACTIONS, HIDDEN_SIZE)
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── 关卡评估 ────────────────────────────────────────────────────────

def evaluate_level(
    model: MarioNET,
    world: int,
    stage: int,
    num_episodes: int = 5,
    device: str = "cpu",
    render: bool = False,
) -> list[dict]:
    """在指定关卡上评估模型。

    LSTM 隐藏状态 (hx, cx) 在每个回合开始时重置为零，
    在回合内的每步之间持续传递——这是模型泛化的关键。
    """
    level_name = f"{world}-{stage}"
    results: list[dict] = []

    for ep in range(1, num_episodes + 1):
        env = build_a3c_env(world, stage, render=render)
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        info = {}

        # LSTM 状态初始化为零
        hx = torch.zeros(1, HIDDEN_SIZE, device=device)
        cx = torch.zeros(1, HIDDEN_SIZE, device=device)

        recent_actions: deque = deque(maxlen=MAX_REPEAT)

        while not done and steps < MAX_STEPS:
            # 准备输入张量
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_t.ndim == 3:
                obs_t = obs_t.unsqueeze(0)  # (4,80,80) → (1,4,80,80)

            with torch.no_grad():
                _critic, logits, hx, cx = model(obs_t, hx, cx)

            # 贪心动作选择
            action = torch.argmax(logits, dim=1).item()
            recent_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            # 卡住检测
            if (
                len(recent_actions) == MAX_REPEAT
                and recent_actions.count(recent_actions[0]) == MAX_REPEAT
            ):
                done = True

        env.close()

        flag = info.get("flag_get", False)
        x_pos = info.get("x_pos", 0)
        results.append({
            "episode": ep,
            "reward": round(ep_reward, 1),
            "steps": steps,
            "x_pos": x_pos,
            "flag_get": flag,
        })

        status = "通关!" if flag else f"x={x_pos}"
        print(f"  {level_name} Ep{ep:2d}: reward={ep_reward:7.1f}  "
              f"steps={steps:4d}  {status}")

    return results


def print_summary(level_name: str, results: list[dict]) -> None:
    """打印单个关卡的汇总统计。"""
    clears = sum(1 for r in results if r["flag_get"])
    total = len(results)
    if total == 0:
        return
    avg_r = np.mean([r["reward"] for r in results])
    avg_x = np.mean([r["x_pos"] for r in results])
    max_x = max(r["x_pos"] for r in results)
    ttf = (
        np.mean([r["steps"] for r in results if r["flag_get"]])
        if clears > 0
        else 0
    )
    print(f"  {level_name} 汇总: 通关={clears}/{total}, "
          f"平均奖励={avg_r:.1f}, 平均X={avg_x:.0f}, 最大X={max_x}, "
          f"平均通关步数={ttf:.0f}")


# ── 主入口 ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MarioNET 模型评估")
    parser.add_argument(
        "--model", type=str,
        default="weights/a3c/mario_net_a3c.pth",
        help="转换后的 .pth 模型路径",
    )
    parser.add_argument(
        "--levels", type=str, nargs="+",
        default=["1-1", "1-2", "1-3", "1-4"],
        help="要评估的关卡（如 1-1 1-2 1-3 1-4）",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="每个关卡的评估回合数",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="显示游戏画面",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="推理设备: cpu 或 cuda",
    )
    args = parser.parse_args()

    # 解析关卡列表
    levels: list[tuple[int, int]] = []
    for level_str in args.levels:
        parts = level_str.split("-")
        if len(parts) != 2:
            print(f"[错误] 无效的关卡格式: {level_str}（应为如 1-1）")
            sys.exit(1)
        levels.append((int(parts[0]), int(parts[1])))

    # 加载模型
    device = args.device
    model_path = args.model
    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 scripts/convert_a3c.py 来下载并转换模型")
        sys.exit(1)

    model = load_model(model_path, device)
    print(f"模型已加载: {model_path}")
    print(f"评估关卡: {[f'{w}-{s}' for w, s in levels]}")
    print(f"每关回合数: {args.episodes}\n")

    # 逐关评估
    all_results: dict[str, list[dict]] = {}
    for world, stage in levels:
        level_name = f"{world}-{stage}"
        print(f"{'=' * 50}")
        print(f"  关卡 {level_name}")
        print(f"{'=' * 50}")
        results = evaluate_level(
            model, world, stage, args.episodes, device, args.render
        )
        all_results[level_name] = results
        print_summary(level_name, results)
        print()

    # 总体汇总
    print(f"{'=' * 50}")
    print(f"  总汇总")
    print(f"{'=' * 50}")
    total_clears = sum(
        sum(1 for r in results if r["flag_get"])
        for results in all_results.values()
    )
    total_eps = sum(len(results) for results in all_results.values())
    if total_eps > 0:
        print(f"  总通关: {total_clears}/{total_eps} ({total_clears / total_eps:.1%})")


if __name__ == "__main__":
    main()
