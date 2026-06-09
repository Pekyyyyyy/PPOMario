"""A3C+LSTM 多关卡连续演示 (pygame 渲染)。

用法:
    python demo.py                              # 连续运行 1-1 ~ 1-4
    python demo.py --all                        # 运行全部 32 关
    python demo.py --levels 1-1                 # 仅运行 1-1
    python demo.py --levels 1-1 1-2 1-3         # 指定关卡列表
    python demo.py --no-render                  # 无渲染模式 (仅打印结果)
    python demo.py --slow                       # 慢速模式 (5 FPS)
"""

from __future__ import annotations

import logging
logging.getLogger("gymnasium").setLevel(logging.ERROR)
logging.getLogger("gym").setLevel(logging.ERROR)

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mario_rl.a3c.model import MarioNET
from mario_rl.a3c.env import build_a3c_env, NUM_ACTIONS

# ── 常量 ────────────────────────────────────────────────────────────

MODEL_PATH = Path("weights/a3c/mario_net_a3c.pth")
HIDDEN_SIZE = 512
MAX_STEPS = 4000
MAX_REPEAT = 200
DEFAULT_LEVELS = [(1, 1), (1, 2), (1, 3), (1, 4)]

# 动作显示名
ACTION_NAMES = ["NOOP", "R", "R+A", "R+B", "R+A+B", "A", "L+A", "L+B", "L+A+B", "DN"]


# ── Pygame 渲染器 ───────────────────────────────────────────────────

class PygameRenderer:
    """用 pygame 渲染 gym rgb_array 帧。"""

    def __init__(self):
        import pygame
        self.pygame = pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.screen: pygame.Surface | None = None

    def show_frame(self, frame: np.ndarray, title: str, fps: int = 60) -> bool:
        """显示一帧。返回 False 表示用户请求退出。"""
        if self.screen is None:
            h, w = frame.shape[:2]
            self.screen = self.pygame.display.set_mode((w * 2, h * 2))
            self.pygame.display.set_caption("Super Mario Bros - A3C+LSTM")

        # 处理事件
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_q:
                    return False

        # numpy (H, W, 3) RGB → pygame Surface
        surf = self.pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))  # HWC → WHC
        )
        # 2x 放大
        surf = self.pygame.transform.scale(surf, (frame.shape[1] * 2, frame.shape[0] * 2))

        self.screen.blit(surf, (0, 0))

        # 标题文字
        text = self.font.render(title, True, (255, 255, 0))
        text_bg = self.font.render(title, True, (0, 0, 0))
        self.screen.blit(text_bg, (6, 6))
        self.screen.blit(text, (5, 5))

        self.pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self):
        self.pygame.quit()


# ── 模型加载 ────────────────────────────────────────────────────────

def load_model(path: str = str(MODEL_PATH)) -> MarioNET:
    if not Path(path).exists():
        print(f"[错误] 模型文件不存在: {path}")
        print("请先运行: python scripts/convert_a3c.py")
        sys.exit(1)

    model = MarioNET(4, NUM_ACTIONS, HIDDEN_SIZE)
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── 单关运行 ────────────────────────────────────────────────────────

def run_level(
    model: MarioNET,
    world: int,
    stage: int,
    renderer: PygameRenderer | None = None,
    fps: int = 60,
) -> dict:
    level_name = f"{world}-{stage}"
    env = build_a3c_env(world, stage, render=renderer is not None)

    obs = env.reset()
    hx = torch.zeros(1, HIDDEN_SIZE)
    cx = torch.zeros(1, HIDDEN_SIZE)

    ep_reward = 0.0
    steps = 0
    info = {}
    done = False
    recent_actions: deque[int] = deque(maxlen=MAX_REPEAT)
    start_time = time.time()

    while not done and steps < MAX_STEPS:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _critic, logits, hx, cx = model(obs_t, hx, cx)
        action = int(torch.argmax(logits, dim=1).item())
        recent_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1

        # ── pygame 渲染 ──
        if renderer is not None:
            frame = env.render()
            if frame is not None:
                title = (f"Level {level_name}  Step {steps}  "
                         f"Reward {ep_reward:.0f}  X={info.get('x_pos', 0)}  "
                         f"Action: {ACTION_NAMES[action]}")
                if not renderer.show_frame(frame, title, fps):
                    print("\n[用户退出]")
                    env.close()
                    renderer.close()
                    sys.exit(0)

        # 卡住检测
        if (len(recent_actions) == MAX_REPEAT
                and recent_actions.count(recent_actions[0]) == MAX_REPEAT):
            done = True

    elapsed = time.time() - start_time
    env.close()

    return {
        "level": level_name, "world": world, "stage": stage,
        "reward": round(ep_reward, 1), "steps": steps,
        "x_pos": info.get("x_pos", 0),
        "flag_get": info.get("flag_get", False),
        "time": elapsed,
    }


# ── 主入口 ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="A3C+LSTM 多关卡演示")
    parser.add_argument("--levels", type=str, nargs="*",
                        help="要运行的关卡列表 (如 1-1 1-2)")
    parser.add_argument("--all", action="store_true",
                        help="运行全部 32 关 (1-1 ~ 8-4)")
    parser.add_argument("--slow", action="store_true",
                        help="慢速模式 (5 FPS)")
    parser.add_argument("--no-render", action="store_true",
                        help="无渲染模式 (仅打印结果)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="模型权重路径")
    args = parser.parse_args()

    # 解析关卡
    if args.all:
        levels = [(w, s) for w in range(1, 9) for s in range(1, 5)]
    elif args.levels:
        levels = []
        for level_str in args.levels:
            parts = level_str.split("-")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                print(f"[错误] 无效关卡格式: {level_str} (应为如 1-1)")
                sys.exit(1)
            levels.append((int(parts[0]), int(parts[1])))
    else:
        levels = DEFAULT_LEVELS

    # 渲染器
    renderer = None
    fps = 5 if args.slow else 60
    if not args.no_render:
        try:
            renderer = PygameRenderer()
            print("按 Q 或关闭窗口退出")
        except Exception as e:
            print(f"[提示] pygame 初始化失败 ({e})，使用无渲染模式")

    # 加载模型
    print(f"加载模型: {args.model}")
    model = load_model(args.model)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 逐关运行
    all_results: list[dict] = []
    total_start = time.time()

    for i, (world, stage) in enumerate(levels):
        level_name = f"{world}-{stage}"
        print(f"\n{'=' * 50}")
        print(f"  [{i + 1}/{len(levels)}] 关卡 {level_name}")
        print(f"{'=' * 50}")

        result = run_level(model, world, stage, renderer=renderer, fps=fps)
        all_results.append(result)

        status = "通关!" if result["flag_get"] else f"x={result['x_pos']}"
        print(f"  结果: {status}  |  "
              f"奖励={result['reward']:.0f}  |  步数={result['steps']}  |  "
              f"耗时={result['time']:.1f}s")

    # 汇总
    total_time = time.time() - total_start
    clears = sum(1 for r in all_results if r["flag_get"])
    total = len(all_results)
    print(f"\n{'=' * 50}")
    print(f"  汇总: {clears}/{total} 通关 ({clears / total:.1%})")
    print(f"  总耗时: {total_time:.1f}s")
    if all_results:
        avg_steps = np.mean([r["steps"] for r in all_results])
        print(f"  平均步数: {avg_steps:.0f}")
    print(f"{'=' * 50}")

    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
