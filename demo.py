"""PPO 马里奥演示 - 手动 Pygame 渲染通关 1-1

用法:
    python demo.py              # 正常速度
    python demo.py --slow       # 慢速演示
    python demo.py --episodes=1 # 只跑一集

退出: 关闭游戏窗口 或 Ctrl+C
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mario_rl.ppo_model import PPONetwork
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# ── 命令行 ────────────────────────────────────────────────────────────
SLOW = "--slow" in sys.argv
EP_FLAG = [a for a in sys.argv if a.startswith("--episodes")]
NUM_EP = int(EP_FLAG[0].split("=")[1]) if EP_FLAG else 3
FRAME_DELAY = 0.03 if SLOW else 0.005  # 慢速 vs 正常帧间隔
SCALE = 2  # 256×240 → 512×480

# ── 配置 ──────────────────────────────────────────────────────────────
WORLD, STAGE = 1, 1
MODEL_PATH = "weights/pretrained/ppo_super_mario_bros_1_1.pth"
STACK, SKIP = 4, 4
ACTIONS = SIMPLE_MOVEMENT

# ── 帧预处理 ──────────────────────────────────────────────────────────
def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return (cv2.resize(gray, (84, 84)) / 255.0).astype(np.float32)

def init_stack(obs):
    return np.stack([preprocess(obs)] * STACK, axis=0)

def push_frame(stack, buf):
    half = SKIP // 2
    pooled = np.max(np.stack(buf[-half:], axis=0), axis=0)
    return np.concatenate([stack[1:], pooled[np.newaxis, :, :]], axis=0)

# ── 加载模型 ──────────────────────────────────────────────────────────
print("Loading model...")
model = PPONetwork(STACK, len(ACTIONS))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
model.eval()
print(f"Ready: {len(ACTIONS)} actions, {sum(p.numel() for p in model.parameters()):,} params")

# ── 环境 (rgb_array 模式，手动渲染) ───────────────────────────────────
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make(
    f"SuperMarioBros-{WORLD}-{STAGE}-v0",
    render_mode="rgb_array",
)
env = JoypadSpace(env, ACTIONS)

# ── Pygame 窗口 ───────────────────────────────────────────────────────
pygame.init()
W, H = 256 * SCALE, 240 * SCALE
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption(f"Mario RL Demo - World {WORLD}-{STAGE}  |  PPO Agent")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 20)

def draw_frame(rgb):
    """将 RGB (240,256,3) 转成 pygame surface，缩放后显示"""
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (W, H))
    screen.blit(surf, (0, 0))
    # 叠加文字
    if 'ep' in draw_frame.__dict__:
        txt = font.render(f"Episode {draw_frame.ep}  |  Steps {draw_frame.steps}", True, (255,255,255))
        screen.blit(txt, (10, 10))
    pygame.display.flip()

# ── 演示 ──────────────────────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  WINDOW: {W}x{H}  |  {FRAME_DELAY*1000:.0f}ms/frame")
print(f"  Close window to exit")
print(f"{'='*45}")
time.sleep(1)

results = []

for ep in range(1, NUM_EP + 1):
    obs, _ = env.reset()
    frame_stack = init_stack(obs)
    buf = []
    done = False
    total_rew = 0
    game_steps = 0
    draw_frame.ep = ep
    draw_frame.steps = 0

    # 初始动作
    tensor = torch.as_tensor(frame_stack, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(tensor)
    action = torch.argmax(logits, dim=1).item()

    while not done:
        for _ in range(SKIP):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rew += reward
            game_steps += 1
            buf.append(preprocess(obs))
            draw_frame.steps = game_steps

            # 实时渲染
            draw_frame(obs)
            clock.tick()  # 不设上限，依赖 delay

            # 处理关闭事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit(0)

            if done:
                break
            time.sleep(FRAME_DELAY)

        if done:
            break

        frame_stack = push_frame(frame_stack, buf)
        buf.clear()
        tensor = torch.as_tensor(frame_stack, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(tensor)
        action = torch.argmax(logits, dim=1).item()

    flag = info.get("flag_get", False)
    x_pos = info.get("x_pos", 0)
    results.append({"ep": ep, "reward": total_rew, "steps": game_steps,
                    "agent": game_steps // SKIP, "x": x_pos, "flag": flag})
    s = "CLEAR!" if flag else f"died x={x_pos}"
    print(f"  Episode {ep}: {s} ({game_steps} frames / {game_steps//SKIP} actions)")

env.close()

clears = sum(1 for r in results if r["flag"])
print(f"\n{'='*40}")
print(f"  {clears}/{NUM_EP} cleared ({clears/NUM_EP:.0%})")
if clears:
    print(f"  Avg agent steps to flag: {np.mean([r['agent'] for r in results if r['flag']]):.0f}")
print(f"{'='*40}")
print("\n  Press any key or close window to exit...")

# 保持窗口
while True:
    for event in pygame.event.get():
        if event.type in (pygame.QUIT, pygame.KEYDOWN):
            pygame.quit()
            sys.exit(0)
    time.sleep(0.1)
