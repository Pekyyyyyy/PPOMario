"""PPO 训练入口

基于 uvipen/Super-mario-bros-PPO-pytorch，适配单 GPU 环境。

用法：
    python train_ppo.py                                    # 默认 1-1
    python train_ppo.py --world 1 --stage 1 --lr 1e-4      # 自定义参数
    python train_ppo.py --action-type complex --lr 1e-3     # 复杂动作集
"""

from __future__ import annotations

import argparse

from mario_rl.ppo.trainer import train_ppo
from mario_rl.utils import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="PPO for Super Mario Bros")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action-type", default="simple", choices=["right_only", "simple", "complex"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (try 1e-3, 1e-4, 1e-5, 7e-5)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--tau", type=float, default=1.0, help="GAE lambda")
    parser.add_argument("--beta", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--num-epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--num-steps", type=int, default=512, help="Steps per PPO update")
    parser.add_argument("--total-steps", type=int, default=5_000_000, help="Total training steps")
    parser.add_argument("--eval-every", type=int, default=10000, help="Evaluate every N steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument("--save-dir", default="weights/ppo", help="Save directory")
    parser.add_argument("--device", help="Torch device (cpu/cuda/cuda:1)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    train_ppo(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
        epsilon=args.epsilon,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
