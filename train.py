import argparse
from pathlib import Path

from mario_rl.agent import MarioAgent
from mario_rl.config import AgentConfig, EnvConfig, TrainingConfig
from mario_rl.env import build_env
from mario_rl.trainer import Trainer
from mario_rl.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDQN agent for Super Mario Bros.")

    # Environment
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0", help="Gym environment id.")
    parser.add_argument(
        "--action-set",
        default="simple",
        choices=["right_only", "simple", "complex"],
        help="Discrete Mario action set.",
    )

    # Training loop
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (None = unlimited).")
    parser.add_argument("--checkpoint", help="Optional checkpoint path to resume from.")
    parser.add_argument("--save-dir", default="weights", help="Directory used to save checkpoints.")
    parser.add_argument("--checkpoint-period", type=int, default=20, help="Save checkpoint every N episodes.")
    parser.add_argument("--log-window", type=int, default=10, help="Window size for moving average.")

    # Evaluation
    parser.add_argument("--eval", dest="eval_checkpoint", help="Evaluate a checkpoint and exit (no training).")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of episodes per evaluation.")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate every N training episodes.")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stop patience (0=disabled).")

    # Agent hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--replay-size", type=int, default=100_000, help="Replay buffer capacity.")
    parser.add_argument("--epsilon", type=float, default=0.75, help="Initial exploration rate.")
    parser.add_argument("--epsilon-decay", type=float, default=0.999995, help="Exploration decay factor (per step).")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum exploration rate.")
    parser.add_argument("--sync-steps", type=int, default=10, help="Target network sync frequency (steps).")
    parser.add_argument("--tau", type=float, default=0.0, help="Soft update coefficient (0 = hard sync).")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping max norm.")
    parser.add_argument("--dueling", action="store_true", help="Use dueling DQN architecture.")

    # Device / render
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--render", action="store_true", help="Render the game during training.")
    parser.add_argument("--device", help="Torch device, for example cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_config = EnvConfig(
        env_id=args.env_id,
        action_set=args.action_set,
        seed=args.seed,
    )
    agent_config = AgentConfig(
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.lr,
        replay_buffer_size=args.replay_size,
        exploration_rate=args.epsilon,
        exploration_rate_decay=args.epsilon_decay,
        exploration_rate_min=args.epsilon_min,
        sync_steps=args.sync_steps,
        tau=args.tau,
        grad_clip=args.grad_clip,
        dueling=args.dueling,
    )
    training_config = TrainingConfig(
        episodes=args.episodes,
        checkpoint_period=args.checkpoint_period,
        checkpoint_path=args.checkpoint,
        save_dir=Path(args.save_dir),
        device=resolve_device(args.device),
        render=args.render,
        max_steps_per_episode=args.max_steps,
        log_window=args.log_window,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        early_stop_patience=args.early_stop,
    )

    env = build_env(env_config)
    agent = MarioAgent(
        action_dim=env.action_space.n,
        config=agent_config,
        device=training_config.device,
        save_dir=training_config.save_dir,
        input_channels=env.observation_space.shape[0],
    )

    trainer = Trainer(env=env, agent=agent, config=training_config)

    # Eval-only mode: load checkpoint, evaluate, print results, exit
    if args.eval_checkpoint:
        metrics = trainer.eval_checkpoint(args.eval_checkpoint, num_episodes=args.eval_episodes)
        print("\nEvaluation results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return

    if training_config.checkpoint_path:
        agent.load(training_config.checkpoint_path)

    trainer.train()


if __name__ == "__main__":
    main()
