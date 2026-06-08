"""Sync local code to remote server and launch training.

Uses paramiko for SSH/SCP. Requires:
  pip install paramiko scp

Server: 10.122.242.98:20002
Account: stu_519
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import paramiko
from scp import SCPClient

# ── Remote server config ──────────────────────────────────────────────
REMOTE_HOST = "10.122.242.98"
REMOTE_PORT = 20002
REMOTE_USER = "stu_519"
REMOTE_PASSWORD = "519123"
REMOTE_PROJECT_DIR = "/home/stu_519/mario_rl"
REMOTE_PYTHON = "python3"

# ── Local project root ────────────────────────────────────────────────
LOCAL_PROJECT_DIR = Path(__file__).resolve().parent

# ── Files/dirs to sync ────────────────────────────────────────────────
SYNC_ITEMS = [
    "mario_rl",
    "scripts",
    "train.py",
    "train_ppo.py",
    "env_test.py",
    "requirements.txt",
    "run_experiments.py",
]


def get_ssh_client() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        password=REMOTE_PASSWORD,
        timeout=15,
    )
    return ssh


def sync_code(ssh: paramiko.SSHClient) -> None:
    """Upload local project files to remote server."""
    print(f"Syncing code to {REMOTE_HOST}:{REMOTE_PROJECT_DIR} ...")
    scp = SCPClient(ssh.get_transport())

    # Ensure remote project dir exists
    ssh.exec_command(f"mkdir -p {REMOTE_PROJECT_DIR}/mario_rl")
    ssh.exec_command(f"mkdir -p {REMOTE_PROJECT_DIR}/weights")

    for item in SYNC_ITEMS:
        local_path = LOCAL_PROJECT_DIR / item
        if local_path.is_dir():
            scp.put(str(local_path), recursive=True, remote_path=REMOTE_PROJECT_DIR)
            print(f"  [OK] {item}/ (directory)")
        else:
            scp.put(str(local_path), remote_path=f"{REMOTE_PROJECT_DIR}/{item}")
            print(f"  [OK] {item}")

    scp.close()
    print("Sync complete.\n")


def run_remote_train(
    ssh: paramiko.SSHClient,
    episodes: int = 1000,
    action_set: str = "simple",
    checkpoint: str | None = None,
    render: bool = False,
    device: str = "cuda",
    save_dir: str = "weights",
    background: bool = False,
    extra_args: str = "",
) -> None:
    """Launch training on remote server."""
    cmd_parts = [
        f"cd {REMOTE_PROJECT_DIR}",
        f"{REMOTE_PYTHON} train.py",
        f"--episodes {episodes}",
        f"--action-set {action_set}",
        f"--device {device}",
        f"--save-dir {save_dir}",
    ]
    if checkpoint:
        cmd_parts.append(f"--checkpoint {checkpoint}")
    if render:
        cmd_parts.append("--render")
    if extra_args:
        cmd_parts.append(extra_args)

    cmd = " && ".join(cmd_parts)

    if background:
        session_name = f"mario_train_{int(time.time())}"
        log_file = f"{REMOTE_PROJECT_DIR}/train_log_{session_name}.txt"
        # Use screen for reliable background execution
        cmd = (
            f"screen -dmS {session_name} bash -c \""
            f"cd {REMOTE_PROJECT_DIR} && "
            f"{REMOTE_PYTHON} -u train.py --episodes {episodes} --action-set {action_set} "
            f"--device {device} --save-dir {save_dir}"
        )
        if checkpoint:
            cmd += f" --checkpoint {checkpoint}"
        if extra_args:
            cmd += f" {extra_args}"
        cmd += f' > {log_file} 2>&1"'
        print(f"Starting background training on remote (screen: {session_name})...")
        print(f"Log: {REMOTE_USER}@{REMOTE_HOST}:{log_file}")
        print(f"Reattach: ssh -p {REMOTE_PORT} {REMOTE_USER}@{REMOTE_HOST} screen -r {session_name}")
    else:
        print(f"Running training on remote...")

    stdin, stdout, stderr = ssh.exec_command(cmd)
    if not background:
        print(stdout.read().decode())
        err = stderr.read().decode()
        if err:
            print("STDERR:", err)


def check_remote_gpus(ssh: paramiko.SSHClient) -> None:
    """Show GPU utilization on remote."""
    stdin, stdout, stderr = ssh.exec_command(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv"
    )
    print(stdout.read().decode())


def sync_results(ssh: paramiko.SSHClient, latest_only: bool = False) -> None:
    """Download checkpoints from remote to local weights/."""
    local_weights = LOCAL_PROJECT_DIR / "weights"
    local_weights.mkdir(exist_ok=True)
    scp = SCPClient(ssh.get_transport())
    if latest_only:
        # Download only metrics files and latest checkpoint per experiment
        scp.get(f"{REMOTE_PROJECT_DIR}/weights/*/metrics.json", local_path=str(local_weights), recursive=True)
        scp.get(f"{REMOTE_PROJECT_DIR}/weights/*/metrics.csv", local_path=str(local_weights), recursive=True)
        # Get latest checkpoint for each experiment dir
        stdin, stdout, stderr = ssh.exec_command(
            f"find {REMOTE_PROJECT_DIR}/weights -name 'checkpoint_*.pth' -printf '%h %p\n' | "
            "sort | awk '{dir=$1; file=$2; if(dir!=last){print file; last=dir}}'"
        )
        latest_ckpts = stdout.read().decode().strip().split("\n")
        for ckpt in latest_ckpts:
            if ckpt.strip():
                scp.get(ckpt.strip(), local_path=str(local_weights), recursive=True)
        print(f"Downloaded latest checkpoints and metrics to {local_weights}")
    else:
        scp.get(f"{REMOTE_PROJECT_DIR}/weights/", local_path=str(local_weights), recursive=True)
        print(f"Results downloaded to {local_weights}")
    scp.close()


def sync_metrics_only(ssh: paramiko.SSHClient) -> None:
    """Download only metrics.json/csv files from remote, organized by round name, without weights."""
    local_results = LOCAL_PROJECT_DIR / "results"
    local_results.mkdir(exist_ok=True)

    scp = SCPClient(ssh.get_transport())

    # List remote experiment dirs
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -d {REMOTE_PROJECT_DIR}/weights/*/ 2>/dev/null || echo ''"
    )
    remote_dirs = stdout.read().decode().strip().split("\n")

    count = 0
    for remote_dir in remote_dirs:
        remote_dir = remote_dir.strip()
        if not remote_dir:
            continue
        round_name = remote_dir.rstrip("/").split("/")[-1]
        local_round_dir = local_results / round_name
        local_round_dir.mkdir(parents=True, exist_ok=True)

        # Download metrics files
        for fname in ["metrics.json", "metrics.csv"]:
            remote_path = f"{remote_dir}/{fname}"
            stdin, stdout, stderr = ssh.exec_command(f"test -f {remote_path} && echo 'yes' || echo 'no'")
            if stdout.read().decode().strip() == "yes":
                scp.get(remote_path, local_path=str(local_round_dir / fname))
                count += 1

    scp.close()
    print(f"已下载 {count} 个指标文件到 {local_results}/")


def check_status(ssh: paramiko.SSHClient) -> None:
    """Show remote training sessions and recent logs."""
    # Check screen sessions
    stdin, stdout, stderr = ssh.exec_command(
        "screen -ls 2>&1 | grep -E 'mario_train|ppo_train' || echo 'No training screen sessions'"
    )
    print("Screen sessions:")
    print(stdout.read().decode())

    # Check for running python training processes
    stdin, stdout, stderr = ssh.exec_command(
        "ps aux | grep -E 'train.py|train_ppo.py' | grep -v grep | grep stu_519 || echo 'No stu_519 training processes'"
    )
    print("Training processes:")
    print(stdout.read().decode())

    # Show latest log tail if exists
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -t {REMOTE_PROJECT_DIR}/train_log_*.txt 2>/dev/null | head -1 | xargs tail -5 2>/dev/null || echo 'No logs found'"
    )
    print("Latest log tail:")
    print(stdout.read().decode())


def watch_training(ssh: paramiko.SSHClient, interval: int = 5) -> None:
    """Continuously tail the latest training log (Ctrl+C to stop)."""
    import time as time_module

    # Find the latest log file
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -t {REMOTE_PROJECT_DIR}/train_log_*.txt 2>/dev/null | head -1"
    )
    log_file = stdout.read().decode().strip()
    if not log_file:
        print("没有找到训练日志文件。")
        return

    print(f"监控日志: {log_file}")
    print(f"刷新间隔: {interval}s | Ctrl+C 停止\n")

    # Get current line count
    stdin, stdout, stderr = ssh.exec_command(f"wc -l < {log_file}")
    last_lines = int(stdout.read().decode().strip() or 0)

    try:
        while True:
            stdin, stdout, stderr = ssh.exec_command(
                f"tail -n +{last_lines + 1} {log_file} 2>/dev/null"
            )
            new_content = stdout.read().decode()
            if new_content:
                print(new_content.rstrip())
                last_lines += new_content.count("\n")
                # Also show if training ended
                if "Training complete" in new_content:
                    print("\n--- 训练已完成 ---")
                    break

            # Check if process still alive
            stdin, stdout, stderr = ssh.exec_command(
                "ps aux | grep 'train.py' | grep -v grep | grep stu_519 | wc -l"
            )
            alive = int(stdout.read().decode().strip() or 0)
            if alive == 0 and last_lines > 0:
                print("\n--- 训练进程已结束 ---")
                break

            time_module.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止（训练仍在后台运行）。")
        print(f"重新连接: python remote_train.py watch")


def launch_ppo(
    ssh: paramiko.SSHClient,
    world: int = 1,
    stage: int = 1,
    action_type: str = "simple",
    lr: float = 1e-4,
    total_steps: int = 5_000_000,
    num_steps: int = 512,
    eval_every: int = 10000,
    save_dir: str = "weights/ppo",
    device: str = "cuda",
    background: bool = False,
) -> None:
    """Launch PPO training on remote server."""
    cmd = (
        f"cd {REMOTE_PROJECT_DIR} && "
        f"{REMOTE_PYTHON} -u train_ppo.py "
        f"--world {world} --stage {stage} --action-type {action_type} "
        f"--lr {lr} --total-steps {total_steps} --num-steps {num_steps} "
        f"--eval-every {eval_every} --save-dir {save_dir} --device {device}"
    )

    if background:
        session_name = f"ppo_train_{int(time.time())}"
        log_file = f"{REMOTE_PROJECT_DIR}/train_log_{session_name}.txt"
        full_cmd = (
            f"screen -dmS {session_name} bash -c \"{cmd} > {log_file} 2>&1\""
        )
        print(f"Starting background PPO training (screen: {session_name})...")
        print(f"Log: {REMOTE_USER}@{REMOTE_HOST}:{log_file}")
        print(f"Monitor: python remote_train.py watch")
    else:
        full_cmd = cmd
        print(f"Running PPO training on remote (foreground)...")

    stdin, stdout, stderr = ssh.exec_command(full_cmd)
    if not background:
        print(stdout.read().decode())
        err = stderr.read().decode()
        if err:
            print("STDERR:", err)


def kill_remote(ssh: paramiko.SSHClient, session_name: str | None = None) -> None:
    """Kill remote training sessions."""
    if session_name:
        stdin, stdout, stderr = ssh.exec_command(f"screen -S {session_name} -X quit 2>&1")
        print(stdout.read().decode())
        print(f"Killed screen session: {session_name}")
    else:
        # Kill all mario_train and ppo_train screen sessions
        stdin, stdout, stderr = ssh.exec_command(
            "screen -ls 2>&1 | grep -E 'mario_train|ppo_train' | awk '{print $1}' | xargs -I{} screen -S {} -X quit"
        )
        print("Killed all training screen sessions")
        # Also kill orphaned train.py / train_ppo.py processes for this user
        stdin, stdout, stderr = ssh.exec_command(
            "pkill -f 'stu_519.*train(_ppo)?\\.py' 2>/dev/null; echo 'Also killed orphan train processes'"
        )
        print(stdout.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote training helper for Super Mario RL."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # sync
    sub.add_parser("sync", help="Upload local code to remote server")

    # train
    train_p = sub.add_parser("train", help="Launch training on remote server")
    train_p.add_argument("--episodes", type=int, default=1000)
    train_p.add_argument("--action-set", default="simple", choices=["right_only", "simple", "complex"])
    train_p.add_argument("--checkpoint", help="Remote checkpoint path to resume from")
    train_p.add_argument("--save-dir", default="weights")
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--background", action="store_true", help="Run in background via nohup")
    train_p.add_argument("--extra-args", default="", help="Extra arguments to pass to train.py (e.g. '--eval-every 25 --gamma 0.99')")

    # gpus
    sub.add_parser("gpus", help="Check remote GPU utilization")

    # status
    sub.add_parser("status", help="Check remote training sessions and recent logs")

    # watch
    watch_p = sub.add_parser("watch", help="Continuously monitor training log (Ctrl+C to stop)")
    watch_p.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds (default: 5)")

    # train_ppo
    ppo_p = sub.add_parser("train_ppo", help="Launch PPO training on remote server")
    ppo_p.add_argument("--world", type=int, default=1)
    ppo_p.add_argument("--stage", type=int, default=1)
    ppo_p.add_argument("--action-type", default="simple")
    ppo_p.add_argument("--lr", type=float, default=1e-4)
    ppo_p.add_argument("--total-steps", type=int, default=5_000_000)
    ppo_p.add_argument("--num-steps", type=int, default=512)
    ppo_p.add_argument("--eval-every", type=int, default=10000)
    ppo_p.add_argument("--save-dir", default="weights/ppo")
    ppo_p.add_argument("--device", default="cuda")
    ppo_p.add_argument("--background", action="store_true")

    # kill
    kill_p = sub.add_parser("kill", help="Stop remote training sessions")
    kill_p.add_argument("--session", help="Specific screen session name to kill")

    # download
    download_p = sub.add_parser("download", help="Download checkpoints from remote to local weights/")
    download_p.add_argument("--latest-only", action="store_true", help="Only download latest checkpoint per experiment")

    # results (metrics only, no weights)
    sub.add_parser("results", help="Download metrics.json/csv only (no weights) from all remote experiments")

    # full (sync + train)
    full_p = sub.add_parser("full", help="Sync code then train")
    full_p.add_argument("--episodes", type=int, default=1000)
    full_p.add_argument("--action-set", default="simple", choices=["right_only", "simple", "complex"])
    full_p.add_argument("--checkpoint", help="Remote checkpoint path to resume from")
    full_p.add_argument("--save-dir", default="weights")
    full_p.add_argument("--device", default="cuda")
    full_p.add_argument("--background", action="store_true")
    full_p.add_argument("--extra-args", default="", help="Extra arguments to pass to train.py")

    args = parser.parse_args()

    ssh = get_ssh_client()
    try:
        if args.command == "sync":
            sync_code(ssh)
        elif args.command == "train":
            run_remote_train(
                ssh,
                episodes=args.episodes,
                action_set=args.action_set,
                checkpoint=getattr(args, "checkpoint", None),
                save_dir=args.save_dir,
                device=args.device,
                background=args.background,
                extra_args=getattr(args, "extra_args", ""),
            )
        elif args.command == "gpus":
            check_remote_gpus(ssh)
        elif args.command == "status":
            check_status(ssh)
        elif args.command == "watch":
            watch_training(ssh, interval=getattr(args, "interval", 5))
        elif args.command == "train_ppo":
            launch_ppo(
                ssh,
                world=args.world,
                stage=args.stage,
                action_type=args.action_type,
                lr=args.lr,
                total_steps=args.total_steps,
                num_steps=args.num_steps,
                eval_every=args.eval_every,
                save_dir=args.save_dir,
                device=args.device,
                background=args.background,
            )
        elif args.command == "kill":
            kill_remote(ssh, session_name=getattr(args, "session", None))
        elif args.command == "download":
            sync_results(ssh, latest_only=getattr(args, "latest_only", False))
        elif args.command == "results":
            sync_metrics_only(ssh)
        elif args.command == "full":
            sync_code(ssh)
            run_remote_train(
                ssh,
                episodes=args.episodes,
                action_set=args.action_set,
                checkpoint=getattr(args, "checkpoint", None),
                save_dir=args.save_dir,
                device=args.device,
                background=args.background,
                extra_args=getattr(args, "extra_args", ""),
            )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
