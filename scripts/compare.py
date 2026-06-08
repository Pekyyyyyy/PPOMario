"""跨实验对比工具：从多个实验目录读取 metrics.json，生成排名对比表。

用法：
    python scripts/compare.py dir1 dir2 dir3                  # 对比指定目录
    python scripts/compare.py --all weights/                  # 自动发现所有实验
    python scripts/compare.py --all weights/ --top 5          # 仅展示前 5 名
    python scripts/compare.py --all weights/ --csv comparison.csv  # 导出 CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def discover_experiments(parent_dir: Path) -> list[Path]:
    """发现父目录下所有包含 metrics.json 的子目录。"""
    experiments = []
    if not parent_dir.is_dir():
        return experiments
    for subdir in sorted(parent_dir.iterdir()):
        if subdir.is_dir() and (subdir / "metrics.json").exists():
            experiments.append(subdir)
    return experiments


def load_experiment(exp_dir: Path) -> dict | None:
    """加载单个实验的指标。"""
    json_path = exp_dir / "metrics.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    training = data.get("training", [])
    evaluation = data.get("evaluation", [])

    # 训练级指标
    train_cr = summary.get("completion_rate", 0)
    train_completions = summary.get("completions", 0)

    info = {
        "experiment": exp_dir.name,
        "path": str(exp_dir),
        "episodes": summary.get("episodes", 0),
        "avg_reward": summary.get("avg_reward", 0),
        "max_reward": summary.get("max_reward", 0),
        "train_completion_rate": train_cr,
        "train_completions": train_completions,
        "avg_x_pos": max((r.get("x_pos", 0) for r in training[-20:]), default=0) if training else 0,
        "total_time_s": summary.get("total_time_seconds", 0),
        "has_eval": bool(evaluation),
        "eval_completion_rate": None,
        "eval_completions": 0,
        "eval_num_episodes": 0,
        "eval_avg_x_pos": None,
        "eval_avg_time_to_flag": None,
    }

    # 评估级指标（取最佳的一次评估）
    if evaluation:
        best_eval = max(evaluation, key=lambda e: (e.get("completion_rate", 0), e.get("avg_x_pos", 0)))
        info["eval_completion_rate"] = best_eval.get("completion_rate", 0)
        info["eval_completions"] = best_eval.get("completions", 0)
        info["eval_num_episodes"] = best_eval.get("num_episodes", 0)
        info["eval_avg_x_pos"] = best_eval.get("avg_x_pos", 0)
        info["eval_avg_time_to_flag"] = best_eval.get("avg_time_to_flag", None)

    return info


def sort_key(info: dict):
    """排序键：评估通关率 > 评估平均 X > 训练通关率 > 平均奖励。"""
    ecr = info["eval_completion_rate"] or -1
    eax = info["eval_avg_x_pos"] or 0
    tcr = info["train_completion_rate"]
    ar = info["avg_reward"]
    return (ecr, eax, tcr, ar)


def format_cr(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1%}"


def format_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}m{sec}s"


def print_table(results: list[dict], top_n: int | None = None) -> None:
    """打印排名对比表。"""
    if not results:
        print("没有找到任何实验结果。")
        return

    results_sorted = sorted(results, key=sort_key, reverse=True)
    if top_n:
        results_sorted = results_sorted[:top_n]

    # 表头
    header = (
        f"{'排名':<4} {'实验名称':<35} {'Ep':<5} {'训练奖励':<10} {'训练CR':<8} "
        f"{'平均X':<7} {'评估CR':<8} {'评估X':<7} {'通关步数':<8} {'耗时':<8}"
    )
    sep = "-" * len(header)

    print(f"\n按评估通关率降序排列：\n")
    print(header)
    print(sep)

    for i, r in enumerate(results_sorted, 1):
        ecr_str = format_cr(r["eval_completion_rate"])
        ecr_detail = ""
        if r["eval_completion_rate"] is not None:
            ecr_detail = f" ({r['eval_completions']}/{r['eval_num_episodes']})"

        etf_str = f"{r['eval_avg_time_to_flag']:.0f}" if r['eval_avg_time_to_flag'] else "N/A"

        print(
            f"{i:<4} "
            f"{r['experiment'][:34]:<35} "
            f"{r['episodes']:<5} "
            f"{r['avg_reward']:<10.1f} "
            f"{format_cr(r['train_completion_rate']):<8} "
            f"{r['avg_x_pos']:<7.0f} "
            f"{ecr_str}{ecr_detail:<0}           "
            f"{r['eval_avg_x_pos'] or 'N/A':<7} "
            f"{etf_str:<8} "
            f"{format_time(r['total_time_s']):<8}"
        )
        # Pad eval CR column properly
    print()


def export_csv(results: list[dict], path: Path) -> None:
    """导出对比表为 CSV。"""
    fieldnames = [
        "experiment", "episodes", "avg_reward", "max_reward",
        "train_completion_rate", "avg_x_pos",
        "eval_completion_rate", "eval_completions", "eval_num_episodes",
        "eval_avg_x_pos", "eval_avg_time_to_flag",
        "total_time_s", "path",
    ]
    results_sorted = sorted(results, key=sort_key, reverse=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_sorted)
    print(f"CSV 已导出到 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="跨实验对比工具")
    parser.add_argument("exp_dirs", nargs="*", type=Path, help="实验目录列表")
    parser.add_argument("--all", type=Path, default=None, metavar="PARENT_DIR", help="自动发现父目录下所有实验")
    parser.add_argument("--top", type=int, default=None, metavar="N", help="仅展示前 N 个实验")
    parser.add_argument("--csv", type=Path, default=None, metavar="PATH", help="导出对比表为 CSV")
    args = parser.parse_args()

    # 收集实验目录
    exp_dirs: list[Path] = []
    if args.all:
        exp_dirs = discover_experiments(args.all)
        if not exp_dirs:
            print(f"在 {args.all} 中未找到实验结果。")
            sys.exit(0)
    else:
        exp_dirs = args.exp_dirs

    if not exp_dirs:
        parser.print_help()
        sys.exit(1)

    # 加载
    results = []
    for d in exp_dirs:
        info = load_experiment(d)
        if info is None:
            print(f"警告：{d} 中没有 metrics.json，跳过。")
            continue
        results.append(info)

    if not results:
        print("没有成功加载任何实验结果。")
        sys.exit(0)

    # 输出
    print_table(results, top_n=args.top)

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()
