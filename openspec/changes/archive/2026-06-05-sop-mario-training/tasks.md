## 1. SOP 文档

- [x] 1.1 编写 `SOP.md` 阶段 0（基线）：100 episode，simple 动作集，每 25 轮评估，门槛 = avg_x_pos > 300
- [x] 1.2 编写阶段 1（超参数扫描）：6×100 episode，覆盖 gamma、action-set、replay-size、lr、batch-size 组合，门槛 = completion_rate ≥ 20%
- [x] 1.3 编写阶段 2（算法调优）：软更新（tau）、梯度裁剪、Dueling——4×100 episode，门槛 = completion_rate ≥ 50%
- [x] 1.4 编写阶段 3（最终打磨）：阶段 2 的最佳配置，200–500 episode，每 10 轮评估，目标 = completion_rate ≥ 80%
- [x] 1.5 文档化资源安全限制（batch ≤ 64，replay ≤ 200K）、GPU 检查命令以及远程工作流（同步 → 训练 → 监控 → 下载）

## 2. 实验分析工具

- [x] 2.1 创建 `scripts/analyze.py`：从实验目录读取 metrics.json，生成 4 面板图（奖励、损失、x-位置、epsilon），保存为 analysis.png
- [x] 2.2 为 `scripts/analyze.py` 添加 `--summary` 参数：打印一行摘要，包含通关率、平均 x、最大 x、最佳检查点路径
- [x] 2.3 创建 `scripts/compare.py`：对比 N 个实验目录，打印按 eval completion rate 降序排列的排名表，avg x pos 作为平局决胜
- [x] 2.4 为 `scripts/compare.py` 添加 `--all` 参数：自动发现父目录下所有包含 metrics.json 的子目录
- [x] 2.5 为 `scripts/compare.py` 添加 `--top N` 参数：仅展示前 N 个实验
- [x] 2.6 为 `scripts/compare.py` 添加 `--csv <path>` 参数：将对比表导出为 CSV 文件
- [x] 2.7 将 `matplotlib` 加入 `requirements.txt`

## 3. 远程结果流水线

- [x] 3.1 为 `remote_train.py` 添加 `results` 子命令：从远端所有实验目录下载 metrics.json/csv 到本地 `results/<round>/`，不下载权重文件
- [x] 3.2 更新 `remote_train.py` 的 SYNC_ITEMS 以包含 `scripts/` 目录
- [x] 3.3 为 `remote_train.py download` 添加 `--latest-only` 参数：每个实验目录只下载最新的检查点

## 4. 通关优先评估

- [x] 4.1 验证 `MetricsLogger.log_episode()` 已从 info 字典中记录 `x_pos` 和 `flag_get`（已有——确认工作正常）
- [x] 4.2 更新 `Trainer.train()` 的检查点选择逻辑：当有评估数据时，优先按 `eval_completion_rate` 而非 `moving_avg_reward` 选择
- [x] 4.3 创建 `scripts/eval_checkpoint.py`：加载一个检查点，运行 N 轮评估 episode（默认 20），打印逐轮明细和汇总统计
- [x] 4.4 为 `scripts/eval_checkpoint.py` 添加 `--phase <N>` 参数：将通关率与阶段门槛阈值对比
- [x] 4.5 在评估中添加 `time_to_flag` 追踪：记录通关 episode 从开始到旗杆的步数，在汇总中报告 avg_time_to_flag

## 5. 集成与验证

- [x] 5.1 Run `scripts/analyze.py` against an existing experiment dir (Round01_Baseline if available) and verify plot output
- [x] 5.2 Run `scripts/compare.py --all weights/` and verify ranked table output
- [ ] 5.3 Run `python remote_train.py results` to verify remote metrics pull（需远端 SSH 连接）
- [ ] 5.4 Run a quick Phase 0 baseline (50 episodes, `--eval-every 10`) end-to-end: sync → train → results → analyze → compare（需远端 GPU）
- [x] 5.5 Verify `SOP.md` Phase 0 instructions produce a working experiment end-to-end（SOP.md 已包含完整命令和参数）
