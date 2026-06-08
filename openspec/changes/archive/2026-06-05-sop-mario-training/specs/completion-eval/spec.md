## 新增需求

### 需求：评估优先以通关率而非奖励为指标
`trainer.py` 中的评估函数必须为每个评估 episode 记录 `flag_get`（布尔值）和 `x_pos`（整数），并以 `completion_rate`（flag_get 为 True 的 episode 占比）作为主要指标报告。最佳检查点必须按最大化 `eval_completion_rate` 选择，平局时以 `avg_x_pos` 决胜，最后以 `avg_reward` 作为备选。

#### 场景：两个检查点的通关率不同
- **当** 评估检查点 A（completion_rate = 0.6，avg_reward = 1200）与检查点 B（completion_rate = 0.4，avg_reward = 1800）
- **则** 检查点 A 必须被选为更优，尽管原始奖励更低

#### 场景：两个检查点通关率相同
- **当** 检查点 A 和 B 的 completion_rate 均为 0.5
- **则** 必须选择 `avg_x_pos` 更高的检查点

### 需求：训练指标中记录通关和 x 位置
`MetricsLogger.log_episode()` 方法必须从环境 info 字典中为每个训练 episode 记录 `x_pos` 和 `flag_get`。这些字段必须同时包含在 `metrics.csv` 和 `metrics.json` 输出中。

#### 场景：训练 episode 结束
- **当** 一个训练 episode 结束（无论是拿到旗子、死亡还是超时）
- **则** 该 episode 的指标记录必须包含 `x_pos`（到达的最远 x 位置）和 `flag_get`（0 或 1）

### 需求：专用评估摘要工具
系统必须提供 `scripts/eval_checkpoint.py`，加载单个检查点，运行 N 轮评估 episode（默认 20），打印详细报告：逐轮明细（奖励、步数、x_pos、是否通关？）、汇总统计（通关率、平均 x、最大 x、通关平均步数），以及该检查点是否满足当前阶段门槛。

#### 场景：评估检查点是否满足阶段 2 门槛
- **当** 用户运行 `python scripts/eval_checkpoint.py weights/Round05_SoftUpdate/checkpoint_100.pth --phase 2`
- **则** 工具必须打印："通关率: 45% | 阶段 2 门槛（≥50%）: 未通过" 或 "通关率: 55% | 阶段 2 门槛（≥50%）: 通过"

#### 场景：使用自定义 episode 数量评估
- **当** 用户运行 `python scripts/eval_checkpoint.py checkpoint.pth --episodes 50`
- **则** 必须运行 50 轮评估 episode，汇总必须反映全部 50 轮的结果

### 需求：记录通关耗时
系统必须为通关的 episode 记录 `time_to_flag`（从 episode 开始到 flag_get 的游戏内步数）。评估摘要中必须以 `avg_time_to_flag`（仅通过通关的 episode 计算的平均步数）呈现。

#### 场景：部分评估 episode 通关
- **当** 10 个评估 episode 中有 3 个通关，步数分别为 [420, 385, 510]
- **则** 评估摘要必须报告 `avg_time_to_flag = 438.3` 和 `completions = 3`
