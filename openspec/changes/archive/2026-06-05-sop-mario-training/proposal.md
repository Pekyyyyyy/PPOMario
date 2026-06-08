## 为什么

当前训练流程是临时拼凑的：每次实验都需要手工 SSH 命令、手动复制结果、凭记忆重新推导超参数选择。远程服务器（2× RTX 2080 Ti，学生共享）要求资源感知的调度策略。需要一个带有可复用工具的系统化标准操作流程（SOP），高效迭代出一个能够**稳定通关** Super Mario Bros 1-1 的 DDQN 智能体，以通关率（首要指标）和到达旗杆耗时（次要指标）衡量。

## 改什么

- 创建正式的 **SOP 文档**，定义分阶段训练流程：基线 → 超参数扫描 → 算法调优 → 最终打磨，阶段之间有明确的决策门槛
- 构建**可复用的 CLI 分析工具**：对比实验结果、绘制训练曲线、下载并归档带元数据的检查点
- 优化训练流水线，优先采取**小批次、快速迭代**的实验策略（50–200 episode），带自动评估和摘要
- 集中管理**结果组织**：每轮实验都有结构化的目录，包含 metrics.json/csv、检查点和自动生成的摘要
- 引入**通关优先的评估标准**：flag_get 率和 avg_x_pos 为主要 KPI，通关耗时作为次要指标

## 能力

### 新增能力
- `sop-workflow`：SOP 文档，指定分阶段训练策略、决策门槛、超参数搜索范围以及资源约束（对 2080 Ti 安全）
- `experiment-tools`：可复用的 Python 脚本，用于对比实验结果、生成对比表、绘制奖励/通关曲线、从远端拉取指标
- `metrics-dashboard`：CSV/JSON 聚合和快速预览工具，按通关率和 x 位置展示表现最优的实验
- `completion-eval`：专用评估协议，以 flag_get 率和 x-pos 优先于原始奖励，每个检查点的评估结果与权重文件并列存储

### 修改的能力
<!-- 无需修改已有规格 -->

## 影响范围

- `mario_rl/metrics.py`：增强以记录通关相关字段（x_pos、flag_get、time_to_flag）
- `mario_rl/trainer.py`：评估循环已捕获 `completion_rate` 和 `avg_x_pos`——这些将成为主要的早停和检查点选择标准
- `remote_train.py`：新增 `results` 子命令以便从远端拉取和聚合实验结果
- 新增文件：`scripts/analyze.py`、`scripts/compare.py`、`scripts/eval_checkpoint.py`、`SOP.md`
- 依赖：`matplotlib` 加入 `requirements.txt` 用于绘图（远端已有）
