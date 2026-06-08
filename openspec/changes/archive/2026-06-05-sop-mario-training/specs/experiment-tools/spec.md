## 新增需求

### 需求：跨实验轮次对比结果
系统必须提供 `scripts/compare.py`，从多个实验目录读取 metrics.json，生成按 completion_rate（首要）和 avg_x_pos（次要）排序的排名对比表。

#### 场景：对比三个实验轮次
- **当** 用户运行 `python scripts/compare.py weights/Round01_Baseline weights/Round02_Gamma099 weights/Round03_ComplexActions`
- **则** 必须打印一张表，列包含：轮次、Episodes、平均奖励、最大奖励、通关率、平均 X 位置、评估通关率、最佳评估 X 位置、耗时（秒），按评估通关率降序排列

#### 场景：某轮无评估数据
- **当** 某轮的 metrics.json 中没有 `evaluation` 条目
- **则** 对比工具必须展示训练级指标（来自训练的 completion_rate），并将评估列标记为 "N/A"

### 需求：绘制实验训练曲线
系统必须提供 `scripts/analyze.py`，从单个实验的 `metrics.json` 生成多面板图表：(a) 奖励随 episode 变化及移动平均，(b) 损失随 episode 变化，(c) x-位置随 episode 变化，(d) epsilon 衰减曲线。图表必须保存为实验目录下的 `analysis.png`。

#### 场景：分析已完成的实验
- **当** 用户运行 `python scripts/analyze.py weights/Round05_SoftUpdate`
- **则** 必须在 `weights/Round05_SoftUpdate/analysis.png` 中创建包含四个子图的文件：奖励曲线、损失曲线、x-位置、epsilon 衰减

#### 场景：实验目录缺少 metrics.json
- **当** 实验目录中没有 `metrics.json` 文件
- **则** analyze 工具必须打印错误："在 <dir> 中未找到 metrics.json。训练是否已完成？"

### 需求：拉取远程实验结果
系统必须扩展 `remote_train.py`，增加 `results` 子命令，从远端所有实验目录下载 metrics.json 和 metrics.csv 到本地 `results/` 目录，按轮次名称组织，不下载权重文件。

#### 场景：拉取所有远程结果
- **当** 用户运行 `python remote_train.py results`
- **则** 远端所有 `weights/*/metrics.json` 和 `weights/*/metrics.csv` 必须下载到本地 `results/<round_name>/`，按轮次目录名组织

### 需求：最佳检查点快速摘要
系统必须为 `scripts/analyze.py` 提供 `--summary` 参数，给定一个实验目录后打印一行摘要：通关率、平均 x_pos、最大 x_pos 以及最佳检查点文件路径。

#### 场景：快速查看实验摘要
- **当** 用户运行 `python scripts/analyze.py weights/Round08_Complex+Soft+Clip --summary`
- **则** 必须打印一行输出："通关率: 45% (9/20) | 平均 X: 2450 | 最大 X: 3168 | 最佳检查点: weights/Round08_Complex+Soft+Clip/checkpoint_100.pth"
