## 新增需求

### 需求：将所有实验聚合为一张对比表
系统必须提供 `scripts/compare.py`，当以 `--all` 参数运行（指定一个父目录如 `weights/`）时，发现所有包含 `metrics.json` 的子目录，聚合其摘要，输出排名表。

#### 场景：全量实验对比
- **当** 用户运行 `python scripts/compare.py --all weights/`
- **则** `weights/` 下所有包含 `metrics.json` 的子目录必须被发现并纳入对比表，按评估通关率降序排列

#### 场景：未发现任何实验
- **当** 用户对不包含任何含 `metrics.json` 子目录的目录使用 `--all`
- **则** 工具必须打印"在 <dir> 中未找到实验结果"并退出

### 需求：Top-N 摘要视图
系统必须为 `scripts/compare.py` 提供 `--top N` 参数，限制输出仅展示通关率排名前 N 的实验。

#### 场景：仅展示前 3 个实验
- **当** 用户运行 `python scripts/compare.py --all weights/ --top 3`
- **则** 仅展示评估通关率最高的 3 个实验

### 需求：对比表导出为 CSV
系统必须支持 `scripts/compare.py` 的 `--csv <path>` 参数，将对比表导出为 CSV 文件以供进一步分析。

#### 场景：导出对比结果
- **当** 用户运行 `python scripts/compare.py --all weights/ --csv results/comparison.csv`
- **则** 必须将 CSV 文件写入 `results/comparison.csv`，列与打印表格一致
