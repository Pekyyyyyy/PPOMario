# 超级马里奥 DDQN 训练标准操作流程 (SOP)

> **目标**：训练一个能够**稳定通关** Super Mario Bros 1-1 的 DDQN 智能体  
> **首要指标**：通关率（flag_get rate）  
> **次要指标**：平均 x 位置 → 通关耗时  
> **硬件**：NVIDIA RTX 2080 Ti (22GB)，学生共享服务器  
> **原则**：小批次、快速验证、多次迭代

---

## 资源安全限制

| 参数 | 上限 | 原因 |
|---|---|---|
| `batch_size` | ≤ 64 | 2080 Ti 22GB 安全范围 |
| `replay_buffer_size` | ≤ 200,000 | 避免内存压力 |
| `stack_frames` | 4（固定） | 标准 Atari 配置 |
| `--max-steps` | **≤ 2000**（必须设置） | 防止单集卡死，超时自动截断 |
| `--eval-episodes` | **≤ 10** | 减少评估开销，快速迭代 |
| 单次实验时长 | ≤ 2 小时 | 共享服务器公平性 |

> ⚠️ **强制要求**：所有训练命令必须带 `--max-steps`。不设此参数会导致马里奥卡地形时单集跑满环境默认上限（~4000 步），训练慢 5-10 倍。

**GPU 检查命令**（每次训练前执行）：
```bash
python remote_train.py gpus
```
如果两张卡都被占用，联系占用者或等待。优先使用 `--device cuda:1`（GPU 1 通常空闲）。

---

## 远程工作流速查

| 步骤 | 命令 |
|---|---|
| ① 检查 GPU | `python remote_train.py gpus` |
| ② 同步代码 | `python remote_train.py sync` |
| ③ 启动训练 | `python remote_train.py train --episodes 100 --background` |
| ④ 监控进度 | `python remote_train.py status` |
| ⑤ 停止训练 | `python remote_train.py kill` |
| ⑥ 下载权重 | `python remote_train.py download` |
| ⑦ 拉取结果 | `python remote_train.py results` |
| ⑧ 分析实验 | `python scripts/analyze.py weights/<实验名>` |
| ⑨ 对比实验 | `python scripts/compare.py --all weights/` |
| ⑩ 评估检查点 | `python scripts/eval_checkpoint.py <检查点路径>` |

---

## 阶段 0：基线验证

**目标**：确认训练流水线正常工作，智能体能学到基本移动策略。  
**Episode 预算**：50  
**评估频率**：每 10 episode  
**预计耗时**：~5-10 分钟  
**门槛**：`avg_x_pos > 300`（能稳定走过前半段）

### 命令

```bash
# 后台运行：
python remote_train.py train --episodes 200 --action-set simple --device cuda:1 \
    --save-dir weights/phase0 \
    --extra-args "--max-steps 2000 --eval-every 20 --eval-episodes 5 --epsilon 0.5 --epsilon-decay 0.9999 --epsilon-min 0.02" \
    --background

# 监控：
python remote_train.py status
```

### 超参数

```
--action-set simple
--episodes 200
--max-steps 2000          ← 防止单集超时（足够到旗子）
--gamma 0.95
--lr 1e-4
--batch-size 32
--replay-size 100000
--epsilon 0.5             ← 从 0.5 开始（降低随机性）
--epsilon-decay 0.9999    ← 加快衰减（~80集后 ε<0.1）
--epsilon-min 0.02
--eval-every 20           ← 每 20 集评估
--eval-episodes 5         ← 5 集快速评估（非 20）
--device cuda:1
```

### 判断标准

| 结果 | 行动 |
|---|---|---|
| 评估 avg_x_pos > 300 | ✅ 进入阶段 1 |
| 训练 X 高但评估 X 低 | ⚠️ ε 衰减太慢——网络依赖随机探索，加快 `--epsilon-decay` |
| avg_x_pos < 300 | ⚠️ 增加 episodes 到 400；检查 epsilon 是否 > 0.3 |
| 训练崩溃/无进展 | ❌ 检查环境兼容性，重跑 `env_test.py` |

---

## 阶段 1：超参数扫描

**目标**：找到让智能体**偶尔能通关**的超参数组合。  
**Episode 预算**：6 轮 × 100 episode  
**评估频率**：每 25 episode  
**门槛**：**评估通关率 ≥ 20%**（20 轮评估中至少 4 次拿到旗子）

### 扫描计划

| 轮次 | 名称 | 探索变量 | 命令参数 |
|---|---|---|---|
| R1 | Baseline | 基线复现 | `--episodes 100 --action-set simple --eval-every 25` |
| R2 | Gamma099 | gamma 0.95→0.99 | `--gamma 0.99` |
| R3 | ComplexActions | 动作空间扩展 | `--action-set complex` |
| R4 | BigReplay | 回放缓冲区 2× | `--replay-size 200000` |
| R5 | LowLR+BigBatch | 小学习率+大批次 | `--lr 5e-5 --batch-size 64` |
| R6 | Gamma099+Complex | gamma+动作组合 | `--gamma 0.99 --action-set complex` |

### 自动化运行

```bash
# 远程一键启动全部 6 轮扫描
python remote_train.py sync
python remote_train.py ssh "cd /home/stu_519/mario_rl && python3 run_experiments.py"
```

或者逐轮手动：
```bash
python remote_train.py full --episodes 100 --action-set simple --eval-every 25 --device cuda:1 --background
python remote_train.py full --episodes 100 --action-set simple --gamma 0.99 --eval-every 25 --device cuda:1 --background
# ... 依此类推
```

### 判断标准

| 结果 | 行动 |
|---|---|
| 任一实验 eval_completion_rate ≥ 20% | ✅ 选出最佳配置，进入阶段 2 |
| 所有实验通关率 < 20% | ⚠️ 调整顺序：先调 gamma（0.95/0.99），再调 lr（1e-4/5e-5），最后调 exploration_decay（0.999995/0.99999） |
| avg_x_pos 显著下降 | ❌ 回退到阶段 0 参数，检查是否是复杂动作集导致探索不足 |

### 分析命令

```bash
# 下载结果并对比
python remote_train.py results
python scripts/compare.py --all results/
python scripts/analyze.py results/<最佳轮次>
```

---

## 阶段 2：算法调优

**目标**：提升稳定性，让智能体**经常能通关**。  
**Episode 预算**：4 轮 × 100 episode  
**评估频率**：每 25 episode  
**门槛**：**评估通关率 ≥ 50%**

### 调优计划

基于阶段 1 的最佳配置（记为 `BASE`），叠加以下改进：

| 轮次 | 名称 | 改进项 | 命令参数（在 BASE 基础上加） |
|---|---|---|---|
| R1 | SoftUpdate | 软更新替代硬同步 | `--tau 0.005` |
| R2 | GradClip | 梯度裁剪 | `--grad-clip 10.0` |
| R3 | Soft+Clip | 软更新+梯度裁剪 | `--tau 0.005 --grad-clip 10.0` |
| R4 | Dueling | Dueling 架构 | `--dueling` |

### 示例命令

```bash
# 假设阶段 1 最佳是 R2 (gamma=0.99)
BASE="--gamma 0.99 --action-set simple --episodes 100 --eval-every 25 --device cuda:1"
python remote_train.py full $BASE --tau 0.005 --background
python remote_train.py full $BASE --grad-clip 10.0 --background
python remote_train.py full $BASE --tau 0.005 --grad-clip 10.0 --background
python remote_train.py full $BASE --dueling --background
```

### 判断标准

| 结果 | 行动 |
|---|---|
| 任一实验 eval_completion_rate ≥ 50% | ✅ 选出最佳配置，进入阶段 3 |
| completion_rate 20-50% | ⚠️ 组合最优配置再跑 200 episode |
| 所有实验 < 20% | ❌ 考虑引入内在奖励（ICM/Curiosity），或回退阶段 1 重扫参数 |

---

## 阶段 3：最终打磨

**目标**：**稳定通关**，达到生产可用水平。  
**Episode 预算**：200–500 episode  
**评估频率**：每 10 episode  
**目标**：**评估通关率 ≥ 80%**

### 命令

```bash
# 加载阶段 2 最佳检查点，加大训练量
python remote_train.py full \
    --episodes 500 \
    --action-set <阶段2最佳> \
    --gamma <阶段2最佳> \
    --tau <阶段2最佳> \
    --grad-clip <阶段2最佳> \
    --dueling \
    --eval-every 10 \
    --checkpoint-period 50 \
    --device cuda:1 \
    --background
```

### 评估与收尾

```bash
# 下载最新权重
python remote_train.py download --latest-only

# 全面评估最佳检查点
python scripts/eval_checkpoint.py weights/checkpoint_500.pth --episodes 50

# 生成完整分析图
python scripts/analyze.py weights/
```

### 判断标准

| 结果 | 行动 |
|---|---|
| completion_rate ≥ 80% | ✅ **SOP 完成！** 保存模型，记录最终参数 |
| 60-80% | ⚠️ 继续训练至 500 episode，或微调 exploration_decay |
| < 60% | ❌ 回退阶段 2，尝试不同算法组合 |

---

## 检查点选择策略

**最佳检查点按以下优先级选择：**

1. **`eval_completion_rate`**（来自 `--eval-every` 的评估）—— 最主要
2. **`eval_avg_x_pos`**（通关率相同时的决胜指标）
3. **`training_completion_rate`**（无评估数据时的回退）
4. **`moving_avg_reward`**（最后手段，仅在没有其他指标时使用）

⚠️ **注意**：训练过程中的高奖励不一定意味着能通关（可能是在刷金币）。始终跑 10-20 轮贪婪评估来判断。

---

## 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|---|---|---|
| 单集耗时 > 30 秒 | 未设 `--max-steps`，马里奥卡地形超时 | **必须加 `--max-steps 1000`，杀掉重跑** |
| 进度日志长时间不更新 | episode 步数过长卡死 | `python remote_train.py kill`，加大 `--max-steps` 限制后重跑 |
| 智能体原地不动 | epsilon 衰减过快 | 提高 `exploration_rate_min` 到 0.05 |
| 奖励很高但不通关 | 在刷金币/踩乌龟 | 检查 x_pos 是否持续增长；降低 gamma |
| GPU 显存不足 | batch 或 replay 过大 | 降低 batch 到 32，replay 到 100K |
| 训练越来越慢 | 内存泄漏或 replay 过大 | 重启 screen 会话，减小 replay |
| 远端连接断开 | SSH 超时 | 使用 `--background`（screen 模式），训练不受影响 |

---

## 训练速度监控

正常速度参考（2080 Ti，simple 动作集）：

| max-steps | 预计速度 | 100 集耗时 | 200 集耗时 |
|---|---|---|---|
| 1000 | ~10-15 秒/集 | ~20 分钟 | ~40 分钟 |
| 2000 | ~15-25 秒/集 | ~35 分钟 | ~70 分钟 |

> 实际速度取决于马里奥存活时间。前期死得快（50-100 步/集），后期存活久（500-2000 步/集）。

---

## PPO 训练（推荐替代 DDQN）

基于 [uvipen/Super-mario-bros-PPO-pytorch](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)，已适配本项目。

**为什么用 PPO？** DDQN 在 200 集后评估通关率仍是 0%。PPO 是在线策略算法，无需回放缓冲，收敛更稳定。

### 快速启动

```bash
# 同步代码 + 启动 PPO（5M 步，约 2-4 小时）
python remote_train.py sync
python remote_train.py train_ppo --device cuda:1 --total-steps 5000000 --background
```

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--lr` | 1e-4 | 学习率（作者推荐 1e-3 → 1e-4 → 1e-5 逐级试） |
| `--total-steps` | 5,000,000 | 总训练步数（~5M 步通常能通关 1-1） |
| `--num-steps` | 512 | 每次 PPO 更新前的采样步数 |
| `--gamma` | 0.9 | 折扣因子 |
| `--action-type` | simple | right_only / simple / complex |
| `--eval-every` | 10000 | 每 N 步评估一次 |

### 自定义奖励

PPO 环境使用专用的奖励函数：
- **分数增量**：`+score_delta / 40`
- **通关**：`+50`（flag_get = True）
- **死亡**：`-50`
- 所有奖励除以 10 缩放

---

## 速查：train.py 主要参数（DDQN）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--episodes` | 1000 | 训练 episode 数 |
| `--action-set` | simple | right_only / simple / complex |
| `--gamma` | 0.95 | 折扣因子 |
| `--lr` | 1e-4 | 学习率 |
| `--batch-size` | 32 | 批次大小 |
| `--replay-size` | 100000 | 回放缓冲区大小 |
| `--tau` | 0.0 | 软更新系数（0=硬同步） |
| `--grad-clip` | None | 梯度裁剪阈值 |
| `--dueling` | False | 启用 Dueling DQN |
| `--eval-every` | None | 每 N episode 评估一次 |
| `--device` | cpu | cpu / cuda / cuda:1 |
| `--checkpoint` | None | 从检查点恢复训练 |
| `--save-dir` | weights | 权重和指标保存目录 |
