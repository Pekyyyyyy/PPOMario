# Super Mario Bros — 强化学习三模型

基于 `gym-super-mario-bros` 和 PyTorch，实现了三种强化学习算法来训练 Super Mario Bros 通关智能体。

## 项目结构

```text
├── train.py                   # DDQN 训练入口
├── train_ppo.py               # PPO 训练入口
├── demo.py                    # A3C+LSTM 多关卡连续演示 (默认 1-1~1-4)
├── remote_train.py            # 远端 GPU 服务器训练辅助
├── mario_rl/
│   ├── __init__.py            # 顶层 API 导出
│   ├── actions.py             # 共享：动作集映射
│   ├── utils.py               # 共享：设备/seed/Gym API 兼容
│   ├── metrics.py             # 共享：训练指标日志
│   │
│   ├── ddqn/                  # Double DQN
│   │   ├── agent.py           #   Agent + 经验回放
│   │   ├── model.py           #   DoubleDQNNetwork (在线/目标网络)
│   │   ├── config.py          #   EnvConfig / AgentConfig / TrainingConfig
│   │   ├── env.py             #   环境构建 (SkipFrame + 灰度 + 缩放 + 帧堆叠)
│   │   └── trainer.py         #   训练循环 + checkpoint 策略
│   │
│   ├── ppo/                   # PPO (Proximal Policy Optimization)
│   │   ├── model.py           #   PPONetwork (Actor-Critic, 4 层卷积)
│   │   ├── env.py             #   环境构建 (CustomReward + CustomSkipFrame)
│   │   └── trainer.py         #   PPOAgent + GAE + Clipped Surrogate
│   │
│   └── a3c/                   # A3C+LSTM (MarioNET, 跨关卡泛化)
│       ├── model.py           #   MarioNET (ResBlock + LSTMCell + Actor/Critic)
│       └── env.py             #   环境包装器 (80x80 裁剪 + RunningMeanStd 归一化)
│
├── scripts/
│   ├── eval_a3c.py            # A3C 多关卡评估
│   ├── eval_checkpoint.py     # DDQN checkpoint 评估
│   ├── analyze.py             # 单实验训练曲线分析
│   └── compare.py             # 跨实验对比排名
│
└── weights/
    ├── ddqn/                  # DDQN 权重 + 实验记录
    │   ├── checkpoint_662.pth
    │   ├── phase0/
    │   └── phase0_v2/
    ├── ppo/                   # PPO 权重
    │   ├── pretrained/        #   uvipen 预训练
    │   └── 1-1/               #   本地训练产物
    └── a3c/                   # A3C 权重 + ONNX + 归一化参数
```

## 三种模型对比

| 特性 | DDQN | PPO | A3C+LSTM (MarioNET) |
|---|---|---|---|
| **算法** | Double DQN + ε-greedy | PPO + GAE + Clipped Surrogate | A3C + LSTM |
| **网络结构** | 3 层卷积 → FC(512) → Q值 | 4 层卷积 → FC(512) → Actor/Critic | 3×ResBlock → FC(3200)→LSTMCell(512)→Actor/Critic |
| **观测尺寸** | 4×84×84 (uint8) | 4×84×84 (float32 [0,1]) | 4×80×80 (float32 [-10,10]) |
| **动作空间** | 7 (simple) | 7 (simple) | 10 (CUSTOM_MOVEMENT) |
| **记忆机制** | 无 (纯 CNN) | 无 (纯 CNN) | ✅ LSTM (跨关卡泛化核心) |
| **跨关卡泛化** | ❌ | ❌ | ✅ 单模型通全部 32 关 |
| **预训练权重** | checkpoint_662.pth (仅 1-1) | ppo_super_mario_bros_1_1.pth (仅 1-1) | mario_net_a3c.pth (全部 32 关) |
| **权重来源** | 本仓库训练 | uvipen/Super-mario-bros-PPO-pytorch | dgriff777/SuperMarioRL |
| **ONNX 导出** | — | — | ✅ |

## 环境依赖

Python 3.9 ~ 3.11。安装依赖：

```bash
pip install -r requirements.txt
```

可选依赖（ONNX 导出验证）：
```bash
pip install onnx onnxscript
```

## 快速开始

### A3C+LSTM 可视化演示

```bash
# 连续运行 1-1 ~ 1-4 (默认)
python demo.py

# 连续运行所有关卡
python demo.py --all

# 运行指定关卡
python demo.py --levels 1-1 1-2

# 慢速模式 (每帧暂停)
python demo.py --slow

# 无渲染模式 (仅结果)
python demo.py --no-render
```


### DDQN 训练

```bash
# 默认训练 (1-1, simple 动作集)
python train.py

# 自定义超参数
python train.py --episodes 500 --action-set complex --device cuda
python train.py --lr 5e-5 --gamma 0.99 --replay-size 200000

# 从 checkpoint 恢复
python train.py --checkpoint weights/ddqn/checkpoint_662.pth
```

### PPO 训练

```bash
# 默认训练 (1-1, simple 动作集)
python train_ppo.py

# 自定义关卡和超参数
python train_ppo.py --world 1 --stage 2 --lr 1e-4
python train_ppo.py --world 1 --stage 3 --lr 7e-5    # 1-3 需要更低 lr
python train_ppo.py --action-type complex --device cuda
```

### A3C+LSTM 评估（跨关卡泛化）

```bash
# 下载预训练权重、转换为 .pth、导出 ONNX
python scripts/convert_a3c.py

# 评估 1-1 ~ 1-4 (每个关卡 5 回合)
python scripts/eval_a3c.py

# 仅评估 1-1
python scripts/eval_a3c.py --levels 1-1 --episodes 10

# 查看实时画面
python scripts/eval_a3c.py --levels 1-1 --render
```

### DDQN checkpoint 评估

```bash
python scripts/eval_checkpoint.py weights/ddqn/checkpoint_662.pth --episodes 10
```

## 远端训练 (Win-SCP)

```bash
# 查看 GPU 状态
python remote_train.py gpus

# 查看训练进程
python remote_train.py status

# 同步代码 + 启动训练
python remote_train.py full --episodes 1000 --action-set simple --background

# 下载训练结果
python remote_train.py download
```

详见 [CLAUDE.md](CLAUDE.md) 中的完整远端操作说明。

## 架构说明

### DDQN 数据流

```
train.py → 组装 EnvConfig / AgentConfig / TrainingConfig
  → build_env() 包装 gym-super-mario-bros:
      JoypadSpace → SkipFrame(4) → GrayScale → Resize(84) → StepCompat → FrameStack(4)
  → MarioAgent: ε-greedy 动作选择 + deque 经验回放 + DDQN 学习
  → Trainer: 回合循环 + 移动平均奖励 checkpoint
```

### PPO 数据流

```
train_ppo.py → build_ppo_env() 包装 gym-super-mario-bros:
    JoypadSpace → CustomReward → CustomSkipFrame(4)
  → PPOAgent: Actor-Critic 采样 + GAE + Clipped Surrogate 更新
  → 定期评估 + 最佳 checkpoint 保存
```

### A3C+LSTM 数据流

```
scripts/convert_a3c.py
  → 下载 SuperMarioBros-v0.dat (dgriff777 预训练权重)
  → MarioNET 加载 + 校验 → 保存 mario_net_a3c.pth → 导出 mario_net_a3c.onnx

scripts/eval_a3c.py
  → build_a3c_env() 包装 gym-super-mario-bros:
      JoypadSpace → A3CSkipFrame(4) → A3CFrameStack(4) → A3CNormalize
  → MarioNET 单步推理 (LSTM 状态跨时间步传递)
  → 贪心动作选择 + 卡住检测
```

## 关键设计决策

- **DDQN**：在线网络选动作、目标网络评估，解耦动作选择和值估计以减少过估计偏差
- **PPO**：单进程实现，Actor-Critic 共享卷积骨干，GAE 优势估计，0.5×critic_loss 权重
- **A3C+LSTM**：LSTM 记忆单元是跨关卡泛化的核心——模型学到了关卡无关的导航策略而非特定关卡的视觉模式
- **Gym API 兼容**：`step_env()` / `reset_env()` 同时处理 4-tuple 和 5-tuple 返回值
- **checkpoint 格式**：统一 `{"model": state_dict}` 字典包装

## 权重文件说明

| 文件 | 模型 | 来源 | 能力 |
|---|---|---|---|
| `weights/ddqn/checkpoint_662.pth` | DDQN | 本仓库训练 | 1-1 (未通关) |
| `weights/ppo/pretrained/ppo_super_mario_bros_1_1.pth` | PPO | uvipen 预训练 | 1-1 100% 通关 |
| `weights/a3c/mario_net_a3c.pth` | A3C+LSTM | dgriff777 预训练 | 全部 32 关 (1-1~1-4 已验证 100%) |
| `weights/a3c/mario_net_a3c.onnx` | A3C+LSTM | ONNX 导出 | 同上 (需搭配 .onnx.data) |

## License

MIT
