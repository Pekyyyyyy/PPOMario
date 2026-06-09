# SOP — 标准操作流程

## 一、环境准备

### 1.1 本地环境

```bash
# 创建虚拟环境 (Python 3.9~3.11)
conda create -n mlpython python=3.10 -y
conda activate mlpython

# 安装依赖
pip install -r requirements.txt

# (可选) ONNX 导出验证
pip install onnx onnxscript
```

### 1.2 远端服务器

```bash
# 查看 GPU 状态
python remote_train.py gpus

# 同步本地代码到远端
python remote_train.py sync

# 首次远程部署时可能需要兼容性补丁 (见 CLAUDE.md)
```

## 二、训练流程

### 2.1 DDQN 训练

```bash
# 基准训练 (1-1, 1000 episodes)
python train.py

# 完整训练命令
python train.py \
  --episodes 2000 \
  --action-set simple \
  --lr 1e-4 \
  --gamma 0.99 \
  --replay-size 200000 \
  --batch-size 64 \
  --device cuda \
  --save-dir weights/my_experiment \
  --eval-every 20

# 从 checkpoint 恢复训练
python train.py --checkpoint weights/ddqn/checkpoint_662.pth --episodes 500
```

### 2.2 PPO 训练

```bash
# 基准训练 (1-1)
python train_ppo.py

# 完整训练命令
python train_ppo.py \
  --world 1 --stage 1 \
  --lr 1e-4 \
  --total-steps 5000000 \
  --num-steps 512 \
  --eval-every 10000 \
  --save-dir weights/ppo/1-1 \
  --device cuda

# 困难关卡 (1-3) 需要更低学习率
python train_ppo.py --world 1 --stage 3 --lr 7e-5 --device cuda

# 远端后台训练
python remote_train.py train_ppo \
  --world 1 --stage 2 --lr 1e-4 --device cuda:1 --background
```

### 2.3 训练监控

```bash
# 查看远端训练状态
python remote_train.py status

# 实时监控日志
python remote_train.py watch

# 停止训练
python remote_train.py kill
```

## 三、评估流程

### 3.1 DDQN 评估

```bash
# 评估 checkpoint
python scripts/eval_checkpoint.py weights/ddqn/checkpoint_662.pth --episodes 20

# 带渲染的评估
python scripts/eval_checkpoint.py weights/ddqn/checkpoint_662.pth --render
```

### 3.2 A3C+LSTM 评估（推荐）

```bash
# 首次使用：下载并转换权重
python scripts/convert_a3c.py

# 评估全部 4 关
python scripts/eval_a3c.py --episodes 10

# 仅评估 1-1，显示画面
python scripts/eval_a3c.py --levels 1-1 --episodes 5 --render

# GPU 推理
python scripts/eval_a3c.py --device cuda
```

## 四、权重管理

### 4.1 权重文件组织

```
weights/
├── ddqn/                        # DDQN 权重
│   ├── checkpoint_662.pth
│   ├── phase0/
│   └── phase0_v2/
├── ppo/                         # PPO 权重
│   ├── pretrained/              #   uvipen 预训练
│   └── 1-1/                     #   本地训练产物
└── a3c/                         # A3C 预训练权重 (推荐)
    ├── SuperMarioBros-v0.dat    #   原始权重 (dgriff777)
    ├── mario_net_a3c.pth        #   转换后权重
    ├── mario_net_a3c.onnx       #   ONNX 模型
    ├── mario_net_a3c.onnx.data  #   ONNX 权重数据
    └── obs_rms.pkl              #   归一化统计量
```

### 4.2 从远端下载权重

```bash
# 下载所有 checkpoint
python remote_train.py download

# 仅下载最新 checkpoint
python remote_train.py download --latest-only

# 仅下载指标文件
python remote_train.py results
```

### 4.3 权重格式

所有 `.pth` 文件统一使用 `{"model": state_dict}` 格式：
```python
ckpt = torch.load(path, map_location='cpu')
model.load_state_dict(ckpt["model"])
```

## 五、ONNX 模型使用

### 5.1 导出

```bash
python scripts/convert_a3c.py --skip-download
```

### 5.2 ONNX 推理示例

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("weights/a3c/mario_net_a3c.onnx")

obs = np.random.randn(1, 4, 80, 80).astype(np.float32)
hx = np.zeros((1, 512), dtype=np.float32)
cx = np.zeros((1, 512), dtype=np.float32)

critic, logits, hn, cn = session.run(
    ["critic", "logits", "hn", "cn"],
    {"observation": obs, "hx": hx, "cx": cx}
)
action = np.argmax(logits)
```

## 六、常见问题

### Q: gym / gymnasium 兼容性问题

项目同时兼容 gym 0.21 和 gymnasium。如果遇到 `ValueError: not enough values to unpack`，检查 `mario_rl/ddqn/env.py` 中的 `StepCompatWrapper` 是否正确加载。

### Q: OpenMP 冲突

```
OMP: Error #15: Initializing libiomp5md.dll
```

设置环境变量：`KMP_DUPLICATE_LIB_OK=TRUE`

### Q: A3C 评估时模型不工作

1. 确认 `weights/a3c/obs_rms.pkl` 存在（206 字节）
2. 确认预处理使用 `process_frame_a3c()` 而非 PPO 的 `process_frame()`
3. 确认观测 shape 为 `(4, 80, 80)` 而非 `(4, 84, 84)`

### Q: 远端训练 remote_train.py 报错

```bash
pip install paramiko scp
```

### Q: `ModuleNotFoundError: No module named 'onnxscript'`

```bash
pip install onnxscript
```

## 七、开发指南

### 7.1 添加新模型

1. 在 `mario_rl/` 下创建新的子目录 `<model_name>/`
2. 实现 `model.py`、`env.py`（和其他必要模块）
3. 创建 `__init__.py` 导出公共 API
4. 更新 `mario_rl/__init__.py` 添加顶层导入
5. 创建训练/评估入口脚本

### 7.2 修改现有模型

| 需求 | 修改位置 |
|---|---|
| 改 DDQN 超参数 | `mario_rl/ddqn/config.py` |
| 改 DDQN 网络结构 | `mario_rl/ddqn/model.py` |
| 改 DDQN 探索策略 | `mario_rl/ddqn/agent.py` → `act()` |
| 改 PPO 网络结构 | `mario_rl/ppo/model.py` |
| 改 PPO 环境预处理 | `mario_rl/ppo/env.py` |
| 改 PPO 训练逻辑 | `mario_rl/ppo/trainer.py` |
| 改 A3C 环境预处理 | `mario_rl/a3c/env.py` |
| 改动作集定义 | `mario_rl/actions.py` |

### 7.3 代码风格

- Python 文件使用 UTF-8 编码
- 模块 docstring 说明用途和关键设计决策
- checkpoint 统一使用 `{"model": state_dict}` 格式
