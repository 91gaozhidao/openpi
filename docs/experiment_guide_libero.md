# LIBERO 实验执行方案

本文档提供在 LIBERO 基准测试上对 `pi0_fast_libero` 和 `pi05_libero` 进行实验验证的详细步骤。
**不需要训练模型**，直接使用已有的微调好的检查点（checkpoints）进行推理和评估。

---

## 目录

- [前置条件](#前置条件)
- [环境搭建](#环境搭建)
- [实验一：pi05\_libero 标准评估（LIBERO 仿真环境）](#实验一pi05_libero-标准评估libero-仿真环境)
- [实验二：pi0\_fast\_libero 标准评估（LIBERO 仿真环境）](#实验二pi0_fast_libero-标准评估libero-仿真环境)
- [实验三：pi05 流匹配 vs pi05+FAST-detokenizer 对齐对比](#实验三pi05-流匹配-vs-pi05fast-detokenizer-对齐对比)
- [结果分析](#结果分析)
- [常见问题](#常见问题)

---

## 前置条件

### 硬件要求
- **GPU**: NVIDIA GPU，至少 8 GB 显存（推理模式）
- 推荐：RTX 4090 / A100 / H100
- **系统**: Ubuntu 22.04

### 预训练检查点

以下检查点可以直接使用，无需训练：

| 模型 | 配置名 | 检查点路径 | 说明 |
|------|--------|-----------|------|
| π₀.₅-LIBERO | `pi05_libero` | `gs://openpi-assets/checkpoints/pi05_libero` | 流匹配 VLA，LIBERO 上 SOTA（96.85%） |
| π₀-FAST-LIBERO | `pi0_fast_libero` | `gs://openpi-assets/checkpoints/pi0_fast_base` | 自回归 VLA + FAST tokenizer |

> **注意**: 检查点会自动下载到 `~/.cache/openpi`，可通过 `OPENPI_DATA_HOME` 环境变量自定义路径。

---

## 环境搭建

### 方法一：使用 uv（推荐）

```bash
# 1. 克隆代码
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# 如果已经克隆了，初始化子模块
git submodule update --init --recursive

# 2. 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 方法二：使用 Docker

```bash
# 授权 X11
sudo xhost +local:docker

# 使用 Docker Compose 运行
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

### LIBERO 仿真环境安装（实验一和实验二需要）

```bash
# 创建 Python 3.8 虚拟环境（LIBERO 仿真器需要）
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

# 安装 LIBERO 依赖
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

---

## 实验一：pi05_libero 标准评估（LIBERO 仿真环境）

### 目的
在 LIBERO 仿真环境中评估 π₀.₅ 模型的任务成功率，复现官方报告的 96.85% 平均成功率。

### 步骤

#### 1.1 启动策略服务器（终端 1）

```bash
# 使用默认的 pi05_libero 检查点
uv run scripts/serve_policy.py --env LIBERO

# 或者指定自定义检查点
uv run scripts/serve_policy.py --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir gs://openpi-assets/checkpoints/pi05_libero
```

> 服务器启动后会在 `0.0.0.0:8000` 监听 WebSocket 连接。

#### 1.2 运行 LIBERO 评估（终端 2）

依次在 4 个任务套件上运行评估：

```bash
# 确保 LIBERO 环境已激活
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 评估 libero_spatial（预期成功率 ~98.8%）
python examples/libero/main.py --args.task-suite-name libero_spatial

# 评估 libero_object（预期成功率 ~98.2%）
python examples/libero/main.py --args.task-suite-name libero_object

# 评估 libero_goal（预期成功率 ~98.0%）
python examples/libero/main.py --args.task-suite-name libero_goal

# 评估 libero_10（预期成功率 ~92.4%）
python examples/libero/main.py --args.task-suite-name libero_10
```

#### 1.3 使用 Docker 运行（替代方法）

```bash
# libero_spatial
SERVER_ARGS="--env LIBERO" CLIENT_ARGS="--args.task-suite-name libero_spatial" \
    docker compose -f examples/libero/compose.yml up --build

# libero_object
SERVER_ARGS="--env LIBERO" CLIENT_ARGS="--args.task-suite-name libero_object" \
    docker compose -f examples/libero/compose.yml up --build

# libero_goal
SERVER_ARGS="--env LIBERO" CLIENT_ARGS="--args.task-suite-name libero_goal" \
    docker compose -f examples/libero/compose.yml up --build

# libero_10
SERVER_ARGS="--env LIBERO" CLIENT_ARGS="--args.task-suite-name libero_10" \
    docker compose -f examples/libero/compose.yml up --build
```

#### 1.4 输出
- **日志**：终端输出每个 task 和 episode 的成功/失败
- **视频**：保存在 `data/libero/videos/` 目录（rollout 回放）
- **成功率**：每个 task suite 的总成功率

#### 1.5 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--args.task-suite-name` | `libero_spatial` | 任务套件：spatial, object, goal, 10, 90 |
| `--args.num-trials-per-task` | `50` | 每个任务的评估轮次 |
| `--args.seed` | `7` | 随机种子 |
| `--args.video-out-path` | `data/libero/videos` | 视频保存路径 |
| `--args.replan-steps` | `5` | 每隔 N 步重新规划 |

---

## 实验二：pi0_fast_libero 标准评估（LIBERO 仿真环境）

### 目的
在 LIBERO 仿真环境中评估 π₀-FAST 模型的任务成功率，作为自回归模型的基线。

### 步骤

#### 2.1 启动策略服务器（终端 1）

```bash
# 使用 pi0_fast_libero 配置和检查点
uv run scripts/serve_policy.py --env LIBERO \
    policy:checkpoint \
    --policy.config pi0_fast_libero \
    --policy.dir gs://openpi-assets/checkpoints/pi0_fast_base
```

> **注意**：`pi0_fast_libero` 使用 `pi0_fast_base` 检查点作为基础模型。如果你有专门微调的
> `pi0_fast_libero` 检查点，请替换为你的检查点路径。

#### 2.2 运行 LIBERO 评估（终端 2）

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 评估 libero_spatial
python examples/libero/main.py \
    --args.task-suite-name libero_spatial \
    --args.video-out-path data/libero/videos_pi0fast

# 评估 libero_object
python examples/libero/main.py \
    --args.task-suite-name libero_object \
    --args.video-out-path data/libero/videos_pi0fast

# 评估 libero_goal
python examples/libero/main.py \
    --args.task-suite-name libero_goal \
    --args.video-out-path data/libero/videos_pi0fast

# 评估 libero_10
python examples/libero/main.py \
    --args.task-suite-name libero_10 \
    --args.video-out-path data/libero/videos_pi0fast
```

#### 2.3 输出
与实验一相同格式，视频保存在 `data/libero/videos_pi0fast/` 目录。

---

## 实验三：pi05 流匹配 vs pi05+FAST-detokenizer 对齐对比

### 目的
比较 pi05_libero 的两种推理路径：
- **Way1（baseline）**：pi05_libero 原生流匹配推理 → 连续动作轨迹
- **Way2（对齐测试）**：pi05_libero 的 VLM 自回归生成 tokens → FAST detokenizer 解码 → 动作轨迹

### 比较指标
- **Token 级别**：生成 token 数量、token ID 范围、EOS 位置、唯一 token 数
- **轨迹级别**：MSE、最大绝对误差、每步 L2 距离

### 步骤

#### 3.1 使用虚拟观测数据运行（快速验证）

```bash
# 运行对比脚本（使用内置的虚拟观测数据）
uv run scripts/compare_pi05_pi0fast_libero.py

# 可以指定输出目录
uv run scripts/compare_pi05_pi0fast_libero.py --output-dir data/compare_results
```

#### 3.2 使用真实观测数据运行

首先从 LIBERO 环境中采集一个观测并保存：

```python
# save_observation.py — 从 LIBERO 仿真环境中采集一个观测
import math
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import pathlib

def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# 初始化 LIBERO 环境
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
init_states = task_suite.get_task_init_states(0)

task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
env.seed(7)
env.reset()
obs = env.set_init_state(init_states[0])

# 等待物体稳定
for _ in range(10):
    obs, _, _, _ = env.step([0.0]*6 + [-1.0])

# 确保输出目录存在
pathlib.Path("data").mkdir(exist_ok=True)

# 保存观测
element = {
    "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
    "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
    "observation/state": np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    )),
    "prompt": str(task.language),
}
np.save("data/libero_obs_sample.npy", element)
print(f"Saved observation to data/libero_obs_sample.npy")
print(f"Task: {task.language}")
```

然后运行对比：

```bash
uv run scripts/compare_pi05_pi0fast_libero.py \
    --obs-path data/libero_obs_sample.npy \
    --output-dir data/compare_results_real
```

#### 3.3 输出文件

运行后在 `--output-dir` 目录下生成：

```
data/compare_results/
├── comparison_results.json   # 完整的对比指标（JSON 格式）
├── way1_actions.npy          # Way1 (流匹配) 生成的动作轨迹
├── way2_actions.npy          # Way2 (AR+FAST) 生成的动作轨迹
├── way2_tokens.npy           # Way2 生成的 tokens
└── way2_full_tokens.npy      # Way2 完整 token 序列（prefix + generated）
```

#### 3.4 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pi05-config-name` | `pi05_libero` | 模型配置名 |
| `--pi05-checkpoint-dir` | `gs://openpi-assets/checkpoints/pi05_libero` | 检查点路径 |
| `--fast-action-dim` | `7` | FAST detokenizer 动作维度（LIBERO=7） |
| `--fast-action-horizon` | `10` | FAST detokenizer 动作时域 |
| `--fast-max-token-len` | `180` | FAST tokenizer 最大 token 长度 |
| `--ar-max-decode-steps` | `256` | AR 生成最大步数 |
| `--ar-temperature` | `0.0` | 采样温度（0=贪心解码） |
| `--flow-num-steps` | `10` | 流匹配去噪步数 |
| `--obs-path` | `None` | 真实观测文件路径（.npy） |
| `--output-dir` | `data/compare_pi05_pi0fast` | 输出目录 |
| `--seed` | `42` | 随机种子 |

---

## 结果分析

### 实验一和实验二的结果表格

| 模型 | Libero Spatial | Libero Object | Libero Goal | Libero 10 | 平均 |
|------|---------------|---------------|-------------|-----------|------|
| π₀.₅ (预期) | 98.8% | 98.2% | 98.0% | 92.4% | 96.85% |
| π₀-FAST | ? | ? | ? | ? | ? |

### 实验三的分析要点

1. **Token 对比分析**
   - 生成的 token 总数和序列长度
   - Token ID 分布范围
   - EOS token 是否正确生成
   - 与 FAST tokenizer 编码规范的一致性

2. **轨迹对比分析**
   - MSE（均方误差）：衡量整体偏差
   - 最大绝对误差：衡量最差情况
   - 每步 L2 距离：衡量逐帧对齐程度
   - 每维 MSE：找出哪些动作维度差异最大

3. **关键问题**
   - Way2 的 AR 生成是否产生了有效的 FAST tokens？
   - 解码后的轨迹是否在合理范围内？
   - 两种方式的推理时间差异？

### 可视化分析

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# 加载结果
with open("data/compare_results/comparison_results.json") as f:
    results = json.load(f)

way1 = np.array(results["way1_actions"])
way2 = np.array(results["way2_actions"])
metrics = results["metrics"]

# 绘制轨迹对比
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
dim_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
for i, (ax, name) in enumerate(zip(axes.flat, dim_names)):
    min_t = min(len(way1), len(way2))
    ax.plot(way1[:min_t, i], label="Way1 (Flow)", alpha=0.8)
    ax.plot(way2[:min_t, i], label="Way2 (AR+FAST)", alpha=0.8)
    ax.set_title(name)
    ax.legend()
axes.flat[-1].text(0.5, 0.5, f"MSE: {metrics['trajectory']['mse']:.6f}",
                   transform=axes.flat[-1].transAxes, ha='center', fontsize=12)
plt.tight_layout()
plt.savefig("data/compare_results/trajectory_comparison.png", dpi=150)
print("Saved trajectory_comparison.png")
```

---

## 常见问题

### Q: 检查点下载失败怎么办？
检查点从 `gs://openpi-assets` 自动下载。如果网络不通：
- 手动下载后放到 `~/.cache/openpi/` 对应目录
- 或设置 `OPENPI_DATA_HOME` 环境变量指定缓存路径

### Q: LIBERO 仿真器出现 EGL 错误怎么办？
```bash
# 改用 GLX 渲染
MUJOCO_GL=glx python examples/libero/main.py --args.task-suite-name libero_spatial
```

### Q: 推理服务器端口被占用怎么办？
```bash
# 使用不同端口
uv run scripts/serve_policy.py --env LIBERO --port 8001

# 客户端也要指定相同端口
python examples/libero/main.py --args.port 8001
```

### Q: GPU 显存不足怎么办？
- 使用 `bfloat16` 推理（默认已启用）
- 确保同一 GPU 上没有其他程序占用显存
- 对于 A100/H100，单卡足够运行推理

### Q: Docker 容器无法访问 GPU？
```bash
# 确认 nvidia-docker 已安装
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Q: 实验三中 Way2 生成的 tokens 全是 0 或无效怎么办？
这可能说明 pi05 的 PaliGemma backbone 没有被训练来生成 FAST 格式的 action tokens。
这正是实验三需要验证的关键点——pi05 的 VLM 是否能够通过 AR 方式产生可被 FAST detokenizer 解码的有效 tokens。
如果结果不理想，说明两种模型在 token 空间上的对齐存在差距。
