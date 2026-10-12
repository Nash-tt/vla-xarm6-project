# xArm6 超声扫描 VLA 微调

在自采的 xArm6 超声扫描数据集上微调 **π0** 与 **SmolVLA** 两类 VLA 模型，训练框架为
**LeRobot**；同时构建一条「推理与执行分离」的部署链路：GPU 机器仅承担策略推理，
工控机负责驱动机械臂。

本仓库是项目实施过程中的**问题与解决方案记录**及配套工具，不构成通用软件库。

文档分两套，分工明确：

| 目录 | 回答的问题 | 组织方式 |
|---|---|---|
| [`docs/`](docs/) | **当初为什么这么改**——每个问题按「现象 → 原因 → 处理 → 验证 → 结论」记一遍 | 按时间顺序，一篇一个问题 |
| [`design/`](design/) | **现在这套东西是怎么搭的**——分几块、每块之间传什么、关键参数为什么取这个值 | 按层次，一篇一个环节 |

**查实现细节看 `design/`，查历史决策看 `docs/`。**
`docs/` 是按时间顺序写的，早期文档描述的实现**可能已经不是现在的实现**；
两者冲突时以 `design/` 与代码为准。

---

## 一、任务与硬件

| 项目 | 内容 |
|---|---|
| 机械臂 | xArm6，TCP 六维位姿控制（`xarm6_ultrasound_tcp_control`） |
| 任务 | 超声探头沿体表扫描，输出连续六维动作 |
| 数据 | LeRobot **v3.0** 格式，15 FPS，28 个回合，裁剪后 27312 帧 |
| 相机 | 两路：RealSense 彩色（`observation.images.rgb`，720×1280）与超声解码图（`observation.images.ultrasound`，256×256）；深度流存在于采集端，未写入数据集 |
| 训练机 | 单张 RTX 3090 24GB |
| 部署 | GPU 机器作为 policy server；Win10 工控机（无独显）作为客户端 |

**推理与执行分离的原因**：工控机无独立显卡，无法承载 π0/SmolVLA 的推理开销；
而机械臂控制程序与相机均位于工控机。因此采用远程推理方案——
工控机发送观测，GPU 机器返回动作块（action chunk），工控机按块开环回放。

---

## 二、目录结构

```
.
├── tools/                       # 数据集处理工具（与训练框架解耦）
│   ├── trim_scan_segments.py    # 裁除接近段，仅保留扫描段
│   ├── recompute_stats_local.py # 重算 meta/stats.json（必须指定 --donor）
│   └── verify_trim.py           # 裁剪一致性校验（T1–T10）
├── pi0_ultrasound_lora/
│   ├── serve.py                 # policy server（含两处上游缺陷的修复）
│   ├── run_train.sh             # π0 + LoRA 微调（未裁剪数据集，即 run1）
│   ├── run_train_scan.sh        # π0 + LoRA 微调（裁剪数据集，即 run2）
│   ├── replay_client.py         # 本机回放测试客户端
│   ├── remote/                  # 工控机侧的轻量客户端；仓库里唯一的传输层
│   │   ├── remote_client.py     # 真机客户端也 import 这一份
│   │   ├── netprobe.py          # 链路带宽与延迟探测
│   │   ├── test_remote_client_timing.py
│   │   └── services_pb2*.py     # proto 生成的 gRPC 桩（免装 grpcio-tools）
│   └── measure_inference.py     # 推理耗时测量
├── run_train_smolvla_scan.sh    # SmolVLA 微调（smolvla 版扫描段数据集）
├── run_train_smolvla_pi0scan.sh # SmolVLA 微调（pi0 版扫描段数据集，指令措辞对照）
├── run_train_act_scan.sh        # ACT 微调（从零训练，chunk_size=50）
├── plot_pred_vs_demo.py         # 预测与示范对比图
├── robot/                       # 工控机侧代码副本（见 robot/README.md）
│   ├── my_code/                 # 示教点采集 → 轨迹生成 → 三路同步采集 → 对齐 → 建集
│   ├── realsense_collector/     # 独立 RealSense RGB-D 采集器
│   └── real_robot/              # 真机远程策略客户端（默认影子模式；不含传输层）
├── design/                      # 实现方案，见 design/README.md
└── docs/                        # 问题记录，见下表
```

---

## 三、文档索引

按问题在流程中出现的顺序排列，建议顺序阅读。

| 编号 | 文档 | 摘要 |
|---|---|---|
| 01 | [数据集裁剪：接近段](docs/01-dataset-trim.md) | 前 44% 的轨迹在教模型输出远端目标点，必须裁除 |
| 02 | [stats.json：崩溃与精度](docs/02-stats-json.md) | 重算统计量丢键导致 `KeyError`；以及 float32 累加器引入的 3.5% 偏差 |
| 03 | [roll 角度表示](docs/03-roll-angle.md) | 同批轨迹的 wrap 版使 roll 的 std 放大 22 倍，两版现已统一为展开口径 |
| 04 | [时间戳精度与 tolerance_s](docs/04-timestamp-tolerance.md) | float32 时间戳的 ULP 超过默认容差，导致视频解码失败 |
| 05 | [LoRA 加载与 rename_map](docs/05-lora-and-rename-map.md) | `from_pretrained` 对 LoRA 权重静默返回随机初始化 |
| 06 | [远程轻量客户端](docs/06-remote-client.md) | 单帧传输量由 2.96MB 降至 0.30MB；及一处导致观测静默丢弃的缺陷 |
| 07 | [环境与工具链](docs/07-environment.md) | conda 环境污染、torchcodec 失效、代理与镜像配置 |
| 08 | [openpi 兼容性评估](docs/08-openpi-compat.md) | 改用 openpi 时，现有客户端与服务端的可复用性 |
| 09 | [两端接口契约](docs/09-interface-contract.md) | 数据格式、维序、相机键名映射、roll 口径与安全阈值 |
| 10 | [两台机器的协作方式](docs/10-collaboration.md) | 代码交换通道、提交前消毒清单、绝不入库的项 |

### 实现方案（`design/`）

按链路层次组织，描述**代码现在长什么样、为什么这么设计**。索引见
[`design/README.md`](design/README.md)。

| 篇 | 覆盖 |
|---|---|
| [01 采集链路](design/01-collection.md) | 示教点 → 轨迹 → 三路同步采集 |
| [02 对齐与建集](design/02-align-and-build.md) | 时钟映射、四路取法、`quality.valid`、超声解码 |
| [03 数据集工具](design/03-dataset-tools.md) | 裁剪判据、为何不重编码、`--unwrap-roll`、`--donor` |
| [04 训练](design/04-training.md) | π0-LoRA / SmolVLA / ACT 三条链路与它们的差异 |
| [05 推理服务端](design/05-inference-server.md) | `serve.py` 的覆写与对上游两处缺陷的修复 |
| [06 传输层](design/06-transport-layer.md) | 瘦客户端、pickle 垫片、三个同步机制 |
| [07 真机客户端](design/07-real-robot-client.md) | 三道安全层、三路数据源、控制环 |

---

## 四、复现路径

### 0. 前置条件

需要一份 LeRobot v3.0 格式的自采数据集，以及可运行 `lerobot-train` 的环境。
本仓库脚本均通过环境变量定位资源，默认值仅为占位，**需按实际路径覆盖**：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `LEROBOT_BIN` | `lerobot-train` 可执行文件 | `/opt/conda/envs/lerobot/bin/lerobot-train` |
| `LEROBOT_SRC` | lerobot 源码树（经 pip 安装时可忽略） | `/opt/lerobot/src` |
| `DATASET_ROOT` | 数据集根目录 | `/data/vla/<数据集名>` |
| `WEIGHTS` | 预训练权重（HF repo id 或本地 snapshot 目录） | `lerobot/pi0_base` |

> 仓库中的 IP `100.64.0.1` / `100.64.0.2` 与主机名 `gpu-server` / `robot-pc` 均为**占位示例**，
> 需替换为实际的 Tailscale 地址与主机名。

### 1. 裁除接近段

```bash
python tools/trim_scan_segments.py --src /data/vla/<原数据集> --dst /data/vla/<裁剪版>
```

### 2. 重算统计量（**必须指定 `--donor`**）

```bash
python tools/recompute_stats_local.py \
    --root  /data/vla/<裁剪版> \
    --donor /data/vla/<原数据集>
```

省略 `--donor` 将在训练启动时抛出 `KeyError: 'observation.images.rgb'`，
原因见[文档 02](docs/02-stats-json.md)。

### 3. 校验

```bash
python tools/verify_trim.py --src /data/vla/<原数据集> --dst /data/vla/<裁剪版>
```

### 4. 训练

```bash
DATASET_ROOT=/data/vla/<裁剪版> ./run_train_smolvla_scan.sh
```

---

## 五、当前状态

**已完成并经验证：**

- 数据裁剪链路（仅修改元数据、不重编码；逐像素验证 168/168 一致）
- 统计量重算与完整性校验（`verify_trim.py` 的 T10）
- π0-LoRA 与 SmolVLA 两条训练链路在 RTX 3090 上跑通
- 远程推理链路：轻量客户端可连接 policy server 并接收动作块

**工控机侧独立完成**：

- 三路同步采集链路：xArm 状态与下发目标、RealSense RGB-D、PhaseLink 超声原始包
  ——脱敏副本已收入 [`robot/my_code/`](robot/README.md)（现行采集链路的 13 个模块 + 对应测试）
- 真机远程策略客户端：默认影子模式，具备力/力矩上限、单步限幅、跟踪误差约束与二次确认
  ——脱敏副本已收入 [`robot/real_robot/`](robot/README.md#三real_robot)；
  入库前的安全阈值逐行审阅已完成，结论与依据记在该节 3.1

**尚未完成**（详见[文档 06](docs/06-remote-client.md) 第七节）：

- 传输需由 `--wire numpy` 切换至 `--wire torch`，方可支持全分辨率图像
  （当前真机客户端仍走 224×224 预缩路径）
- 时间集成（temporal ensembling）尚未实现，官方默认的 `weighted_average` 未做成可配置项
- 两处上游缺陷尚未提交 issue

---

## 六、许可

代码供参考。数据集与模型权重不在本仓库内。
