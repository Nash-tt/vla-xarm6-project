# xArm6 超声扫描 VLA 微调

在自采的 xArm6 超声扫描数据集上微调 **π0** 与 **SmolVLA** 两类 VLA 模型，训练框架为
**LeRobot**；同时构建一条「推理与执行分离」的部署链路：GPU 机器仅承担策略推理，
工控机负责驱动机械臂。

本仓库是项目实施过程中的**问题与解决方案记录**及配套工具，不构成通用软件库。
`docs/` 下每篇文档对应一个曾导致训练失败或结果偏差的具体问题，
统一按「现象 → 原因 → 处理 → 验证 → 结论」组织。

---

## 一、任务与硬件

| 项目 | 内容 |
|---|---|
| 机械臂 | xArm6，TCP 六维位姿控制（`xarm6_ultrasound_tcp_control`） |
| 任务 | 超声探头沿体表扫描，输出连续六维动作 |
| 数据 | LeRobot **v3.0** 格式，15 FPS，28 个回合，裁剪后 27312 帧 |
| 相机 | 单路 RGB（`observation.images.rgb`）；另有深度流，未启用 |
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
│   ├── run_train.sh             # π0 + LoRA 微调
│   ├── replay_client.py         # 本机回放测试客户端
│   ├── remote/                  # 部署于工控机的轻量客户端
│   │   ├── remote_client.py
│   │   ├── netprobe.py          # 链路带宽与延迟探测
│   │   └── services_pb2*.py     # proto 生成的 gRPC 桩（免装 grpcio-tools）
│   └── measure_inference.py     # 推理耗时测量
├── run_train_smolvla_scan.sh    # SmolVLA 微调（扫描段数据集）
├── plot_pred_vs_demo.py         # 预测与示范对比图
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

**尚未完成**（详见[文档 06](docs/06-remote-client.md) 第七节）：

- xArm6 的 `Robot` 实现尚未接入，当前以 `ReplayRobot` 做回放测试
- 传输需由 `--wire numpy` 切换至 `--wire torch`，方可支持全分辨率图像
- 时间集成（temporal ensembling）尚未实现，官方默认的 `weighted_average` 未做成可配置项
- 两处上游缺陷尚未提交 issue

---

## 六、许可

代码供参考。数据集与模型权重不在本仓库内。
