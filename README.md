# xArm6 超声扫描 VLA 微调

在自采的 xArm6 超声扫描数据集上微调 **π0** 与 **SmolVLA** 两个 VLA 模型（框架用 **LeRobot**），
并搭建一条「**GPU 机器只跑推理、工控机驱动机械臂**」的远程部署链路。

本仓库不是一份成品库，而是这个项目**踩过的坑 + 配套工具**的沉淀。
每个 `docs/` 下的文档都对应一个真实把训练卡死或把结果带偏的问题，
写法统一为：**现象 → 根因 → 解决 → 验证 → 教训**。

---

## 一、任务与硬件

| 项 | 内容 |
|---|---|
| 机械臂 | xArm6，TCP 6 维位姿控制（`xarm6_ultrasound_tcp_control`） |
| 任务 | 超声探头沿体表扫描，输出连续 6 维动作 |
| 数据 | LeRobot **v3.0** 格式，15 FPS，28 个回合，裁剪后 27312 帧 |
| 相机 | 单路 RGB（`observation.images.rgb`），另有深度流未启用 |
| 训练机 | 单张 RTX 3090 24GB |
| 部署 | GPU 机器当 policy server；Win10 工控机（无独显）跑客户端驱动机械臂 |

**为什么要把推理和机械臂分开**：工控机没有独显，跑不动 π0/SmolVLA；
而机械臂的控制程序和相机都在工控机上。于是走「远程推理」——
工控机把观测发给 GPU 机器，GPU 机器返回动作块（action chunk），工控机按块开环回放。

---

## 二、目录结构

```
.
├── tools/                      # 数据集处理工具（与训练框架解耦）
│   ├── trim_scan_segments.py   # 裁掉接近段，只留扫描段
│   ├── recompute_stats_local.py# 重算 meta/stats.json（必须带 --donor）
│   └── verify_trim.py          # 裁剪后一致性校验（T1~T10）
├── pi0_ultrasound_lora/
│   ├── serve.py                # policy server（含两处上游 bug 的修复）
│   ├── run_train.sh            # π0 + LoRA 微调
│   ├── replay_client.py        # 本机回放测试客户端
│   ├── remote/                 # 部署到工控机的瘦客户端
│   │   ├── remote_client.py
│   │   ├── netprobe.py         # 链路带宽/延迟探测
│   │   └── services_pb2*.py    # 由 proto 生成的 gRPC 桩（免装 grpcio-tools）
│   └── measure_inference.py    # 推理耗时测量
├── run_train_smolvla_scan.sh   # SmolVLA 微调（扫描段数据集）
├── plot_pred_vs_demo.py        # 预测 vs 示范 对比图
└── docs/                       # 踩坑记录，见下表
```

---

## 三、文档索引

按「问题出现在流程的哪一步」排序，建议顺序阅读。

| # | 文档 | 一句话 |
|---|---|---|
| 01 | [数据集裁剪：接近段](docs/01-dataset-trim.md) | 前 44% 的轨迹在教模型输出远处的点，必须裁掉 |
| 02 | [stats.json：崩溃与精度](docs/02-stats-json.md) | 重算 stats 丢键 → `KeyError` 崩溃；以及 float32 累加器带来的 3.5% 误差 |
| 03 | [roll 角度约定](docs/03-roll-angle.md) | π0 要展开、SmolVLA 要 wrap，选错会让归一化尺度放大 22 倍 |
| 04 | [时间戳精度与 tolerance_s](docs/04-timestamp-tolerance.md) | float32 时间戳的 ULP 超过默认容差，视频解码直接失败 |
| 05 | [LoRA 静默加载与 rename_map](docs/05-lora-and-rename-map.md) | `from_pretrained` 对 LoRA 权重会静默返回随机初始化 |
| 06 | [远程瘦客户端](docs/06-remote-client.md) | 把每帧 2.96MB 压到 0.30MB；以及一个会让观测被静默丢弃的坑 |
| 07 | [环境与工具链](docs/07-environment.md) | conda 环境污染、torchcodec 坏掉、代理与镜像的正确组合 |
| 08 | [openpi 兼容性评估](docs/08-openpi-compat.md) | 换 openpi 的话，现有客户端/服务端还能不能用 |

---

## 四、复现路径

### 0. 前置

需要一份 LeRobot v3.0 格式的自采数据集，以及一份能跑 `lerobot-train` 的环境。
本仓库的脚本都通过环境变量定位资源，默认值只是占位，**请按自己的路径覆盖**：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `LEROBOT_BIN` | `lerobot-train` 可执行文件 | `/opt/conda/envs/lerobot/bin/lerobot-train` |
| `LEROBOT_SRC` | lerobot 源码树（用 pip 安装时可不管） | `/opt/lerobot/src` |
| `DATASET_ROOT` | 数据集根目录 | `/data/vla/<数据集名>` |
| `WEIGHTS` | 预训练权重（HF repo id 或本地 snapshot 目录） | `lerobot/pi0_base` |

> 仓库里的 IP `100.64.0.1` / `100.64.0.2` 与主机名 `gpu-server` / `robot-pc` 都是**占位示例**，
> 请替换成自己的 Tailscale 网段地址与主机名。

### 1. 裁掉接近段

```bash
python tools/trim_scan_segments.py --src /data/vla/<原数据集> --dst /data/vla/<裁剪版>
```

### 2. 重算统计量（**必须带 `--donor`**）

```bash
python tools/recompute_stats_local.py \
    --root  /data/vla/<裁剪版> \
    --donor /data/vla/<原数据集>
```

漏掉 `--donor` 会在训练启动时抛 `KeyError: 'observation.images.rgb'`，原因见 [文档 02](docs/02-stats-json.md)。

### 3. 校验

```bash
python tools/verify_trim.py --src /data/vla/<原数据集> --dst /data/vla/<裁剪版>
```

### 4. 训练

```bash
DATASET_ROOT=/data/vla/<你的裁剪版> ./run_train_smolvla_scan.sh
```

---

## 五、诚实的现状

**已经跑通并验证的：**

- 数据裁剪链路（只改元数据不重编码，逐像素验证过 168/168）
- stats 重算 + 完整性校验（`verify_trim.py` 的 T10）
- π0-LoRA 与 SmolVLA 两条训练链路在 3090 上跑通
- 远程推理：瘦客户端能连上 policy server 并收到动作块

**还没做完的（见文档 06 第七节）：**

- xArm6 的 `Robot` 实现尚未接入（目前用 `ReplayRobot` 回放测试）
- 传输从 `--wire numpy` 切到 `--wire torch` 才能上全分辨率图像
- 时间集成（temporal ensembling）未实现，官方默认 `weighted_average` 尚未做成可配置
- 两个上游 bug 尚未提交 issue

---

## 六、许可

代码供参考。数据集与模型权重不在此仓库内。
