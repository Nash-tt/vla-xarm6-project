# 04 · 训练

**跑在哪：** GPU 机器（单张 RTX 3090 24GB）。
**输入：** [03](03-dataset-tools.md) 产出的裁剪版 LeRobot 数据集。
**产出：** `checkpoints/<step>/pretrained_model/`，由 [05 serve.py](05-inference-server.md) 加载。

三个策略各一份启动脚本，本层没有 Python 代码，全是 shell：

| 脚本 | 策略 | 数据集（实测，取自 checkpoint 里的 `train_config.json`） |
|---|---|---|
| `pi0_ultrasound_lora/run_train.sh` | π0 + LoRA | `vla_batch_20261007_pi0_6d_rollunwrap`（**未裁剪**） |
| `run_train_smolvla_scan.sh` | SmolVLA 全量 | `vla_batch_20261007_smolvla_6d_no_depth_rollunwrap_scan` |
| `run_train_act_scan.sh` | ACT 从零 | `vla_batch_20261007_pi0_6d_rollunwrap_scan` |

> **π0 用的是未裁剪数据集**，另两个用的是裁剪版。
> `docs/03` 第 72 行写 π0 用了 `..._rollunwrap_scan`，与脚本和 checkpoint 记录都不符——
> `run1/checkpoints/012250/pretrained_model/train_config.json` 里
> `dataset.repo_id = local/vla_batch_20261007_pi0_6d_rollunwrap`，
> 且 `dataset.root` 指向 `/data/vla/vla_batch_20261007_pi0_6d_rollunwrap`。
> **以 checkpoint 为准。** 两个数据集都在盘上，别按目录名的相似性猜。

---

## 一、三者的真实差异：一张表

| | π0-LoRA | SmolVLA | ACT |
|---|---|---|---|
| 参数 | 3B 级（LoRA 只训 0.4%） | ~450M | **51.6M** |
| 骨干 | PaliGemma（预训练） | SmolVLM2（预训练） | **resnet18（ImageNet，从头）** |
| 预训练权重 | `lerobot/pi0_base` | `lerobot/smolvla_base` | **无** |
| 微调方式 | LoRA r=16 / α=16 | **全量** | 全量（本来就没东西可冻） |
| `rename_map` | **必须**（见第二节） | 不需要（`{}`） | 不需要 |
| batch_size | 8 | 32 | 24 |
| steps | 24500（实际停在 12250） | 20000 ✅ | 20000（进行中） |
| 用语言 | 否（但模型结构里有） | **是** | 否 |
| 时间集成 | 无 | 无 | **独有** |

**ACT 是三者里最省事的**：不要预训练权重、不要 `rename_map`、不要 PEFT，
而且它是唯一支持时间集成的策略。代价是它能力最弱（51.6M vs 3B）。

---

## 二、`rename_map`：只有 π0 需要的那个补丁

这是三个脚本最容易搞混的一处。

π0 的 `pi0_base` 配置里**把图像槽位名写死了**：`base_0_rgb` / `left_wrist_0_rgb` /
`right_wrist_0_rgb`。而数据集里的键是 `observation.images.rgb` /
`observation.images.ultrasound`。对不上，所以 π0 必须改名：

```json
"rename_map": {
  "observation.images.rgb":       "observation.images.base_0_rgb",
  "observation.images.ultrasound": "observation.images.left_wrist_0_rgb"
}
```

**SmolVLA 和 ACT 不需要。** SmolVLA 实测 `rename_map = {}`；
ACT 是从零训练、特征由数据集 meta 现推的（`factory.py:530-531`），
实测直接得到 `observation.images.rgb` / `observation.images.ultrasound`，
数据集自己的名字就是策略的特征名。

### 2.1 `input_features` / `output_features`：三个策略三种写法

这是比 `rename_map` 更容易踩的一处——**三个脚本对同样两个参数的写法互不相同**：

| 策略 | 写法 | 原因 |
|---|---|---|
| π0 | **不传** | `pi0_base` 的 config 已声明固定槽位，传 `null` 会清掉并报 `All image features are missing from the batch` |
| SmolVLA | **传 `null`** | 特征要从数据集 meta 推导，不传就用不上数据集的特征 |
| ACT | **不传** | `ACTConfig` 两者默认本就是空 dict，`factory.py:531` 只在为空时才用数据集推导；显式传 `null` 反而多一处出错点 |

**别互相照搬。** 三种写法的差异来自三个策略对"特征从哪来"的不同设计，
不是历史遗留的随意。

---

## 三、π0 + LoRA

`pi0_ultrasound_lora/run_train.sh`。做法与理由见
[文档 05](../docs/05-lora-and-rename-map.md)，这里只列实测超参
（读自 `run1/checkpoints/012250/pretrained_model/train_config.json`）：

| 项 | 值 |
|---|---|
| `policy.type` / `path` | `pi0` / `None`（权重由 `WEIGHTS` 环境变量给） |
| `batch_size` | **8** |
| `steps` | 24500（**实际停在 12250**） |
| `num_workers` | 8 |
| `tolerance_s` | 1e-3 |
| 优化器 | AdamW，lr **1e-4**，weight_decay 0.01，grad_clip 1.0，betas [0.9, 0.95] |
| 调度器 | `cosine_decay_with_warmup`，warmup **1000**，decay 30000，peak 1e-4 → 1e-5 |
| PEFT | **LORA**，r=16，alpha=16 |
| 全量训练的模块 | `state_proj`、`action_in_proj`、`action_out_proj`、`action_time_mlp_in`、`action_time_mlp_out` |
| `rename_map` | 见第二节 |

学习率比 π0 全量微调的默认值（2.5e-5）高 4 倍，是 LoRA 的惯例
（可训练参数少，需要更大的步长）；`full_training_modules` 列出的五项是**投影层**——
它们必须全量训练，否则状态与动作的投影矩阵冻在预训练分布上，LoRA 的秩补不回来。

**run1 没跑完 24500，停在 12250**（`checkpoints/` 最后一个目录是 `012250`，
之后是 `last`）。部署时加载的是 `run1/checkpoints/012250/pretrained_model`
——一个中途存档，不是收尾存档。这一点在 [05](05-inference-server.md) 里会再提。

---

## 四、SmolVLA：全量微调，已跑完 20k

`run_train_smolvla_scan.sh`。实测超参（读自
`smolvla_ultrasound_scan_20k/checkpoints/020000/pretrained_model/train_config.json`）：

| 项 | 值 |
|---|---|
| `policy.type` | `smolvla` |
| `load_vlm_weights` | **true**（从 SmolVLM2 视觉语言塔加载） |
| `batch_size` | **32** |
| `steps` | 20000 ✅（checkpoint 到 `020000`） |
| 优化器 | AdamW，lr **1e-4**，weight_decay **1e-10**，grad_clip **10.0** |
| 调度器 | cosine + warmup 1000 / decay 30000，peak 1e-4 → **2.5e-6** |
| PEFT | **null**（不做 LoRA） |
| `rename_map` | **`{}`** |

weight_decay 是 1e-10（几乎为零）、grad_clip 是 10.0（很宽）——**这两个值是 SmolVLA
上游配置带来的，不是本站点调的**。全量微调 450M 参数配 1e-4 是常规量级。

它是三者中**唯一用语言的**，所以数据集里的 task 字符串对它有意义；
另两个策略里那串 214 字符的指令等于闲置。

> 盘上另有一个 `smolvla_ultrasound_20k_tol1ms` 目录，checkpoint 同样到 `020000`。
> 从名字看是用 `tolerance_s=1e-3` 重跑的一份对照。选用哪一份要做实验记录里的决定，
> 不要按目录名猜。

---

## 五、ACT：从零训练

`run_train_act_scan.sh`（2026-10-12 新增）。ACT 与另两个的根本差别是**没有预训练权重可用**：

- Hub 上的 `lerobot/act_*` 都是 **ALOHA 特定机器人**的（14 维双臂关节），
  套不到 6 维 TCP 位姿控制上
- 所以 `--policy.type=act` 而**不传 `--policy.path`**
- resnet18 的 ImageNet 骨干权重由 `torchvision` 自动下载，缓存在
  `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。
  脚本会 `unset` 代理并置 `HF_HUB_OFFLINE=1`，所以**首次运行前该文件必须已在缓存里**，
  否则加载骨干时失败。这是这个脚本唯一的外部依赖，容易被忽略。

超参：

| 项 | 值 | 说明 |
|---|---|---|
| 参数量 | 51.6M | resnet18 骨干 + transformer |
| `chunk_size` | **50** | 与真机客户端的 `--actions-per-chunk 50` 对齐 |
| `n_action_steps` | **50** | 必须 ≤ chunk_size（`configuration_act.py:143` 会校验）；取满 = 不做时间集成 |
| `batch_size` | **24** | 实测上限，见下 |
| `steps` | 20000 | |
| lr / wd / lr_backbone | **1e-5 / 1e-4 / 1e-5** | 全部取 `ACTConfig` 默认，未调 |
| `use_amp` | **False** | 见下 |
| `rename_map` | 不传 | 不需要 |

### 5.1 batch_size=24 是实测出来的上限

在原生 720×1280 输入下（**不是**先缩到 224）：

| batch | 显存 | 步时 |
|---|---|---|
| 16 | 11.3 GiB | 389 ms |
| **24** | **16.8 GiB** | **577 ms** |
| 32 | **OOM** | — |

显存几乎全在影像塔上——它把 720×1280 切成 **23×40 = 920 个 token**。
`chunk_size` 从 50 加到 100 只多占 0.4 GiB，说明 chunk 维度不是瓶颈。

实测训练速度 **~1.9 step/s**（`nvidia-smi` 显示 19778 MiB / 97%），
20000 步约 **2.9 小时**。

### 5.2 为什么不开 AMP

`ACTConfig` 默认 `use_amp=False`。开着大概率能省掉近一半步时，
但 **VAE 分支的 log-prob 项在 AMP 下容易出 NaN**。要开请先冒烟确认 loss 不发散。

### 5.3 时间集成：ACT 独有，但默认没开

ACT 是这套代码里**唯一**支持 `temporal_ensemble_coeff` 的策略。
要开的话：

```bash
--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
```

`n_action_steps` 必须为 1（`configuration_act.py:138` 会校验），这是集成的前提——
每个时刻都重新推理一次，再把多个 chunk 按指数权重融合。

**当前默认整块开环执行**（`n_action_steps = chunk_size = 50`），与真机客户端
`--actions-per-chunk 50` 的行为一致。要拿到集成的收益，客户端也得改成每步都请求。
这属于尚未实现的部分。

---

## 六、三个脚本共用的部分

三份脚本的骨架完全一样，改动任何一份时另两份要跟着看：

```bash
set -euo pipefail
LEROBOT_BIN="${LEROBOT_BIN:-/opt/conda/envs/lerobot/bin/lerobot-train}"   # 绝对路径
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # ALL_PROXY 的 socks:// 会让 httpx 报错
export HF_HUB_OFFLINE=1
```

**`LEROBOT_BIN` 用绝对路径而不是"当前激活的环境"**，是 2026-10-11 那次环境事故的产物：
装 openpi 时把依赖灌进了 lerobot 环境，毁了 torch / transformers / lerobot 三件套。
现在脚本不依赖激活状态，环境坏了会**在开跑前**报一句清楚的错，而不是训练到一半崩。
（详见[文档 08](../docs/08-openpi-compat.md)。）

每个脚本还各有一张前置检查：权重目录存在、数据集目录存在、`lerobot-train` 可执行。

### 6.1 `tolerance_s=1e-3`

三个脚本都传 `--tolerance_s=1e-3`，理由不是数据脏，是 **LeRobot 的 float32 边界 bug**：

`decode_video_frames_pyav` 里 `torch.tensor(timestamps)` 把 float64 降成 float32，
而 float32 在 t≈1277 s 处的 ULP = **1.22e-4 > 默认的 1e-4**。
裁剪后的数据集实测最大偏差 **3.052e-05**（0/27312 帧超 1e-4），
所以默认值其实过得去；保留 1e-3 是留 33 倍余量，**零代价**——
容差只影响校验，不影响选帧（解码器总是取最近帧）。

见[文档 04](../docs/04-timestamp-tolerance.md)。

### 6.2 SMOKE 模式

三份脚本都支持 `SMOKE=1 ./run_train_*.sh`，把步数、batch、worker 压到最小，
只验证「显存 + 流程」跑得通。**换环境、换数据集、改配置后先冒烟**，
别直接起 20k 的正式跑——上面 ACT 那节的 batch_size 上限就是这么试出来的。

---

## 七、跑起来与看进度

```bash
cd <仓库根目录>
nohup setsid ./run_train_act_scan.sh > act_train_20k.log 2>&1 < /dev/null &
```

`setsid` 让进程脱离终端会话，断开 SSH 也不会被杀。日志里 LeRobot 用 `\r` 刷新进度条，
读的时候先 `tr '\r' '\n'`：

```bash
tr '\r' '\n' < act_train_20k.log | tail -3
```

判完成的标志是日志里出现 `End of training`。checkpoint 每 `save_freq` 步落一个
`checkpoints/<step>/`，每个约 591 MB（ACT）。

产物目录结构（三个策略一致）：

```text
<OUTPUT_DIR>/
├── checkpoints/
│   ├── 002000/  ...  020000/     # 每 save_freq 步一个
│   │   ├── pretrained_model/     #   ← 部署时加载这个
│   │   └── training_state/       #   ← 断点续训用
│   └── last -> 最新一个
└── ...
```

**部署时加载的是 `pretrained_model/`，不是 `last/` 的指针。**
服务端加载哪个 checkpoint 由 `serve.py` 的默认路径或 `--checkpoint` 决定，
见 [05](05-inference-server.md)。
