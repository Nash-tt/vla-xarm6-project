# π0 (LoRA) 微调 — xArm6 超声扫描

单卡 RTX 3090 (24GB) 上微调 `lerobot/pi0_base`。

## 为什么是 LoRA

π0 是 3B 参数的 VLA。官方显存要求：推理 >8GB / **LoRA 微调 >22.5GB** / 全量微调 >70GB。
24GB 单卡只够 LoRA：paligemma(2B) 全冻结，只对 action expert 的 `q_proj`/`v_proj`
挂 adapter，权重以 bf16 加载后约 6GB，激活靠 gradient checkpointing 压。

## 文件

| 文件 | 作用 |
|---|---|
| `run_train.sh` | 训练入口。`SMOKE=1 ./run_train.sh` 跑冒烟测试（20 步） |
| `unwrap_roll.py` | 修数据：解卷绕 roll 角的 ±180 分支切割 |
| `recompute_stats.py` | 在全量数据上用 float64 重算 state/action 统计量 |
| `check_dataset.py` | 验证数据集可加载、视频可解码 |
| `verify_dataset.py` | 逐维对比实测分布 vs stats.json |
| `shorten_task.py` | 把 214 字符的任务描述换成 11 token 的短句（两处存储都改） |
| `fetch_tokenizer.py` | 预下载 PaliGemma tokenizer，让训练可全程离线 |
| `check_timestamps.py` | 扫全量数据集，量化 parquet 时间戳 vs 视频 PTS 的偏差 |
| `measure_inference.py` | 离线测推理显存/延迟/动作 MAE + 消融 + pi0_base 对照 |
| `serve.py` | **远程推理服务端**（gRPC），修掉上游加载 LoRA 的静默 bug |
| `replay_client.py` | **回放客户端**：纯软件验证 wire 契约，不碰硬件 |
| `make_replay_bundle.py` | 把数据集一小段导成自包含 `.npz`（约 0.19MB/帧），供**另一台主机**发过来 |
| `remote/` | **瘦客户端**：对面只要 `grpcio + numpy`，不需要 torch / lerobot / 数据集 |

## 数据

`/data/vla/vla_batch_20261007_pi0_6d_rollunwrap`
（由 `vla_batch_20261007_smolvla_6d_no_depth` 复制后修正，原始数据集未改动）

28 episodes / 49053 帧 / 15fps / 单任务。双相机 → pi0 图像槽：

| 数据集 | pi0 槽位 |
|---|---|
| `observation.images.rgb` (720×1280) | `base_0_rgb` |
| `observation.images.ultrasound` (256×256) | `left_wrist_0_rgb` |
| — | `right_wrist_0_rgb`（`--policy.empty_cameras=1` 生成的 mask=0 空图） |

pi0_base 预训练用 3 个图像槽，补一个空槽能让前缀布局与预训练完全一致
（openpi 处理少相机机器人的做法）。

### roll 角修复

原始数据里 TCP roll 真实姿态集中在 +165°..+193°，但采集跨越了 ±180° 欧拉角
分支切割，同一物理姿态被记成 +178 或 −178 两个符号（`[-180,-90)` 有 21474 帧，
`(90,180]` 有 27579 帧，**中间一帧都没有**）。

MEAN_STD 归一化下这直接把该维度废掉：

```
roll 原始:    mean=17.79  std=172.81   range [-180.00, 180.00]
roll 解卷绕:  mean=175.39 std=  7.73   range [ 163.97, 193.04]
```

修法是负角 +360（`unwrap_roll.py`），state/action 一致处理，再重算 stats。
除 roll 外所有维度的统计量与原值逐位一致，说明改动是精准的。

## 四个容易踩的坑

1. **代理**：环境里 `ALL_PROXY=socks://127.0.0.1:7890/` 的 scheme 前缀 httpx 不认，
   会让 HF 请求直接报 `Unknown scheme for proxy URL`。
   注意**只需删 `ALL_PROXY`，`HTTP(S)_PROXY=http://127.0.0.1:7890` 必须保留**——
   本机直连 HF 是不通的，把代理全 unset 会退化成静默卡死（比报错更难查）。
2. **hf-xet**：`snapshot_download` 默认走 xet 协议，在本机代理下会静默卡死
   （`.incomplete` 分片 0 字节不动）。加 `HF_HUB_DISABLE_XET=1` 回落普通 HTTP 即可，
   实测 ~2MB/s。

3. **别用 `--policy.input_features=null` 覆盖特征表**。`make_policy` 里是
   `if not cfg.input_features: cfg.input_features = {从数据集推导}`。pi0_base 的
   `config.json` 本来就声明好了 3 个图像槽 + `state(32)` + `action(32)`，一旦传 null
   把它清空，数据集的名字（`observation.images.rgb` 等）就会盖掉 pi0 的槽位名。
   结果是 `rename_map` 把 batch 改成了 `base_0_rgb`，而 policy 还在找 `rgb`，直接报：

   ```
   ValueError: All image features are missing from the batch.
   ```

   同理 `--policy.empty_cameras=1` 也是多余的：`_preprocess_images` 会把 batch 中
   缺失的图像键自动补成 mask=0 的占位图（SigLIP 用 -1 填充），`right_wrist_0_rgb`
   天然被 mask。state/action 的 6 维由模型内部 pad 到 `max_state_dim=32`。

4. **`tolerance_s` 默认值 1e-4 在长视频上必然翻车**（LeRobot 的 float32 边界 bug）。
   第一次正式训练在 step 455 崩了：

   ```
   FrameTimestampError: One or several query timestamps unexpectedly violate the
   tolerance (tensor([0.0001]) > tolerance_s=0.0001)
   video: videos/observation.images.ultrasound/chunk-000/file-000.mp4
   ```

   **数据本身没问题**——`check_timestamps.py` 扫全量 49053 帧，真实偏差最大
   只有 7.1e-6 s。问题出在精度：`decode_video_frames_pyav` 里
   `torch.tensor(timestamps)` 把 float64 时间戳降成 **float32**，而 float32 在
   t≈1277s 处的 ULP 正好是 **1.22e-4 > 1e-4**。两个本该相差 5e-6 的数被量化到
   相邻浮点格点，算出来恰好是 1 个 ULP；判定又是严格小于
   （`is_within_tol = min_ < tolerance_s`），于是卡死。

   全量只有 **7 帧**踩中（0.014%），全部落在那条 **3270s 长的超声视频**上。
   RGB 拆成 10 个文件、每个最长 366s，ULP 只有 3.05e-5，所以从不触发——
   也就是说：**任何超过 ~14 分钟（float32 ULP 达到 1e-4 的时长）的拼接视频
   都会踩这个坑**，跟数据质量无关。

   修法是 `run_train.sh` 里加 `--tolerance_s=1e-3`（TrainPipelineConfig 的顶层
   参数，不是 `--dataset.tolerance_s`）。1e-3 是实测最大偏差的 8 倍，又只占一帧
   间隔 66.7ms 的 1.5%，不可能选错帧。上游更干净的修法是把 `decode_video_frames`
   里那个 `torch.tensor(timestamps)` 显式指定 `dtype=torch.float64`。

## 依赖

- conda env `lerobot`（torch 2.11.0+cu130, transformers 5.5.4, numpy 2.2.6）
- `peft` —— lerobot 的 PEFT 支持需要它，通过 `lerobot[peft]` extra 装
- `grpcio>=1.73.1,<2.0.0` —— **远程部署**（`serve.py`/`replay_client.py`）需要，
  来自 `lerobot[async]` extra。重建环境时用的
  `.[training,pi,peft,evaluation,viz]` **不含它**，要单独补：
  `pip install "grpcio>=1.73.1,<2.0.0"`（只加这一个包，不动 torch/transformers）
- **另一台主机**（跑 `remote/`）：只要 `grpcio>=1.73.1` + `protobuf>=6.31.0` + `numpy`，
  见 `remote/requirements.txt`。**不需要 torch / lerobot / CUDA / 数据集**。
  版本下限不是随手写的：生成的 `services_pb2_grpc.py` 写死
  `GRPC_GENERATED_VERSION='1.73.1'` 会主动拒绝更旧的 grpc，`services_pb2.py` 调
  `ValidateProtobufRuntimeVersion(6,31,0)` 会拒绝更旧的 protobuf。需要 Python ≥ 3.9。

### ⚠️ 2026-10-11 环境事故与重建

当天 11:12–11:22 在 openpi 目录下执行 `uv pip install -e .` 时，conda 的 `lerobot`
环境处于激活状态。`uv pip` 的环境发现顺序是：

```
--python  →  VIRTUAL_ENV  →  CONDA_PREFIX  →  .venv  →  系统 Python
```

**`CONDA_PREFIX` 排在项目 `.venv` 之前**（openpi 自己的 `.venv` 在 11:10 已经建好，
完全正常，但根本没进候选）。于是 openpi 的依赖全灌进了这个 conda 环境：
torch 2.11.0+cu130→2.7.1+cu126、transformers 5.5.4→4.53.2、numpy 2.2.6→1.26.4，
并把 editable 的 lerobot 换成了 GitHub commit `0cf8648` 的**非 editable 副本**——
那个 commit 没有 `policies/` 目录，`import lerobot.policies` 直接 ModuleNotFoundError。
（openpi 的 `.venv` 本身没受损，它自带 python 3.11 + torch 2.7.1+cu126 + jax 0.5.3。）

**重建方式**（新 conda 环境 + 当前 pyproject）：

```bash
pip install -e ".[training,pi,peft,evaluation,viz]" transformers==5.5.4
```

**不要用 `requirements-ubuntu.txt`**。那份 lock 生成于更早的提交，已自相矛盾：
`gym-aloha==0.1.3` vs extra 现在要 `>=0.1.4`；`transformers==5.3.0` vs 要 `>=5.4.0`；
`accelerate==1.13.0` vs 要 `>=1.14.0`；且完全没收录 `python-can`。`pip install -r`
必然 ResolutionImpossible。按当前 pyproject 范围解出来的版本反而**正好等于原训练环境**。

**防呆**：跑 openpi 一律先 `conda deactivate`，或显式
`uv pip install --python openpi-main/.venv/bin/python`。
`UV_PROJECT_ENVIRONMENT` 只对 `uv run`/`uv sync` 有效，**管不住 `uv pip`**。

坏掉的旧环境保留为 `lerobot-broken-20261011`（约 11G）作安全网，确认无碍后可删。
- tokenizer 来自 gated 仓库 `google/paligemma-3b-pt-224`（pi0_base 自身不带
  tokenizer 文件）。需要两步：
  1. `hf auth login --force` —— 本机缓存的 OAuth token 已于 2026-07-31 过期，
     且 refresh_token 失效，必须重新登录
  2. 在 https://huggingface.co/google/paligemma-3b-pt-224 用**同一账号**
     接受 Gemma 许可，否则登录了也是 403

  然后 `python fetch_tokenizer.py` 预下载（几十 MB，不下 3B 权重）。
  预下载好之后 `run_train.sh` 里的 `HF_HUB_OFFLINE=1` 就能正常生效

## 训练

```bash
# 冒烟测试（先确认显存够）
SMOKE=1 ./run_train.sh

# 正式训练
STEPS=24500 BATCH_SIZE=4 ./run_train.sh
```

## 实测结果（`measure_inference.py`）

```bash
python measure_inference.py                 # 默认每帧采样 5 次, seed=0
python measure_inference.py --no-base       # 跳过 pi0_base 对照
```

### ⚠️ 两个口径，别混用

pi0 是 flow matching，**每次 `predict_action_chunk` 都现抽一组高斯噪声**，所以结果本身是
随机的。脚本同时给两个口径：

| 口径 | 含义 | 用途 |
|---|---|---|
| **单次MAE** | 抽一次就执行整个 chunk | **部署时会拿到的误差，引用这个** |
| 均值MAE | 平均 5 次抽样 | 只用于横向对比；噪声互相抵消，实机拿不到 |

没固定种子时踩过：同一 checkpoint、同一批帧，两轮单次抽样测出 **1.641 / 2.091** —— 全是采样噪声。

### 推理显存与延迟（确定性，四轮一致：349.9 / 354.2 / 356.9 / 357.9 ms）

```
峰值显存        8.44 GiB          (权重本身 8.30 GiB)
每 chunk        357.9 ms (2.8 Hz)
chunk=50 步 -> 3.33 s 轨迹, 实时余量 9.3x
冷启动首帧      770 ms (含 cudnn autotune)
底座加载        104 ~ 109 s
```

**部署目标可以是 12 GB 卡。** `gradient_checkpointing` 对推理**零影响**（357.9 vs 357.8 ms）——
因为 `modeling_pi0.py:630` 那条分支只在 `self.training` 下生效，不必为部署特意关它。

### 三态对照（`--samples 5 --seed 0`，6 帧各取一条 episode）

| 模型状态 | **单次MAE** | 均值MAE | vs「照抄当前动作」(3.083) |
|---|---|---|---|
| **LoRA 微调后** | **1.191** | 0.703 | **0.39x** |
| LoRA 关掉（`lora_B` 置零） | 58.637 | 57.523 | 19.02x |
| pi0_base（原版，无 adapter） | 39.957 | 38.987 | 12.96x |

- **微调总收益：39.957 → 1.191（-97.0%）**
- 对照是公平的：pi0 内部恒在 `max_action_dim=32` 的 padded 空间计算，两边形状一致，
  pi0_base 的预训练权重**全部加载成功**（日志 "All keys loaded successfully"），
  不存在「动作头被随机初始化」的问题。

### 消融结论：LoRA 增量挣了几乎全部收益

`lora_B` 全置零、只保留 `full_training_modules` 里重训的投影层 → MAE 反而**恶化**
（部署口径 1.191 → 58.637，**-98.0%**）。更值得注意的是 58.637 **比没微调的
pi0_base（39.957）还差**。

> 含义：投影层是**和 LoRA 增量共同适配**出来的，被训练成"预期输入已带上 LoRA 位移"。
> 一旦抽掉位移，输入分布对不上就崩。**这个 adapter 拆不开** —— 别想着导出时
> "留投影层、丢掉 LoRA"。

另一个副产品：三个状态下「抽样平均」的收益完全不同 ——

```
LoRA 微调后   1.191 → 0.703   平均抹掉 41%   ← 误差主要是采样抖动
LoRA 关掉    58.637 → 57.523  平均抹掉 2%    ← 误差是系统性的, 平均救不回来
pi0_base     39.957 → 38.987  平均抹掉 2%
```

即：**微调后剩下的误差以采样噪声为主，不是系统性偏差** —— 所以多采样平均对它是真有效的。

### 逐维误差（部署口径）

```
            单次MAE   均值MAE   该维50步行程    占行程
y_mm         4.799     2.583       34.490      13.9%   ← 主扫描轴
pitch_deg    0.162     0.132        1.017      15.9%
x_mm         1.011     0.484        0.000      n/a     ← 真值不动, 这是抖动
z_mm         1.004     0.921        0.671    149.5%
roll_deg     0.145     0.077        0.035      n/a
yaw_deg      0.022     0.020        0.006      n/a
```

两个可读的结论：

- **真正在动的轴（y / pitch）误差占行程 14% / 16%**，跟得住。
- **采样噪声集中在 x 和 y**（这两维从均值口径到单次口径差约 2 倍，z/roll/yaw 几乎不变）。
- ⚠️ **部署隐患**：x 和 z 真值基本不动，但模型有 **~1 mm 抖动**。对超声扫描，
  探头 z 向 1 mm 抖动可能影响贴合压力一致性 —— 训练集拟合数字看不出来，需实机验证。

### 加载 PEFT checkpoint 的坑（踩过）

`PI0Policy.from_pretrained(<LoRA ckpt>)` 是**错的**：它只找 `model.safetensors`，
而 LoRA checkpoint 里只有 `adapter_model.safetensors`。找不到时**不报错**，只打印一句
"Returning model without loading pretrained weights"，然后返回一个**随机初始化的模型**
（实测 MAE 52.7，真实是 1.2）。正确入口是 `policies/factory.py:565` 的 `make_policy()`；
`measure_inference.py` 复刻了那三行。`lerobot-eval` 走的是 `make_policy`，不受影响。

另外两个注册上的坑：config 必须取自 **checkpoint**（取自 pi0_base 的话
`output_features.action.shape` 是 32 而非 6，归一化映射也不同），且必须用**基类**
`PreTrainedConfig.from_pretrained` 读（ckpt 的 `config.json` 带 `"type":"pi0"`，
直接调 `PI0Config.from_pretrained` 会因 draccus 解不了判别字段而报错），
并先 `import lerobot.policies.pi0.configuration_pi0` 注册子类。

### ⚠️ 最大的限制

**以上全部是训练集数字。** 这个 run 的 `eval_split=0.0`，28 条 episode 全进了训练，
**测不出泛化**。1.191 只说明"记住了"，不说明"会做"。要真实数字必须留 holdout 重训。

## 远程部署（`serve.py` + `replay_client.py`）

架构：**这台机器只做推理**（policy server），另一台机器驱动机械臂（robot client）。
LeRobot 自带这套 gRPC 协议，实现在 `lerobot/async_inference/`。`pi0` 在
`SUPPORTED_POLICIES` 里，`SUPPORTED_ROBOTS` 里**没有 xArm6**。

```bash
# 本机自测（假机器人在本进程里，走真机同款的 RobotClient）
python serve.py --host 0.0.0.0 --port 8080 --device cuda &
python replay_client.py --episode 0 --n-frames 300

# 跨机测试（数据从另一台主机发过来，对面不需要 torch/lerobot/数据集）
python serve.py --wire numpy --host 0.0.0.0 --port 8080 --device cuda &
python make_replay_bundle.py --episode 0 --n-frames 300 --out replay_bundle.npz
# 把 replay_bundle.npz + remote/ 拷到对面，见 remote/README.md
```

### 上游 `policy_server.py` 的两个静默 bug

**1（致命）：加载 LoRA 会返回随机初始化的模型。**
上游第 152 行是 `policy_class.from_pretrained(path)`，而 `PI0Policy.from_pretrained`
只找 `model.safetensors`，LoRA checkpoint 里只有 `adapter_model.safetensors`。
**找不到时它不报错**，只打印一句 "Returning model without loading pretrained weights"，
然后返回随机权重的模型 —— 接着一本正经地给真机下发垃圾动作。`serve.py` 复刻了
`policies/factory.py:565` 的 PEFT 分支，并**硬校验** `lora_B` 范数非零
（PEFT 初始化时 B 恒为 0，读进来还全零就说明 adapter 没生效），把静默失败变成启动报错。

**2：`rename_map` 被清空。**
上游第 162 行无条件用 client 传来的 rename_map 覆盖 ckpt 存档的那份，而
`RobotClient` 构造 `RemotePolicyConfig` 时是**位置传参**（`robot_client.py:103`），
压根没传这个字段 ⇒ 恒为 `{}` ⇒ ckpt 里存的
`{"observation.images.rgb": "observation.images.base_0_rgb", ...}` 被清掉。
`serve.py` 改成：client 明确传了才覆盖。

### wire 契约（都是从源码挖出来的，不是猜的）

| 项 | 要求 | 违反的后果 |
|---|---|---|
| **相机名** | 必须逐字等于 policy 的键：`base_0_rgb`、`left_wrist_0_rgb` | `helpers.py:166` 的 `policy_image_features[key]` 是直接索引，**且发生在 preprocessor 之前** ⇒ KeyError。`rename_map` 救不了这里 |
| **第三个图像槽** | 机器人不用给 `right_wrist_0_rgb` | policy 自动补 mask=0 空图 |
| **特征表顺序** | `observation_features` 的插入顺序 = `observation.state` 维序 | `feature_utils.py:93` 的 `names = list(joint_fts)` 取字典顺序，`build_dataset_frame` 按 names 逐位组装 ⇒ **静默换轴，不报错** |
| **图像格式** | **HWC uint8** | 数据集里是 CHW float[0,1]，而 `resize_robot_observation_image` 内部 `permute(2,0,1)` 要求入参 HWC；`prepare_image` 只在 uint8 时才 /255 |
| **fps** | **15** | `DEFAULT_FPS=30`、`RobotClientConfig.fps` 默认也是 30 ⇒ 动作的时间语义整体错一倍 |
| **动作语义** | 绝对 TCP（`use_relative_actions=False`） | 输出是位姿目标不是增量，client 不能再叠 delta |
| **动作顺序** | `action_features` 顺序 = 模型输出维序 | `_action_tensor_to_action_dict` 按 `robot.action_features` 逐位映射 ⇒ 静默换轴 |
| **roll** | 必须解卷绕（训练分布 [163.97, 193.04]） | SDK 报 −180..180，同一物理姿态在 ±180 两侧差 360°，模型直接看到分布外输入。真机驱动必须自己实现 `unwrap_roll.py` 的逻辑 |
| **归一化 stats** | 由 server 从 ckpt 读 | 已处理 ✓ |

### 回放自测结果（纯软件，不碰硬件）

`replay_client.py` 把 `ReplayRobot`（从数据集回放的假机器人）接进**真机同款的
`RobotClient`**，让整条链路原样跑 300 帧 / 20s，再逐帧对比下发动作与数据集真值。

```
下发动作      300 步    收到 chunk 21 个（每个 50 步）
发出观测      83 帧（28%）  ← 客户端只在队列降到阈值以下才问策略，设计如此
控制循环      308 次，均周期 67.7 ms（目标 66.7 ms）
队列饿死      8 次  ← 首帧推理 0.35s 的空窗，真机上表现为"启动后头几步没动作"
整体 MAE      1.478 / 1.498（两次）   ← 离线 measure_inference.py 的部署口径 = 1.191
```

**MAE 落在离线值同量级 ⇒ 契约对齐。** 比离线略差是正常的：图像被量化成 uint8，
而且走的是真实的异步队列路径（离线是同步单帧）。两次跑出 1.498/1.478 的小差异来自
pi0 本身的采样随机性 + 问策略的帧不完全固定。

逐维：`y_mm` 占行程 2.5%、`z_mm` 9.8%、`pitch_deg` 10.2% 跟得住；
`x_mm`（真值段内不动，MAE 2.4mm）和 `roll_deg`（真值只动 0.078°，MAE 0.16°）是**抖动**，
和离线测到的一致 —— 超声探头 z/x 向毫米级抖动仍需实机验证贴合压力。

### 跨机测试结果（数据从另一台主机发过来）

`make_replay_bundle.py` 把一段回放导成自包含 `.npz`（300 帧 / 56MB），
`remote/remote_client.py` 在**只装了 `grpcio + numpy`**（约 80MB，无 torch / 无 lerobot /
无数据集）的机器上把它发过来、收动作、对比真值。Tailscale 网段，本机 `100.64.0.1:8080`。

同一段（episode 0 第 1301 帧起 300 帧）三方对照：

| 路径 | MAE | 说明 |
|---|---|---|
| 离线 `measure_inference.py` | **1.191** | 部署口径上限 |
| 本机闭环 `replay_client.py` | **1.478** | 真机同款 RobotClient + 完整 gRPC |
| 跨机 `remote/remote_client.py` | **1.663** | 19 chunk / 72 次观测 / 执行 14.68 Hz |

> 第三行是在 `remote_client.py` **合流前**的版本上测的（当时的 679 行版）。
> 合流后这一份同时是真机的传输层，启动了预热、`--chunk-size-threshold` 默认由
> 0.5 提到 0.9，观测上传也移出了动作时钟——MAE 应仍在同量级，但表格末列那三个
> 计数会变，复测时以新输出为准。细节见 `remote/README.md`。

**链路零降质**：包里的图像预先缩到 224×224，而服务端收到后还会再缩一次 ——
224→224 的 bilinear（align_corners=False, scale=1）映射到原像素中心、权重恰为 1，
**实测是精确恒等**。所以送进模型的张量与本机全分辨率路径逐位相同，
单帧却从 2.96MB 降到 0.30MB（15Hz 峰值 36Mbps vs 上游原格式的 355Mbps）。

1.663 与本机 1.478 的差属于 π0 自身采样噪声（本机跑两次就是 1.478/1.498），
不是链路引入的。**两条路都走通的结论：wire 契约对齐。**

跨机路上踩的两个坑（都已修，详见 `remote/README.md`）：

- **pickle 垫片只抄字段没抄方法**：对面没有 `lerobot` 模块，靠 `sys.modules` 塞同名
  垫片让 unpickle 能解析类路径。只写字段的话 unpickle 正常、对象看着也对，一调
  `get_timestep()` 就 AttributeError —— 而且崩在**接收线程**里。连锁反应是
  队列永远空 → 观测 `timestep` 恒为 0 → 撞上服务端 `_predicted_timesteps` 里的 0 →
  **后续观测全部静默丢弃**，表现为"观测一直在发，动作一个都不来"。
- **回放有尽头，真机没有**：`self.t` 跑到帧数后 `get_observation()` 会越界。

另外：**每次 client 连接服务端都会重载 3B 模型，约 108s**（上游 `SendPolicyInstructions`
的实现如此）。真机部署时这是一次性成本，调试时很磨人。

### 真机还缺什么

0. **把 `--wire numpy` 换回 `torch`**。`numpy` 只服务于"对面没装 torch"的测试场景；
   真机的 `RobotClient` 收到的动作必须是 `torch.Tensor`（不换的话，
   `RobotClient.receive_actions` 里的 `.to(client_device)` 会炸）。
1. **xArm6 的 `Robot` 实现**。`make_robot_from_config`（`robots/utils.py:25`）是
   **硬编码 if/elif**，第三方机器人只能靠 `lerobot_robot_*` 发行包注册。
   照 `ReplayRobot` 抄即可：`get_observation` 换成读相机+读 SDK（记得 HWC uint8 + roll 解卷绕），
   `send_action` 换成 SDK 下发。官方 SDK 在 `/opt/lerobot/xArm-Python-SDK-master/`（原版，无用户代码）。
2. **`observations_similar` 的 `atol` 默认 1**（`helpers.py:282`）：它拿 state 范数差判断
   "这帧和上帧一样吗、要不要跳过推理"。我们的 state 是 mm/度，atol=1 就是 **1mm**，
   和实测的 x/z 向 ~1mm 抖动同量级，可能造成推理触发抖动，真机上要调。
3. **`eval_split=0.0` ⇒ 没有任何泛化证据**。真机第一次跑，分不清看到的误差是分布外失效
   还是别的问题。
4. 控制循环的 `time.sleep(max(0, dt - elapsed))` **不补偿累积漂移**：实测 300 帧用了
   20.9s（14.4 Hz，目标 15Hz）。短程无所谓，长时间跑会累积。

