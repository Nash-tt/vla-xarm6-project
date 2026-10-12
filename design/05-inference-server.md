# 05 · 推理服务端

**跑在哪：** GPU 机器，`pi0_ultrasound_lora/serve.py`（255 行）。
**做什么：** 加载微调后的策略，通过 gRPC 接收观测、回传动作块。
**对面是谁：** [06 传输层](06-transport-layer.md) 的瘦客户端 → [07 真机客户端](07-real-robot-client.md)。

---

## 一、为什么是"远程推理"

**这台机器不接机械臂，只算策略。** 驱动机械臂的是工控机，它没有独立显卡
（见 [design/README](README.md) 第一节）。

LeRobot 自带这套 gRPC 协议，实现全在 `lerobot/async_inference/` 里，
**本项目没有重新发明它**。`serve.py` 继承上游 `PolicyServer`，
只覆写了 `SendPolicyInstructions` 一个方法，加了一个 `_time_action_chunk` 的钩子。

```
工控机                                      GPU 机器
┌────────────────────┐                    ┌──────────────────────┐
│ real_robot_client  │                    │  serve.py            │
│  └ remote_client   │  ── 观测 ───────►  │   └ PolicyServer     │
│      (瘦客户端)     │  ◄── 动作块 ─────  │       └ 策略模型      │
└────────────────────┘    gRPC/Tailscale  └──────────────────────┘
```

**服务端不决定"什么时候推理"**——它被动收观测、被动回动作。
节奏完全由客户端掌握，见 [06 第三节](06-transport-layer.md)。

---

## 二、只改了三处，其中两处是上游的静默 bug

`serve.py` 的 docstring 说"只有两处差异"，但实际有三处刻意的改动
（前两处是修 bug，第三处是加硬化检查）：

### 差异 1（致命）：LoRA 加载的静默失败

上游 `policy_server.py:152`：

```python
self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
```

而 `PI0Policy.from_pretrained` **只找 `model.safetensors`**，
LoRA checkpoint 里只有 `adapter_model.safetensors`。找不到时它**不报错**，
只打印一句 `Returning model without loading pretrained weights`，
**然后返回一个随机初始化的模型**。

`measure_inference.py` 里实测过后果：MAE **52.7**，而真实值是 **1.19**。
40 倍的差距，但接口一切正常——服务端会一本正经地给真机下发垃圾动作。

正确入口是 `policies/factory.py:565` 那段 PEFT 分支，`serve.py:81-98` 复刻了它的三行核心：

```python
cfg    = PreTrainedConfig.from_pretrained(path)     # 必须用基类读
policy = PI0Policy.from_pretrained(base, config=cfg)
policy = PeftModel.from_pretrained(policy, path, config=peft_cfg, is_trainable=False)
inner  = policy.get_base_model()                    # LoRA 层就地注入，取回底座即可
```

两个容易错的细节写在注释里：

- **`config` 必须取自 checkpoint 且用基类读。** ckpt 的 `config.json` 带
  `"type":"pi0"`，直接调 `PI0Config.from_pretrained` 会因 draccus 解不了判别字段而报错。
- **底座路径来自 `adapter_config.json` 的 `base_model_name_or_path`**，读不到就直接 `RuntimeError`。

### 差异 2：`rename_map` 被静默清空

上游 `policy_server.py:162` 无条件用 client 传来的 `rename_map` 覆盖 checkpoint 存档的那份：

```python
preprocessor_overrides={"rename_observations_processor": {"rename_map": policy_specs.rename_map}}
```

但 `RobotClient` 构造 `RemotePolicyConfig` 时是**位置传参**
（`robot_client.py:103`），压根没传 `rename_map`，于是**恒为 `{}`**。
结果是 ckpt 里存的

```json
{"observation.images.rgb": "observation.images.base_0_rgb",
 "observation.images.ultrasound": "observation.images.left_wrist_0_rgb"}
```

被**清空**。而 π0 没有这个名字映射就跑不起来（见 [04 第二节](04-training.md)）。

`serve.py:180-186` 改成：**client 明确传了才覆盖，否则保留 checkpoint 自带的**。
两种情形都会打日志，事后能看出走的是哪条路。

> 这也解释了为什么客户端侧不需要传 `rename_map`——**服务端从 checkpoint 里读**，
> 那份才是训练时真正用的。

### 差异 3：把静默失败变成硬报错

`serve.py:100-119`，一段上游没有的检查：

```python
norms = [(n, p.detach().float().norm().item())
         for n, p in inner.named_parameters() if "lora_B" in n]
if not norms:
    raise RuntimeError("... 找不到任何 lora_B 层 —— adapter 没注入")
if len(zeros) == len(norms):
    raise RuntimeError("全部 N 个 lora_B 都是 0 —— adapter 权重没加载进来")
```

原理：**PEFT 初始化时 `lora_B` 恒为 0**，所以如果读进来还全零，
说明 adapter 权重根本没加载 —— 此时模型等价于未微调。
注释把取舍写得很清楚：

> *宁可起不来，也不要给真机发垃圾动作。*

日志会打出 `lora_B` 的范数区间与全零个数（`serve.py:116-119`）。**看这一行就知道加载对不对。**

---

## 三、两处"服务端说了算"

### 3.1 设备不信任客户端

```python
self.device = self.forced_device or policy_specs.device     # :166
```

`--device` 默认 `cuda`，而且**写死不给 client 覆盖的机会**。理由在注释里：

> *对面要是传个 cpu，3B 模型一帧要跑几十秒，表现成"卡死"而不是报错。*

这是个真实的误导模式——不是崩溃，是"看起来没反应"。

### 3.2 先加载策略，再起服务

`main()` 的顺序（`:225-245`）：

```python
server_cfg = PolicyServerConfig(...)
policy_server = LoraPolicyServer(server_cfg, ...)     # 构造
if not os.path.exists(args.ckpt): raise SystemExit(...)
grpc_server = grpc.server(...)                        # 之后才起 gRPC
```

注释：*宁可启动失败，也不要让 client 连上一个没加载好策略的空壳。*

但要注意：**真正的策略加载发生在客户端连上并下发 `SendPolicyInstructions` 时**
（`:176`），不在 `main()` 里。所以启动服务端时只校验了 `--ckpt` 路径存在，
策略是否加载成功要等第一个客户端连上才见分晓——**看日志的那行 lora_B 范数**。

---

## 四、参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt` | `pi0_ultrasound_lora/run1/checkpoints/012250/pretrained_model` | 见下 |
| `--host` | `0.0.0.0` | 远程部署必须 |
| `--port` | `8080` | |
| `--device` | `cuda` | 写死 |
| `--fps` | 15 | **数据集就是 15 fps，别用上游默认的 30** |
| `--obs-queue-timeout` | 2.0 | |
| `--wire` | `torch` | 见第五节 |

### `DEFAULT_CKPT` 指向一个中途存档

默认值是 `run1/checkpoints/012250/pretrained_model`。
如 [04 第三节](04-training.md) 所述，π0 的目标是 24500 步，
**实际停在 12250**，所以 012250 就是最后一个存档。
它不是"收尾存档"，是"跑到一半停下的存档"——这一点在读结果时要记住。

---

## 五、`--wire`：为瘦客户端留的一条路

`_time_action_chunk` 的覆写（`:133-145`）。上游默认 `wire=torch`：
每个 `TimedAction.action` 都是 `torch.Tensor`，对面 unpickle 时**必须有 torch**（2GB 起）。

`--wire numpy` 时把回传动作转成 numpy 数组，对面只要 numpy 就能收。
代价是**这条路不再是真机的字节格式**——注释明确写了
*只用于远程连通性测试*。

真机部署用默认的 `torch`。`numpy` 这条是给 `remote/` 里的瘦客户端测试用的。

> **带宽的下降不来自这个开关。** 观测上行是原始 uint8 ndarray 字节（pickle），
> **没有图像编码**；从 2.96 MB 降到 0.30 MB 全部来自客户端侧预先缩到 224×224。
> `docs/06` 曾把这个归功于 `--wire numpy`，是错的。见 [06 第五节](06-transport-layer.md)。

---

## 六、环境变量的处理

`serve.py:45-52`，在 `import lerobot` **之前**执行：

```python
for _v in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)          # socks:// 这个 scheme httpx 不认
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```

**只删 `ALL_PROXY`，`HTTP(S)_PROXY` 要留着**——本机直连 HF 不通。
这与其他训练脚本的 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY ...`（全删）**不一样**，
别照抄。原因见 [design/README](README.md) 与 [10 文档第六节](../docs/10-collaboration.md)：
PyPI 走清华镜像直连，GitHub HTTPS 走代理，而 HF 需要代理但 httpx 不认 socks scheme。

---

## 七、起服务

```bash
cd <仓库根目录>/pi0_ultrasound_lora
python serve.py --ckpt run1/checkpoints/012250/pretrained_model \
                --host 0.0.0.0 --port 8080 --device cuda --fps 15
```

起来后等客户端连。客户端连上并下发策略指令时，日志会依次出现：

```text
Client <peer> | type=pi0 | ckpt=... | actions_per_chunk=... | device=cuda
底座：<base_model_name_or_path>
adapter 已注入：N 个 lora_B，范数 [x, y]，全零 0 个        ← 这一行是健康标志
策略就绪，耗时 X.Xs | 图像键 [...] | state ... | action ...
```

**第二行和第三行缺一不可。** 只有第一行没有后两行，说明客户端还没下发策略指令。

接着读 [06 传输层](06-transport-layer.md)。
