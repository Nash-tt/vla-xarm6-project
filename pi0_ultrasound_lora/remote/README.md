# 远程推理测试（另一台主机 → 本机 GPU）

把数据集里的一小段回放**从另一台主机**发过来，本机推理，再把动作传回去，最后对比
数据集真值。用来验证两台机器之间的链路和 wire 契约，**不碰机械臂**。

```
   gpu-server (100.64.0.1)                   另一台主机
   ┌──────────────────────────┐                  ┌────────────────────────┐
   │ serve.py --wire numpy    │  ← 观测(图+state) │ remote_client.py       │
   │ LoRA 微调后的 pi0 / 3090 │  → 动作 chunk     │ 只有 numpy + grpcio    │
   └──────────────────────────┘                   └────────────────────────┘
             走 Tailscale 100.x 网段
```

对面**不需要** LeRobot、不需要 torch、不需要 CUDA、不需要数据集。

---

## 一、本机（gpu-server）先做的事

### 1. 导出回放包

```bash
conda activate lerobot
cd pi0_ultrasound_lora
python make_replay_bundle.py --episode 0 --n-frames 300 --out replay_bundle.npz
```

产物约 53MB（300 帧）。想跑久一点就加大 `--n-frames`。

包里的图像是**预先缩到 224×224** 的，这不是降质：服务端收到后还会再缩一次，
而 224→224 的 bilinear 是精确恒等（已实测），所以送进模型的张量和本机全分辨率
路径**逐位相同**。好处是单帧从 2.96MB 降到 0.30MB。

### 2. 起服务端

```bash
python serve.py --wire numpy --host 0.0.0.0 --port 8080 --device cuda
```

`--wire numpy` 是**必须的**：默认（`torch`）回传的动作是 `torch.Tensor`，
对面没 torch 就 unpickle 不了。这条路上唯一为"对面没 torch"做的让步。

### 3. 把包和 `remote/` 目录拷过去

```bash
# 对面是 Windows + OpenSSH 的情况
scp replay_bundle.npz remote/ <用户名>@100.64.0.2:~/pi0_remote/
```

---

## 二、对面（那台主机）做的事

### 1. 确认网络通

```powershell
# Windows PowerShell
Test-NetConnection 100.64.0.1 -Port 8080
```
```bash
# Linux / macOS
nc -vz 100.64.0.1 8080
```

本机 Tailscale 状态（当前）：
- `gpu-server` = `100.64.0.1`  ← 跑 serve.py 的这台
- `robot-pc` = `100.64.0.2`

### 2. 装依赖（只有三个包）

```bash
python3 -m pip install -r requirements.txt
```

| 包 | 版本下限 | 为什么 |
|---|---|---|
| `grpcio` | ≥1.73.1 | `services_pb2_grpc.py` 里写死 `GRPC_GENERATED_VERSION='1.73.1'`，会主动拒绝更旧的运行时 |
| `protobuf` | ≥6.31.0 | `services_pb2.py` 调 `ValidateProtobufRuntimeVersion(6,31,0)` |
| `numpy` | ≥1.24 | state / 动作向量 |

需要 **Python ≥ 3.9**（`protobuf` 6.x 的要求）。装完约 80MB —— 对比 torch 的 2GB+。

### 3. 跑

```bash
python3 remote_client.py --server 100.64.0.1:8080 --bundle replay_bundle.npz
```

先 `--dry-run` 可以只验包不连网：

```bash
python3 remote_client.py --bundle replay_bundle.npz --dry-run
```

---

## 三、输出怎么读

```
【7】判定
  ✅ 远程链路通，且契约对齐。MAE 1.xxx，与离线 1.191 同量级
```

MAE 是"收到的动作 vs 数据集真值（同一帧）"的逐维平均绝对误差，单位 mm / deg。

### 已经在本机验证过的结果

用 `--episode 0 --n-frames 300`（自动挑到 episode 内第 1301 帧起，**与本机
`replay_client.py` 测的是同一段**，可直接对比）：

| 基准 | MAE | 说明 |
|---|---|---|
| `measure_inference.py`（离线） | **1.191** | 部署口径（单次采样），上限 |
| `replay_client.py`（本机闭环，同段） | **1.478** | 走完整 gRPC + 队列，图像量化成 uint8 |
| **本脚本（跨机，同段）** | **1.663** | 19 个 chunk、72 次观测、执行 14.68 Hz |

三者同量级。跨机比本机慢一点点属于 π0 自身的采样噪声 —— 同样的 episode 0 在本机
跑两次就分别是 1.478 / 1.498，所以 1.663 在正常波动内，**不是链路引入的偏差**。

链路本身没有任何降质：包里的图像是 224×224，服务端再缩一次是精确恒等，
送进模型的张量与全分辨率路径逐位相同。

明显超过 3.0 就是契约没对齐，按提示逐条查。

---

## 四、带宽

| 路径 | 单帧 | 15Hz 峰值 | 说明 |
|---|---|---|---|
| 本方案（图像缩到 224） | 0.30 MB | **36 Mbps** | 客户端只在队列降到阈值以下才发观测，实测约占 28% 的迭代 → 均值约 10 Mbps |
| 上游默认（原分辨率 720×1280） | 2.96 MB | **355 Mbps** | 协议对图像**不做任何压缩**，直接 pickle 原始 uint8 |

第二行是真机部署时要留意的事：720×1280 的相机在 15Hz 下要 355Mbps，Wi-Fi 基本扛不住。

---

## 五、排错

| 现象 | 原因 |
|---|---|
| `ModuleNotFoundError: No module named 'lerobot...'` | 垫片没生效 —— 必须 `import remote_client`，不能把这几个文件拆散 |
| 卡在「连接」不动，服务端日志停在 `底座：...` | 正常，服务端在加载 3B 模型，约 **110 秒**。每次 client 连接都会重载一次 |
| `一步都没收到` | 服务端是不是忘了 `--wire numpy`？看服务端日志有没有 `adapter 已注入：36 个 lora_B` |
| MAE 几十以上 | 服务端加载成了随机初始化模型（日志里没有 `adapter 已注入`），或 `--ckpt` 指错了 |
| `端口绑定失败` | 8080 被占：`ss -ltnp \| grep 8080` |

服务端的实时日志在跑 `serve.py` 的那个终端里。

---

## 六、代码里三个容易踩的点

1. **pickle 垫片**（`remote_client.py` 的 `_install_pickle_stubs`）
   传输对象是 `pickle.dumps(TimedObservation)`，pickle 记的是**类路径 + 类名**
   （`lerobot.async_inference.helpers.TimedObservation`）。对面没这个模块，
   所以在 `sys.modules` 里塞了同名模块 + **字段名完全一致**的 dataclass。
   dataclass 的 pickle 走 NEWOBJ + `__dict__`，字段顺序和基类都不影响。

   ⚠️ **方法也必须抄**。踩过一次：垫片只写了字段没写 `get_timestep()`，
   unpickle 一切正常、对象看着也对，一调用就 AttributeError —— 而且崩在
   **接收线程**里，主线程只看到"动作再也不来"，极难归因。
   现在 `check_stubs_work()` 会把 `get_timestep/get_timestamp/get_action/
   get_observation` 全部实调一遍，别再退回只查字段的版本。

   那次崩溃的连锁反应值得记住：接收线程死 → 动作队列永远空 →
   观测的 `timestep` 恒为 `max(latest_action, 0) = 0` → 服务端
   `_predicted_timesteps` 里已经有 0 → 后续观测**全部被静默丢弃** →
   表现为"观测一直在发，动作一个都不来"。

2. **相机键名必须是 policy 的键**（`base_0_rgb` / `left_wrist_0_rgb`）
   因为 `helpers.py:166` 是 `policy_image_features[key]` **直接索引**，且发生在
   preprocessor **之前** —— 键名不对只有 KeyError，`rename_map` 救不了。
   真机是在自己那边做映射的，这里直接把键名写成 policy 的键（见包里的 meta）。

3. **`must_go` 是活性机制，不是优化**
   上游 `robot_client.py:337` 在收到 chunk 后 `must_go.set()`。队列一空，
   下一个观测就带 `must_go=True`，服务端 `_enqueue_observation` 无条件收下 →
   必然再推理一次。少这一句，后续观测只能指望 `_obs_sanity_checks` 放行，
   一旦 timestep 落进服务端的 `_predicted_timesteps` 就会被**静默丢弃**，
   表现为"观测一直发，动作再也不回来"。

4. **回放有"尽头"，真机没有**
   上游 `RobotClient` 永远在循环，观测源不会结束。这里 `self.t` 跑到帧数后
   `get_observation()` 会越界（`IndexError: index 300 is out of bounds`）。
   所以 `_ready_to_send_observation()` 里加了 `exhausted` 判断 ——
   跑完就不再发观测。这个越界本身无害（发生在收尾阶段），但会打一大段
   traceback 淹掉真正的结果。

---

## 七、这跟真机部署的差距

这条链路验证的是**协议和契约**，不是真机路径。真要接机械臂还差：

1. **`--wire numpy` 换回 `torch`** —— 真机的 `RobotClient` 要 torch.Tensor。
2. **写 xArm6 的 `Robot` 实现** —— 模板就是本机 `replay_client.py` 里的
   `ReplayRobot`：把 `get_observation` 换成读相机 + 读 SDK，`send_action` 换成
   下发 SDK，接口一个字不用改。别忘了 roll 的 `unwrap_roll.py` 逻辑。
3. **图像走原分辨率** —— 真机相机给的就是原图，那时要面对上面 355Mbps 的问题。
