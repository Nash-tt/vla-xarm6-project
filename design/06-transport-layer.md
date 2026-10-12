# 06 · 传输层

**代码位置：** `pi0_ultrasound_lora/remote/`，`remote_client.py`（815 行）。
**跑在哪：** 工控机（真机部署时）或任意一台机器（连通性测试时）。
**上游对应物：** `lerobot/async_inference/robot_client.py`。
**对面是谁：** [05 serve.py](05-inference-server.md)。**这两者是必须同版本的一对。**

> 这一份在本仓库里**只有一处**。历史上的两份（679 行 / 815 行）已经分叉过，
> 统一时按 815 行那版为准，`robot/real_robot/` 不再保留副本。
> 理由见 [07](07-real-robot-client.md) 与 [10 文档第二节](../docs/10-collaboration.md)。

---

## 一、它要解决的问题：对面什么都不能装

工控机没有独立显卡，也没有 LeRobot、没有 torch。而 LeRobot 的客户端
`RobotClient` 是长在 `lerobot` 包里的。**传输层的存在就是为了让对面不需要装这些。**

依赖只有 `grpcio` + `numpy`（+ protobuf，grpcio 会带）——见 `requirements.txt`。
对面不需要 1.8 G 数据集、不需要 LeRobot、不需要 CUDA。

代价是：它**复刻的是 `robot_client.py` 的线上行为，不是那个类**。
对面没装 LeRobot，所以要用本目录里两个 protobuf 生成文件 + 一段 pickle 垫片手搓。
**两条路必须字节级同构**，这是本篇所有细节的来源。

```
LeanRobotClient                                    PolicyServer
├─ receive_actions      ◄── GetActions ────────    （回传动作块）
├─ send_observations_loop ── SendObservations ──►  （上行观测，独立线程）
└─ control_loop（15 Hz）                            （服务端被动响应）
```

**服务端不决定"什么时候推理"**——客户端把观测推上去，服务端才跑。
所以整个系统的节奏、活性、节流全在客户端这一侧，见第四、五节。

---

## 二、pickle 垫片：最难归因的一处

观测和动作在传输时是 `pickle.dumps(TimedObservation / TimedAction)`，
而 pickle 里记的是**类的模块路径 + 类名**：
`lerobot.async_inference.helpers.TimedObservation`。对面没有这个模块，
unpickle 直接 `ModuleNotFoundError`。

解法（`_install_pickle_stubs`，`:133`）：在 `sys.modules` 里塞一个同名模块，
里面放**属性名完全一致**的 dataclass。pickle 对 dataclass 走的是
NEWOBJ + `__dict__` 状态，所以**只要属性名对得上就行**——
字段顺序、有没有基类、有没有方法都不影响。

代码往 `sys.modules` 里补了这些名字（`:133-163`）：
`lerobot`、`lerobot.async_inference`、`lerobot.async_inference.helpers`（真垫片），
以及 `constants` / `transport` / `transport.utils` / `utils.feature_utils` /
`utils.constants`（空模块占位，万一服务端在别处引用）。

### 2.1 `check_stubs_work()`：字段和方法都要抄

这是那个**踩过的坑**留下的自检（`:165`）：

> *只查字段是不够的 —— 只抄字段的垫片能通过"往返"检查，却会在接收线程里
> 以 AttributeError 崩掉。*

`_STUB_SRC` 里的注释说得更直白：

> **字段和方法都要抄**：pickle 恢复的是字段，但代码调用的是方法。
> 只抄字段的话 unpickle 能成功、对象看着对，一调 `get_timestep()` 就
> AttributeError —— 而且是在接收线程里崩，主线程只看到"动作再也不来"。

`check_stubs_work()` 把 `get_timestep()` / `get_timestamp()` / `get_observation()` /
`get_action()` 全调一遍，就是为了把这个失败模式钉死在启动时。

**这是本层最贵的教训**：症状是"动作不来"，病因在 pickle 垫片少了个方法。

---

## 三、与真机路径的三个刻意的差异

文档字符串里列了三条，都不是妥协，是设计：

### 3.1 相机的键名用的是 policy 的键

发出去的是 `base_0_rgb` / `left_wrist_0_rgb`——**policy 侧的名字**，
不是数据集里的 `observation.images.rgb`。

这不是真机的做法（真机在自己那边用 `rename_map` 映射），而是因为
`helpers.py:166` 的 `policy_image_features[key]` 是**直接索引**，
且发生在 **preprocessor 之前**——键名不匹配只有 `KeyError`，`rename_map` 救不了。

见 [design/README 约定③](README.md)。

### 3.2 图像预先缩到 224×224

服务端还会再缩一次，而 224→224 是**精确恒等**，所以给模型的张量与本机全分辨率路径
**逐位相同**。省 10 倍带宽（见 `make_replay_bundle.py`）。

### 3.3 服务端必须用 `--wire numpy` 起

默认格式回传的是 `torch.Tensor`，对面没 torch 就 unpickle 不了。
这是唯一为"对面没 torch"做的**协议让步**——代价是这条路不再是真机的字节格式
（见 [05 第五节](05-inference-server.md)）。

> **注意这三条只适用于瘦客户端这条路。** 真机部署走的是 `--wire torch` 的默认格式，
> 键名映射由服务端从 checkpoint 读（[05 差异 2](05-inference-server.md)）。

---

## 四、三个同步机制

`LeanRobotClient` 的线程模型。**理解这三个，就理解了整个传输层。**

### 4.1 观测发送与动作时钟解耦

```python
def control_loop_observation(self, force_must_go=False):   # :468
    """把观测交给独立发送线程，绝不阻塞动作时钟。"""
    if self.observation_pending.is_set():
        return False
    ...
    self.observation_queue.put_nowait(obs)

def send_observations_loop(self):                          # :496
    """在动作控制线程之外执行可能很慢的 gRPC 上传。"""
```

`observation_queue` 的 `maxsize=1`（`:334`）——**只留一个在途观测**。
配合 `observation_pending` 事件，形成"上一帧还没发完就不发下一帧"的背压。

为什么必须解耦：上行一个观测是 0.3 MB 的 gRPC 调用，网络一抖动就是几十毫秒。
**控制环是 15 Hz（66 ms 一拍）**，同步发必然把动作时钟拖垮。
解耦后动作时钟只做 `put_nowait`，永不阻塞。

**发送失败会让整个客户端停**（`send_observations_loop` 里
`self.shutdown_event.set()`）——不是重试，是停。理由见第七节。

### 4.2 启动顺序：先起接收，再启动时钟

```python
self.start_barrier = threading.Barrier(1)   # :330
# The receive and control loops no longer need to start together.
# Receiving starts first so the initial action chunk can be primed
# before the fixed-rate action clock begins.
```

`control_loop` 与 `receive_actions` 都在开头 `start_barrier.wait()`，
但屏障只有一个参与者（`Barrier(1)`），所以它**不构成同步**——
是一个显式的"解除起跑线"。

真正的顺序控制靠 `first_actions_event`：**先收到第一个动作块，才启动固定频率时钟**。
不这么做的话，时钟会先跑掉几拍、动作队列还空着，等于开环空转。

### 4.3 `must_go`：活性机制，不是优化

```python
# 关键：上游 robot_client.py:337 就在这里 set。
# 它不是"优化"，是**活性机制**：队列一空，本次 obs 必带 must_go=True，
# 服务端 _enqueue_observation 无条件收下 → 必然再跑一次推理。
# 少了这一句，后续 obs 全部 must_go=False，只能指望 _obs_sanity_checks
# 放行；一旦 obs 的 timestep 落进服务端的 _predicted_timesteps 就会被
# 静默丢弃，表现为"发了一堆观测，动作再也不来"。
self.must_go.set()
```

**这里必须纠正一处文档错误：** `docs/06` 把 `must_go` 描述成"心跳"。
它不是心跳，**语义是"跳过服务端的两个 sanity check 并无条件入队"**
（`policy_server.py:291`）。

服务端默认会对观测做合理性检查，包括"这个 timestep 是不是已经预测过"
（`_predicted_timesteps`）。而 `_predicted_timesteps` 是**会话单调**的，
只在 `_reset_server()` 里清空。配合 `observations_similar`（atol=1mm），
**这就是静默丢观测的机制**——而且丢的时候只记 `logger.debug`，
而根 logger 是 INFO，所以**日志里什么都看不到**。

`must_go=True` 是绕过这一整套的唯一出口。少了它，系统会在"看起来一切正常"
的情况下停止产出动作。

客户端侧的用法（`:485-488`）：

```python
obs.must_go = force_must_go or (
    self.must_go.is_set() and self.action_queue.empty()
)
```

即**只在动作队列空了的时候**才强制。收到新动作块后 `must_go.clear()`。

---

## 五、上行带宽：0.3 MB 从哪来

观测是**原始 uint8 ndarray 字节**（pickle），**没有任何图像编码**。

| | 大小 |
|---|---|
| 全分辨率（720×1280×3 ×2 路 + depth） | ~2.96 MB |
| 预缩到 224×224 后 | **~0.30 MB** |

**10 倍的下降全部来自客户端侧预先缩到 224×224**，不是来自 `--wire numpy`。
`docs/06` 把这个归功于 `--wire numpy` 是错的——那个开关管的是**下行动作的类型**，
与上行观测体积无关。

`send_bytes_in_chunks`（`:212`）按 2 MB 切块，注意它的判定顺序是
**先判 END 再判 BEGIN**，所以小于 2 MB 的载荷只会有一个 `TRANSFER_END` 块。
本客户端的观测约 0.3 MB，走的正是这条路。

---

## 六、动作侧：聚合策略是 `latest_only`

`receive_actions`（`:523`）收到动作块后的处理：

```python
with self.action_queue_lock:
    current = {a.get_timestep(): a.get_action() for a in self.action_queue.queue}
future = Queue()
for new in timed_actions:
    if new.get_timestep() <= latest:       # 比已经执行过的旧，丢
        continue
    if new.get_timestep() not in current:
        future.put(new)
    else:
        # aggregate_fn = latest_only ⇒ 取后到的那个
        future.put(type(new)(...))
```

三条规则：

1. **不超过已执行的最新 timestep 的动作一律丢**（`:latest_action`）
2. 与队列中已有的一样的 timestep，**后到的覆盖先到的**（`latest_only`）
3. 整个队列被**替换**，不是追加

不这么做，两个动作块的时间范围重叠时，队列里会出现两条指向同一时刻的动作，
执行顺序由入队顺序决定——那就是随机的。

`action_chunk_size = max(self.action_chunk_size, len(timed_actions))`（`:552`）：
以**实际收到的最大块长**为准，不信客户端启动时的参数。

---

## 七、错误处理：为什么失败等于停

整个客户端有两条错误通道：

| 通道 | 字段 | 谁看 |
|---|---|---|
| 接收线程 | `rx_error` | 主循环 |
| 发送线程 | `observation_error` | 主循环（真机客户端第 722 行转成 `SafetyStop`） |

**接收线程的崩溃特别难归因**，代码专门写了注释（`:545`）：

> *这个线程一旦死掉，主线程只会看到"动作再也不来"，极难归因
> （第一次调试时就是垫片少了 `get_timestep` 在这里崩的）。*

所以异常处理里除了设 `rx_error` 还额外 `traceback.print_exc()`——
不让它变成一条没有上下文的"动作不来"。

**两个线程的错误都直接 `shutdown_event.set()`，不重试。** 这是刻意的一侧选择：
传输层已经不可信时，继续跑等于让机械臂在不确定的状态下执行动作，
而这台机器的另一端挂着探头。失败就该停，由 [07](07-real-robot-client.md) 的安全包络接手。

---

## 八、回放包：让"没机器"也能验链路

`Bundle`（`:264`）+ `make_replay_bundle.py`（在 `pi0_ultrasound_lora/` 下）
把数据集的一段录成 `replay_bundle.npz`（约 53 MB，**不入库**）。
`LeanRobotClient` 的 `get_observation` 从包里第 t 帧取，
`send_action` 只记录不下发。

这样整条链路（垫片、节流、must_go、聚合）可以在**不接机械臂**的情况下验通，
且能算出端到端 MAE 与本地推理对照：

```python
LOCAL_REPLAY_MAE = 1.478     # 本机 replay_client.py 测到的
OFFLINE_MAE      = 1.191     # 离线推理
```

`Bundle.exhausted` 后不再发观测（`:420`）——上游没这个问题，
因为真机的观测没有"尽头"；回放包有，越界会崩。这是回放路径专有的一处改动。

---

## 九、参数

| 参数 | 说明 |
|---|---|
| `--server` | `host:port`，默认指向 GPU 机的 Tailscale 地址 |
| `--bundle` | 回放包路径 |
| `--ckpt` | **服务端本地路径**（服务端拿它加载模型），不是对面的路径 |
| `--fps` | 15，控制环频率 |
| `--actions-per-chunk` | 与服务端 `chunk_size` 对齐（[04](04-training.md) 里 ACT 取 50） |
| `--chunk-size-threshold` | 队列剩余比例低于它才发新观测 |
| `--policy-device` | 发过去但服务端不采信（[05 3.1](05-inference-server.md)） |
| `--grpc-compression` | `none` / `gzip` |
| `--connect-timeout` | 握手超时 |

`--ckpt` 那一条容易误解：**它发的是服务端的路径**，对面根本没有这个文件。
`DEFAULT_CKPT` 因此与服务端一致地指向
`pi0_ultrasound_lora/run1/checkpoints/012250/pretrained_model`
——那个中途存档（[04 第三节](04-training.md)）。

> `docs/09` 的 132 行与 165 行在 `--grpc-compression` 上互相矛盾，
> 以本节为准（参数确实存在，默认 `none`）。

接着读 [07 真机客户端](07-real-robot-client.md)。
