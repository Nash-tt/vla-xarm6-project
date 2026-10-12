# 08 · 如果改用 openpi 训练，现有的客户端/服务端还能沿用吗？

这是一个**评估结论**，不是一次实施记录。结论先行：

| 层面 | 能否沿用 | 说明 |
|---|---|---|
| 数据格式 | ❌ **硬阻塞** | 已实测：openpi 依赖的 lerobot 0.1.0 只认 v2.1 |
| 服务端（serve.py） | ❌ 不能直接用 | openpi 没有 gRPC，也没有 LeRobot 的 `PolicyServer` 桥 |
| 客户端（remote/） | ⚠️ 有条件下能 | 写一层适配器可以救活整条链路 |
| 训练脚本 | ✅ 基本无关 | 但数据集仓库要能用 |

以下每条都标了**实测**还是**推断**。

---

## 一、数据侧：真正的硬阻塞（**已实测**）

openpi 自带的 `.venv` 里 pin 的 `lerobot` 是 **0.1.0**，
它的 `CODEBASE_VERSION = "v2.1"`，要求数据集里有：

```
meta/tasks.jsonl
meta/episodes.jsonl
```

而我们的自采数据集是 **v3.0** 格式，目录长这样：

```
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
```

实测结果（用 openpi 的 `.venv` 里的 lerobot 去读我们的 v3.0 数据集）：

| 函数 | 结果 |
|---|---|
| `load_info` | ✅ 通过 |
| `load_tasks` | ❌ 失败 |
| `load_episodes` | ❌ 失败 |

### ⚠️ 这个失败会伪装成网络错误

`LeRobotDatasetMetadata.load_metadata()` 外面套着：

```python
try:
    ... load_info / load_tasks / load_episodes ...
except (FileNotFoundError, NotADirectoryError):
    # 转而去 HF hub 拉取
    get_safe_version(...) → pull_from_repo(...)
```

所以「文件不存在」这个真实原因会被**吞掉**，转成一次 hub 拉取尝试，
最终报出来的是一个 `OfflineModeIsEnabled` 之类的**网络错误**。

我们第一版测试就因此被误导，以为是对不上 HF。**排查时要把这几步拆开单独调。**

另外：**版本检查只 `warn` 不报错**，所以你不会在一开始就被拦住，
而是一路走到某个深处才炸。

---

## 二、服务端：openpi 没有 gRPC

全仓库 grep `async_inference` / `PolicyServer` / `grpc` —— **零命中**。
openpi 的 `Policy` 继承的是它自己的 `openpi_client.base_policy.BasePolicy`，
和 LeRobot 的 `PolicyServer` 没有任何关系。

openpi 用的是 **WebSocket**（默认端口 8000）：

- 连接后服务端先推一个**元数据帧**
- 线格式是 **msgpack + 自定义 ndarray 编码**
  （`b"__ndarray__"` / `b"data"` / `b"dtype"` / `b"shape"`），
  **不是 pickle，也不是 protobuf**
- 响应形如 `{"actions": (H, D), "server_timing": ..., "policy_timing": ...}`
- 健康检查是 `GET /healthz`

### 两条路

**方案 A：写适配器（推荐，改动小）**
把 openpi 的 policy 包装成一个 LeRobot `Policy`，
这样 `serve.py` 和**整个客户端链路原样存活**。

**方案 B：改用 openpi 的 websocket**
要重写客户端（wire 格式、握手、帧结构全变），
并且**控制循环的归属要反过来**——openpi 由客户端负责开环回放。

### ⚠️ 双归一化陷阱

openpi 内部有 `Normalize`（它会按 `x.shape[-1]` 把 stats 切片），
LeRobot 侧有 pre/post processor。**只能有一边做归一化。**

搞错了是**静默的**——不报错，只是结果不对。
上方案 A 的时候这是第一个要确认的事。

---

## 三、框架无关的风险

### roll 口径（同样要紧）

openpi 的 `compute_norm_stats.py` 也只算 `keys = ["state", "actions"]`，
roll 的展开/wrap 问题**一模一样存在**。见[文档 03](03-roll-angle.md)。

### float32 累加器（**推断，未实测**）

openpi 的 `RunningStats`（`src/openpi/shared/normalize.py`）用的是和 LeRobot
**完全相同的公式**：

```python
std = sqrt(mean_of_squares - mean**2)
```

所以[文档 02-B](02-stats-json.md) 里那个 float32 相消问题**很可能同样存在**。

> **但这一条没有实测过**，只是一个基于公式相同的推断。
> 真要换 openpi，这个值得花十分钟量一下。

---

## 四、动作维度：必须自定义输入输出

openpi 的 pi0：

```python
action_dim: int = 32
action_horizon: int = 50
```

它通过 `PadStatesAndActions` 把动作**补齐到 32 维**——
但这个 transform **只在 PI0 / PI05 的 `model_transforms` 里注册**。
**pi0-FAST 不做 padding**，这点要注意。

我们的动作是 **6 维**，所以不管走哪条路，**都必须写自定义的 Inputs / Outputs**
把 6 维映射到模型期望的形状。最近的模板是 `examples/ur5/`。

---

## 五、时间集成：openpi 里也没有

全仓库 grep，**openpi 里不存在 temporal ensembling**。
开环回放由客户端负责（`ActionChunkBroker` 按 `self._cur_step` 取动作）。

这一点和 LeRobot 的 `async_inference` 是一致的（那边也没有）。
**所以这不是选框架的理由，两边都得自己写。**

---

## 六、建议

**LeRobot 本身已经有 π0**，我们这条链路（数据 → 训练 → 服务端 → 远程客户端）已经通了。

换 openpi 的成本，粗估**约两周**的适配量：
自定义 Inputs/Outputs、数据格式转换（v3.0 → v2.1，或改 openpi 侧的 lerobot 版本）、
websocket 或适配器、归一化归属的确认。

**所以换的理由需要非常具体。** 如果只是「openpi 是官方实现」，
那不足以支撑两周——我们的链路已经在跑，而 openpi 那边要重新踩一遍
数据格式和归一化的坑。

反过来，如果目标是 pi0-FAST、或者 openpi 的某个特定 checkpoint / 训练配方，
那这笔投入是值得的，但**建议先花半天做数据侧的可行性验证**
（打开本文第一节说的那个「伪装成网络错误」的坑），再决定。
