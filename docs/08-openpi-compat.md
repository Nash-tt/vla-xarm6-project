# 08 · 改用 openpi 时现有客户端与服务端的可复用性评估

本篇是一份**评估结论**，而非实施记录。结论先行：

| 层面 | 可复用性 | 说明 |
|---|---|---|
| 数据格式 | 否，**硬阻塞** | 已实测：openpi 依赖的 lerobot 0.1.0 仅支持 v2.1 |
| 服务端（`serve.py`） | 否，不可直接使用 | openpi 无 gRPC，亦无 LeRobot `PolicyServer` 桥接 |
| 客户端（`remote/`） | 有条件可行 | 编写一层适配器可保留整条链路 |
| 训练脚本 | 基本无关 | 但数据集需可用 |

以下各项均标注**实测**或**推断**。

---

## 一、数据侧：硬阻塞（**已实测**）

openpi 自带的 `.venv` 中 pin 的 `lerobot` 为 **0.1.0**，
其 `CODEBASE_VERSION = "v2.1"`，要求数据集包含：

```
meta/tasks.jsonl
meta/episodes.jsonl
```

而本项目的自采数据集为 **v3.0** 格式：

```
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
```

实测结果（以 openpi 的 `.venv` 中的 lerobot 读取 v3.0 数据集）：

| 函数 | 结果 |
|---|---|
| `load_info` | 通过 |
| `load_tasks` | 失败 |
| `load_episodes` | 失败 |

### 该失败会伪装为网络错误

`LeRobotDatasetMetadata.load_metadata()` 的外层为：

```python
try:
    ... load_info / load_tasks / load_episodes ...
except (FileNotFoundError, NotADirectoryError):
    # 转而去 HF hub 拉取
    get_safe_version(...) → pull_from_repo(...)
```

因此「文件不存在」这一真实原因被吞掉，转为一次 hub 拉取尝试，
最终报出 `OfflineModeIsEnabled` 一类的**网络错误**。

初次测试即因此被误导。**排查时应将这几步拆分后单独调用。**

另需注意：**版本检查仅 `warn`，不报错**，
因此不会在起始处被拦截，而是在较深处才暴露。

---

## 二、服务端：openpi 不含 gRPC

全仓库检索 `async_inference` / `PolicyServer` / `grpc`——**零命中**。
openpi 的 `Policy` 继承自其自身的 `openpi_client.base_policy.BasePolicy`，
与 LeRobot 的 `PolicyServer` 无任何关系。

openpi 使用 **WebSocket**（默认端口 8000）：

- 连接后服务端先推送一个**元数据帧**
- 线格式为 **msgpack + 自定义 ndarray 编码**
  （`b"__ndarray__"` / `b"data"` / `b"dtype"` / `b"shape"`），**非 pickle，亦非 protobuf**
- 响应形如 `{"actions": (H, D), "server_timing": ..., "policy_timing": ...}`
- 健康检查为 `GET /healthz`

### 两条路径

**方案 A：编写适配器（推荐，改动较小）。**
将 openpi 的 policy 包装为 LeRobot `Policy`，
`serve.py` 与**整条客户端链路可原样保留**。

**方案 B：改用 openpi 的 websocket。**
需重写客户端（线格式、握手、帧结构均不同），
且**控制循环的归属需反转**——openpi 由客户端负责开环回放。

### 双归一化陷阱

openpi 内部含 `Normalize`（按 `x.shape[-1]` 对 stats 切片），
LeRobot 侧含 pre/post processor。**仅允许一侧执行归一化。**

选错是**静默的**——不报错，仅结果不正确。
采用方案 A 时，这是首要确认项。

---

## 三、与框架无关的风险

### roll 口径（同样适用）

openpi 的 `compute_norm_stats.py` 亦仅计算 `keys = ["state", "actions"]`，
roll 的展开/wrap 问题**同样存在**。见[文档 03](03-roll-angle.md)。

### float32 累加器（**推断，未实测**）

openpi 的 `RunningStats`（`src/openpi/shared/normalize.py`）使用与 LeRobot
**完全相同的公式**：

```python
std = sqrt(mean_of_squares - mean**2)
```

因此[文档 02-B](02-stats-json.md) 中的 float32 抵消问题**很可能同样存在**。

> 该条**未实测**，仅为公式相同所导出的推断。若确定改用 openpi，此项值得实测确认。

---

## 四、动作维度：必须自定义输入输出

openpi 的 pi0：

```python
action_dim: int = 32
action_horizon: int = 50
```

其通过 `PadStatesAndActions` 将动作**补齐至 32 维**，
但该 transform **仅在 PI0 / PI05 的 `model_transforms` 中注册**，
**pi0-FAST 不做 padding**，此点需注意。

本项目动作为 **6 维**，因此无论采用哪条路径，
**均须编写自定义 Inputs / Outputs** 将 6 维映射至模型期望的形状。
最接近的模板是 `examples/ur5/`。

---

## 五、时间集成：openpi 中亦不存在

全仓库检索，**openpi 中不存在 temporal ensembling**。
开环回放由客户端负责（`ActionChunkBroker` 依 `self._cur_step` 取动作）。

这与 LeRobot 的 `async_inference` 一致，
**因此不构成选择框架的理由，两侧均需自行实现。**

---

## 六、建议

**LeRobot 自身已包含 π0**，且本项目链路（数据 → 训练 → 服务端 → 远程客户端）已打通。

改用 openpi 的成本粗估**约两周**适配工作量：
自定义 Inputs/Outputs、数据格式转换（v3.0 → v2.1，或调整 openpi 侧的 lerobot 版本）、
websocket 或适配器、归一化归属的确认。

**因此切换的理由需要足够具体。** 若仅为「openpi 是官方实现」，
不足以支撑两周投入——现有链路已在运行，而 openpi 侧需重新处理数据格式与归一化问题。

反之，若目标是 pi0-FAST，或 openpi 的特定 checkpoint / 训练配方，
则该投入是合理的；但**建议先以半天时间完成数据侧可行性验证**
（需注意第一节所述「伪装为网络错误」的陷阱），再行决定。
