# 06 · 远程轻量客户端：带宽、静默丢帧与真机接入缺口

> 代码：`pi0_ultrasound_lora/serve.py`、`pi0_ultrasound_lora/remote/`
> 背景文档：`pi0_ultrasound_lora/remote/README.md`

## 架构

工控机（Win10，无独显）无法承载 π0/SmolVLA 的推理开销，而机械臂与相机均接于其上。
因此将推理环节分离：

```
   工控机 / robot-pc (100.64.0.2)              GPU 机器 / gpu-server (100.64.0.1)
   ┌────────────────────────┐                 ┌──────────────────────────┐
   │ 相机 → 观测            │  ──gRPC──────▶  │ serve.py                 │
   │ 机械臂 ← 动作块        │  ◀────────────  │  (LeRobot PolicyServer)  │
   │ remote_client.py       │                 │  π0 / SmolVLA + LoRA     │
   └────────────────────────┘                 └──────────────────────────┘
```

采用的是 LeRobot 的 `async_inference`：服务端注册
`services_pb2_grpc.add_AsyncInferenceServicer_to_server`，
`TimedObservation` / `TimedAction` / `RemotePolicyConfig` 经 **pickle** 序列化。

---

## 一、带宽：自 355 Mbps 降至 36 Mbps

官方客户端单帧需发送 **2.96 MB**（约 355 Mbps），在无线链路上无法稳定。

本项目的轻量客户端将单帧降至 **0.30 MB（约 36 Mbps）**，减少约 10 倍：

- 图像在客户端侧完成编码压缩，不以明文数组传输
- `--wire numpy` 采用 numpy 的二进制布局，省去 torch 的序列化开销
- 仅发送推理实际需要的键

这是**部署可行性的前提条件**，而非性能优化。

---

## 二、最隐蔽的缺陷：接收线程终止导致观测被静默丢弃

### 现象

客户端连接成功、握手完成，但**策略始终不产生动作**。
无异常、无报错，日志无任何输出。

### 原因

工控机未安装 LeRobot，因此 `remote/` 中提供了一份手写的
**pickle stub**（`services_pb2*.py` 及客户端中的骨架类），
用于在反序列化时复现 `TimedObservation` / `TimedAction` 等类的结构。

pickle 的反序列化机制是按**模块名 + 类名 + 属性名**还原对象。
若 stub 中的**方法名或属性名与真实类不一致**（LeRobot 版本升级时即可能发生），
反序列化会抛出 `AttributeError`。

关键在于：**该异常发生在接收线程内部。**
主线程运行自身的循环（发送观测、等待动作），无法感知接收线程已终止。
其表现为——**观测不再进入，且没有任何报错。**

### 处理

1. 接收线程的异常必须**向上传播**，不可就地吞掉。
   线程内发生的一切异常都应能传递至主循环并触发显式失败。
2. stub 的字段需与真实类**逐字对齐**，并在两侧版本变化时重新核对。

### 结论

**在多线程结构中，「静默终止」比「崩溃」更危险。**
某一线程终止而主线程继续运行，系统会以「未发生任何事件」的形态停摆。
编写此类代码时，线程的异常出口应作为一等设计要素处理。

---

## 三、`must_go`：活性机制

官方 `PolicyServer` 中的 `must_go` 标志属于**活性（liveness）机制**：
客户端须持续表明自身存活，否则服务端将判定连接失效并重置状态。

远程客户端需正确实现该机制——**它不是可选的心跳，而是协议的一部分。**
忽略它会导致服务端在某一时刻单方面重置，而在客户端侧仅表现为「响应变慢」。

---

## 四、上游缺陷：`_reset_server()` 重置不完整

`_reset_server()` 未重置 `last_processed_obs` 与 `fps_tracker`。

后果：客户端重连后，服务端会以**上一会话的 `last_processed_obs`** 进行比对，
将新会话的观测误判为陈旧或重复而丢弃；`fps_tracker` 亦会携带旧数据继续统计，
使 FPS 读数失去意义。

**尚未提交 upstream issue。**

---

## 五、一处未解释的现象

远程客户端的测试进程曾以**退出码 144（SIGSTKFLT）**结束。
未找到复现路径，亦未定位原因。在此记录，以免日后作为新问题重复排查。

---

## 六、相机键名必须直接使用策略的键名

客户端发来的观测键名会被 `helpers.py:166` 在预处理**之前**用于索引
策略的图像特征表，即 `rename_map` 在该位置尚未生效。
详见[文档 05-B](05-lora-and-rename-map.md)。

**结论：客户端的 camera key 必须直接写为策略的键名**（如 `observation.images.rgb`），
不应依赖服务端重映射。

---

## 七、真机接入尚缺的工作

当前状态：**轻量客户端可连接 policy server 并接收动作块**，
即链路已通，但尚未接入真实机械臂。缺口如下：

1. **xArm6 的 `Robot` 实现。** 目前以 `ReplayRobot` 进行离线回放测试。
   真机接入需参照 `ReplayRobot` 的接口编写实现，将 `send_action`
   替换为 xArm SDK 的下发调用。（官方 SDK 为原版，不含用户代码。）
2. **传输由 `--wire numpy` 切换至 `--wire torch`。**
   当前为节省带宽采用 numpy；接入真机需传输全分辨率图像时应切换，
   并重新测定带宽余量。
3. **时间集成（temporal ensembling）尚未实现。** 官方默认为 `weighted_average`，
   当前实现固定为 `latest_only`，未做成可配置项。
   需注意 LeRobot 的 `async_inference` 中同样**没有** temporal ensembling，
   开环回放完全由客户端负责。
4. 两处上游缺陷值得提交：`_reset_server()` 重置不完整、
   `from_pretrained` 对 PEFT checkpoint 静默返回随机权重。
