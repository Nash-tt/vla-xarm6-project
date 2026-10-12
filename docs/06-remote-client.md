# 06 · 远程瘦客户端：带宽、静默丢帧，和上真机还差什么

> 代码：`pi0_ultrasound_lora/serve.py`、`pi0_ultrasound_lora/remote/`
> 背景文档：`pi0_ultrasound_lora/remote/README.md`

## 架构

工控机（Win10，无独显）跑不动 π0/SmolVLA，但机械臂和相机都接在它上面。
所以把推理拆出去：

```
   工控机 / robot-pc (100.64.0.2)              GPU 机器 / gpu-server (100.64.0.1)
   ┌────────────────────────┐                 ┌──────────────────────────┐
   │ 相机 → 观测            │  ──gRPC──────▶  │ serve.py                 │
   │ 机械臂 ← 动作块        │  ◀────────────  │  (LeRobot PolicyServer)  │
   │ remote_client.py       │                 │  π0 / SmolVLA + LoRA     │
   └────────────────────────┘                 └──────────────────────────┘
```

用的是 LeRobot 的 `async_inference`：服务端注册
`services_pb2_grpc.add_AsyncInferenceServicer_to_server`，
`TimedObservation` / `TimedAction` / `RemotePolicyConfig` 走 **pickle** 序列化。

---

## 一、带宽：从 355 Mbps 压到 36 Mbps

官方客户端每帧要发 **2.96 MB**（约 355 Mbps）。在 Wi-Fi 上这是不可能稳的。

我们的瘦客户端把每帧压到 **0.30 MB（约 36 Mbps）**，降了约 10 倍：

- 图像在客户端就编码压缩，而不是发明文数组
- `--wire numpy` 走 numpy 的二进制布局，省掉 torch 的序列化开销
- 只发推理真正需要的键

这是**部署能不能成的前提**——不是优化，是必要条件。

---

## 二、最阴的一个坑：接收线程死掉，观测被静默丢弃

### 现象

客户端连上了，握手也成功，但**策略就是不动**。
没有异常，没有报错，日志一片安静。

### 根因

工控机上没有装 LeRobot，所以 `remote/` 里放了一份手写的
**pickle stub**（`services_pb2*.py` 和客户端里的骨架类），
用来在反序列化时「复现」`TimedObservation` / `TimedAction` 这些类的形状。

pickle 反序列化的机制是：按**模块名 + 类名 + 属性名**去还原对象。
如果 stub 里的**方法名或属性名和真实类对不上**（LeRobot 版本一升级就可能对不上），
反序列化时会抛 `AttributeError`。

要命的是：**这个异常发生在接收线程里。**
主线程有自己的循环（发观测、等动作），它不会知道接收线程已经死了。
表现出来就是——**观测再也进不来，也没有任何报错。**

### 解决

1. 接收线程的异常必须**向上传播**，不能吞掉。
   线程里发生的一切异常都要能冒到主循环，触发显式失败。
2. stub 的字段要跟真实类**逐字对齐**，并在两边版本变化时重新核对。

### 教训

**多线程里，「静默死亡」比「崩溃」危险得多。**
一个线程挂了而主线程继续跑，整个系统会以「什么都没发生」的姿态停摆。
写这类代码时，线程的异常出口要当成一等公民来设计。

---

## 三、`must_go`：活性机制

官方 `PolicyServer` 里有个 `must_go` 标志，是**活性（liveness）机制**：
客户端必须持续表现「我还在」，否则服务端会认为连接已死并重置状态。

我们的远程客户端需要正确实现它——**不是可选的心跳，是协议的一部分。**
漏了它，服务端会在某个时刻单方面 reset，而客户端侧看起来只是「响应变慢了」。

---

## 四、上游 bug：`_reset_server()` 重置不全

`_reset_server()` 没有重置 `last_processed_obs` 和 `fps_tracker`。

后果是：客户端重连之后，服务端会拿**上一个会话的 `last_processed_obs`**
去做比对，把新会话的观测误判为「过期/重复」而丢弃；
`fps_tracker` 也会带着旧数据继续统计，让 FPS 读数失去意义。

**尚未提交 upstream issue。**

---

## 五、一个没解释的现象

远程客户端的测试进程有一次以**退出码 144（SIGSTKFLT）**结束。
没有找到复现路径，也没有定位到原因。记录在案，避免以后当成新问题重新查一遍。

---

## 六、相机键名必须直接用策略的键名

客户端发来的观测键名会被 `helpers.py:166` 在预处理**之前**用来索引
策略的图像特征表——也就是说 `rename_map` 在这个位置还没生效。
详见[文档 05-B](05-lora-and-rename-map.md)。

**结论：客户端的 camera key 必须直接写成策略的键名**（如 `observation.images.rgb`），
不要指望在服务端重映射。

---

## 七、上真机还差什么

当前状态：**瘦客户端能连上 policy server 并收到动作块**——
也就是链路通了，但还没接到真机械臂上。缺的是：

1. **xArm6 的 `Robot` 实现。** 目前用 `ReplayRobot` 做离线回放测试。
   真机接入要照 `ReplayRobot` 的形状写一个，把
   `send_action` 换成 xArm SDK 的下发调用。
   （官方 SDK 是原版，无用户代码。）
2. **传输从 `--wire numpy` 切到 `--wire torch`。**
   现在为了省带宽用的是 numpy；上真机要传全分辨率图像时得换过去，
   同时重新测带宽够不够。
3. **时间集成（temporal ensembling）没做。** 官方默认是 `weighted_average`，
   我们这边写死了 `latest_only`，还没做成可配置项。
   注意 LeRobot 的 `async_inference` 里也**没有** temporal ensembling ——
   开环回放完全由客户端负责。
4. 两个上游 bug 值得报：`_reset_server()` 重置不全、
   `from_pretrained` 对 PEFT checkpoint 静默返回随机权重。
