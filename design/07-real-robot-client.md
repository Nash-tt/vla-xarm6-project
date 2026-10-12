# 07 · 真机客户端

**代码位置：** `robot/real_robot/real_robot_client.py`（1002 行）。
**跑在哪：** 工控机，接 xArm6 + RealSense + PhaseLink。
**上游是谁：** [06 传输层](06-transport-layer.md) 的 `LeanRobotClient`（它继承）、
[05 serve.py](05-inference-server.md)（远端策略）。

**默认影子模式**——连真机、读传感器、调远端策略、只记录，**不调用任何运动 API**。

---

## 一、这一层为什么存在

`LeanRobotClient` 只会"发观测、收动作"，它对机械臂一无所知。
真机客户端补上的是**从"动作数组"到"机械臂运动"之间那道必须存在的闸**。

继承关系（`:607`）：

```python
class RealRobotClient(transport.LeanRobotClient):
```

它覆写三个方法：

| 覆写 | 作用 |
|---|---|
| `get_observation()` | 从**真传感器**取，而不是回放包的第 t 帧 |
| `pop_action()` | 动作不只要弹出来，还要**通过包络检查并下发** |
| `run_control_loop()` | 固定频率时钟，带 `--max-seconds` 与错误通道 |

其余（垫片、节流、must_go、动作聚合）全部继承自 [06](06-transport-layer.md)。

**这也解释了为什么传输层必须和它同版本**：`pop_action` 覆写的签名、
`first_actions_event`、`observation_error` 这些名字一变，这里就崩。

---

## 二、两态：影子 / 执行

```python
print("  mode:", "EXECUTE" if args.execute else "SHADOW (no robot commands)")
```

| 模式 | 触发 | 行为 |
|---|---|---|
| **SHADOW**（默认） | 什么都不加 | 连设备、读传感器、调策略、记录，**不下发任何运动命令** |
| **EXECUTE** | `--execute` **且** `--confirm <EXECUTE_CONFIRMATION>` | 真正下发伺服目标 |

两个开关**必须同时给出，字符串精确匹配**：

```python
if args.execute and args.confirm != EXECUTE_CONFIRMATION:
    parser.error(f"real execution requires --confirm {EXECUTE_CONFIRMATION}")
```

这是 `parse_args` 里的 `parser.error`，**在连接任何设备之前就退出**。
推荐流程（`README_REAL_ROBOT.md`）：

```text
起 PhaseLink → --validate-only → 影子测试 → 有界真机测试
```

`--validate-only` 只校验文件、import 与安全配置，**不连接设备也不连服务端**。

---

## 三、三道独立的安全层

三层不是冗余——它们拦的是**不同种类的错误**：

```
① 启动门      parse_args 的两个开关 + prepare_execute 只读
              拦: 人误操作 / 起始状态不对
                        ↓
② 每步限幅    PolicyTargetRateLimiter + SafetyEnvelope.validate
              拦: 策略输出离谱 / 单步跳变
                        ↓
③ 状态守门    _require_healthy（每次下发前）
              拦: 控制器本身出问题
```

### ① 启动门

`prepare_execute()`（`:536`）**只读不写**：查状态、查力传感器、
`motion_enable`、`set_mode(1)`、`set_state(0)`——**全程没有 `set_position`**。
即它**不会自动把机械臂挪到起始位姿**。

如果给了 `--safety-trajectory`，还会检查实际 TCP 是否在示教起点附近
（`require_at_start`，容差 8 mm / 5°），不在就 `SafetyStop`。

### ② 每步限幅

**`SafetyEnvelope.validate`（`:234`）**，每个动作下发前都要过：

| 检查 | 默认阈值 | 拦什么 |
|---|---|---|
| 轨迹管（需 `--safety-trajectory`） | 15 mm / 8° | 策略目标离开示教轨迹太远 |
| 跟踪误差 | **12 mm / 5°** | 目标距实际 TCP 太远 |
| 单步限幅 | **3 mm / 2°** | 相邻两次目标之间的跳变 |
| 轨迹进度跳变 | 回退 20 点 / 前跳 100 点 | 策略把进度拉回去或猛地跳到末尾 |

跟踪误差那一条是关键：**它不信任策略输出的绝对位姿**。
哪怕策略给了一个轨迹管内的点，只要它离实际 TCP 超过 12 mm，就停。
不这么做，一个"正确但超前"的目标会让机械臂一步跨过去。

`PolicyTargetRateLimiter`（`:88`）是**另一套、可选**的机制，默认关闭：

> *The limiter is deliberately opt-in. In the default/direct mode it returns the raw policy target unchanged.*

打开（`--rate-limited-policy`）后，它把"远处的绝对策略目标"转成
**连续的位姿流**，按 `--policy-linear-speed`（2.0 mm/s）与
`--policy-angular-speed`（1.0 deg/s）限速逼近。

**注意这是与上面那套完全不同的思路**：上面的限幅是"超了就停"，
这个是"平滑地追过去"。默认走的是前者——**停比追更安全**，
且这个开关的存在是因为真机上观察到绝对目标跳得远，不是策略错了。

### ③ 状态守门

`_require_healthy`（`:506`），**每次下发前**复查：

| 条件 | 行为 |
|---|---|
| 断连 | `SafetyStop` |
| `error_code` 非零 | `SafetyStop` |
| `warn_code` 非零 | `SafetyStop` |
| `state` 是 4 或 6 | `SafetyStop`（5 仅在尚未下发任何目标时允许检视） |
| 力/力矩不可用 | `SafetyStop` |
| **力读数全零** | `SafetyStop`，除非显式 `--allow-zero-ft` |

最后一条值得单说：**传感器坏了要停，不能因为"读数没超限"就放行**。
没有这条，力传感器失效后所有力检查都被静默架空——
看起来全程正常，实际上根本没在检查。

---

## 四、三路数据源

`get_observation()`（`:623`）把它们拼成观测。

| 源 | 类 | 关键约束 |
|---|---|---|
| RealSense | `RealSenseSource` `:286` | 默认 1280×720 @30，缩到 224×224 |
| 超声 | `UltrasoundSource` `:391` | 读 PhaseLink 的 live 目录，取**最后一条完整**记录 |
| xArm | `XArmInterface` `:446` | SDK 上报回调，`snapshot()` 取最近一条 |

### 4.1 陈旧数据一律停

三路都有时效检查，超时即 `SafetyStop`：

- RealSense：`RealSense frame is stale: {age:.3f}s`
- 超声：`ultrasound packet is stale: {age:.3f}s`
- xArm：`xArm report is stale: {age:.3f}s`

**这是这一层最重要的性质**：任何一路数据不新鲜，整个系统停，
而不是"用上一次的凑合"。凑合的结果是策略看着旧图像对着一台已经动过的机械臂下命令。

### 4.2 ⚠️ xArm 的时效是硬编码 0.25 s，不是 `--sensor-max-age`

**这里要纠正一处文档错误。** `--sensor-max-age` 默认 0.5，
但它**只作用于相机与超声**（`:632`、`:633`），以及预检（`:940`、`:941`）。

xArm 那条走的是 `snapshot(self, max_age_s=0.25)`（`:492`）的**函数默认值**，
调用处没有传 `--sensor-max-age`。所以：

| 路 | 实际时限 | 是否受 `--sensor-max-age` 影响 |
|---|---|---|
| RealSense | `--sensor-max-age`（0.5） | ✅ |
| 超声 | `--sensor-max-age`（0.5） | ✅ |
| **xArm** | **0.25 s（写死）** | ❌ |

`docs/09` 第 224 行的参数表把三路并列在 `--sensor-max-age` 0.5 s 下，是错的。实际是 xArm 更严——
这也合理，机械臂状态是最不能过时的一路。

### 4.3 超声的"最后一条完整记录"

`_last_complete_jsonl_record`（`:371`）从文件**尾部**读 256 KB，
逐行解析，返回最后一条能解析成功的记录。理由是 PhaseLink 在写盘过程中，
文件末尾随时可能是半行。取最后一条**完整**的，而不是简单读最后一行。

---

## 五、控制环

`run_control_loop`（`:713`）。与 [01 的伺服环](01-collection.md) 同一个思路：
**对齐绝对时刻，不累积漂移**。

```python
next_tick = self.start_monotonic
while self.running:
    now = time.monotonic()
    if now - self.start_monotonic >= self.args.max_seconds:   # ① 总时长上限
        break
    if self.observation_error: raise SafetyStop(...)          # ② 发送线程死了
    if self.rx_error:          raise SafetyStop(...)          # ③ 接收线程死了
    if self.actions_available():
        self.pop_action()
    elif self.args.execute and self.args.queue_empty_stop > 0:
        ...                                                    # ④ 队列空太久
    if self._ready_to_send_observation():
        self.control_loop_observation()
    next_tick += self.environment_dt                           # ← 累加，不重置
    delay = next_tick - time.monotonic()
    time.sleep(delay) if delay > 0 else (next_tick := time.monotonic())
```

四道出口：

| # | 条件 | 默认 |
|---|---|---|
| ① | 运行超过 `--max-seconds` | **10.0 s** |
| ② | 观测发送线程异常 | — |
| ③ | 动作接收线程异常 | — |
| ④ | 动作队列空超过 `--queue-empty-stop` | **0.0**（关闭） |

④ 默认关闭，**且这不危险**：队列空时 `pop_action` 返回 `None`，控制环不做事——
**结构上就不会重放陈旧动作**（见 [06 第六节](06-transport-layer.md) 的聚合逻辑）。
加上 `--max-seconds` 兜底，以及断供本身会经 `rx_error` 触发停止。
需要恢复该超时：`--queue-empty-stop 3.0`。

`next_tick += self.environment_dt` 是**累加**。`delay` 已经为负（跑慢了）时，
重置 `next_tick` 到当前时刻，不让它无限追债。

---

## 六、异常与中断：进停止态，不自动清错

任何异常或 Ctrl+C 都走 `stop_motion(emergency=True)`
→ `set_state(4)` + `set_mode(0)`，**不自动清错**。

理由：**错误码被清掉之后，故障就看不见了。** 停下来让人去看那块屏，
比自动恢复后接着跑要安全得多。这也意味着恢复生产要人工介入——
对于一个挂着探头、压着工件的机械臂，这是对的取舍。

---

## 七、参数速查

### 安全相关

| 参数 | 默认 | 说明 |
|---|---|---|
| `--execute` + `--confirm` | 关 | 两把钥匙 |
| `--max-seconds` | **10.0** | 单次运行时长上限 |
| `--max-force-n` / `--max-torque-nm` | 40 / 5 | 每次下发前复查 |
| `--allow-zero-ft` | 关 | 允许力读数全零（危险） |
| `--trajectory-tube-mm` / `-deg` | 15 / 8 | 需 `--safety-trajectory` |
| `--max-action-step-mm` / `-deg` | 3 / 2 | 单步限幅 |
| `--max-tracking-error-mm` / `-deg` | 12 / 5 | 目标距实际 TCP |
| `--start-tolerance-mm` / `-deg` | 8 / 5 | 起始位姿容差 |
| `--max-backtrack-points` / `--max-forward-jump-points` | 20 / 100 | 轨迹进度 |
| `--queue-empty-stop` | 0.0 | 关闭 |
| `--safety-trajectory` | 无 | 不给就走"直接执行策略目标 + 实时限幅" |

### 速率限制（可选）

| 参数 | 默认 |
|---|---|
| `--rate-limited-policy` | 关 |
| `--policy-linear-speed` | 2.0 mm/s |
| `--policy-angular-speed` | 1.0 deg/s |

### 链路

| 参数 | 默认 |
|---|---|
| `--server` | `100.64.0.1:8080`（脱敏） |
| `--robot-ip` | `192.168.1.100`（脱敏） |
| `--fps` | 15 |
| `--actions-per-chunk` | **50**（与 ACT/SmolVLA 的 chunk 对齐） |
| `--chunk-size-threshold` | 0.9 |
| `--grpc-compression` | **gzip**（这里默认开，与瘦客户端不同） |
| `--image-size` | 224 |
| `--sensor-max-age` | 0.5（**不含 xArm**，见 4.2） |

---

## 八、跑真机单测

`robot/real_robot/` **不自包含**——它 `import remote_client`，
那一份在 [`pi0_ultrasound_lora/remote/`](../pi0_ultrasound_lora/remote/remote_client.py)。
这是**想要的行为**：单独拿这个目录去跑会停在 import，
不会静默加载到一份旧副本。

```bash
cd robot/real_robot
PYTHONPATH=<repo>/pi0_ultrasound_lora/remote:<repo>/robot/my_code \
  python -m unittest test_real_robot_client      # 14 项
```

第二处路径是因为它还 `from export_ultrasound_images import ...`，
而那个模块按 `XARM_SDK_ROOT` 去找 `<xArm-SDK>/my_code`。
在工控机上这条路径本来就在，**只有在本仓库里需要手工补**。

---

## 九、这一层的边界

| 不做的事 | 为什么 |
|---|---|
| 不自动去起始位姿 | `prepare_execute` 只读；挪机械臂必须是人的决定 |
| 不重试失败的传输 | [06 第七节](06-transport-layer.md)：不可信时继续跑比停下更危险 |
| 不清错误码 | 清掉故障就看不见了 |
| 不自动清 `_predicted_timesteps` | 那是服务端状态，客户端够不着 |
| **不实现时间集成** | 需要每步都请求推理，当前是整块开环（[04 5.3](04-training.md)） |

最后一条是**已知的未完成项**：ACT 支持时间集成，但客户端按
`--actions-per-chunk 50` 整块执行，拿不到集成的收益。
要做需要改这一层的时钟逻辑。
