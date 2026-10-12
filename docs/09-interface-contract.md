# 09 · 两端接口契约

本篇是 GPU 机器与工控机之间的**接口约定**，不是问题记录。

两端的代码分别由不同的人（或不同的 AI 会话）维护，任何一侧擅自改动下列任一项，
另一侧都会以**静默错误**的形式失效（最典型的是模型照常输出动作，但数值无意义）。
因此凡涉及本表内容的改动，**必须同时更新本文档**。

> 本仓库统一使用占位地址与主机名（`100.64.0.1` / `100.64.0.2`、`gpu-server` / `robot-pc`）。
> 实际地址由各自的命令行参数给出，不写入仓库。

---

## 一、数据契约

### 格式与规模

| 项 | 值 |
|---|---|
| 格式 | LeRobot **v3.0**（`meta/tasks.parquet`、`meta/episodes/`，非 v2.1 的 `.jsonl`） |
| 帧率 | **15 FPS** |
| 回合数 | 28 |
| 帧数 | 裁剪前 49053；裁剪后（`_scan`）27312 |
| 任务文本 | `Follow the fitted scan trajectory with the ultrasound probe.` |

### 特征表

```
observation.images.rgb          video     [720, 1280, 3]
observation.images.ultrasound   video     [256, 256, 3]
observation.state               float32   [6]
action                          float32   [6]
timestamp                       float32   [1]
frame_index / episode_index / index / task_index    int64  [1]
```

**数据集有两路相机**，不是单路。深度流存在于采集端，但**未写入数据集**
（smolvla 变体的目录名即 `no_depth`）。

### 维序（最易出错处）

```python
state.names  = ["tcp_x_mm", "tcp_y_mm", "tcp_z_mm", "tcp_roll_deg", "tcp_pitch_deg", "tcp_yaw_deg"]
action.names = ["target_x_mm", "target_y_mm", "target_z_mm",
                "target_roll_deg", "target_pitch_deg", "target_yaw_deg"]
```

- 单位是 **mm 与 度**，不是米与弧度。
- 顺序为**平移三分量在前、姿态三分量在后**；姿态内部为 roll → pitch → yaw。
- `state` 是当前 TCP 位姿，`action` 是**目标** TCP 位姿。
  **不得用观测状态替代 action**——采集端记录的 `robot_actions.jsonl`
  就是实际下发的目标值，与 `robot_records.jsonl` 是两份独立数据。

### 相机键名映射（关键）

数据集里的键名与 policy 期望的键名**不同**，映射必须由发送方完成：

| 数据集键 | policy 键 | 内容 |
|---|---|---|
| `observation.images.rgb` | `base_0_rgb` | RealSense 彩色，720×1280 |
| `observation.images.ultrasound` | `left_wrist_0_rgb` | PhaseLink S 扫解码图，256×256 |

**原因**：`helpers.py:166` 以 `policy_image_features[key]` **直接索引**，
且发生在 preprocessor **之前**，此时 `rename_map` 尚未生效。
键名不匹配只会抛 `KeyError`，映射表救不了。详见[文档 05-B](05-lora-and-rename-map.md)。

### roll 口径：展开（unwrapped）

xArm 上报的 roll 会跨越 ±180°，**必须先展开为连续序列**再建集。
采用 wrap 口径会让同一批轨迹的 roll 标准差放大约 22 倍，
进而使归一化尺度失真（详见[文档 03](03-roll-angle.md)）。

校验两个角度序列是否一致时，用**环形距离**，不要直接相减：

```python
delta = (a - b + 180.0) % 360.0 - 180.0
```

### 数据集变体

| 目录 | 帧数 | roll | 说明 |
|---|---|---|---|
| `vla_batch_20261007_pi0_6d_rollunwrap` | 49053 | 展开 | π0 用，未裁剪 |
| `vla_batch_20261007_pi0_6d_rollunwrap_scan` | 27312 | 展开 | π0 用，已裁接近段 |
| `vla_batch_20261007_smolvla_6d_no_depth` | 49053 | 展开 | SmolVLA 用，未裁剪 |
| `vla_batch_20261007_smolvla_6d_no_depth_rollunwrap_scan` | 27312 | 展开 | SmolVLA 用，已裁剪 |
| `vla_batch_20261007_smolvla_6d_no_depth_scan` | 27312 | **wrap** | **旧版，勿再使用** |

---

## 二、超声图像的产生链路

超声图不是相机拍的，是**从原始数据包解码渲染**得到的。这条链路在采集端：

```
ultrasound_packets.bin            # PhaseLink 原始数据包
  → decode_s_scan(packet, beam_count, display_max_amplitude_percent)
  → amplitude_u8                   # 幅度量化，按显示动态范围
  → color_Amplitude 调色板         # 256 级，QRgb 0xAARRGGBB → BGR
  → 256×256×3 uint8
  → 编码入数据集 observation.images.ultrasound
```

调色板从厂商源码 `amplitudpalette.cpp` 的 `color_Amplitude[]` 提取，
解码逻辑**对齐 PhaseLink 的 `ViewWidget::showEScan`**。

**含义**：超声图是**渲染结果**，不是测量值。模型学到的是该调色板与动态范围下的
视觉表征。改变 `display_max_amplitude_percent`、增益或量程，
即使物理扫查完全一致，进模型的图也会不同。
**推理时必须使用与训练时一致的 `ultrasound_metadata.json` 配置。**

---

## 三、推理链路契约

```
   robot-pc (100.64.0.2)                      gpu-server (100.64.0.1)
   ┌────────────────────────┐                 ┌──────────────────────────┐
   │ 相机 / 超声 / xArm     │  ──gRPC──────▶  │ serve.py                 │
   │ 动作块 → 机械臂        │  ◀────────────  │  (LeRobot PolicyServer)  │
   └────────────────────────┘                 └──────────────────────────┘
```

### 传输层

| 项 | 值 |
|---|---|
| 协议 | LeRobot `async_inference`，gRPC |
| 注册 | `services_pb2_grpc.add_AsyncInferenceServicer_to_server` |
| 序列化 | **pickle**（`TimedObservation` / `TimedAction` / `RemotePolicyConfig`） |
| 默认端口 | 8080 |
| 压缩 | 工控机侧默认 `--grpc-compression gzip` |

pickle 记录的是**类路径 + 类名**。瘦客户端没有 LeRobot，
因此必须在 `sys.modules` 中注入同名的垫片模块，且**字段名与方法名都要一致**。
只补字段不补方法的后果见[文档 06 第二节](06-remote-client.md)——异常发生在接收线程内，
表现为「观测一直在发、动作一个都不来」，无任何报错。

### 服务端参数（`serve.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `8080` | |
| `--device` | `cuda` | 写死，不给客户端覆盖 |
| `--fps` | `15` | **不要用上游默认的 30** |
| `--obs-queue-timeout` | `2.0` | |
| `--wire` | `torch` | 瘦客户端**必须**传 `numpy` |

`--wire` 是唯一为「对面没有 torch」做的让步：`torch` 回传 `torch.Tensor`，
`numpy` 回传 `np.ndarray`。真机的 `RobotClient` 需要 torch。

### 客户端参数

瘦客户端（`remote/remote_client.py`）与真机客户端（`real_robot_client.py`）
共用同一套协议，差异只在下列两项：

| 参数 | 瘦客户端 | 真机客户端 | 说明 |
|---|---|---|---|
| `--fps` | 15 | 15 | |
| `--actions-per-chunk` | 50 | 50 | |
| `--chunk-size-threshold` | **0.5** | **0.9** | 真机更保守：队列保持在 90% 以上才不发新观测 |
| `--wire` | 必须 `numpy` | `torch` | |

### 三个必须遵守的协议行为

1. **`must_go` 是活性机制，不是可选优化。**
   上游 `robot_client.py:337` 在收到动作块后置位。队列一空，下一个观测即带
   `must_go=True`，服务端无条件收下并保证再推理一次。缺这一句，后续观测
   一旦落进服务端的 `_predicted_timesteps` 就会被**静默丢弃**。

2. **相机键名必须直接写为 policy 的键名**（见第一节映射表）。

3. **`_reset_server()` 重置不完整。**
   上游未重置 `last_processed_obs` 与 `fps_tracker`，重连后新会话的观测
   会被误判为陈旧而丢弃。详见[文档 06 第四节](06-remote-client.md)。

---

## 四、带宽预算

| 路径 | 单帧 | 15Hz 峰值 | 说明 |
|---|---|---|---|
| 图像预缩至 224×224 | **0.30 MB** | 36 Mbps | 实测约占 28% 迭代 → 均值约 10 Mbps |
| 原分辨率 720×1280 | **2.96 MB** | **355 Mbps** | 协议对图像不做压缩，直接 pickle 原始 uint8 |

224→224 的 bilinear 是**精确恒等**（已实测），因此预缩不构成降质，
送进模型的张量与全分辨率路径逐位相同。

> 真机路径若要传原分辨率，**必须先解决 355 Mbps 这一项**，无线链路无法承载。
> 当前的 `real_robot_client.py` 默认 `--image-size 224`，即仍在走预缩路径。

---

## 五、真机路径的安全性契约

`real_robot_client.py` 是本项目风险最高的文件。**默认是影子模式**：
连接 xArm 读取状态、读 RealSense 与超声、调用远程策略、记录预测动作，
但**不调用任何运动 API**。

真正下发运动需要**两个条件同时满足**：

```powershell
--execute --confirm EXECUTE_REAL_XARM_POLICY
```

且**不会自动移动到起始位姿**——机械臂必须先由已验证的定位流程摆好。

### 保护阈值（均为保守调试值）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--sensor-max-age` | 0.5 s | RealSense / 超声 / xArm 任一数据超龄立即停止 |
| `--max-force-n` | 40 N | 外力合力上限 |
| `--max-torque-nm` | 5 N·m | 外力矩上限 |
| `--max-action-step-mm` | 3.0 mm | 平移单步限幅 |
| `--max-action-step-deg` | 2.0° | 转角单步限幅 |
| `--max-tracking-error-mm` | 12.0 mm | 目标距实际 TCP 上限 |
| `--max-tracking-error-deg` | 5.0° | 同上（角度） |
| `--start-tolerance-mm` / `-deg` | 8.0 / 5.0 | 起始位姿容差 |
| `--trajectory-tube-mm` / `-deg` | 15.0 / 8.0 | 历史轨迹走廊（需显式启用） |
| `--max-backtrack-points` | 20 | 走廊内回退上限 |
| `--max-forward-jump-points` | 100 | 走廊内前跳上限 |
| `--servo-speed` / `--servo-acceleration` | 10.0 / 500.0 | |
| `--max-seconds` | 10.0 | 单次运行时长上限 |
| `--initial-action-timeout` | 180.0 s | 等首个动作块（服务端要加载 3B 模型，约 110 s） |

另有：动作队列为空时**保持等待，不重复旧动作**；默认不因断供超时退出；
错误码、警告码、断联、SDK 返回非零一律停止；异常或 Ctrl+C 进入 xArm 停止状态，
**不自动清错**。

> **这些阈值不可为了让异常策略「跑起来」而放宽。**
> 若确需调整，应连同理由一并记入文档，而不是就地改数字。

---

## 六、改动同步清单

修改下列任一项时，两侧必须同步，并更新本文档：

- [ ] `state` / `action` 的维序、单位、含义
- [ ] 相机键名映射，或数据集特征增删
- [ ] roll 口径
- [ ] 数据集的 `fps` 或 `codebase_version`
- [ ] 超声解码的调色板或动态范围配置
- [ ] `--fps`、`--actions-per-chunk`、`--chunk-size-threshold`
- [ ] 任何安全阈值
