# 01 · 采集链路

**跑在哪：** 工控机（Win10），代码在 `<xArm-SDK>\my_code\`，仓库副本 `robot/my_code/`。
**产出：** 一个 episode 目录，里面是四路**各自带时间戳**的原始记录。
**下一棒：** [02 对齐与建集](02-align-and-build.md)。

---

## 一、核心设计：四路各记各的，不共用时钟

采集时不试图把四路数据"同步着写进一个文件"。xArm、RealSense、超声三者各有自己的时间基准，
硬凑只会把误差藏进数据里：

| 路 | 有没有设备时钟 | 记什么 |
|---|---|---|
| xArm 状态 | **无**。SDK 回调只给一个 `count`，不是时间 | SDK 回调里立刻打主机 `time.time_ns()` |
| xArm 下发目标 | 同左 | 每次 `set_servo_cartesian` 前后打主机时间 |
| RealSense | **有**，设备钟毫秒（带亚毫秒小数） | 设备时间戳 + 收到帧时的主机时间戳 |
| 超声 | **有**，PhaseLink 的 200 MHz 计数 | 帧索引里的设备计数 + 写盘时的主机时间戳 |

所以主机 wall clock 是唯一的公共参照，**每一路都要在到达的那一刻被"盖一个主机时间戳"**。
设备钟留在记录里不用，直到对齐那一步才用最小二乘映射到主机钟（见 [02](02-align-and-build.md)）。

这样做的代价是文件多、格式碎；收益是**任何一步的错误都能回溯**——
时钟拟合残差大就直接不用设备钟，退回主机到达时间，不必重采。

---

## 二、四步

```
① collect_teach_points_any_count.py     手动拖到各位姿按 Enter
        → teach_points_*.json（实测 TCP 位姿，含 robot_ip / tcp_offset）
        │
        ▼
② generate_intermediate_trajectories.py 相邻实测轨迹之间插值
        → generated_set_*/intermediate_*.json（只内插，不外推）
        │
        ▼
③ run_single_curve_from_latest_points_servo_soft_pose.py
        │                                单条曲线的伺服连续扫描（姿态沿路径插值）
        │                ★ 这里定义 SERVO_* 全部参数，④ 直接 import 复用
        ▼
④ run_fitted_trajectory_vla_sync.py     ★ 三路同步采集器（链路主线，1059 行）
        → episode 目录
```

③ 和 ④ **共用同一套伺服参数和同一条 `build_curve()`**：④ 从 ③
`import` 了 `SERVO_PERIOD_SECONDS`、`SERVO_SPEED_MM_S`、`SERVO_ACC_MM_S2`、
`PATH_POINT_SPACING_MM`、`DRY_RUN_LIFT_MM`、`build_curve` 等十余项
（`run_fitted_trajectory_vla_sync.py:33-51`）。

**这不是"复用"，是刻意的绑定。** ③ 是离线把轨迹跑一遍的调试入口，④ 是带采集的正式跑法。
两者若参数不同，④ 的采集结果就与调试时看到的运动对不上。要调参就在 ③ 里改，④ 自动跟着变。

---

## 三、伺服环：50 Hz，绝对时刻驱动

`run_single_curve_from_latest_points_servo_soft_pose.py:27-35` 定死了这套参数：

| 常量 | 值 | 含义 |
|---|---|---|
| `SERVO_PERIOD_SECONDS` | **0.02** | 50 Hz 伺服周期 |
| `SERVO_SPEED_MM_S` | 10.0 | 伺服速度上限 |
| `SERVO_ACC_MM_S2` | 500.0 | 伺服加速度上限 |
| `PATH_POINT_SPACING_MM` | 0.2 | 路径点间距 |
| `POINTS_PER_SEGMENT` | 50 | 每段插值点数 |
| `DRY_RUN_LIFT_MM` | 0.0 | 现行版按示教高度实跑 |
| `START_DELAY_SECONDS` | 5 | 起跑前留 5 秒 |

**下发不是"每 20 ms 循环一次"，而是对齐到绝对时刻。** `run_fitted_trajectory_vla_sync.py:899-967`：

```python
servo_start_monotonic_ns = time.monotonic_ns()
next_deadline_ns = servo_start_monotonic_ns
for index, target in enumerate(curve):
    ...
    while time.monotonic_ns() < next_deadline_ns:   # 等到这一拍的绝对时刻
        pass
    arm.set_servo_cartesian(target, speed=..., mvacc=...)
    next_deadline_ns += int(round(SERVO_PERIOD_SECONDS * 1e9))   # 累加，不重置
```

关键在最后一行是 `+=` 而不是 `= now + period`。前者让每拍的时刻由**起点**唯一决定，
延迟不会被累积进下一个周期（不漂移）；后者会把每次的抖动加进时间轴，跑 1000 拍就偏出去。
这在伺服连续流里是必须的——`set_servo_cartesian` 的语义是"以固定周期喂点"，
喂的节奏一漂，控制器就会插值出非预期的路径。

> **动作日志不占伺服环。** `robot_actions.jsonl` 的序列化在独立线程做
> （`:226` 注释即为此）。50 Hz 的环里不能有文件 I/O。

---

## 四、三路数据是怎么进来的

### 4.1 xArm 状态：`RobotReportRecorder`（来自 `collect_teach_vla_sync`）

起在 `:822-826`，输出 `robot_records.jsonl`。SDK 上报回调里打主机时间戳并塞进队列，写盘在另一个线程。
元数据里记 `robot_record_count` 与 `robot_dropped_count`——**丢帧数要盯着**，
队列满时是静默丢的，只有这个计数能看出来。

### 4.2 RealSense：`RealSenseRecorder`（`realsense_sync_recorder.py`，461 行）

起在 `:792-807`。默认 `1280×720 @ 30 FPS`（`:604-620`），彩色与深度对齐。
每帧同时记设备时间戳与主机时间戳，写 `realsense_frames.jsonl`（帧本体落盘为图像文件）。
预览窗口是可选独立线程（`--no-realsense-preview` 关掉），**不阻塞采集**。

元数据里记 `realsense_frame_count` / `received_count` / `dropped_count` /
`queue_capacity` / `depth_png_compression`，还专门记了
`realsense_preview_closed_by_user` 与 `preview_error`——预览窗口被用户关掉是个真实故障源，
单独记一个字段以免和采集失败混为一谈。

### 4.3 超声：PhaseLink 写盘 + 本进程旁路监控

这是三路里最绕的一路。**本进程不通过 API 拿超声数据**，而是：

1. 启动 `Phaselink.exe`（`launch_phaselink`，可用 `--no-launch-phaselink` 跳过）
2. 通过 `PHASELINK_VLA_DIR` 告诉它把原始包写到哪
3. **提前拒绝开动**：`:186` `require evidence of raw PhaseLink packets before servo motion starts`
   ——文件里必须已经有包，否则不起伺服。理由很直白：超声没在工作而机械臂已经跑完，
   这一趟就白采了，不如在动之前失败。
4. 监控 `ultrasound_packets.bin` + `ultrasound_frames.jsonl` 的增长

超声的采集参数（增益、量程、阵元、TCG 曲线……）在
`run_fitted_trajectory_vla_sync.py:56-120` 以 `ULTRASOUND_REQUESTED_SETTINGS`
预设写死，预设名 `contact_scan_v1`。**改动这里等于改动数据的物理含义**，
必须同步[接口契约](../docs/09-interface-contract.md)。

### 4.4 采集窗口

`capture_window` 在 `metadata.json` 里，记成对的 `start_unix_ns` / `start_monotonic_ns`
与 `end_*`。**窗口包含接近段**——从 capture-start 位姿经"定高平移 → 垂直下压"
到轨迹第一点的全过程都在窗口内（`:689-694` 的 `staged_approach`，
且 `included_in_capture_window: True`）。

这是有意为之：让策略看到"探头怎么接近工件"，而不只是"已经在扫了"。
代价是数据集里混着大量接近段——**所以裁剪是必须的**，见 [03](03-dataset-tools.md)。

---

## 五、开动前会拒绝的几种情况

采集脚本在真正下伺服指令前有一串门，全部是"拒绝执行"而不是"警告后继续"：

| 门 | 位置 | 行为 |
|---|---|---|
| 机器人 IP 与示教点记录不一致 | `:640-647` | 抛错拒动，除非显式 `--allow-robot-ip-mismatch` |
| 磁盘剩余 < 10 GB | `prepare_output_root` / `DEFAULT_MIN_FREE_GB` | 拒动 |
| 宽/高/FPS 非正 | `:633-637` | 拒动 |
| 超声没有出包证据 | `:186` | 拒动 |
| 机器人未就绪（10 s 超时，需稳定 0.5 s） | `ROBOT_READY_*` | 等，超时失败 |
| 超声未就绪（10 s 超时） | `ULTRASOUND_READY_*` | 等，超时失败 |

IP 那条值得单说：它比对的是**示教点 JSON 里记的 robot_ip**。
换了一台臂而沿用旧示教点，位姿标定整个作废，这时候宁可跑不起来。

---

## 六、episode 目录长什么样

`DEFAULT_OUTPUT_ROOT` 默认 `D:\fitted_trajectory_vla_runs`（可用环境变量
`VLA_OUTPUT_ROOT` 覆盖），每个 episode 一个子目录：

```text
<output_root>/<episode_id>/
├── metadata.json            # schema: xarm_realsense_phaselink_fitted_trajectory_episode_v1
│                            #   指令、采集窗口、各路的计数/丢帧、预设参数
├── trajectory.json          # schema: fitted_pose_trajectory_v1
│                            #   示教点、TCP 偏移、伺服参数、逐点目标位姿
├── robot_records.jsonl      # xArm 状态观测（SDK 回调，带主机时间戳）
├── robot_actions.jsonl      # 下发的伺服目标（schema: xarm_cartesian_servo_action_v1）
├── realsense_frames.jsonl   # RGB-D 帧索引（设备钟 + 主机钟）
├── ultrasound_packets.bin   # PhaseLink 原始包（未解码）
├── ultrasound_frames.jsonl  # 帧索引（设备计数 + 主机钟）
└── ultrasound_metadata.json # 超声侧解码配置
```

`metadata.json` 里的 `files` 段（`:735-741`）是这些文件名的**权威声明**，
对齐脚本按它找文件，不靠猜。

---

## 七、已知问题

**① `run_robot_trajectory_with_realsense_preview.py` 跑不起来（ImportError）。**
第 22 行 `from run_fitted_trajectory_vla_sync import prepare_robot_at_start`，
而 `run_fitted_trajectory_vla_sync.py` **没有这个函数**——它定义的是
`prepare_robot_for_capture`。全仓库 `grep` 只有这一个文件的两处引用
（第 22 行的 import 与第 219 行的调用）。

这不是本次整理引入的，是原件里就存在的。**没有在仓库里就地修**：
`robot/my_code/` 是工控机的镜像（见 [10 文档](../docs/10-collaboration.md)），
在这里改会与机器上的原件分叉。要修就在工控机上修，再同步回来。

**② 示教模式采的 episode 接不上对齐。**
`collect_teach_vla_sync.py` 全程不写 `robot_actions`（`grep -c robot_actions` 为 0），
而 `align_vla_episode.py:275` 在动作记录为空时直接抛错。
即：手拖示教采出来的数据**当前无法进入训练集**。现行流程走的是自动扫描
（③④），不依赖示教模式，所以不影响现状——但那条路是断的。

**③ 采集产物与运行日志不入库。** 见 [10 文档第四节](../docs/10-collaboration.md)。
