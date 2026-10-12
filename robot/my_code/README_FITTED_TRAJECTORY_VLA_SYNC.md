# 拟合轨迹运行阶段的超声 VLA 三路同步采集

## 对应的实际流程

本方案针对以下流程：

1. 提前使用 `collect_teach_points_any_count.py` 采集若干示教点；
2. `run_single_curve_from_latest_points_servo_soft_pose.py` 读取
   `teach_points_any_count.json`；
3. 使用 Catmull-Rom 曲线拟合位置和姿态，并按距离重新采样；
4. 机械臂以 `set_servo_cartesian` 自动运行拟合轨迹；
5. 轨迹运行期间同步采集机械臂、RealSense 和超声数据。

示教点采集阶段只是轨迹生成的准备步骤，不是本 VLA episode 的有效采集窗口。

## 代码入口

- `run_fitted_trajectory_vla_sync.py`：自动运行拟合轨迹并采集三路原始数据；
- `realsense_sync_recorder.py`：独立记录 RealSense RGB-D；
- `align_vla_episode.py`：采集后对齐成固定帧率样本；
- PhaseLink 的 `vladatapacketrecorder.h/.cpp`：保存超声原始包。

`run_scan_collect_vla_realsense.py` 是早期两路原型：它把“最新相机帧”挂到伺服
步骤，并在伺服循环中读取机器人属性，不符合本三路原速采集架构。实际采集只使用
`run_fitted_trajectory_vla_sync.py`。

轨迹采集程序直接复用以下已有参数，不重新实现轨迹拟合：

- `build_curve()`；
- `DRY_RUN_LIFT_MM`；
- `SERVO_SPEED_MM_S`；
- `SERVO_ACC_MM_S2`；
- `SERVO_PERIOD_SECONDS`；
- `PATH_POINT_SPACING_MM`；
- `TEACH_POINTS_PATH`。

因此修改原轨迹文件中的安全抬高量、速度或点间距后，三路采集程序会使用相同配置。

当前原轨迹文件中的 `DRY_RUN_LIFT_MM = 30.0`，探头会比示教曲线抬高
30 mm，适合无接触安全试跑，但通常无法采到有效耦合超声。完成轨迹、TCP、探头
安装和急停验证后，真实接触采集才可将该值改为经过确认的接触偏移（常见为
`0.0`，但应以实际 TCP/探头标定为准）。

当前静态配置为：机器人 IP `192.168.1.100`；伺服周期 `0.02 s`；轨迹点距
`0.2 mm`（对应名义轨迹推进速度约 `10 mm/s`）；传给 SDK 的 `speed=10 mm/s`、
`mvacc=500 mm/s²`。本 SDK 1.17.6 文档将 `set_servo_cartesian` 的 speed/mvacc
标为 reserved，因此真正的流式推进主要由点距和周期决定。移动到起点的
`set_position` 速度为 `10 mm/s`，未显式指定起点移动加速度。当前示教点文件是
`my_code/teach_points_any_count.json`，其中 TCP offset 为 `[0,0,0,0,0,0]`；采集
程序会在移动前恢复该 TCP，但这仍表示“未完成探头接触点 TCP 标定”，不能直接视为
接触采集配置。

## 一次性编译 PhaseLink

使用 Qt Creator 打开：

```text
PAU_SDK - dt/Phaselink/Phaselink.pro
```

重新执行 qmake 并 Build。必须确保新生成的 `Phaselink.exe` 包含
`vladatapacketrecorder.cpp`。采集脚本会比较 exe 与 recorder、mainwindow、pro
等接入源码的修改时间；如果 exe 仍是旧版本，会拒绝开始采集，避免生成缺少超声
数据的 episode。当前源码使用独立写盘线程，因此 `Client::DataPacket` 回调只在最
前面取主机时间并入队，不执行二进制或 JSON 写盘。

## 运行

```powershell
cd "<xArm-SDK>\my_code"
python run_fitted_trajectory_vla_sync.py
```

程序默认打开名为 `RealSense RGB - <episode_id>` 的实时彩色预览窗口。预览使用
独立线程和容量为 1 的“最新帧”队列，不会等待 JPEG/PNG 写盘，也不会阻塞相机取帧
或机械臂伺服循环。按 `Q`、`Esc` 或关闭窗口只会关闭预览，RGB-D 记录继续进行。
如现场不需要窗口，可显式使用：

```powershell
python run_fitted_trajectory_vla_sync.py --no-realsense-preview
```

运行顺序：

1. 根据最新的 `teach_points_any_count.json` 生成拟合轨迹；
2. 创建唯一 episode，并保存一份 `trajectory.json`；
3. 用当前 episode 目录启动 PhaseLink；
4. 启动并预热 RealSense；
5. 连接 xArm，启动高频 robot report 记录；
6. 机械臂以位置模式移动到拟合轨迹起点；
7. 恢复示教点文件记录的 TCP offset，并切换到笛卡尔伺服模式；
8. 在 PhaseLink 中连接超声设备、点击 `Start scan` 并检查耦合；
9. 回到终端按 Enter，程序开启有效时间窗口并自动执行拟合轨迹；
10. 轨迹结束后停止有效窗口；
11. 在 PhaseLink 中点击 `Stop scan` 并关闭程序，完成超声文件刷新。

如果 PhaseLink exe 不在默认位置：

```powershell
python run_fitted_trajectory_vla_sync.py `
  --phaselink-exe "D:\path\to\Phaselink.exe"
```

默认采用 xArm `real` report，优先保证运动状态采样率。如果控制器固件不兼容：

```powershell
python run_fitted_trajectory_vla_sync.py --robot-report-type rich
```

## 原始文件

```text
fitted_scan_vla_YYYYMMDD_HHMMSS/
  metadata.json
  trajectory.json
  robot_records.jsonl
  robot_actions.jsonl
  realsense_metadata.json
  realsense_frames.jsonl
  color/*.jpg
  depth/*.png
  ultrasound_metadata.json
  ultrasound_frames.jsonl
  ultrasound_packets.bin
```

其中：

- `robot_records.jsonl` 是控制器实际上报的机器人状态；
- `robot_actions.jsonl` 是每次实际调用 `set_servo_cartesian` 的目标、调用时间、
  返回码和相对调度延迟；
- `trajectory.json` 是本 episode 使用的完整拟合轨迹；
- `ultrasound_packets.bin` 是连续保存的原始超声包；
- `ultrasound_frames.jsonl` 保存每个超声包的二进制偏移、长度、主机时间戳和
  200 MHz 设备时间戳。

不要把下发的目标位姿当成实际机械臂状态。VLA 样本中：

- observation.robot 来自 `robot_records.jsonl`；
- action 来自 `robot_actions.jsonl`。

对齐器要求 `robot_actions.jsonl` 非空，不再用“下一时刻观测状态”伪造 action。
某个目标时刻早于第一条命令完成时间时，该样本会标为无效；action 的生效时间以
`command_end_unix_ns`（SDK 调用返回）为准。

## 对齐成 15 Hz

```powershell
python align_vla_episode.py `
  --episode "fitted_trajectory_vla_runs\fitted_scan_vla_YYYYMMDD_HHMMSS" `
  --fps 15
```

输出：

- `aligned_samples.jsonl`；
- `alignment.json`。

每个样本包含：

- 目标时间；
- 插值后的实际机器人状态；
- 最近的 RealSense RGB-D 帧；
- 最近的超声原始包索引；
- 当时生效的最近一条笛卡尔伺服命令；
- 四路时间偏差和 `quality.valid`。

建议第一次保留无效样本，先检查 `alignment.json`。确认阈值合理后再运行：

```powershell
python align_vla_episode.py `
  --episode "fitted_trajectory_vla_runs\fitted_scan_vla_YYYYMMDD_HHMMSS" `
  --fps 15 `
  --drop-invalid
```

## 同步原理

三个数据源不能在 Python 循环中互相等待：

- xArm report 按控制器上报频率独立记录；
- RealSense 取帧线程按相机帧率复制 RGB-D 缓冲，JPEG/PNG 压缩与写盘由另一线程执行；
- PhaseLink 按超声包到达频率入队，原始二进制和 JSON 索引由另一线程连续写盘；
- 伺服控制循环只负责按 0.02 秒周期下发轨迹，不进行图像保存或超声解码。

所有记录都带 `host_unix_ns`。RealSense 和超声还保留各自设备时钟，后处理通过
线性拟合将设备时钟映射到主机时间轴，减少 USB/网络接收抖动。机械臂状态在相邻
report 之间插值，RealSense 和超声使用最近邻。

这是适用于 15 Hz VLA 数据的软件同步。若要求小于约 1–5 ms 的严格同步，还需
增加公共硬件触发，并标定不同设备的固定采集/传输延迟。

## 首次短采必须检查

- `robot_dropped_count` 应为 0；
- `realsense_dropped_count` 应接近 0；
- `robot_action_count` 应等于 `trajectory_points`；
- `trajectory_completed` 应为 true；
- `alignment.json` 中两种 device clock fit 应启用；
- `schedule_lag_ms` 不应持续增大；
- 15 Hz 输出中的 action 偏差通常应小于一个 0.02 秒伺服周期附近；
- 超声包数应覆盖完整 `capture_window`。
