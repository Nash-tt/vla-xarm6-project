# 手拖示教模式采集（非当前拟合轨迹运行流程）

当前项目需要的是“提前采点后自动运行拟合轨迹”的同步采集，请优先使用
`README_FITTED_TRAJECTORY_VLA_SYNC.md` 和
`run_fitted_trajectory_vla_sync.py`。

本文件及 `collect_teach_vla_sync.py` 仅保留给以后确实需要记录手拖示教过程的场景。

## 结论

不要让机械臂、RealSense、超声三个采集线程互相等待。三路数据应按各自原始频率连续写盘，并在数据到达采集电脑时立即记录同一台电脑的时间。采集完成后，再统一生成 15 Hz（或其他频率）的训练样本。

本目录新增：

- `collect_teach_vla_sync.py`：采集示教模式机械臂报告、RealSense RGB-D，并启动 PhaseLink。
- `align_vla_episode.py`：将三路原始数据对齐为固定频率的 VLA 样本。

PhaseLink Qt 工程新增：

- `vladatapacketrecorder.h/.cpp`：保存每个超声原始包和接收时间。

## 为什么现有脚本不够

`run_scan_collect_vla_realsense.py` 在机械臂发送一个伺服目标后读取“最新 RealSense 帧”。这会把机械臂循环频率、JPEG/PNG 写盘时间和相机线程调度混在一起。它能做近似关联，但无法量化两路数据相差多少。

PhaseLink 包尾的 `timestamp` 是设备侧 200 MHz 计数。它不是 Unix 时间，不能直接与 Python 的 `time.time()` 相减。新方案同时保留设备计数和主机接收时间，后处理用线性时钟映射去除大部分接收抖动。

## 一次性准备

1. 使用 Qt Creator 打开：

   `超声设备/cs_phaselink/Phaselink/Phaselink.pro`

2. 重新执行 qmake，然后 Build。

3. 确认生成的 `Phaselink.exe` 是重新编译后的版本。未重新编译的旧 exe 不会读取 `PHASELINK_VLA_DIR`，因此不会生成超声同步文件。

4. 安装 Python 依赖：

   ```powershell
   python -m pip install -r requirements_vla_realsense.txt
   ```

## 采集

在 `my_code` 目录运行：

```powershell
python collect_teach_vla_sync.py
```

流程如下：

1. 程序创建唯一 episode 目录。
2. 程序用 `PHASELINK_VLA_DIR=<episode目录>` 启动 PhaseLink。
3. RealSense 完成预热并开始保存 RGB-D。
4. xArm 进入 `mode=2` 关节示教模式，并通过 SDK 的 `real` report callback 连续保存高频状态。
5. 在 PhaseLink 中连接设备并点击 `Start scan`。
6. 回到终端按 Enter，开始有效同步窗口，然后拖动机械臂完成示教。
7. 再按 Enter，停止有效同步窗口。
8. 在 PhaseLink 中点击 `Stop scan` 并关闭窗口，使末尾缓冲和统计信息完整写盘。

固定时长采集示例：

```powershell
python collect_teach_vla_sync.py --duration 30
```

如果 PhaseLink 的 exe 不在默认位置：

```powershell
python collect_teach_vla_sync.py --phaselink-exe "D:\path\to\Phaselink.exe"
```

## 对齐为 15 Hz

```powershell
python align_vla_episode.py --episode "teach_vla_runs\teach_vla_YYYYMMDD_HHMMSS" --fps 15
```

建议第一次不要加 `--drop-invalid`。先查看 `alignment.json` 中的：

- 三路输入数量；
- device clock fit 是否启用；
- 接收时间残差；
- 无效样本数量；
- 各路最大时间偏差阈值。

确认数据质量后，可输出只包含有效样本的文件：

```powershell
python align_vla_episode.py --episode "teach_vla_runs\teach_vla_YYYYMMDD_HHMMSS" --fps 15 --drop-invalid
```

## 每个 episode 的文件

```text
teach_vla_YYYYMMDD_HHMMSS/
  metadata.json
  robot_records.jsonl
  realsense_metadata.json
  realsense_frames.jsonl
  color/*.jpg
  depth/*.png
  ultrasound_metadata.json
  ultrasound_frames.jsonl
  ultrasound_packets.bin
  aligned_samples.jsonl
  alignment.json
```

`ultrasound_frames.jsonl` 中的 `binary_offset` 和 `binary_size` 用于从 `ultrasound_packets.bin` 随机读取某一帧完整原始包。训练前可以再按 PhaseLink 的包格式解码成 A-scan、门控值或渲染图像；不要在实时采集回调里做耗时的图像转换。

## 对齐策略

- 机械臂：使用 report callback 的主机到达时间；在相邻报告间插值 TCP 和关节角。
- RealSense：保留设备毫秒时间戳和主机到达时间，拟合设备时钟到主机时钟；每个目标时刻选最近 RGB-D 帧。
- 超声：保留包尾 200 MHz 时间戳和主机到达时间，拟合设备时钟到主机时钟；每个目标时刻选最近原始包。
- 示教没有“下发动作”，所以 VLA 的 action 使用下一个采样时刻的实际关节状态、TCP 状态和关节增量作为行为克隆标签。

## 精度边界

这是软件时间同步，适合 15 Hz VLA 数据采集，通常目标是把偏差控制在一帧以内，并且让每条样本都带可检查的 `alignment_skew_ms`。它不是硬件同步：

- 主机接收时间包含 USB、网络和线程调度延迟；
- RealSense 图像曝光时刻与帧送达电脑的时刻之间有固定延迟；
- 超声设备时间戳与网络包送达之间也有固定延迟；
- 软件拟合能减小抖动，但无法自动消除不同设备的固定传输延迟。

默认使用 `--robot-report-type real`，优先保证运动状态频率。如果控制器固件与 real report 不兼容，可临时改为 `--robot-report-type rich`，但必须在 `alignment.json` 中确认机械臂采样率和 `robot_skew_ms` 是否满足 15 Hz 对齐要求。

如果实验要求毫秒级相位一致，应增加公共硬件触发脉冲，并让支持外触发的 RealSense、超声设备和机械臂控制箱 I/O 同时记录该脉冲。软件时间戳仍应保留，用于检测漏触发和恢复帧对应关系。

## 采集前质量检查

- 采集电脑关闭自动休眠和 USB 选择性暂停。
- RealSense 使用 USB 3.x，避免与超声网卡共享拥塞的扩展坞。
- episode 目录写入本地 SSD。
- 机械臂、相机和探头标定结果单独版本化：相机外参、TCP 到探头成像平面的外参、深度尺度。
- 每次采集保存任务文本、患者/仿体匿名 ID、探头型号、超声参数和标定版本。
- 原始数据只追加不覆盖；对齐和转换属于可重复生成的派生数据。
