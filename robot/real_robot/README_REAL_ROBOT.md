# 真机远程策略客户端

入口：`real_robot_client.py`

默认是影子模式：连接xArm读取状态、读取RealSense和PhaseLink、调用远程策略、
记录预测动作，但不会调用任何机器人运动API。

执行模式不会自动移动到起始位姿。机械臂必须先由已经验证过的定位流程放到
本次任务的起始位姿。

> **实现方案见 [`design/07-real-robot-client.md`](../../design/07-real-robot-client.md)**——
> 三道安全层各自拦什么、三路数据源的时效检查、控制环的四道出口，以及一处文档纠正
> （xArm 的时效是硬编码 0.25 s，**不受** `--sensor-max-age` 影响）。
> 本文是操作步骤，与 `design/` 冲突时以 `design/` 和代码为准。

## 0. 依赖的两个模块

本文件 `import remote_client`（传输层）与 `export_ultrasound_images`（超声解码）：

- `remote_client.py` —— 本仓库里**只有一份**，位于 `pi0_ultrasound_lora/remote/`；
- `export_ultrasound_images.py` —— 在 `<xArm-SDK>\my_code\`。

所以**本目录不能单独跑**：必须把整个 `pi0_ultrasound_lora/remote/` 与本文件放进
同一个目录（工控机上即 `<deploy-dir>`）。若只拷贝本文件，会在 `import remote_client`
处直接失败——这是有意为之，避免静默加载到一份旧副本。详见 `robot/README.md` 第 3.3 节。

## 1. 启动PhaseLink原始包输出

先在PowerShell创建本次实时目录，再从同一个PowerShell启动带VLA记录器的
Release版本PhaseLink：

```powershell
$liveDir = "<deploy-dir>\live_ultrasound_current"
New-Item -ItemType Directory -Path $liveDir -Force
$env:PHASELINK_VLA_DIR = $liveDir

& "<PAU_SDK>\build-Phaselink-Desktop_Qt_5_12_10_MSVC2015_64bit-Release\release\Phaselink.exe"
```

在PhaseLink中连接设备、设为训练时的参数并点击开始扫查。确认目录中
`ultrasound_packets.bin` 和 `ultrasound_frames.jsonl` 正在增长。

## 2. 只检查代码和配置（不连接任何设备）

```powershell
$python = "<xArm-SDK>\.venv_vla\Scripts\python.exe"

& $python <deploy-dir>\real_robot_client.py `
  --validate-only
```

## 3. 影子测试（连接真机，但绝不下发运动命令）

远端服务必须用 `--wire numpy` 启动。然后运行：

```powershell
& $python <deploy-dir>\real_robot_client.py `
  --server 100.64.0.1:8080 `
  --robot-ip 192.168.1.100 `
  --grpc-compression gzip `
  --max-seconds 10
```

先检查 `<deploy-dir>\real_runs\...\live_policy_records.jsonl`：

- `shadow_valid` 应持续为 `true`；
- 目标位姿应靠近实际位姿和参考轨迹；
- 观测RPC最大延迟不能让动作队列归零；
- RealSense和超声帧不能过期；
- 力传感器不能全零。

## 4. 有界真机测试

只有影子测试通过、机械臂已经位于任务起始位姿、急停可触达、负载/TCP/力传感器
设置正确时才使用。以下命令会真实运动：

```powershell
& $python <deploy-dir>\real_robot_client.py `
  --server 100.64.0.1:8080 `
  --robot-ip 192.168.1.100 `
  --grpc-compression gzip `
  --max-seconds 3 `
  --execute `
  --confirm EXECUTE_REAL_XARM_POLICY
```

首次执行建议只跑3秒。默认保护：

- 动作队列为空时保持等待，不重复旧动作；默认不因断供超时退出；
- RealSense、超声或xArm报告超过0.5秒立即停止；
- 平移单步最大3 mm，转角单步最大2°；
- 目标距离实际TCP最大12 mm / 5°；
- 默认直接执行策略输出，不读取历史轨迹；
- 外力合力最大40 N，外力矩最大5 N·m；
- 错误码、警告码、断联、SDK返回非零都会停止；
- 异常或Ctrl+C进入xArm停止状态，不自动清错。

这些阈值是保守调试值，不应为了让异常策略“跑起来”而随意放宽。

如果希望额外启用历史轨迹走廊约束，可显式增加：

```powershell
--safety-trajectory "D:\某个episode\trajectory.json"
```

实时超声默认使用已经从训练数据复制出来的固定解码配置：

```text
<deploy-dir>\config\training_ultrasound_metadata.json
```

来源为 `<data-root>\generated_set_1_01_01\ultrasound_metadata.json`。
正常运行不再需要传入 `--ultrasound-settings-metadata`；仅当以后更换训练数据的
超声配置时才需要覆盖该参数或替换固定配置文件。

实时超声原始包目录也固定为：

```text
<deploy-dir>\live_ultrasound_current
```

正常运行不再需要传入 `--ultrasound-live-dir`。但必须在启动 PhaseLink 之前设置
`PHASELINK_VLA_DIR` 为这个固定目录；客户端不会自动启动或重启 PhaseLink。

如果后续需要恢复断供超时保护，可增加例如
`--queue-empty-stop 3.0`，表示连续3秒没有动作时停止。

## 可选的限速策略模式

现有“直接执行模型绝对目标”的行为仍然是默认模式。需要把远距离模型目标
转换为连续的小步笛卡尔伺服目标时，显式增加：

```powershell
--rate-limited-policy `
--policy-linear-speed 2 `
--policy-angular-speed 1
```

在15 Hz下，上述设置将每条实际命令限制为约0.133 mm和0.067°（RPY向量范数）。
不设置单独的累计行程上限，仍由 `--max-seconds` 限制本次实验时长。

动作日志中的 `target_tcp_pose` 和 `raw_policy_target` 保留模型原始输出；
`commanded_tcp_pose` 是通过限速后、实际接受安全检查并在执行模式下发给xArm的目标。
`rate_limit` 保存原始距离、实际步长以及是否触发平移/旋转限速。

该模式不会关闭机械臂状态、力/力矩、传感器时效、跟踪误差、单步变化、SDK返回码
和急停相关保护。
