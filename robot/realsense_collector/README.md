# RealSense VLA 图像数据采集器

已为 Intel RealSense D435I 配置 RGB-D 数据采集。每个 episode 包含同步、对齐的彩色图与深度图，以及语言指令、时间戳、相机内参和可选机器人动作。

## 在另一台 Windows 电脑部署

1. 安装 **64 位 Python 3.11 或 3.12**，安装时勾选 `Add Python to PATH`。
2. 将整个 `vla_realsense_collector` 文件夹复制到新电脑。
3. 使用 USB 3.x 数据线连接 RealSense；避免仅支持充电的线和无供电 USB Hub。
4. 在该目录打开 PowerShell，执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

安装脚本会在当前目录创建 `.venv`、安装固定版本依赖，并列出连接的相机。这个虚拟环境不要从旧电脑复制，应在新电脑上重新生成。

## 快速开始

在本目录打开 PowerShell：

```powershell
.\run_collector.ps1 -Instruction "把红色方块放入盒子" -Output ".\dataset"
```

预览窗口按 `Q` 或 `Esc` 停止；也可按 `Ctrl+C`。固定采集 20 秒：

```powershell
.\run_collector.ps1 -Instruction "抓取水杯" -Output ".\dataset" -Duration 20
```

无预览采集（适合远程运行）：

```powershell
.\.venv\Scripts\python.exe .\collector.py --output .\dataset --instruction "抓取水杯" --duration 20 --no-preview
```

列出相机：

```powershell
.\.venv\Scripts\python.exe .\collector.py --list-devices
```

## 输出格式

```text
dataset/episode_YYYYMMDD_HHMMSS/
  metadata.json       # 设备信息、相机内参、深度比例、语言指令
  frames.jsonl        # 每帧时间戳、文件路径、指令和动作
  summary.json        # 帧数、耗时、平均帧率
  color/000000.jpg    # BGR8 彩色帧
  depth/000000.png    # 对齐到彩色图的 uint16 深度帧
```

米制深度的恢复公式为：`depth_m = depth_png * depth_scale_meters_per_unit`，其中比例记录在 `metadata.json`。

## 同步机器人动作

机器人控制程序可持续原子更新一个 JSON 文件，例如 `latest_action.json`：

```json
{"joint_position": [0, 0, 0, 0, 0, 0], "gripper": 0.4}
```

采集时传入：

```powershell
.\.venv\Scripts\python.exe .\collector.py --action-file .\latest_action.json --instruction "抓取水杯"
```

采集器会在每帧的 `frames.jsonl` 中记录当时读取到的 `action`。为避免读到半写入内容，控制程序应先写临时文件，再用重命名替换目标文件。

## 常用参数

- `--serial`：多相机时指定序列号。
- `--width/--height/--fps`：默认 `640×480@30`。
- `--color-only`：只保存 RGB。
- `--jpeg-quality`：默认 95。
- `--warmup`：正式录制前丢弃的帧数，默认 30。

如需重建本机虚拟环境，删除 `.venv` 后重新运行：

```powershell
.\setup_windows.ps1
```

若设备可以列出但持续提示 `Frame didn't arrive within 5000`，先关闭 RealSense Viewer、相机软件或其他占用相机的进程。仍无效时可拔插 USB 3.x 线，或执行一次 SDK 复位：

```powershell
.\.venv\Scripts\python.exe -c "import pyrealsense2 as rs; rs.context().query_devices()[0].hardware_reset()"
```

等待约 5 秒，让 Windows 重新枚举设备后再开始采集。
