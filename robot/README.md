# 工控机侧代码

采集与真机控制运行在工控机上（Win10），本文档与下面两个子目录是那批代码的**副本**，
收入本仓库是为了让「数据是怎么采出来的」有据可查。

> **这里是副本，不是工作版本。** 真正的代码在工控机上，改这里不会影响采集。
> 两侧的对应关系与回溯方式见[文档 10](../docs/10-collaboration.md)。

| 子目录 | 内容 | 脱敏 |
|---|---|---|
| `my_code/` | 采集链路：示教点 → 轨迹 → 三路同步采集 → 对齐 → 建集 | 两处替换（见第一节） |
| `realsense_collector/` | 独立 RealSense RGB-D 采集器，可脱离机器人单独用 | **零脱敏**，与原件逐字节相同 |

---

## 一、`my_code/`

### 1.1 与工控机原件的差异

仅两处替换，逐条可核对：

| 原件 | 此处 |
|---|---|
| xArm 局域网地址（`192.168.1.*`） | `192.168.1.100` |
| `C:\Users\<用户名>\Desktop\develop_dt\Xarm_SDK_dt\xArm-Python-SDK-master` | `<xArm-SDK>` |

涉及 8 个文件。其余内容逐字节相同。

> 与[文档 10](../docs/10-collaboration.md) 第四节同理：替换表的左列只写**可检索的前缀**，
> 不写全值——否则清单自己就成了泄漏点。用 `grep -rn '192.168.1.'` 照样能定位。

### 1.2 采集链路

拟合轨迹运行阶段（当前流程）的完整顺序：

```
① collect_teach_points_any_count.py      手动拖到各位姿按 Enter，存成 teach_points_*.json
                │
                ▼
② generate_intermediate_trajectories.py  在相邻两条实测轨迹之间插值，只内插不外推
                │                         → generated_set_*/intermediate_*.json
                ▼
③ run_single_curve_from_latest_points_servo_soft_pose.py
                │                        单条曲线的伺服连续扫描（姿态沿路径插值）
                │                        工作副本 DRY_RUN_LIFT_MM = 0.0，即按示教高度实跑
                ▼
④ run_fitted_trajectory_vla_sync.py      ★ 自动扫描阶段的三路同步采集器
                │                        复用 ③ 的轨迹与伺服参数，同时记录
                │                        xArm 状态/下发目标、RealSense RGB-D、PhaseLink 原始超声包
                ▼
⑤ align_vla_episode.py                   三路各自带主机时间戳，此处对齐成固定 15 Hz
                │                        设备时钟线性映射到主机时钟，去掉到达抖动
                ▼
⑥ build_vla_training_dataset.py          生成 LeRobot 训练集，只收 quality.valid 为真的样本
```

模块依赖（不是可选组合，缺一个就跑不起来）：

```
run_fitted_trajectory_vla_sync.py
├── collect_teach_vla_sync.py ── realsense_sync_recorder.py
└── run_single_curve_from_latest_points_servo_soft_pose.py

build_vla_training_dataset.py ── export_ultrasound_images.py
```

#### 各文件

| 文件 | 作用 |
|---|---|
| `collect_teach_points_any_count.py` | 示教点采集，任意点数，按 Enter 存当前 TCP 位姿 |
| `generate_intermediate_trajectories.py` | 相邻实测轨迹之间生成中间轨迹，**不外推**；附带 SVG 预览 |
| `run_single_curve_from_latest_points_servo_soft_pose.py` | 单条曲线伺服扫描，柔和调参 + 姿态沿路径插值（现行版） |
| `run_single_curve_from_latest_points_servo_soft.py` | 同上但**固定姿态**（`DRY_RUN_LIFT_MM` 仍为 30.0 的抬升空跑）；与上一行构成对照，说明「姿态跟随 / 固定」与「实跑 / 抬升空跑」是两个独立选项 |
| `realsense_sync_recorder.py` | RealSense RGB-D 记录，同时打设备时间戳与主机时间戳 |
| `collect_teach_vla_sync.py` | 手拖示教模式的三路同步采集；也被 ④ 当作模块引用 |
| `run_fitted_trajectory_vla_sync.py` | **自动扫描阶段的三路同步采集器**（链路主线） |
| `align_vla_episode.py` | 三路对齐到固定采样率；位姿插值、图像取最近帧、设备时钟线性映射 |
| `build_vla_training_dataset.py` | 生成 LeRobot 训练集，仅收 `quality.valid` 的样本 |
| `export_ultrasound_images.py` | PhaseLink 原始包 → S-scan 图像，解码口径见[文档 09](../docs/09-interface-contract.md) |
| `check_xarm_force_readonly.py` | **只读**力/力矩诊断。明确不使能运动、不改模式、不清零、不发任何运动指令 |
| `run_robot_trajectory_with_realsense_preview.py` | 只跑轨迹 + RGB 预览、不写盘不上超声的安全测试入口；预览在独立线程，不阻塞 20 ms 伺服环 |
| `recollect_teach_points_after.py` | 重采某个示教点之后的部分；替换前先备份原 JSON，且需最终确认 |

`test_*.py` 与上表模块一一对应，覆盖对齐、建集、轨迹生成、录像、同步采集与伺服扫描。

### 1.3 没收录的

| 文件 | 原因 |
|---|---|
| `run_single_curve_from_4_points.py` | 4 点版，最早的一版 |
| `run_single_curve_from_latest_points.py` | 9 点版，逐点 `set_position` |
| `run_single_curve_from_latest_points_servo.py` | 伺服版，尚未调柔 |

以上三者是 `..._servo_soft_pose.py` 的前身，构成一条演化链；差异已由现行版与本文档体现。

| `run_scan_collect_vla_realsense.py` | 单进程 RGB-D + 机器人采集，**不含超声**，已被三路同步采集器取代 |
| `build_vla_training_dataset_dscan.py` | D-scan 分支（按体数据「深度 × 仰角 × 扫查」切片的另一套图像口径），非现行的 S-scan |
| `surface_scan_demo_no_probe.py` | 无探头演示脚本 |
| `test.py` | 名为 test，实为示教点采集的早期版本，且写死了机器人地址 |
| 示教点与轨迹 JSON | 是本站点的实测标定数据，不是代码 |

`setup_vla_env.ps1` 末尾的提示指向 `run_scan_collect_vla_realsense.py`（上面已排除的那个）；
建 venv 与装依赖的部分仍然适用。

### 1.4 运行前提

- xArm Python SDK（`setup_vla_env.ps1` 从本地项目安装）
- RealSense：`requirements_vla_realsense.txt`
- PhaseLink：需用打了 VLA 记录补丁的 Qt 工程重新编译一次，旧 `Phaselink.exe`
  不读 `PHASELINK_VLA_DIR`，不会产出同步文件。步骤见两份 README
- 超声解码调色板与动态范围来自 PhaseLink 侧；若改动需同步[文档 09](../docs/09-interface-contract.md)

采集产物（图像目录、原始包、示教点 JSON）均不入库。

---

## 二、`realsense_collector/`

独立 RealSense RGB-D 采集器，**不含机器人控制、不含超声**，可以在一台只接了相机的
Windows 电脑上单独跑（见其 `README.md` 的部署一节）。它与 `my_code/` 里的
`realsense_sync_recorder.py` 是两条不同的路：后者被三路同步采集器当模块调用，
只出帧流；前者是独立可执行的完整采集器，自带 episode 目录结构、相机内参与元数据。

**与工控机原件逐字节相同，未做任何替换。** 已核对：无写死的 IP、主机名、用户路径或
相机序列号（`--serial` 缺省时取第一台设备）。

| 文件 | 作用 |
|---|---|
| `collector.py` | 采集主程序：彩色 + 对齐后的深度、`frames.jsonl`、`metadata.json`、`summary.json`；可选 `--action-file` 同步机器人动作 |
| `realsense_viewer.py` | 只预览不落盘的取景工具 |
| `setup_windows.ps1` | 建 `.venv`、装固定版本依赖、列出已连接相机；会校验 Python 3.11/3.12 与 64 位 |
| `run_collector.ps1` | 调用 `.venv` 里的解释器启动采集 |
| `requirements.txt` | numpy / opencv-python / pyrealsense2 |
| `README.md` | 部署步骤、参数、输出格式，以及 `Frame didn't arrive` 的处置 |

输出为每帧一个 JPEG（彩色）+ 一个 uint16 PNG（深度），米制深度按
`depth_m = depth_png * depth_scale_meters_per_unit` 还原，比例记在 `metadata.json`。

> 其 `README.md` 里的示例指令（"把红色方块放入盒子"）与动作格式
> （`joint_position` + `gripper`）是通用的抓取任务示例，不是本项目的任务——
> 这个采集器是跨任务复用的，未针对超声扫描改过。

### 2.1 没收录的

`.venv/`（依赖与字节码）。采集产物（`dataset/episode_*/`）不入库。

---

## 三、`real_robot/`

真机远程策略客户端：连 xArm 读状态、读 RealSense 与 PhaseLink 原始包、把观测发给
远端 policy、按动作块下发。**默认影子模式**——连真机、读传感器、调远端策略、只记录，
不调用任何运动 API。

| 文件 | 作用 |
|---|---|
| `real_robot_client.py` | 入口。安全包络、三个数据源、控制环、参数解析 |
| `services_pb2*.py` | proto 生成的桩，免装 `grpcio-tools` |
| `config/training_ultrasound_metadata.json` | 超声解码配置（波束数、动态范围等），从训练数据复制而来 |
| `README_REAL_ROBOT.md` | 运行步骤：起 PhaseLink → `--validate-only` → 影子测试 → 有界真机测试 |
| `test_real_robot_client.py` | 单测（14 项） |
| `requirements.txt` | 只有 grpcio / protobuf / numpy，不需要 torch 与 CUDA |

> **本目录不含传输层。** `real_robot_client.py` 第 45 行 `import remote_client`，
> 那一份在 [`pi0_ultrasound_lora/remote/remote_client.py`](../pi0_ultrasound_lora/remote/remote_client.py)
> ——仓库里唯一一份。理由与取法见第 3.3 节。

### 3.1 安全阈值审阅（入库前的必要一步）

[文档 10](../docs/10-collaboration.md) 第七节要求 `real_robot_client.py` 入库前逐行确认
安全阈值是**通用保守值**而非本站点专用调参。已逐行读完，结论：**是通用保守值，可以入库。**
依据如下（均为代码中的默认值，非文档转述）：

| 性质 | 代码依据 |
|---|---|
| 两道闸门才能动 | `--execute` 与 `--confirm EXECUTE_REAL_XARM_POLICY` 必须同时给出，字符串精确匹配，否则 `parse_args` 直接 `parser.error` 退出 |
| 不自动去起始位姿 | `prepare_execute()` 只读不写：查状态、查力传感器、`motion_enable`、`set_mode(1)`、`set_state(0)`，全程没有 `set_position` |
| 单次运行有时长上限 | `--max-seconds` 默认 10.0 s |
| 力/力矩有上限 | `--max-force-n` 40 N、`--max-torque-nm` 5 N·m，每次下发前用 `_require_healthy` 复查 |
| 力传感器不能是坏的 | 读数全零即停止，除非显式 `--allow-zero-ft`——防止传感器失效后力检查被静默架空 |
| 位置与姿态单步限幅 | 3 mm / 2°；目标距实际 TCP 另有 12 mm / 5° 上限 |
| 起始位姿容差 | 8 mm / 5°，只在与参考轨迹比对时启用（需显式 `--safety-trajectory`） |
| 控制器状态防护 | `state` 为 4/6 即停；5 仅在尚未下发任何目标时允许检视 |
| 错误码即停 | `error_code`、`warn_code`、断联、SDK 返回非零一律 `SafetyStop` |
| 异常/中断进停止态 | 任何异常或 Ctrl+C 都走 `stop_motion(emergency=True)` → `set_state(4)` + `set_mode(0)`，**不自动清错** |
| 不重复旧动作 | 动作队列为空时 `pop_action` 返回 `None`，控制环等待；结构上就不会重放陈旧动作 |

数值量级对一台挂探头的 6 轴机械臂是常见保守调试值（力 40 N、单步 3 mm/2°、
跟踪误差 12 mm/5°），没有出现针对某次实验调过的痕迹，也没有发现被放宽的痕迹。

唯一值得留意的是 `--queue-empty-stop` 默认为 `0.0`（关闭）。这不危险：队列空时
不会重放旧动作，且断供本身会经 `rx_error` 触发停止，加上 `--max-seconds` 兜底。
需要恢复该超时可显式传 `--queue-empty-stop 3.0`。

### 3.2 与工控机原件的差异

脱敏替换如下，逐条可核对：

| 原件 | 此处 |
|---|---|
| `C:\Users\<用户名>\Desktop\develop_dt\Xarm_SDK_dt\xArm-Python-SDK-master` | `<xArm-SDK>` |
| `C:\Users\<用户名>\Desktop\develop_dt\PAU_SDK - dt` | `<PAU_SDK>` |
| `D:\pi0_remote`（部署目录） | `<deploy-dir>` |
| `E:\dt_vla_data`（数据集根） | `<data-root>` |
| GPU 机 Tailscale 地址（`100.127.*`） | `100.64.0.1` |
| xArm 局域网地址（`192.168.1.*`） | `192.168.1.100` |

命中 `real_robot_client.py`（5 处）与 `README_REAL_ROBOT.md`（14 处）。
`requirements.txt`、两个 `services_pb2*.py`、`test_real_robot_client.py` **逐字节相同，未改动**。

> 早先这张表里另有一行 `/home/lab301/vla-training/pi0_ultrasound_lora/run1`，
> 那处替换只出现在 `remote_client.py`（它的 `DEFAULT_CKPT`）里；
> 该文件已移出本目录，见 3.3，此行随之撤销。

### 3.3 传输层只有一份，在服务端那一侧

`real_robot_client.py` 第 45 行 `import remote_client as transport`，第 607 行
`class RealRobotClient(transport.LeanRobotClient)`。它对传输层的依赖不止一处：

| 依赖 | 用处 |
|---|---|
| `send_observations_loop()` | 独立线程上行观测，不阻塞动作时钟 |
| `control_loop_observation(force_must_go=True)` | 预热时强制发一次；**返回值**表示入队是否成功 |
| `first_actions_event` | 等首个动作块落地，之后才启动固定频率时钟 |
| `observation_error` | 发送线程的异常，主循环转成 `SafetyStop`（第 722 行） |
| `observation_send_latencies` | 影子测试要看的观测上行延迟 |

**这份传输层在本仓库里只存在于
[`pi0_ultrasound_lora/remote/remote_client.py`](../pi0_ultrasound_lora/remote/remote_client.py)。**

原来树里有两份，且已经分叉：

| 位置 | 行数 | 状况 |
|---|---|---|
| 工控机 `D:\pi0_remote\remote_client.py`（部署版） | 815 | 上表五项齐全 |
| 工控机 `Desktop\pi0_remote\`，即树里原来的 `pi0_ultrasound_lora/remote/` | 679 | **五项全缺** |

拿 679 那版替换会直接报属性错误——不会静默出错，但也不会有人预料到。
现按 815 版统一：**内容移入 `pi0_ultrasound_lora/remote/`，本目录不再保留副本**；
配套的 `test_remote_client_timing.py` 也一并移过去（它测的正是上表前两项）。

> **工控机不需要改动。** 取回两份原件与仓库副本逐行 diff 核对过：
> `D:\pi0_remote\` 那份与仓库里这份**只差 3 行，且全是脱敏**——docstring 里的示例
> 地址、`DEFAULT_CKPT` 的绝对路径前缀、`--server` 默认值；`Desktop\pi0_remote\`
> 那份与仓库里原来的 679 行版也只差同样 3 行。
> 即：**真机部署目录里跑的本来就是这一版**，这次统一只动仓库，不动机器。

选服务端那一侧而不是本目录，有两个理由：`serve.py` 就在隔壁，而**客户端与 `serve.py`
是必须同版本的一对**；且 `remote/` 整个目录本来就是要拷到工控机的部署单元。

**本目录因此不再自包含。** 单独拿 `robot/real_robot/` 去跑会停在 `import remote_client`
——这是想要的行为，不会静默加载到一份旧副本。工控机上把整个
`pi0_ultrasound_lora/remote/` 放进 `<deploy-dir>` 即可，`real_robot_client.py` 就在它旁边。

在仓库里跑真机单测，要把两处补进 `PYTHONPATH`：

```bash
cd robot/real_robot
PYTHONPATH=<repo>/pi0_ultrasound_lora/remote:<repo>/robot/my_code \
  python -m unittest test_real_robot_client     # 14 项
```

第二处是因为 `real_robot_client.py` 还 `from export_ultrasound_images import ...`，
而它按 `XARM_SDK_ROOT` 去找 `<xArm-SDK>/my_code`。在工控机上这条路径本来就在，
**只有在本仓库里需要手工补**——这个缺依赖在我改动之前就存在，不是本次引入的。

`services_pb2*.py` 由 proto 生成，**两台机器必须同版本**：版本不一致时不会报错，
只会握手失败。

### 3.4 没收录的

| 项 | 原因 |
|---|---|
| `remote_client.py`（部署版，815 行） | **已收入本仓库，但不在此目录**——移到了 `pi0_ultrasound_lora/remote/`，见 3.3 |
| `test_remote_client_timing.py` | 同上，随传输层一起移动 |
| `README.md`（部署目录里那份） | 与 `pi0_remote/README.md` **逐字节相同**，是链路测试的说明；本仓库已有其脱敏版 `pi0_ultrasound_lora/remote/README.md`，不重复收录 |
| `*.20261011_before_rate_limit.bak` | 加限速模式前的备份，现行版本已包含该改动 |
| `live_ultrasound_current/`、`real_runs/`、`replay_bundle.npz`、`.venv/` | 运行产物与依赖 |

> **本目录不含数据集、示教点或轨迹文件。** `--safety-trajectory` 指向的参考轨迹
> 属站点实测数据，需自行提供；不传该参数时走「直接执行策略目标」并仅靠实时限幅保护。
