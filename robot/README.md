# 工控机侧代码

采集与真机控制运行在工控机上（Win10），本文档与 `my_code/` 是那批代码的**脱敏副本**，
收入本仓库是为了让「数据是怎么采出来的」有据可查。

> **这里是副本，不是工作版本。** 真正的代码在工控机上，改这里不会影响采集。
> 两侧的对应关系与回溯方式见[文档 10](../docs/10-collaboration.md)。

## 一、与工控机原件的差异

仅两处替换，逐条可核对：

| 原件 | 此处 |
|---|---|
| xArm 局域网地址（`192.168.1.*`） | `192.168.1.100` |
| `C:\Users\<用户名>\Desktop\develop_dt\Xarm_SDK_dt\xArm-Python-SDK-master` | `<xArm-SDK>` |

涉及 8 个文件。其余内容逐字节相同。

> 与[文档 10](../docs/10-collaboration.md) 第四节同理：替换表的左列只写**可检索的前缀**，
> 不写全值——否则清单自己就成了泄漏点。用 `grep -rn '192.168.1.'` 照样能定位。

## 二、采集链路

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

### 各文件

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

## 三、没收录的

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

## 四、运行前提

- xArm Python SDK（`setup_vla_env.ps1` 从本地项目安装）
- RealSense：`requirements_vla_realsense.txt`
- PhaseLink：需用打了 VLA 记录补丁的 Qt 工程重新编译一次，旧 `Phaselink.exe`
  不读 `PHASELINK_VLA_DIR`，不会产出同步文件。步骤见两份 README
- 超声解码调色板与动态范围来自 PhaseLink 侧；若改动需同步[文档 09](../docs/09-interface-contract.md)

采集产物（图像目录、原始包、示教点 JSON）均不入库。
