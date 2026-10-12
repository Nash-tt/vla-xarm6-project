# 02 · 对齐与建集

**跑在哪：** 工控机，紧接 [01 采集](01-collection.md) 之后。
**输入：** 一个 episode 目录（四路各自带时间戳的原始记录）。
**产出：** 一个**便携训练文件夹**（注意：不是 LeRobot 数据集，见第五节）。
**下一棒：** 拷到 GPU 机器，交给 [03 数据集工具](03-dataset-tools.md)。

两步，两个脚本，职责不重叠：

```
align_vla_episode.py        四路 → 固定 15 Hz 的 aligned_samples.jsonl
        │                   只做时间对齐，不动图像内容
        ▼
build_vla_training_dataset.py   逐样本解码超声 → 便携训练文件夹
                                只收 quality.valid 为真的样本
```

照例，先对齐再建集，是为了**对齐结果可复用**：改了超声解码参数重跑第二步即可，
不必重跑第一步的插值。

---

## 一、对齐：为什么是 15 Hz，怎么插

训练要固定采样率，而四路的原生到达率各不相同（伺服 50 Hz、RealSense 30 Hz、超声不定）。
对齐就是**在一条等间隔的时间轴上，为每一拍从四路各取一个值**。

`--fps` 默认 `15.0`（`align_vla_episode.py:251`）。选定 15 Hz 是采集带宽与数据量的折中，
写入数据集的 `fps` 字段后，训练侧必须与之一致。

时间轴的范围取**四路的重叠区**，再与采集窗口求交（`align_vla_episode.py:329-336`）：

```python
overlap_start = max(requested_start, robot_times[0], rs_times[0], us_times[0])
overlap_end   = min(requested_end,   robot_times[-1], rs_times[-1], us_times[-1])
```

`requested_start/end` 来自 `metadata.json` 的 `capture_window`。**窗口缺了就拒绝执行**，
`:285-290` 明确写了理由：`refusing to align setup or move-to-start data as valid VLA samples`
——没有窗口，对齐脚本无从知道哪一段是"采集"，会把机器人上电初始化的数据也当成样本。

### 1.1 四路各用各的取法

**这是本篇最要紧的一段**，四种取法对应四种物理含义：

| 路 | 取法 | 位置 | 理由 |
|---|---|---|---|
| xArm 状态 | **线性插值** | `interpolate_robot` `:199` | 50 Hz 连续伺服，位姿是时间的连续函数，插值比取最近帧更准 |
| RealSense | **取最近帧** | `nearest_record` `:147` | 图像不可插值。取时间上最近的那一帧，偏差记为 `alignment_skew_ms` |
| 超声 | **取最近帧** | 同上 | 同左，一帧 S-scan 是整块数据 |
| 动作 | **取"当前生效"的那条** | `active_action` `:160` | 见下 |

xArm 的插值里，**姿态维用最短弧**：`interpolate_vector(..., angle_start=3)` 让
`tcp_pose` 的索引 3 起（roll/pitch/yaw）走 `interpolate_angle_degrees`，
即先算 `(b - a + 180) % 360 - 180` 把差值折到 ±180 内再插值，插完再折回。
不这么做，从 179° 到 -179° 的 2° 运动会被插成 358°。

动作那条用 `bisect_right(times, target_ns) - 1`：**取目标时刻之前最后一次真正下发的指令**。
`:402` 的注释把这条钉死了：

> *This is always an actual issued position/servo command. An observed future robot state is never substituted for action.*

即**绝不用"未来的机器人状态"冒充动作**。这是模仿学习里最容易犯的错——
用下一帧状态当标签，训出来的策略是在预测状态而不是在学控制。

### 1.2 动作的有效性政策与它不一套标准

伺服目标每 20 ms 刷新一次，位置目标则是"发一次、一直有效到下一条"。
两种指令用同一个时限卡是错的，所以 `action_is_valid` 分开处理（`:169-174`）：

```python
if action_record.get("type") == "position_target":
    return True                                   # 长期有效，不卡 age
return abs(action_skew_ms) <= servo_max_skew_ms   # 伺服目标卡 80 ms
```

另外 `return_code != 0` 的动作一律无效——**下发了但控制器拒绝了**的指令不是动作。

---

## 二、设备钟映射：最小二乘 + 斜率门

RealSense 与超声都带设备时间戳。**设备钟不能直接用**：它与主机钟有恒定偏移，
还可能有频率差（晶振不同）。但对齐又要跨设备比较时间，所以要把设备钟**线性映射**到主机钟。

`fit_device_clock`（`:79-137`）做的是最朴素的一元最小二乘，但有三处不朴素的细节：

**① 只用采集窗口附近 ±5 s 的记录拟合。**
`records_in_clock_fit_window`（`:56`，`CLOCK_FIT_MARGIN_NS = 5_000_000_000`）。
注释说得很清楚：PhaseLink 关窗时会吐一些非帧包，尾部未必有设备时间戳，
**拿它们参与回归会毒化整条拟合**（原始记录仍全部保留，只是不参与拟合）。

**② 用相对值算，防止 uint64 精度丢失。**
超声设备钟是 200 MHz 计数，`200e6 × 秒数` 早就超出 float64 能精确表示整数秒的范围；
主机钟是 unix 纳秒，同样巨大。所以先各自减去第一对的值，再换算成秒：

```python
device0, host0 = pairs[0]
xs = [(device - device0) / float(ticks_per_second) for device, _ in pairs]
ys = [(host - host0) / 1e9 for _, host in pairs]
```

**③ 斜率必须在 [0.95, 1.05]，否则整条拟合作废。**

```python
if not 0.95 <= slope <= 1.05:
    result["rejected_reason"] = "device clock slope outside 0.95..1.05"
    return None, result
```

斜率是"设备钟走 1 秒时主机钟走几秒"，物理上必须接近 1。
超出 5% 说明设备钟被重置过或计数方案变了——注释一句点明取舍：
*a bad/reset device clock is worse than raw host arrival timestamps*。
拒绝时**退回主机到达时间**（`attach_sync_times` 传 `None` 的 mapper 即不映射），
不是退回失败。

拟合质量写进 `alignment.json` 的 `clock_fits`，逐条可查：
`slope`、`drift_ppm`、`arrival_residual_rms_ms`、`arrival_residual_max_abs_ms`、
`pair_count`、以及是否 `used`。**少于 10 对点直接不拟合**（`:95`）。

---

## 三、`quality.valid`：四道时限

每一拍都要过四道时限检查（`:370-375`）：

```python
valid = (abs(robot_state["nearest_skew_ms"]) <= args.robot_max_skew_ms
         and abs(rs_skew_ms)                 <= args.realsense_max_skew_ms
         and abs(us_skew_ms)                 <= args.ultrasound_max_skew_ms
         and action_valid)
```

| 参数 | 默认 | 物理含义 |
|---|---|---|
| `--robot-max-skew-ms` | 80 | 插值点距最近实测状态多远 |
| `--realsense-max-skew-ms` | 50 | 取到的帧距目标时刻多远 |
| `--ultrasound-max-skew-ms` | 50 | 同上 |
| `--action-max-skew-ms` | 80 | 对伺服目标而言：指令有多旧 |

xArm 给 80 ms 而相机给 50 ms，是因为前者插值（本来就不必贴着实测点），
后者取最近帧（偏差直接等于时间错位）。

**默认不丢，只标记。** 无效样本仍写进 `aligned_samples.jsonl`，`quality.valid` 为 `false`；
加 `--drop-invalid` 才跳过。这样对齐结果保留全貌，丢在哪一步由下一步决定——
建集时会再筛一遍（见第四节），两次的口径不同，所以两处都留了计数。

---

## 四、建集：二层过滤与超声解码

`build_vla_training_dataset.py`（666 行）。它先 `ensure_alignment`（缺 `alignment.json`
就自动跑一遍对齐），然后做**第二层过滤**（`:267-275`）：

```python
valid_samples = [s for s in all_samples
    if s.get("quality", {}).get("valid") is True
    and s.get("action")                    is not None
    and s.get("observation", {}).get("robot")      is not None
    and s.get("observation", {}).get("realsense")  is not None
    and s.get("observation", {}).get("ultrasound") is not None]
```

比对齐那层多的是**四个非空检查**。两层都要，是因为对齐时 `valid` 只保证"时间对得上"，
不保证"数据都在"——比如某路的记录本身是空的，时间却算得出来。

模型无有效样本时直接抛错，不产出一个空数据集。

### 4.1 超声解码

超声这一路，原始包（`ultrasound_packets.bin`）在这里才被解码成 S-scan 图像，
由 `export_ultrasound_images.py`（361 行）完成。解码口径——波束数、动态范围、
调色板、深度量程——全部**从采集时用的 `ULTRASOUND_REQUESTED_SETTINGS` 推导**，
而不是另立一套参数：

- `calculate_beam_count(settings)`、`display_max_amplitude_percent`、`range_start/end_mm`、
  `probe_pitch_mm`、`element_step` 都从 settings 里取
- 调色板来自 PhaseLink 侧（`load_phaselink_palette(DEFAULT_PALETTE_SOURCE)`）
- 脚本会区分 `settings_source` 是 `ultrasound_metadata.applied_settings`
  （**实际生效的**）还是请求值——两者不一致时数据集的物理含义就不对

**这意味着超声解码参数是采集时定下的，事后改不了。** 想换解码口径只能重采，
或用 `preserve_ultrasound_physical_aspect` 在几何上做文章。
解码细节见[接口契约](../docs/09-interface-contract.md)。

### 4.2 力的审计

`audit_force(valid_samples)`（`:113`）在建集时跑一遍力/力矩检查，`--require-force`
可把它变成硬门。这是采集侧的最后一处质量兜底——探头压得太狠的 episode 在训练前就被拦下。

---

## 五、产出：便携文件夹，不是 LeRobot 数据集

**这一点必须说清楚，`robot/README.md` 里原先写的"生成 LeRobot 训练集"是错的。**

`build_vla_training_dataset.py` 写出的是一个自描述的便携文件夹，
`schema` 为 **`xarm_rgbd_ultrasound_vla_training_dataset_v1`**（`:477`）：

```text
<output_dir>/
├── dataset.json          # schema: xarm_rgbd_ultrasound_vla_training_dataset_v1
│                         #   采样率、样本数、过滤掉多少、各模态的形状
├── samples.jsonl         # 逐样本：时间戳、指令、图像相对路径、state、action
├── rgb/                  # 彩色帧图像文件
├── depth/                # uint16 深度 PNG（比例记在 metadata/realsense_metadata.json）
├── ultrasound/           # 解码后的 S-scan
└── metadata/             # 原样复制的 5 份元数据
    ├── metadata.json  trajectory.json  alignment.json
    └── realsense_metadata.json  ultrasound_metadata.json
```

元数据是**原样 `shutil.copy2` 过来**的（`:463-469`），不做加工——
上游参数怎么定的，这里就怎么留着，便于日后回溯。

**转成 LeRobot 格式不在这个仓库里。** 仓库中没有把 `samples.jsonl` 变成
LeRobot 数据集的脚本；那一步是仓库外的（工控机上做完，把结果拷到 GPU 机器）。
GPU 侧看到的是已经成型的 LeRobot 数据集，所以 [03](03-dataset-tools.md) 之后的所有工具
都直接按 LeRobot v3.0 的目录结构操作。

> 这里有个容易混淆的点：`dataset.json` 里的键（`rgb` / `depth` / `ultrasound`）
> 与 LeRobot 里的键（`observation.images.rgb` / `observation.images.ultrasound`）
> **不是同一套**。前者是便携格式的模态目录名，后者是训练格式的 feature 名。
> 两套名字在同一份文档里只隔几行、又都叫「键」，读 `docs/06` 与 `docs/09`
> 的相机键名表时留意别串（`docs/09` 那张表说的是**后者到 policy 键**的映射）。

---

## 六、指令字符串：仓库里有三份，不一致

| 来源 | 内容 |
|---|---|
| `run_fitted_trajectory_vla_sync.py:54-58`（实际写进采集元数据） | "Move from the configured capture-start pose to the first taught scan point using a constant-height approach, descend vertically, then follow the fitted trajectory while collecting synchronized RGB-D and ultrasound." |
| `align_vla_episode.py:18-21`（`DEFAULT_INSTRUCTION`，仅在元数据缺失时兜底） | "Move the ultrasound probe along the demonstrated path while maintaining stable contact and image quality." |
| `docs/09` | "Follow the fitted scan trajectory with the ultrasound probe." |

对齐时 `instruction = args.instruction or metadata.get("instruction") or DEFAULT_INSTRUCTION`
（`:268`）——**元数据里的那份优先**，所以实际进数据集的是第一份（214 字符）。
默认值只在元数据缺失时用得上。

**当前不影响任何训练**：π0 的 LoRA 训练与 ACT 都不吃语言（ACT 根本没有语言塔）。
只有 SmolVLA 用它，而 SmolVLA 的数据集是从同一个 episode 重建的，
指令同样是第一份。三份不一致是文档债，不是数据债。

---

## 七、产物与下一步

对齐产物 `aligned_samples.jsonl` + `alignment.json` **留在 episode 目录里**，与原始记录并列——
这样任何时候都能从原始记录重跑到任意一步，不必回工控机重采。

建集产物拷到 GPU 机器后，进入 [03 数据集工具](03-dataset-tools.md)。
**拷过去的第一件事永远是裁剪**，因为 `capture_window` 里含接近段
（见 [01 第五节](01-collection.md)），而接近段教的是"去一个很远的目标"，
不是扫描动作本身。
