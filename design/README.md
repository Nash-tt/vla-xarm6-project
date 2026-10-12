# 实现方案

`docs/` 记的是**踩过的坑**——每个问题按「现象 → 原因 → 处理 → 验证 → 结论」写一遍。
这里记的是**现在这套东西是怎么搭的**：分几块、每块之间传什么、关键参数为什么取这个值、
哪些地方是有意为之的怪。

两者分工明确：想查「当初为什么这么改」去 `docs/`，想知道「代码现在长什么样、为什么这么设计」看这里。

> 本篇描述的是**代码事实**，行号对应 2026-10-12 的仓库状态。
> 凡属推断而非实测的结论，文中都会标「未实测」。

---

## 一、整体分四层

```
① 采集（工控机）
   示教点 → 轨迹生成 → 伺服扫描 ─┬─ xArm 状态
                                 ├─ RealSense RGB-D      ← 三路各自打主机时间戳
                                 └─ PhaseLink 原始超声包
                                          │
                                          ▼
② 对齐与建集（工控机）        固定 15 Hz 对齐 → 便携训练文件夹
                                          │
                                          ▼
③ 数据集工具（GPU 机器）      裁接近段 → 重算 stats → 校验
                                          │
                                          ▼
④ 训练与部署（GPU 机器 + 工控机）
   π0-LoRA / SmolVLA / ACT 训练 → serve.py ──gRPC──> remote_client → real_robot_client
```

**层与层的边界是数据格式，不是函数调用。** ①②跑在工控机上，③④跑在 GPU 机器上，
两侧不共享代码，只共享[文档 09](../docs/09-interface-contract.md) 约定的文件结构。
这是被硬件逼出来的：工控机没有独立显卡，相机和机械臂又都在它身上。

---

## 二、七篇

| 篇 | 覆盖 | 代码位置 |
|---|---|---|
| [01 采集链路](01-collection.md) | 示教点 → 轨迹 → 三路同步采集 | `robot/my_code/` |
| [02 对齐与建集](02-align-and-build.md) | 三路对齐成 15 Hz、超声解码、数据集封装 | `robot/my_code/` |
| [03 数据集工具](03-dataset-tools.md) | 裁剪、重算统计量、校验 | `tools/` |
| [04 训练](04-training.md) | π0-LoRA、SmolVLA、ACT 三条训练链路 | `pi0_ultrasound_lora/`、根目录 |
| [05 推理服务端](05-inference-server.md) | policy server 与它对上游的两处修复 | `pi0_ultrasound_lora/serve.py` |
| [06 传输层](06-transport-layer.md) | 瘦客户端、三个同步机制、pickle 垫片 | `pi0_ultrasound_lora/remote/` |
| [07 真机客户端](07-real-robot-client.md) | 安全包络、控制环、影子/执行两态 | `robot/real_robot/` |

### 建议阅读顺序

- **只想知道数据是怎么来的** → 01 → 02 → 03
- **想复现训练** → 03 → 04
- **想把模型跑上机器** → 05 → 06 → 07
- **要改传输层** → 06 先读透，它和 `serve.py` 是必须同版本的一对

---

## 三、贯穿全链路的四个约定

这四条在每一层都要成立，改动任何一层时都得回头看它们：

**① roll 用展开口径。** 采集端存的是控制器原生的近 ±180 表示，展开发生在数据集工具那一步
（`tools/trim_scan_segments.py --unwrap-roll`）。理由是三个策略的 action 归一化都是 `MEAN_STD`，
wrap 版会把 std 从 7.77 膨胀到 172。详见 [03](03-dataset-tools.md) 与[文档 03](../docs/03-roll-angle.md)。

**② 六维位序固定：`x, y, z, roll, pitch, yaw`，平移在前、旋转在后。**
`info.json` 里另有全名（`tcp_*` / `target_*`），但**仓库里没有任何工具去读它**——
所有脚本都硬编码短名与索引，roll 固定是索引 3。这是一处刻意的简单化：
代价是改维序会静默出错，收益是不用到处解析 JSON。

**③ 相机键名在发送侧就写成 policy 键。**
观测里的键是 `base_0_rgb` / `left_wrist_0_rgb`（policy 侧的名字），**不是**数据集里的
`observation.images.rgb`。原因是上游 `helpers.py` 拿 `policy_image_features[key]` 做直接索引，
发生在 preprocessor 之前，`rename_map` 救不了。详见 [06](06-transport-layer.md) 第四节。

**④ 时间戳一律用主机 wall clock。** 跨机器比时间戳只能用 `time.time()`；
设备时钟（RealSense 的 ms、超声的 200 MHz ticks）只在对齐那一步用最小二乘映射到主机钟。

---

## 四、和 `docs/` 的关系

写这批文档时核对代码，发现 `docs/` 里有若干处与现行代码对不上——参数名写错、
数值口径过时、把两件事说得互相矛盾等。**`docs/` 那几篇按时间记录的性质不改**，
所以订正**只写在这里**，不在 `docs/` 原处回改（回改会破坏「当时是这么想的」这层记录）。

核对代码后确认的**六处**，逐条给出判据；引用 `docs/` 时请以本表为准：

| `docs/` 位置 | 原文说的 | 实际（判据） |
|---|---|---|
| `docs/03` 第 72 行 | π0 训练用 `..._pi0_6d_rollunwrap_scan` | run1 用的是**未裁剪**的 `..._pi0_6d_rollunwrap`；判据是 `run1/checkpoints/012250/pretrained_model/train_config.json` 里的 `dataset.repo_id` / `dataset.root` |
| `docs/06` 第 30–34 行 | 单帧 2.96MB→0.30MB 归功于「图像编码压缩」+ `--wire numpy` | 两条都不是主因：`--wire` 管的是**下行动作的类型**（torch/numpy），与上行体积无关；图像也**没有编码**，省下来的是发送侧的 224×224 预缩 |
| `docs/06` 第 75–80 行 | `must_go` 是「活性／心跳」机制 | 它是**跳过服务端两道合理性检查、无条件入队**（上游 `policy_server.py:291`）。按心跳实现会得到一个能跑但不是原意的客户端 |
| `docs/09` 第 132 vs 165 行 | 132 行说工控机侧默认 `gzip`，165 行又在一张「两者共用同一套默认值」的表里写 `none` | 两边默认值**本来就不同**，不是矛盾：`real_robot_client.py:761` 默认 `gzip`，`remote_client.py:604` 默认 `none`。165 行那张表的标题才是错的 |
| `docs/09` 第 224 行 | `--sensor-max-age` 0.5 s，RealSense / 超声 / xArm **任一**超龄即停 | xArm 走的是 `snapshot(max_age_s=0.25)` 的**函数默认值**（`real_robot_client.py:492`），不受该参数影响。只有相机与超声受它管 |
| `docs/03` 第 3、65 行 | 把 `tools/trim_scan_segments.py --unwrap-roll` 与 `pi0_ultrasound_lora/unwrap_roll.py` 并列 | **两者不是一回事**：前者逐回合 `np.unwrap` + 整体平移对齐全局参考，且**重算 stats**；后者只是把负数整列 +360，**不写 stats.json**（`unwrap_roll.py` 全文只改 parquet，stats 仅打印） |

最该记住的还是那条总原则：`docs/` 是按**时间顺序**记问题的，
早期文档里描述的实现**可能已经不是现在的实现**。以 `design/` 和代码为准。
