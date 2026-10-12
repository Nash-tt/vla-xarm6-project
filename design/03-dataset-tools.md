# 03 · 数据集工具

**跑在哪：** GPU 机器，代码在 `tools/`（三个脚本，1387 行）。
**输入：** [02](02-align-and-build.md) 产出的 LeRobot 格式数据集。
**产出：** 一个可以直接喂给训练的裁剪版数据集。

三个脚本，**必须按顺序跑**，每一步都有下一步依赖的副作用：

```bash
# ① 裁剪 + 解 wrap —— 只改元数据，不重编码视频
python tools/trim_scan_segments.py \
    --src <原数据集> --dst <新数据集> --unwrap-roll --link-videos

# ② 重算统计量 —— --donor 是必须的，不是可选
python tools/recompute_stats_local.py --root <新数据集> --donor <原数据集>

# ③ 回读验证 —— 十项，任何一项挂都说明前两步有问题
python tools/verify_trim.py --src <原数据集> --new <新数据集>
```

> 注意三个脚本的目标参数名**不统一**：`trim` 用 `--dst`，
> `verify` 用 `--new`，`recompute` 用 `--root` + `--donor`。
> 照抄命令行时容易错。

---

## 一、要解决的问题

每个 episode 的开头有一段**接近段**：机械臂从一个采集起始位姿出发，
经"定高平移 → 垂直下压"，落到位才真正开始扫描（见 [01 第四节](01-collection.md)）。
这段被有意包含在 `capture_window` 里，但它的**动作语义和扫描段完全不同**：

```text
ep0 第 300 帧:
    state  y =    1.924 mm
    action y = -229.240 mm        |action - state| = 231 mm
```

接近段里 `action` 冻结在几百毫米外的目标点上，不跟随 state。全库统计：

- `action` 的 y 维中位数是 **-156.9 mm**（落在冻结值区间）
- `action` 的 roll 维最小值 **32.31°**，而 state 的 roll 只在 **[163.97, 193.04]** 之间
- 扫描段里 `action` 是正常的"当前位置指令"，`|a-s|` 中位 **~0.17 mm**（略微领先实测姿态）

**留着接近段就是在教模型"输出一个几百毫米外的点"**，且在 roll 维上输出一个
物理上根本不存在的值。所以必须裁掉。

---

## 二、判据：主判据是"成片污染"，不是单帧超标

`compute_boundary`（`trim_scan_segments.py:139-220`）。逐 episode 独立定边界 `b`，
保留 `[b, n)`：

```text
d_pos[t] = ||a[t,:3] - s[t,:3]||                位置偏差, mm
d_rot[t] = max_i |a[t,3+i] - s[t,3+i]|          姿态偏差, deg, 按角度环绕
bad[t]   = (d_pos[t] >= 1.0) or (d_rot[t] >= 1.0)
roll[t]  = bad 在窗口 [t-win+1, t] 里的比例      win = round(win_s * fps) = 15
T        = 最后一个 roll[t] > tol 的 t           tol = 0.2
dense    = T + 1                                  ← 主判据
```

再取两个更粗的兜底判据，与主判据取**较晚者**：

| 判据 | 定义 | 防的是什么 |
|---|---|---|
| `dense`（主） | 最后一个"污染密度 > tol"的窗口的下一帧 | 成片的接近段 |
| `rule95` | 第一个 i，使其后 ≥ 80% 的帧都干净 | 密度判据在长尾上的钝感 |
| `unfreeze` | 最后一个 `d_pos > 50 mm` 的帧 + 1 | 尾部零星的大偏差（冻结目标） |

```python
b = max(dense, rule95, unfreeze, 0)
cap = max(1, int(n * max_trim_frac))     # max_trim_frac = 0.85
b = min(b, cap, n - 1)                   # 至多裁 85%，至少留 1 帧
```

**姿态维必须和位置维一起看。** 接近段冻结的是整个末端位姿，有时位置差得不远而姿态差很多
（roll 从 163° 到 32°，差 131°）。但两者的孤立偏差性质不同：

- 位置上 1–3 mm 的过冲是**正常的跟踪误差**
- 姿态上差 100 多度**只可能是"命令还指着另一个目标"**

所以密度判据负责切"成片的污染块"，**孤立帧不在这里切**——切了会把整段扫描陪葬
（视频时间戳连续性会断）。孤立异常由 `find_anomalies`（`:222`）单独列出来，
写进 `trim_report.json`，由人决定。`--trim-rot-anomalies` 可以把边界推到
最后一个 `|姿态偏差| >= hard_deg` 的帧之后，代价是尾部若干帧陪葬，默认关闭。

> `trim_scan_segments.py` 顶部的模块 docstring 只写了"两个条件的较晚者"
> （95% 规则与解冻点），**漏了主判据 dense**。以 `compute_boundary` 的 docstring 与代码为准。

### 实测结果

28/28 个 episode 的 `biting` 全是 **`dense`**——即主判据在每个回合都起了决定作用，
两个兜底判据一次都没咬到。这说明污染确实是成片的、边界是清晰的。

---

## 三、为什么不重编码视频

这是整个工具最关键的一处设计。**裁剪只改元数据，一个字节的视频都不动。**

LeRobot v3.0 的视频是**共享 mp4**：多个 episode 共用同一个文件，靠每个 episode 的
`videos/<key>/from_timestamp` 定位。数据 parquet 里的 `timestamp` 每个 episode 从 0 重计。
读取端 `dataset_reader._query_videos()` 取的是：

```text
from_timestamp + timestamp
```

所以要裁掉前 `b` 帧，只需：

- data parquet 只留边界后的帧，`frame_index` 从 0 重编号，`timestamp = frame_index / fps`
- `episodes` 里 `from_timestamp_new = from_timestamp_old + b / fps`

偏移后的绝对时间戳与原数据集**几乎逐位相同**（差 ~1e-6 s，而 `tolerance_s = 1e-4`），
所以视频内容根本不需要重新编码，解码出来的像素必然一致。

好处是巨大的：重编码 28 个 episode 的 720×1280 视频是小时级的操作，
且有损；现在裁剪是秒级的，且**无损**——`verify_trim.py` 的 T3 会逐像素验证这一点。

`--link-videos` 进一步让视频用硬链接而不是复制，省掉一份 168 个文件的体积。

---

## 四、`--unwrap-roll`：一个必须打开的开关

裁剪的同时把 roll 解成展开口径。**判据用环绕口径算，所以这个开关不影响裁剪结果**——
它只改输出的 state/action 数值。

理由见[文档 03](../docs/03-roll-angle.md)，这里给出量化对比（实测）：

| | wrap 版（控制器原生 ±180） | 解 wrap 后 |
|---|---|---|
| roll 的 std | **172** | **7.7661** |
| 物理区间 [163.967, 193.039] 映射到 | z ∈ [0.819, 0.988] | z ∈ [-1.442, +2.272] |
| z 的跨度 | 0.169，且全挤在 +0.9 附近 | 3.71 |

三个策略的 action 归一化都是 `MEAN_STD`（除以 std）。**wrap 版把 roll 压成了一个近似常数**，
模型在这维上学不到任何东西——不是"学得差"，是输入端就没有信息。

`unwrap_roll_inplace`（`:103-135`）两步走：

1. **每个 episode 内部 `np.unwrap`** —— 消掉回合中间的 ±180 跳变
2. **每个 episode 整体平移 360 的整数倍**，使各回合的中位数对齐到全局参考值

第 2 步不能省。少了它，各回合会各自停在不同的 360 分支上，
算出来的 std 是**分支间距**而不是物理跨度。

> 别把这个和解 wrap 与 `pi0_ultrasound_lora/unwrap_roll.py` 混为一谈。
> 后者是把负数整列 +360 的粗暴映射，**且只改 parquet、不重算 stats**
> （全文不写 `stats.json`，算出来的统计量只打印）。
> `docs/03` 第 3、65 行把两者并列成同一件事，是错的。

---

## 五、`--donor`：不给我就会炸

`recompute_stats_local.py`（192 行）。它**直接调官方 `recompute_stats`**
（`lerobot.datasets.dataset_tools`），不自己写一份——理由是数据集级 stats 是训练时
归一化真正读的那一份，口径必须与官方一致。

但官方实现有个必须绕开的坑（`dataset_tools.py:1562`）：

- `skip_image_video=True` 时只遍历 numeric_keys，**image/video 统计沿用原值、不重算**。
  即使传 `False` 也一样——源码里 `numeric_keys` 把 image/video 排除了，
  并留着 `TODO: enable image and video stats re-computation`。
  **上游本来就假定 image/video 统计是外部预置的。**
- 更关键：它写出的是一份**全新**的 `stats.json`，只有它算出来的键。
  在一个没有 `stats.json` 的数据集上直接跑，会丢掉 7 个键里的 5 个，
  训练时 `make_dataset`（`factory.py:134`）在
  `dataset.meta.stats["observation.images.rgb"][...]` 上抛 `KeyError`。

**这就是 2026-10-12 那次崩溃的原因。**

所以裁剪数据集第一次重算 stats **必须给 `--donor`**：

1. 读源数据集的 `stats.json` 作为合并底座（`load_donor_stats`，`:39`，不落盘）
2. 官方 `recompute_stats` 覆盖 `action` / `observation.state`（这两个一定在 new_stats 里，
   源值不可能污染它们）
3. 剩下的 **meta_keys**（`index` / `episode_index` / `task_index` / `frame_index` / `timestamp`）
   由本脚本的 `recompute_meta_keys`（`:60`）补算——这 5 个键官方明确跳过，
   但裁剪后 `index`/`timestamp` 的分布完全变了，沿用源值就是错的

安全性在于**裁剪不重编码**：源视频与裁剪后逐像素一致，所以沿用源的 image/video 统计是等价的。
脚本最后会断言 `new[k] == donor_stats[k]` 逐位一致，并检查 `stats.json`
覆盖了全部特征，否则返回码 2。

### 5.1 float32 的数值怪癖

官方 `compute_stats.py:115` 用 `variance = mean_of_squares - mean**2` 这个**数值不稳定**的公式。
在 float32 累加下实测：

- gripper 维的 std 误差 **+3.5%**
- yaw 维 std 算出来 **0.95**，而精确池化值是 **1.39**

**校验时别拿精确池化值当判据。** 这不是 bug 修复点，是官方实现的既有特征；
强行"修正"反而会让训练侧与实际 stats 不一致。详见[文档 02](../docs/02-stats-json.md)。

---

## 六、验证：`verify_trim.py` 的十项

472 行，十项检查，任何一项失败即非零返回：

| 项 | 查什么 |
|---|---|
| T1 | 元数据自洽：总数、回合数、每回合 index 区间连续、`timestamp = frame_index/fps` |
| T2 | **状态/动作逐位相同**：新 (ep,j) 对应原 (ep, b+j)，数值必须完全相等 |
| T3 | **视频逐像素相同**：解码同一帧，新旧张量必须完全相等 |
| T4 | 绝对时间戳重建：`from_new + ts_new` 与 `from_old + ts_old` 之差应 ≪ `tolerance_s` |
| T5 | 动作语义：`\|a-s\|` 分布；原数据集里 `action.roll` 的野生值应已消失 |
| T6 | 运动连续：保留区每帧位移仍 ~0.66 mm/帧，说明留下的是真扫描而非静止保持 |
| T7 | 逐回合 stats 与数据自洽（新数据集） |
| T8 | 保留区内无成片污染（窗口密度判据复查） |
| T9 | **`delta_timestamps` 取样本可用**——训练实际走的那条路径 |
| T10 | `meta/stats.json` 完整性：键齐、形状对、数值与裁后数据一致 |

T2、T3、T9、T10 是四根顶梁柱：

- **T2/T3** 证明"只改元数据"这件事真的做到了（不是"应该能"）
- **T9** 不测 parquet，测的是 `delta_timestamps` 取样本——训练真正走的路径。
  前面的检查全过而这一步挂是完全可能的（时间戳重建差一点点就会在取窗口时越界）
- **T10** 是 stats 那个坑的回归测试

`rot_dist`（`:41`）与 trim 工具用**同一个环绕口径**，注释特别说明：
pi0 那版 roll 是展开的（163–193），smolvla 那版是 wrap 的（-180–180），
同一批数据在两版里差 360°，直接相减会得到 359.8° 的假偏差。

---

## 七、裁剪后的数据集长什么样

以 `vla_batch_20261007_pi0_6d_rollunwrap_scan` 为例（实测）：

| | 值 |
|---|---|
| 帧数 | **27312** |
| episode 数 | **28** |
| fps | **15** |
| 特征 | `observation.images.rgb` (3,720,1280)、`observation.images.ultrasound` (3,256,256)、`observation.state` (6,)、`action` (6,) |
| state 名 | `tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_roll_deg, tcp_pitch_deg, tcp_yaw_deg` |
| action 名 | `target_*`（同后缀） |

**注意：`info.json` 里的全名没有工具会去读。** 所有脚本硬编码短名与索引，
roll 固定是索引 3（见 [design/README 约定②](README.md)）。

磁盘上另有一份 smolvla 版（`vla_batch_20261007_smolvla_6d_no_depth_rollunwrap_scan`），
与 pi0 版**结构完全相同**——同样的帧数、回合数、fps、特征、state 名，
解 wrap 后的数值**逐位相同**。唯一的差别是 task 字符串（60 字符 vs 214 字符）。
**ACT 不吃语言，两版对它完全等价**（见 [04 训练](04-training.md)）。

---

## 八、修完这一步之后

裁剪版是训练的实际输入。三个策略脚本都指向它：

- π0-LoRA → `pi0_ultrasound_lora/run_train.sh`
- SmolVLA → `run_train_smolvla_scan.sh`
- ACT → `run_train_act_scan.sh`

继续读 [04 训练](04-training.md)。
