# 02 · `meta/stats.json`：一次崩溃与一处精度陷阱

> 工具：`tools/recompute_stats_local.py`　校验：`tools/verify_trim.py` 的 T10

该文件是训练时执行归一化实际读取的数据，也是本项目中最易出错的一环。
它引发过两类问题：一类是**启动即崩溃**，另一类是**读数不吻合但影响可忽略**。

---

## A. 崩溃：`KeyError: 'observation.images.rgb'`

### 现象

数据裁剪完成后，`lerobot-train` 启动即报错：

```
KeyError: 'observation.images.rgb'
  File ".../lerobot/common/policies/factory.py", line 134, in make_dataset
```

报错位置指向相机配置，实际原因是统计文件缺键。

### 原因

裁剪工具**有意不复制** `meta/stats.json`（该文件记录裁剪前的分布，保留反而有害）。
因此裁剪版不含 stats.json，转而调用官方 `recompute_stats` 重算。
但官方实现产出的是一份**全新文件**，其中**仅含两个键**：

```
action
observation.state
```

三个因素叠加（均位于 `lerobot/datasets/dataset_tools.py:1562` 附近）：

1. `numeric_features` 排除了 5 个 meta 键
   （`index` / `episode_index` / `task_index` / `frame_index` / `timestamp`）；
2. **image / video 统计从不重算**——源码中保留有
   `TODO: enable image and video stats re-computation`。
   即便传入 `skip_image_video=False` 亦无变化，因为 `numeric_keys` 本身已排除
   image/video；
3. 其写出的是**新文件**，而非增量更新。

而训练侧 `factory.py:129-136` 会遍历 `meta.camera_keys` 索引 `meta.stats`：

```python
if cfg.dataset.use_imagenet_stats:
    for key in dataset.meta.camera_keys:
        dataset.meta.stats[key][stats_type] = ...   # 缺键在此抛出
```

缺少 `observation.images.rgb` 即触发 `KeyError`。

### 处理：以 `--donor` 打底

关键认识是：**这并非变通方案，而是上游既有的设计假设。**
上游从不重算 image/video 统计，其前提是这些值由外部预置。

因此正确做法是以**源数据集的 stats.json 打底**，再由官方实现覆盖其可计算的部分：

```bash
python tools/recompute_stats_local.py --root <裁剪版> --donor <原数据集>
```

该做法的安全性来自官方自身的合并规则（`dataset_tools.py:1665-1670`）：

```python
if dataset.meta.stats:
    for key, value in dataset.meta.stats.items():
        if key not in new_stats:      # 新算出的键优先
            new_stats[key] = value
```

即：**新 stats 中已存在的键，源值无法覆盖。**
`action` / `observation.state` 必然存在于 `new_stats`（二者属于 numeric_features），
故源数据集的旧值不可能污染这两个键；留下的仅为 image/video——
而裁剪**不重编码视频**（逐像素验证 168/168），两侧统计完全等价。

实现上无需先向目标目录写入一份再覆盖：将 donor 的 dict 直接赋给 `ds.meta.stats`
即可，因为合并逻辑仅读取该内存对象。

### 补算 meta 键

`recompute_stats` 跳过上述 5 个 meta 键，但**裁剪后其分布已完全改变**
（`index` 重排、`timestamp` 归零重新计时），沿用源值即为错误。

脚本改用官方 `compute_episode_stats` + `aggregate_stats` 逐回合计算后聚合，
与录制时的口径一致。

### 验证

```
OK: stats.json 覆盖全部 9/9 个特征
OK: image/video 统计与源逐位一致
     action / observation.state 与上一次官方重算逐位一致
```

脚本末尾设有**完整性断言**：任一 `features` 键缺失即返回退出码 2 并打印，
以避免该崩溃被带入训练阶段。

### 结论

`verify_trim.py` 的 T7 已覆盖「逐回合 stats 列」，但**数据集级的 `meta/stats.json`
是另一个文件、另一条读取路径**，初期完全未被覆盖。发现问题后补充了 **T10**。
结论是：校验应依「训练实际读取哪些文件」枚举，而非依「本次修改了哪些文件」枚举。

---

## B. 精度陷阱：float32 累加器引入 3.5% 的 `std` 偏差

### 现象

补充 T10 后，其报告一项 std 不吻合：

```
std 实测 0.3900  期望 0.3767     （相差 3.5%）
```

### 原因

`RunningQuantileStats`（`compute_stats.py`）使用的公式为：

```python
std = sqrt(mean_of_squares - mean**2)
```

累加器 dtype **随输入 dtype 而定**，而 parquet 中 `observation.state` 与 `action`
为 **float32**。

对「均值大、方差小」的维度，`mean²` 与 `var` 相差数个数量级，相减时有效位数被抵消。
实测各维的 `mean²/var`：

```
[18.7, 0.3, 265.8, 509.5, 2.8, 59000.4]
                          ↑ gripper 维
```

gripper 维 `mean ≈ −91.5`、`var ≈ 0.142`，抵消比达 59000 倍。
float32 在 `mean²` 尺度上的绝对舍入（约 1.2e-3）已与 `var` 本身同量级。

在 27312 帧数据集上的实测结果：

| 路径 | gripper `std` |
|---|---|
| float64 累加器 | 0.376698（与 float128 参考值一致） |
| float32、单批全量 | **0.0**（方差被完全抵消） |
| `recompute_stats` 逐回合分批 | 0.390023（**+3.5%**） |

最后一行反而优于「单批全量」：逐回合分批时每次累加的尺度较小，抵消程度较轻。
**该偏差的幅度依赖于分批方式**，因此不是一个稳定量。

另需说明：官方 `aggregate_feature_stats` 使用的是正确的池化方差公式
（`sqrt(sum((var + delta_mean²) * count) / total_count)`），与 float128 参考值吻合。
**存在问题的仅为 `RunningQuantileStats` 这一个累加器。**

### 对训练的实际影响：可忽略

归一化为 `(x − mean) / std`，std 相差 3.5% 即该维尺度相差 3.5%。逐维结果：

| 维度 | std 相对偏差 |
|---|---|
| gripper | 3.5% |
| roll | 0.015% |
| dim0 | 0.0095% |

`mean` 同样受 float32 累加影响（逐帧累加 27312 次，绝对误差约 1e-3）。
`min`/`max` 不涉及累加，结果精确。

### 处理：按「判据的实际职责」设定容差

`verify_trim.py` 的 T10 最终采用：

```python
("min",  arr.min(0),  0,     1e-3),   # 精确，容差从严
("max",  arr.max(0),  0,     1e-3),
("mean", arr.mean(0), 1e-4,  1e-3),
("std",  arr.std(0),  5e-2,  1e-6),   # 放宽至 5%
```

该判据的实际职责是检出两类**偏差达数十倍至数千倍**的错误：

1. 沿用了源数据集的陈旧值（裁剪后未重算）
2. roll 口径不符（展开与 wrap 之差，见[文档 03](03-roll-angle.md)，为 22 倍）

相对于这两类，3.5% 属于噪声量级。**放宽容差并非掩盖问题，而是使判据对准其应覆盖的对象。**

### 结论

1. **不应尝试「修复」上游的该数值问题。** 收益为零，且可能使本地行为偏离官方口径，
   导致后续无法对账。
2. 校验某个数值时，应先确定其误差来源与量级，再设定容差；
   而非先设定一个「看起来严格」的容差，再被其反复否定。
3. 该结论同时修正了我们此前写在脚本注释中的错误数值：
   原先记录的 roll std 为 7.83 / 7.727，**裁剪子集上的真值为 7.7661**。
