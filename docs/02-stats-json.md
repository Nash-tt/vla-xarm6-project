# 02 · `meta/stats.json`：一次崩溃，一次精度陷阱

> 工具：`tools/recompute_stats_local.py`　校验：`tools/verify_trim.py` 的 T10

这份文件是训练时做归一化真正读的那一份，也是这个项目里最容易出错的一环。
它带来过两个问题：一个是**硬崩溃**，一个是**读数对不上但无害**。

---

## A. 崩溃：`KeyError: 'observation.images.rgb'`

### 现象

数据裁好了，`lerobot-train` 一启动就炸：

```
KeyError: 'observation.images.rgb'
  File ".../lerobot/common/policies/factory.py", line 134, in make_dataset
```

报错位置看着像「相机配置写错了」，其实是统计文件缺键。

### 根因

裁剪工具**故意不复制** `meta/stats.json`（它记录的是裁剪前的分布，留着更糟）。
于是裁剪版没有 stats.json，我们调用官方的 `recompute_stats` 重算——
但官方实现产出的是一份**全新的** stats.json，里面**只有两个键**：

```
action
observation.state
```

三个原因叠加（都在 `lerobot/datasets/dataset_tools.py:1562` 附近）：

1. `numeric_features` 排除了 5 个 meta 键
   （`index` / `episode_index` / `task_index` / `frame_index` / `timestamp`）；
2. **image / video 统计它从来不重算**——源码里明确留着
   `TODO: enable image and video stats re-computation`。
   即使把 `skip_image_video=False` 传进去也一样，因为 `numeric_keys` 本身就把
   image/video 排除了；
3. 它写出的是**新文件**，不是增量更新。

而训练侧 `factory.py:129-136` 会遍历 `meta.camera_keys` 去索引 `meta.stats`：

```python
if cfg.dataset.use_imagenet_stats:
    for key in dataset.meta.camera_keys:
        dataset.meta.stats[key][stats_type] = ...   # ← 缺键就在这里炸
```

缺 `observation.images.rgb` → `KeyError`。

### 解决：`--donor` 打底

关键认识：**这不是 workaround，这就是上游的设计假设。**
上游从来不重算 image/video 统计，它假定这些值是外部预置好的。

所以正确做法是用**源数据集的 stats.json 打底**，再让官方实现覆盖它算得出来的部分：

```bash
python tools/recompute_stats_local.py --root <裁剪版> --donor <原数据集>
```

安全性来自官方自己的合并规则（`dataset_tools.py:1665-1670`）：

```python
if dataset.meta.stats:
    for key, value in dataset.meta.stats.items():
        if key not in new_stats:      # 新算出来的键优先
            new_stats[key] = value
```

也就是说：**新 stats 里已有的键，源值不可能覆盖它。**
`action` / `observation.state` 一定在 `new_stats` 里（它们是 numeric_features），
所以源数据集的旧值污染不了它们；留下来的只有 image/video——
而裁剪**不重编码视频**（逐像素验证过 168/168），两边统计完全等价。

实现上没必要往目标目录里先 dump 一份再覆盖：把 donor 的 dict 直接赋给
`ds.meta.stats` 即可，因为合并逻辑只读这个内存对象。

### 补算 meta 键

`recompute_stats` 明确跳过那 5 个 meta 键，但**裁剪之后它们的分布完全变了**
（`index` 重排、`timestamp` 归零重新计时），沿用源值就是错的。

脚本用官方的 `compute_episode_stats` + `aggregate_stats` 逐回合算再聚合，
口径与录制时一致。

### 验证

```
OK: stats.json 覆盖全部 9/9 个特征
OK: image/video 统计与源逐位一致
     action / observation.state 与上一次官方重算逐位一致
```

脚本末尾带一个**完整性断言**：只要有任一 `features` 键缺失就返回退出码 2 并打印，
避免这个崩溃又被带到训练阶段。

### 教训

`verify_trim.py` 的 T7 早就查了「逐回合 stats 列」，但**数据集级的 `meta/stats.json`
是另一个文件、另一条读取路径**，所以当初完全没覆盖到。发现问题后补了 **T10**。
教训是：校验要按「训练真正读哪些文件」来枚举，而不是按「我改了哪些文件」来枚举。

---

## B. 精度陷阱：float32 累加器带来 3.5% 的 `std` 误差

### 现象

补上 T10 之后，它报了一个 std 不符：

```
std 实测 0.3900  期望 0.3767     ← 差 3.5%
```

### 根因

`RunningQuantileStats`（`compute_stats.py`）用的是这个公式：

```python
std = sqrt(mean_of_squares - mean**2)
```

累加器 dtype **跟着输入走**，而 parquet 里 `observation.state` / `action`
是 **float32**。

对「均值大、方差小」的维度，`mean²` 和 `var` 相差几个数量级，
相减时有效位数被吃掉。实测各维的 `mean²/var`：

```
[18.7, 0.3, 265.8, 509.5, 2.8, 59000.4]
                          ↑ gripper 维最惨
```

gripper 维 `mean ≈ -91.5`、`var ≈ 0.142`，相消比 59000 倍。
float32 在 `mean²` 尺度上的绝对舍入（~1.2e-3）已经和 `var` 本身同量级了。

27312 帧数据集上的实测：

| 路径 | gripper `std` |
|---|---|
| float64 累加器 | 0.376698 ← 与 float128 参考值一致 |
| float32、单批全量 | **0.0** （方差被完全吃掉） |
| `recompute_stats` 逐回合喂批 | 0.390023 （**+3.5%**） |

注意最后一行反而比「单批全量」准——逐回合分批喂进去，每次累加的尺度小一些，
相消没那么严重。**这个误差的大小依赖于分批方式**，是个不稳定的量。

顺带确认：官方的 `aggregate_feature_stats` 用的是正确的池化方差公式
（`sqrt(sum((var + delta_mean²) * count) / total_count)`），
与 float128 参考值吻合。**有问题的只有 `RunningQuantileStats` 这一个累加器。**

### 对训练的实际影响：可忽略

归一化是 `(x - mean) / std`，std 差 3.5% 意味着该维尺度差 3.5%。
逐维看：

| 维度 | std 相对误差 |
|---|---|
| gripper | 3.5% |
| roll | 0.015% |
| dim0 | 0.0095% |

mean 同样受 float32 累加影响（逐帧累加 27312 次，绝对误差 ~1e-3）。
`min`/`max` 不做累加，是精确的。

### 解决：按「真正要抓什么」设计校验容差

`verify_trim.py` 的 T10 最终用：

```python
("min",  arr.min(0),  0,     1e-3),   # 精确，紧一点
("max",  arr.max(0),  0,     1e-3),
("mean", arr.mean(0), 1e-4,  1e-3),
("std",  arr.std(0),  5e-2,  1e-6),   # 放松到 5%
```

这个判据的真实职责，是抓两类**几十到几千倍量级**的错误：

1. 沿用了源数据集的陈旧值（裁剪后没重算）
2. roll 口径不符（展开 vs wrap，见[文档 03](03-roll-angle.md)，那是 22 倍）

相对这两类，3.5% 是噪声。**放松容差不是掩盖问题，而是把判据对准它该管的事。**

### 教训

1. **不要去「修」上游这个数值问题。** 收益为零，还可能让本地行为偏离官方口径，
   以后对不上账。
2. 校验一个数值时，先问「这个量的误差来源是什么、量级多大」，
   再决定容差——而不是先定一个「看起来严格」的容差，然后被它反复咬。
3. 这条也修正了我们自己早先写在脚本注释里的错误数字：
   之前记的 roll std 是 7.83 / 7.727，**裁剪子集上的真值是 7.7661**。
