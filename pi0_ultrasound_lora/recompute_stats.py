#!/usr/bin/env python
"""在全量数据上重算 observation.state / action 的统计量并写回 meta/stats.json。

不走 lerobot 的 recompute_stats (它逐 episode 算完再用 aggregate_stats 聚合,
而 aggregate_stats 把分位数按 count 加权平均 —— 平均分位数 != 全局分位数)。
这里直接在完整数组上调 get_feature_stats, 分位数也是正确的。

其余特征(图像/索引等)的 stats 原样保留。

用法: python recompute_stats.py <dataset_root>
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets import get_feature_stats, write_stats
from lerobot.datasets.compute_stats import DEFAULT_QUANTILES

FEATURES = ("observation.state", "action")


def main() -> int:
    root = Path(sys.argv[1])
    files = sorted(glob.glob(str(root / "data/chunk-*/file-*.parquet")))
    if not files:
        print(f"没找到 parquet: {root/'data'}", file=sys.stderr)
        return 1

    tables = [pq.read_table(f) for f in files]
    stats = json.load(open(root / "meta/stats.json"))

    for feat in FEATURES:
        # 必须用 float64: lerobot 的 RunningQuantileStats 用 E[x^2]-E[x]^2 求方差,
        # 在 float32 下对「均值大、方差小」的维度会发生灾难性相消 (实测 yaw 的 std
        # 会被算成 0.95 而非正确的 1.39)。float64 下与 numpy 完全一致。
        arr = np.concatenate([np.stack(t.column(feat).to_pylist()) for t in tables]).astype(np.float64)
        # axis=0 => 每维独立统计, 与 lerobot 对向量特征的处理一致
        new = get_feature_stats(arr, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)
        old = stats[feat]
        print(f"\n=== {feat} ===  ({arr.shape[0]} 帧, {arr.shape[1]} 维)")
        for k in ("mean", "std"):
            print(f"  {k}: " + "  ".join(f"{o:9.2f} -> {n:9.2f}" for o, n in zip(old[k], new[k])))
        print(f"  q01: " + "  ".join(f"{o:9.2f} -> {n:9.2f}" for o, n in zip(old["q01"], new["q01"])))
        print(f"  q99: " + "  ".join(f"{o:9.2f} -> {n:9.2f}" for o, n in zip(old["q99"], new["q99"])))
        stats[feat] = new

    write_stats(stats, root)
    print(f"\n已写回 {root/'meta/stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
