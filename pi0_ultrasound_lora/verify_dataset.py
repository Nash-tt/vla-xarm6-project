#!/usr/bin/env python
"""校验数据集: 逐维对比 stats.json 与实际数据, 并检查 roll 是否已解卷绕。"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1])
names = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]

files = sorted(glob.glob(str(root / "data/chunk-*/file-*.parquet")))
tables = [pq.read_table(f) for f in files]
n_rows = sum(t.num_rows for t in tables)
print(f"数据文件 {len(files)} 个, 共 {n_rows} 帧")

stats = json.load(open(root / "meta/stats.json"))

for feat in ["observation.state", "action"]:
    arr = np.concatenate([np.stack(t.column(feat).to_pylist()) for t in tables]).astype(np.float32)
    print(f"\n=== {feat} ===   shape={arr.shape}")
    print(f"{'dim':10s} {'实测 mean':>11s} {'json mean':>11s} {'实测 std':>10s} {'json std':>10s} "
          f"{'实测范围':>22s} {'负值帧':>7s}")
    for i, n in enumerate(names):
        col = arr[:, i]
        rng = f"[{col.min():8.2f}, {col.max():8.2f}]"
        print(f"{n:10s} {col.mean():11.2f} {stats[feat]['mean'][i]:11.2f} "
              f"{col.std():10.2f} {stats[feat]['std'][i]:10.2f} {rng:>22s} {int((col < 0).sum()):7d}")

print("\n=== roll 分支检查 (应无 |roll|<90 的帧) ===")
for feat in ["observation.state", "action"]:
    arr = np.concatenate([np.stack(t.column(feat).to_pylist()) for t in tables]).astype(np.float32)
    r = arr[:, 3]
    mid = int(((r > -90) & (r < 90)).sum())
    print(f"{feat:18s} 落在 (-90,90) 的帧: {mid}  (解卷绕前应为 0, 因为数据全在 ±180 附近)")
