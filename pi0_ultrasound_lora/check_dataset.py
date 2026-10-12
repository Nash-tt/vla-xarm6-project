#!/usr/bin/env python
"""验证数据集副本能被 LeRobotDataset 正常加载, 并检查视频可解码。"""

import sys

from lerobot.datasets import LeRobotDataset

root = sys.argv[1]
repo_id = sys.argv[2]

ds = LeRobotDataset(repo_id=repo_id, root=root)
print(f"episodes : {ds.num_episodes}")
print(f"frames   : {ds.num_frames}")
print(f"fps      : {ds.meta.fps}")
print(f"tasks    : {ds.meta.tasks}")
print("\nfeatures:")
for k, v in ds.meta.features.items():
    print(f"  {k:32s} {v['dtype']:8s} {v['shape']}")

print("\n=== 抽 3 帧验证视频解码 + 数值 ===")
import numpy as np

for i in [0, 1000, 40000]:
    item = ds[i]
    imgs = {k: tuple(v.shape) for k, v in item.items() if k.startswith("observation.images.")}
    st = np.asarray(item["observation.state"]).ravel()
    ac = np.asarray(item["action"]).ravel()
    print(f"  frame {i:6d}: {imgs}  state[roll]={st[3]:7.2f}  action[roll]={ac[3]:7.2f}"
          f"  task={item.get('task', '?')!r}")

print("\n=== stats.json 里 state/action 的 roll 维 ===")
for f in ("observation.state", "action"):
    s = ds.meta.stats[f]
    print(f"  {f:18s} mean[3]={s['mean'][3]:7.2f}  std[3]={s['std'][3]:5.2f}")
