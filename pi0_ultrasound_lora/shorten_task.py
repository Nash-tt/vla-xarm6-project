#!/usr/bin/env python
"""把数据集的任务描述换成短句。

注意: 原描述 214 字符实测是 39 token, 并未超出 tokenizer 的 max_length=48,
所以「改短」不是为了规避截断, 只是让 prefix 更精简 (39 -> 11 token)。

单任务数据集里这句话每帧都相同, 模型完全可以当常数记住, 信息量为 0;
措辞真正影响的是部署时的一致性 —— 推理时输入的指令要和训练时这句对齐,
否则就是语言前缀上的分布漂移。

任务字符串存两处, 都要改:
  1. meta/tasks.parquet           —— 以任务字符串为索引, 列 task_index
  2. meta/episodes/*.parquet      —— tasks 列, list<string>, 每 episode 一条

用法: python shorten_task.py <dataset_root> "<新任务描述>"
"""

import glob
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

new_task = sys.argv[2]
root = Path(sys.argv[1])

# ---- 1. meta/tasks.parquet ----
tasks_path = root / "meta/tasks.parquet"
df = pq.read_table(tasks_path).to_pandas()
old = list(df.index)
if len(old) != 1:
    print(f"预期只有 1 个任务, 实际 {len(old)} 个, 中止", file=sys.stderr)
    raise SystemExit(1)
old_task = old[0]
df = df.rename(index={old_task: new_task})
pq.write_table(pa.Table.from_pandas(df), tasks_path)
print(f"tasks.parquet:\n  旧 ({len(old_task)} 字符): {old_task}\n  新 ({len(new_task)} 字符): {new_task}")

# ---- 2. meta/episodes/*.parquet 的 tasks 列 ----
ep_files = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
for f in ep_files:
    t = pq.read_table(f)
    idx = t.schema.get_field_index("tasks")
    col = t.column(idx).to_pylist()
    new_col = [[new_task if s == old_task else s for s in row] for row in col]
    t = t.set_column(idx, "tasks", pa.array(new_col, type=t.schema.field("tasks").type))
    tmp = Path(f).with_suffix(".parquet.tmp")
    pq.write_table(t, tmp, compression="snappy")
    tmp.replace(f)
    print(f"{Path(f).name}: 更新 {len(new_col)} 个 episode 的 tasks 列")
