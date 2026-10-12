#!/usr/bin/env python
"""解卷绕 LeRobot 数据集里 TCP roll 角的分支切割。

背景: roll_deg 的真实姿态集中在 +165°..+193° 附近, 但采集时跨越了 ±180° 的
欧拉角分支切割, 导致同一物理姿态被记成 +178 或 -178 两个符号。MEAN_STD 归一化
的 std 会因此被膨胀到 172 (真实 std 仅 ~7.7), 归一化后 ±180 两侧符号相反,
该维度基本学不到东西。

本脚本把负角 +360 映射到连续分支 ([-180,0) -> [180,360)), 使 roll 成为连续量。
只改 observation.state / action 的第 3 维 (roll_deg), 其余维度与列结构不动。

用法: python unwrap_roll.py <dataset_root> [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLL_DIM = 3  # info.json 中 names 的顺序: x, y, z, roll, pitch, yaw
FEATURES = ("observation.state", "action")


def unwrap_roll(arr: np.ndarray) -> np.ndarray:
    """把 [-180,0) 的 roll 平移到 [180,360), 消除 ±180 分支切割。"""
    v = arr[:, ROLL_DIM]
    arr[:, ROLL_DIM] = np.where(v < 0, v + 360.0, v)
    return arr


def process_file(path: Path, dry_run: bool) -> dict:
    table = pq.read_table(path)
    stats = {}
    for name in FEATURES:
        idx = table.schema.get_field_index(name)
        if idx < 0:
            raise KeyError(f"{path} 里没有 {name} 列")
        arr = np.stack(table.column(name).to_pylist()).astype(np.float32)
        before = arr[:, ROLL_DIM].copy()
        arr = unwrap_roll(arr)
        after = arr[:, ROLL_DIM]

        flat = pa.array(arr.reshape(-1), type=pa.float32())
        table = table.set_column(idx, name, pa.FixedSizeListArray.from_arrays(flat, arr.shape[1]))

        stats[name] = {
            "翻转帧数": int((np.sign(before) != np.sign(after)).sum()),
            "改前 std": float(before.std()),
            "改后 std": float(after.std()),
            "改后范围": (float(after.min()), float(after.max())),
        }

    if not dry_run:
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="snappy")
        tmp.replace(path)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = args.dataset_root / "data"
    files = sorted(data_dir.glob("*/*.parquet"))
    if not files:
        print(f"没找到 parquet: {data_dir}", file=sys.stderr)
        return 1

    grand = {n: {"翻转帧数": 0} for n in FEATURES}
    for f in files:
        s = process_file(f, args.dry_run)
        for n in FEATURES:
            grand[n]["翻转帧数"] += s[n]["翻转帧数"]
            grand[n].update({k: v for k, v in s[n].items() if k != "翻转帧数"})
        print(f"{f.name}: " + " | ".join(
            f"{n} 改符号 {s[n]['翻转帧数']} 帧, std {s[n]['改前 std']:.2f} -> {s[n]['改后 std']:.2f}"
            for n in FEATURES))

    print("\n=== 汇总 ===")
    for n in FEATURES:
        g = grand[n]
        print(f"{n:18s} 改符号 {g['翻转帧数']:6d} 帧 | std {g['改前 std']:7.2f} -> {g['改后 std']:5.2f} "
              f"| 范围 [{g['改后范围'][0]:.2f}, {g['改后范围'][1]:.2f}]")
    if args.dry_run:
        print("\n(dry-run, 未写入)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
