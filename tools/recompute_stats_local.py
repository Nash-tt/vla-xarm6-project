#!/usr/bin/env python3
"""对本地数据集重算 meta/stats.json —— 直接调官方 recompute_stats，不自己写一份。

为什么必须走官方实现：裁剪工具只重算了"逐回合"的 numeric 统计列
（meta/episodes/*.parquet 里的 stats/<feature>/<stat>），数据集级的
meta/stats.json 是训练时归一化真正读的那一份，必须由官方实现重算，
否则两边口径会不一致。

注意官方行为（dataset_tools.py:1562）：
  - skip_image_video=True 时只遍历 numeric_keys，image/video 统计沿用原值、不重算。
    即使 skip_image_video=False 也一样 —— 源码里 numeric_keys 把 image/video 排除了，
    并留着 "TODO: enable image and video stats re-computation"。
    也就是说**上游本来就假定 image/video 统计是外部预置的**。
  - 更关键：numeric_features 排除了 meta_keys
    {"index","episode_index","task_index","frame_index","timestamp"}，
    而它写出的是一份**全新**的 stats.json（只有它算出来的键）。
    所以在一个没有 stats.json 的数据集上直接跑，你会丢掉 7 个键里的 5 个，
    训练时 make_dataset(factory.py:134) 会在
        dataset.meta.stats["observation.images.rgb"][...]
    上抛 KeyError —— 2026-10-12 就是这么炸的。

因此裁剪数据集必须用 --donor 给源数据集打底（源视频逐像素一致，image/video 统计等价），
再由官方 recompute_stats 覆盖 action/observation.state，最后由本脚本补算 meta_keys。

用法：
    # 裁剪后首次重算（必须给 --donor）
    python recompute_stats_local.py --root <裁剪后> --donor <源数据集>
    # 不给 --donor 时只重算 action/state，会重建出一份残缺的 stats.json，慎用
    python recompute_stats_local.py --root <数据集目录>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# lerobot 源码路径；用 pip 装的 lerobot 时这行是空操作，指到源码树可覆盖它
sys.path.insert(0, os.environ.get("LEROBOT_SRC", "/opt/lerobot/src"))


META_KEYS = ("index", "episode_index", "task_index", "frame_index", "timestamp")


def load_donor_stats(root: Path, donor: Path) -> dict:
    """读源数据集的 stats.json，作为 recompute_stats 的合并底座。

    不落盘：recompute_stats 的合并逻辑只读 `dataset.meta.stats` 这个内存对象
    （dataset_tools.py:1665 `for key, value in dataset.meta.stats.items()`），
    所以把 dict 直接赋给 ds.meta.stats 即可，不必往 target 里 dump 一份再覆盖。

    安全性：合并规则是 "new_stats 里没有的键才从 meta.stats 取"，而 action /
    observation.state 一定在 new_stats 里（它们是 numeric_features），
    所以源数据集的旧值不可能污染这两个；留下来的只有 image/video ——
    裁剪只改元数据不重编码（168/168 逐像素一致），统计等价。
    """
    donor_p = donor / "meta" / "stats.json"
    if not donor_p.is_file():
        raise SystemExit(f"donor 没有 meta/stats.json：{donor_p}")
    return json.loads(donor_p.read_text(encoding="utf-8"))


def recompute_meta_keys(root: Path) -> dict:
    """从裁剪后的 parquet 重算 meta_keys 的统计。

    官方 recompute_stats 明确跳过这 5 个键，但源数据集里它们是有的（9 键之一），
    裁剪后 index/timestamp 的分布完全变了，沿用源值就是错的。
    这里复用官方 compute_episode_stats + aggregate_stats，逐回合算再聚合，口径与录制时一致。
    """
    import numpy as np
    import pandas as pd

    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    keys = [k for k in META_KEYS if k in features]

    parquets = sorted((root / "data").glob("*/*.parquet"))
    if not parquets:
        raise SystemExit(f"找不到 parquet：{root / 'data'}")

    all_ep = []
    for p in parquets:
        df = pd.read_parquet(p)
        for ep in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == ep]
            data = {k: np.asarray(ep_df[k].values) for k in keys if k in ep_df.columns}
            all_ep.append(compute_episode_stats(data, {k: features[k] for k in data}))
    return aggregate_stats(all_ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--donor", type=Path, default=None,
                    help="源数据集（未裁剪），用它完整的 stats.json 打底。裁剪版必须给")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    root = args.root.resolve()
    if not (root / "meta" / "info.json").is_file():
        print(f"不是数据集目录：{root}", file=sys.stderr)
        return 1

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    print(f"数据集 : {root}")
    print(f"帧数   : {info['total_frames']}   回合: {info['total_episodes']}")

    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.donor is not None:
        donor = args.donor.resolve()
        seed = load_donor_stats(root, donor)
        print(f"打底   : {donor}  ({len(seed)} 个特征)")
        print("         image/video 沿用源值（裁剪不重编码，逐像素一致，统计等价）")
    else:
        seed = None
        print("打底   : 无 —— 只会得到 action/observation.state 两个键，训练会 KeyError")

    # 裁剪工具故意不复制 meta/stats.json，所以首次跑时它不存在
    stats_p = root / "meta" / "stats.json"
    if stats_p.exists():
        stats_p.rename(root / "meta" / "stats.json.bak")
    old = json.loads((root / "meta" / "stats.json.bak").read_text(encoding="utf-8")) \
        if (root / "meta" / "stats.json.bak").exists() else {}

    ds = LeRobotDataset("local/recompute", root=str(root), video_backend="pyav")
    if seed is not None:
        ds.meta.stats = seed
    recompute_stats(
        ds,
        skip_image_video=True,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )

    # 官方实现只管 action/state，剩下的 meta_keys 这里补
    if args.donor is not None:
        new = json.loads(stats_p.read_text(encoding="utf-8"))
        mk = recompute_meta_keys(root)
        mk = {k: {s: (v.tolist() if hasattr(v, "tolist") else v) for s, v in d.items()}
              for k, d in mk.items()}
        new.update(mk)
        write_stats(new, root)
        print(f"补算   : meta_keys {sorted(mk)}")

    new = json.loads(stats_p.read_text(encoding="utf-8"))

    def fmt(v):
        """把 stats 值压成一行。image 的统计是 (3,1,1) 嵌套，展平后逐通道显示。"""
        if not isinstance(v, list):
            return str(v)
        while v and isinstance(v[0], list):
            v = [x for c in v for x in c]
        return "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"

    print()
    for feat in sorted(new):
        print(f"{feat}")
        for stat in ("mean", "std", "min", "max"):
            if stat not in new[feat]:
                continue
            b = new[feat][stat]
            a = old.get(feat, {}).get(stat)
            tag = "" if a is None else f"   (原 {fmt(a)})"
            print(f"    {stat:<5} {fmt(b):>34}{tag}")

    # 完整性断言：训练路径 factory.py:129-134 会遍历 meta.camera_keys 去索引
    # meta.stats，缺任何一个就是 KeyError 崩溃。
    missing = [k for k in info["features"] if k not in new]
    print()
    if missing:
        print(f"!! stats.json 缺特征 {missing} —— 训练会在 make_dataset 崩", file=sys.stderr)
        return 2
    print(f"OK: stats.json 覆盖全部 {len(new)}/{len(info['features'])} 个特征")

    if args.donor is not None:
        donor_stats = json.loads((args.donor.resolve() / "meta" / "stats.json").read_text(encoding="utf-8"))
        bad = [k for k in donor_stats if k.startswith("observation.images") and new[k] != donor_stats[k]]
        print("OK: image/video 统计与源逐位一致" if not bad else f"!! image 统计被改动: {bad}")
    print(f"备份   : {root / 'meta' / 'stats.json.bak'}  (重算前)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
