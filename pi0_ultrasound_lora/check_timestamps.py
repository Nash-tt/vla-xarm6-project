#!/usr/bin/env python
"""扫全量数据集, 量化「parquet 时间戳」与「视频实际 PTS」之间的偏差。

训练在 step 455 崩了:
    FrameTimestampError: ... (tensor([0.0001]) > tolerance_s=0.0001)
    video: videos/observation.images.ultrasound/chunk-000/file-000.mp4

LeRobot 的判定是严格小于 (`is_within_tol = min_ < tolerance_s`), 所以偏差
恰好等于 1e-4 也算失败 —— 典型的浮点边界问题。但得先确认偏差到底有多大:
如果只是 1e-4 量级, 放大 tolerance_s 就行; 如果是半帧量级, 那是真的同步错误。

原理:
    query_ts = episodes[f"videos/<key>/from_timestamp"] + parquet["timestamp"]
    (见 dataset_reader._query_videos: shifted_query_ts = from_timestamp + ts)
只 demux 读 PTS, 不做解码, 所以很快。

结论: 真实偏差最大只有 ~7e-6 s, 数据没问题。真正的原因是 float32:
decode_video_frames 里 `torch.tensor(timestamps)` 把 float64 时间戳降成 float32,
而 float32 在 t≈1277s 处的 ULP 恰好是 1.22e-4 > tolerance_s=1e-4。
两个本该相差 5e-6 的数被量化到相邻的浮点格点上, 算出来就是 1 个 ULP。
所以脚本同时报「float64 真值」和「float32 实际算出来的值」。

用法: python check_timestamps.py [dataset_root]
"""

import glob
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/data/vla/vla_batch_20261007_pi0_6d_rollunwrap")

data = pq.read_table(sorted(glob.glob(str(root / "data/*/*.parquet")))[0],
                     columns=["timestamp", "episode_index", "frame_index"]).to_pandas()
eps = pq.read_table(sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))[0]).to_pandas()

pts_cache: dict[str, tuple[np.ndarray, float, int]] = {}


def get_pts(path: str):
    """只 demux 拿 PTS, 不解码像素。返回 (pts_seconds, time_base, n_packets)."""
    if path not in pts_cache:
        with av.open(path) as c:
            s = c.streams.video[0]
            tb = float(s.time_base)
            pts = np.array([float(p.pts * tb) for p in c.demux(s) if p.pts is not None])
        pts_cache[path] = (pts, tb, len(pts))
    return pts_cache[path]


video_keys = [k for k, v in __import__("json").load(open(root / "meta/info.json"))["features"].items()
              if v.get("dtype") == "video"]

print(f"数据集: {root}")
print(f"fps={15}  episodes={len(eps)}  frames={len(data)}\n")

def stats(name: str, dev: np.ndarray, total: int):
    print(f"  {name}")
    print(f"    最大 {dev.max():.3e} s   中位 {np.median(dev):.3e} s   平均 {dev.mean():.3e} s")
    for tol in (1e-4, 3e-4, 5e-4, 1e-3, 1e-2):
        n_bad = int((dev >= tol).sum())
        mark = "  <-- 训练用的默认值" if tol == 1e-4 else ""
        print(f"    偏差 >= {tol:<8.1e}: {n_bad:6d} / {total} 帧 ({100 * n_bad / total:6.3f}%){mark}")


for key in video_keys:
    print(f"=== {key} ===")
    devs64, devs32, total, nfiles, max_ts = [], [], 0, set(), 0.0
    for _, ep in eps.iterrows():
        ci = int(ep[f"videos/{key}/chunk_index"])
        fi = int(ep[f"videos/{key}/file_index"])
        path = str(root / f"videos/{key}/chunk-{ci:03d}/file-{fi:03d}.mp4")
        nfiles.add(path)
        pts, tb, n = get_pts(path)

        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        ts = data["timestamp"].values[a:b].astype(np.float64)
        q64 = float(ep[f"videos/{key}/from_timestamp"]) + ts
        max_ts = max(max_ts, q64.max())

        # 每个 query 找最近的实际 PTS (取左右邻居共 3 个候选)
        idx = np.clip(np.searchsorted(pts, q64), 1, len(pts) - 1)

        # --- float64 真值 (数据本身准不准) ---
        d64 = np.minimum(np.abs(q64 - pts[idx - 1]), np.abs(q64 - pts[idx]))
        devs64.append(d64)

        # --- float32 实际算出来的值 (torch.tensor(list) 会降精度) ---
        q32 = q64.astype(np.float32)
        d32 = np.minimum(
            np.abs(q32 - pts[idx - 1].astype(np.float32)),
            np.abs(q32 - pts[idx].astype(np.float32)),
        )
        devs32.append(d32)
        total += len(d64)

    ulp = float(np.spacing(np.float32(max_ts)))
    print(f"  video files: {len(nfiles)}   time_base={list(pts_cache.values())[0][1]:.3g}s   "
          f"最长视频 {max_ts:.1f}s   float32 ULP @该处 = {ulp:.3e}")
    stats("float64 真值 (数据本身):", np.concatenate(devs64), total)
    stats("float32 实际计算 (LeRobot 判定用的):", np.concatenate(devs32), total)
    print()
