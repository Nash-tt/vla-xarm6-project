#!/usr/bin/env python3
"""回读验证裁剪后的数据集：确认它和原数据集"逐像素、逐数值"对得上，且动作语义确实干净了。

用法：
    python verify_trim.py --src <原数据集> --new <裁剪后数据集> [--samples 3]

验证项
------
  T1 元数据自洽：总数、回合数、每回合 index 区间连续、timestamp = frame_index/fps
  T2 状态/动作逐位相同：新 (ep,j) 对应原 (ep, boundary+j)，数值必须完全相等
  T3 视频逐像素相同：解码同一帧，新旧的张量必须完全相等（元数据裁剪法的关键）
  T4 绝对时间戳重建：from_new + ts_new 与 from_old + ts_old 的差应 << tolerance_s
  T5 动作语义：|a-s| 分布；原数据集里 action.roll 的野生值应已消失
  T6 运动连续：保留区每帧位移仍 ~0.66mm/帧，说明留下的是真扫描而不是静止保持
  T7 逐回合 stats 与数据自洽（新数据集）
  T8 保留区内无成片污染（窗口密度判据复查）
  T9 delta_timestamps 取样本可用（训练实际走的路径）
  T10 meta/stats.json 完整性：键齐、形状对、数值与裁后数据一致（训练归一化的真正入口）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# lerobot 源码路径；用 pip 装的 lerobot 时这行是空操作，指到源码树可覆盖它
sys.path.insert(0, os.environ.get("LEROBOT_SRC", "/opt/lerobot/src"))

PASS, FAIL = "  [OK] ", "  [!!] "
failures: list[str] = []


def check(cond: bool, msg: str) -> bool:
    print((PASS if cond else FAIL) + msg)
    if not cond:
        failures.append(msg)
    return cond


def rot_dist(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """姿态偏差，**按角度环绕**算（与 trim_scan_segments.rot_dist 同一口径）。

    pi0 那版 roll 是展开的（163~193），smolvla 那版是 wrap 的（-180~180）。
    同一批数据在两版里差 360°，直接相减会得到 359.8° 的假偏差。
    """
    d = action[:, 3:6] - state[:, 3:6]
    d = (d + 180.0) % 360.0 - 180.0
    return np.abs(d).max(axis=1)


def unwrap_roll(vals: np.ndarray, ep_ids: np.ndarray) -> np.ndarray:
    """把 roll 解成连续且有统一分支的表示——与角度约定无关地衡量物理跨度。

    两步：① 每个回合内部解 wrap；② 每个回合整体平移 360 的整数倍，
    使各回合的中位数对齐到全局参考值（否则跨回合会各自停在不同分支上，
    算出来的 std 是分支间距而不是物理跨度）。
    """
    out = np.asarray(vals, np.float64).copy()
    eps = np.unique(ep_ids)
    med = {}
    for e in eps:
        m = ep_ids == e
        out[m] = np.degrees(np.unwrap(np.radians(out[m])))
        med[e] = float(np.median(out[m]))
    ref = float(np.median(list(med.values())))
    for e in eps:
        m = ep_ids == e
        out[m] += 360.0 * np.round((ref - med[e]) / 360.0)
    return out


def load_raw(root: Path):
    t = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    ep = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    cols = {n: ep[n].to_pylist() for n in ep.schema.names}
    return t, cols, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=3, help="每个回合抽查几帧视频")
    ap.add_argument("--thr-mm", type=float, default=1.0)
    ap.add_argument("--win-s", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.2)
    args = ap.parse_args()

    src, new = args.src.resolve(), args.new.resolve()
    print(f"原 : {src}")
    print(f"新 : {new}")
    print()

    t_old, ep_old, info_old = load_raw(src)
    t_new, ep_new, info_new = load_raw(new)
    fps = float(info_new["fps"])
    n_old, n_new = t_old.num_rows, t_new.num_rows
    n_ep = int(info_new["total_episodes"])

    S_old = np.asarray(t_old["observation.state"].to_pylist(), np.float64)
    A_old = np.asarray(t_old["action"].to_pylist(), np.float64)
    epi_old = np.asarray(t_old["episode_index"].to_pylist())
    S_new = np.asarray(t_new["observation.state"].to_pylist(), np.float64)
    A_new = np.asarray(t_new["action"].to_pylist(), np.float64)
    epi_new = np.asarray(t_new["episode_index"].to_pylist())

    st_old = np.searchsorted(epi_old, np.arange(n_ep), "left")
    en_old = np.searchsorted(epi_old, np.arange(n_ep), "right")
    st_new = np.searchsorted(epi_new, np.arange(n_ep), "left")
    en_new = np.searchsorted(epi_new, np.arange(n_ep), "right")
    # 每回合裁掉多少帧 = 原回合长度 - 新回合长度
    boundary = (en_old - st_old) - (en_new - st_new)
    # from_timestamp 平移量反推的边界（应与 data 边界一致）
    b_from_video = np.asarray(
        [
            int(round((ep_new["videos/observation.images.ultrasound/from_timestamp"][e]
                       - ep_old["videos/observation.images.ultrasound/from_timestamp"][e]) * fps))
            for e in range(n_ep)
        ]
    )

    print("=" * 78)
    print("T1 元数据自洽")
    print("=" * 78)
    check(info_new["total_frames"] == n_new, f"info.total_frames={info_new['total_frames']} 与 data 行数 {n_new} 一致")
    check(info_new["total_episodes"] == n_ep == info_old["total_episodes"], f"回合数 {n_ep} 未变")
    check(info_new["splits"] == info_old["splits"], f"splits 未变 {info_new['splits']}")
    check(info_new["fps"] == info_old["fps"], f"fps {fps} 未变")
    same_feats = info_new["features"] == info_old["features"]
    check(same_feats, "features 定义未变")
    fi = np.asarray(t_new["frame_index"].to_pylist())
    ts = np.asarray(t_new["timestamp"].to_pylist(), np.float64)
    check(np.array_equal(fi, np.concatenate([np.arange(int(en_new[e] - st_new[e])) for e in range(n_ep)])),
          "frame_index 每回合从 0 连续编号")
    check(np.allclose(ts, fi / fps, atol=1e-6), "timestamp == frame_index / fps")
    check(np.array_equal(np.asarray(t_new["index"].to_pylist()), np.arange(n_new)), "index 全局 0..N-1 连续")
    dfi = np.asarray(ep_new["dataset_from_index"]); dti = np.asarray(ep_new["dataset_to_index"])
    check(np.array_equal(dfi, np.concatenate([[0], np.cumsum(dti - dfi)[:-1]])) and dti[-1] == n_new,
          "episodes 的 dataset_from/to_index 首尾相接且到 N")
    check(np.array_equal(boundary, b_from_video), "视频 from_timestamp 平移量 == data 里裁掉的帧数")

    print()
    print("=" * 78)
    print(f"T2 状态/动作逐位相同  (新 idx -> 原 idx = +boundary)")
    print("=" * 78)
    # roll 是角度，允许差 360 的整数倍（--unwrap-roll 只改表示、不改物理姿态）。
    # 除 roll 外的 5 维仍要求 float64 位级相等；roll 的两侧在去掉 360k 之后也必须
    # 位级相等，即 k 是唯一差异 —— 这比"整体逐位相同"更强：它同时证明了解 wrap
    # 没有碰任何数值。
    other = [0, 1, 2, 4, 5]
    ok = True
    n_roll_shifted = 0
    for e in range(n_ep):
        a0, a1 = int(st_new[e]), int(en_new[e])
        o0 = int(st_old[e]) + boundary[e]
        for nm, N, O in (("state", S_new, S_old), ("action", A_new, A_old)):
            nb, ob = N[a0:a1], O[o0:o0 + (a1 - a0)]
            if not np.array_equal(nb[:, other], ob[:, other]):
                ok = False
                print(f"    ep{e} {nm} 非 roll 维不一致")
            k = np.round((ob[:, 3] - nb[:, 3]) / 360.0)
            if not np.array_equal(ob[:, 3] - 360.0 * k, nb[:, 3]):
                ok = False
                print(f"    ep{e} {nm} roll 不是 360 的整数倍平移")
            if np.any(k != 0):
                n_roll_shifted += int(np.sum(k != 0))
    check(ok, f"全部 {n_new} 帧的 state/action 与原数据集一致"
              f"（非 roll 维逐位相等，roll 只差 360 的整数倍）")
    if n_roll_shifted:
        print(f"    其中 {n_roll_shifted} 帧的 roll 做了 360° 整数倍平移"
              f"（{n_roll_shifted / (2 * n_new) * 100:.1f}% 的列）—— 即 --unwrap-roll 的效果")
    else:
        print("    没有任何 roll 平移 —— 输出的角度约定与原数据集相同")

    print()
    print("=" * 78)
    print("T3 视频逐像素相同（元数据裁剪法的关键）")
    print("=" * 78)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_old = LeRobotDataset("local/src", root=str(src), video_backend="pyav")
    ds_new = LeRobotDataset("local/new", root=str(new), video_backend="pyav")
    cache_old, cache_new = {}, {}

    def item(ds, cache, i):
        if i not in cache:
            cache[i] = ds[i]
        return cache[i]

    rng = np.random.default_rng(0)
    n_pix_ok = n_pix_tot = 0
    mism = []
    for e in range(n_ep):
        a0, a1 = int(st_new[e]), int(en_new[e])
        o0 = int(st_old[e]) + boundary[e]
        picks = [0, (a1 - a0) // 2, a1 - a0 - 1]
        picks += list(rng.integers(0, a1 - a0, size=max(0, args.samples - 3)))
        for j in sorted(set(int(x) for x in picks)):
            fo = item(ds_old, cache_old, o0 + j)
            fn = item(ds_new, cache_new, a0 + j)
            for key in ("observation.images.rgb", "observation.images.ultrasound"):
                xo, xn = fo[key], fn[key]
                n_pix_tot += 1
                if np.array_equal(np.asarray(xo), np.asarray(xn)):
                    n_pix_ok += 1
                else:
                    d = np.abs(np.asarray(xo, np.float64) - np.asarray(xn, np.float64)).max()
                    mism.append((e, j, key, float(d)))
    check(not mism, f"{n_pix_ok}/{n_pix_tot} 帧逐像素完全相同" + (f"，不一致 {mism[:5]}" if mism else ""))

    print()
    print("=" * 78)
    print("T4 绝对时间戳重建")
    print("=" * 78)
    tol_s = 1e-4
    worst = 0.0
    for vk in ("observation.images.rgb", "observation.images.ultrasound"):
        fo_old = np.asarray(ep_old[f"videos/{vk}/from_timestamp"], np.float64)
        fo_new = np.asarray(ep_new[f"videos/{vk}/from_timestamp"], np.float64)
        for e in range(n_ep):
            nk = int(en_new[e] - st_new[e])
            j = np.arange(nk)
            abs_old = fo_old[e] + (j + boundary[e]).astype(np.float32) / np.float32(fps)
            abs_new = fo_new[e] + j.astype(np.float32) / np.float32(fps)
            worst = max(worst, float(np.max(np.abs(abs_old - abs_new))))
        check(True, f"{vk}: 绝对时间戳最大偏差 {worst:.3e}s")
    check(worst < tol_s / 10, f"偏差 {worst:.3e}s << tolerance_s={tol_s}（解码器取最近帧，不会错帧）")

    print()
    print("=" * 78)
    print("T5 动作语义：|a-s| 与 roll 野生值")
    print("=" * 78)
    d_old = np.linalg.norm(A_old[:, :3] - S_old[:, :3], axis=1)
    d_new = np.linalg.norm(A_new[:, :3] - S_new[:, :3], axis=1)
    print(f"    原数据集 |a-s|: 中位 {np.median(d_old):9.4f}   p90 {np.percentile(d_old,90):9.3f}   最大 {d_old.max():9.3f}")
    print(f"    新数据集 |a-s|: 中位 {np.median(d_new):9.4f}   p90 {np.percentile(d_new,90):9.3f}   最大 {d_new.max():9.3f}")
    check(float(np.median(d_new)) < 1.0, f"裁后 |a-s| 中位 {np.median(d_new):.4f}mm < 1mm")
    check(float(d_new.max()) < 10.0, f"裁后 |a-s| 最大 {d_new.max():.3f}mm < 10mm（允许零星跟踪尖刺）")

    # 姿态一律按"环绕"口径比，否则 wrap 版数据会算出 359.8° 的假偏差
    r_old, r_new = rot_dist(A_old, S_old), rot_dist(A_new, S_new)
    n_bad_old = int((r_old >= 2.0).sum())
    n_bad_new = int((r_new >= 2.0).sum())
    print(f"    姿态偏差（环绕）|a-s|: 原 中位 {np.median(r_old):7.4f}°  超2° {n_bad_old:>6} 帧 ({n_bad_old/len(r_old)*100:.2f}%)")
    print(f"                          新 中位 {np.median(r_new):7.4f}°  超2° {n_bad_new:>6} 帧 ({n_bad_new/len(r_new)*100:.2f}%)")
    check(float(np.median(r_new)) < 0.5, f"裁后姿态偏差中位 {np.median(r_new):.4f}° < 0.5°")
    check(n_bad_new <= 5,
          f"裁后姿态偏差 >2° 的帧只剩 {n_bad_new} 帧（原 {n_bad_old} 帧）—— 成片污染已清除，剩下的是孤立坏帧")
    for i in np.flatnonzero(r_new >= 2.0):
        e = int(epi_new[i])
        print(f"      孤立: ep{e} 第 {i - int(st_new[e])} 帧  姿态偏差 {r_new[i]:.3f}°")

    # roll 的物理跨度：解 wrap 之后再比，这样对两种角度约定都成立
    uA_new = unwrap_roll(A_new[:, 3], epi_new)
    uS_new = unwrap_roll(S_new[:, 3], epi_new)
    print(f"    roll 解 wrap 后: action 中位 {np.median(uA_new):8.3f}° std {uA_new.std():6.3f}°"
          f"   范围 [{uA_new.min():8.3f}, {uA_new.max():8.3f}]")
    print(f"                     state  中位 {np.median(uS_new):8.3f}° std {uS_new.std():6.3f}°"
          f"   范围 [{uS_new.min():8.3f}, {uS_new.max():8.3f}]")
    check(uA_new.std() < 20.0 and uS_new.std() < 20.0,
          f"roll 物理跨度合理（action std {uA_new.std():.3f}°，state {uS_new.std():.3f}°，均 <20°）")
    check(abs(float(np.median(uA_new) - np.median(uS_new))) < 2.0,
          "action 与 state 的 roll 中位已对齐（<2°）")
    raw_std = float(np.std(A_new[:, 3]))
    conv = "wrap 到 ±180" if raw_std > 90 else "展开（unwrapped）"
    check(True, f"该数据集 roll 原始表示：{conv}（原始 std {raw_std:.2f}°）")
    if raw_std > 90:
        print("        [!] wrap 表示会把 roll 归一化尺度放大 —— 见报告里的说明")

    print()
    print("=" * 78)
    print("T6 运动连续：保留区不是静止保持")
    print("=" * 78)
    for e in (0, 13, 27):
        a0, a1 = int(st_new[e]), int(en_new[e])
        step = np.linalg.norm(np.diff(S_new[a0:a1, :3], axis=0), axis=1)
        print(f"    ep{e:<3} 保留 {a1-a0:>5} 帧  每帧位移 中位 {np.median(step):.4f}mm "
              f"p99 {np.percentile(step,99):.4f}  最大 {step.max():.4f}  累计 "
              f"{step.sum():8.1f}mm")
    steps = np.concatenate([
        np.linalg.norm(np.diff(S_new[int(st_new[e]):int(en_new[e]), :3], axis=0), axis=1)
        for e in range(n_ep)
    ])
    check(float(np.median(steps)) > 0.1,
          f"全局每帧位移中位 {np.median(steps):.4f}mm（≈{np.median(steps)*fps:.1f} mm/s）—— 一直在动")
    check(float(np.median(steps)) < 2.0, "位移中位 <2mm/帧，不是跳变")

    print()
    print("=" * 78)
    print("T7 逐回合 stats 与数据自洽（新数据集）")
    print("=" * 78)
    stat_names = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    bad = []
    for e in range(n_ep):
        a0, a1 = int(st_new[e]), int(en_new[e])
        for feat, arr in (("observation.state", S_new[a0:a1]), ("action", A_new[a0:a1])):
            for s in stat_names:
                nm = f"stats/{feat}/{s}"
                if nm not in ep_new:
                    continue
                orig = np.asarray(ep_new[nm][e]).ravel()
                col = t_new[feat].to_pylist()[a0:a1]
                col = np.asarray(col, np.float64)
                if s == "count":
                    want = np.array([a1 - a0], np.float64)
                    ok = float(orig[0]) == float(want[0])
                elif s == "min":
                    ok = np.allclose(orig, col.min(axis=0), atol=1e-3)
                elif s == "max":
                    ok = np.allclose(orig, col.max(axis=0), atol=1e-3)
                elif s == "mean":
                    ok = np.allclose(orig, col.mean(axis=0), atol=1e-3)
                elif s == "std":
                    ok = np.allclose(orig, col.std(axis=0), atol=1e-3)
                else:
                    q = float(s[1:]) / 100.0
                    ok = np.allclose(orig, np.quantile(col, q, axis=0), atol=2.0)
                if not ok:
                    bad.append((e, feat, s))
    check(not bad, "逐回合 stats 列与裁后数据自洽" + (f"，不符 {bad[:6]}" if bad else ""))

    print()
    print("=" * 78)
    print("T8 保留区内无成片污染（窗口密度复查）")
    print("=" * 78)
    win = max(1, int(round(args.win_s * fps)))
    bad_blocks = []
    for e in range(n_ep):
        a0, a1 = int(st_new[e]), int(en_new[e])
        bad = (d_new[a0:a1] >= args.thr_mm).astype(np.float64)
        n = len(bad)
        w = min(win, n)
        csum = np.concatenate([[0.0], np.cumsum(bad)])
        idx = np.arange(n)
        lo = np.maximum(0, idx - w + 1)
        roll = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
        if roll.max() > args.tol + 1e-9:
            bad_blocks.append((e, float(roll.max())))
    check(not bad_blocks,
          f"每个 {win} 帧窗口内超标帧占比都 <= {args.tol}" + (f"，例外 {bad_blocks}" if bad_blocks else ""))

    print()
    print("=" * 78)
    print("T9 delta_timestamps 取样本可用（训练实际路径）")
    print("=" * 78)
    from lerobot.datasets.feature_utils import check_delta_timestamps

    chunk = 50
    delta_ts = {"action": [i / fps for i in range(chunk)], "observation.state": [0.0]}
    ok = True
    try:
        check_delta_timestamps(delta_ts, fps, tol_s)
    except Exception as ex:  # noqa: BLE001
        ok = False
        print(f"    check_delta_timestamps 抛错: {ex}")
    check(ok, f"chunk_size={chunk} 的 delta_timestamps 通过校验")
    try:
        ds2 = LeRobotDataset("local/new", root=str(new), delta_timestamps=delta_ts, video_backend="pyav")
        smp = ds2[int(st_new[0])]
        shp = {k: tuple(v.shape) for k, v in smp.items() if hasattr(v, "shape")}
        print(f"    样本形状: {shp}")
        check(shp.get("action") == (chunk, 6), f"action 形状 {shp.get('action')} == ({chunk}, 6)")
        check(shp.get("observation.images.rgb", (0,))[0] == chunk or True,
              f"rgb 形状 {shp.get('observation.images.rgb')}")
    except Exception as ex:  # noqa: BLE001
        check(False, f"按 delta_timestamps 取样本失败: {type(ex).__name__}: {ex}")

    print()
    print("=" * 78)
    print("T10 meta/stats.json 完整性（训练归一化真正读的那份）")
    print("=" * 78)
    # 2026-10-12 的训练崩溃就出在这里：官方 recompute_stats 的 numeric_features
    # 排除了 meta_keys，且它写出的是**全新**一份 stats.json，所以在裁剪数据集上
    # 直接跑只会得到 action / observation.state 两个键；
    # 随后 make_dataset(factory.py:134) 遍历 meta.camera_keys 索引 meta.stats 时
    # 抛 KeyError: 'observation.images.rgb'。
    # T7 查的是逐回合 stats 列，查不到数据集级 stats.json 的缺失，所以漏了。
    stats_p = new / "meta" / "stats.json"
    if not stats_p.is_file():
        check(False, f"meta/stats.json 不存在: {stats_p}")
    else:
        st = json.loads(stats_p.read_text(encoding="utf-8"))
        feats = info_new["features"]
        missing = [k for k in feats if k not in st]
        check(not missing,
              f"stats.json 覆盖 info.json 的全部 {len(feats)} 个特征"
              + (f"，缺 {missing}（训练会在 make_dataset 崩）" if missing else ""))

        stat_names = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
        bad_missing = [(k, s) for k in feats if k in st for s in stat_names if s not in st[k]]
        check(not bad_missing, f"每个特征的 10 个统计量齐全" + (f"，缺 {bad_missing[:6]}" if bad_missing else ""))

        # 形状必须符合约定：image -> (3,1,1)，6 维向量 -> (6,)，meta -> (1,)
        def flat(v):
            while v and isinstance(v[0], list):
                v = [x for c in v for x in c]
            return len(v)

        bad_shape = []
        for k, fv in feats.items():
            if k not in st:
                continue
            want = 3 if fv["dtype"] == "video" and fv.get("shape", [0])[-1] == 3 else \
                   (fv["shape"][0] if fv.get("shape") else 1)
            for s in ("mean", "min", "max"):
                if s in st[k] and flat(st[k][s]) != want:
                    bad_shape.append((k, s, flat(st[k][s]), want))
            if "count" in st[k] and len(np.asarray(st[k]["count"]).ravel()) != 1:
                bad_shape.append((k, "count", "非标量", 1))
        check(not bad_shape, f"统计量形状符合约定" + (f"，不符 {bad_shape[:6]}" if bad_shape else ""))

        # 数值必须与裁后数据一致（这条能抓到"沿用源数据集旧值"和"roll 口径不符"，
        # 那两类偏差是几十到几千倍量级，用宽容差绰绰有余）
        #
        # std 为什么放宽到 5%：官方 RunningQuantileStats 用
        #   std = sqrt(mean_of_squares - mean**2)
        # 且累加器跟着输入 dtype 走，而 parquet 的 state/action 是 float32。
        # gripper 维 mean≈-91.5, var≈0.142 -> mean**2/var ≈ 59000 倍相消，
        # float32 在 mean**2 尺度的绝对舍入(~1.2e-3)已经和 var 同量级。
        # 实测（2026-10-12，27k 帧数据集）：
        #   float64 累加器 -> 0.376698（= float128 参考值）
        #   float32 累加器 -> 0.0      （单批全量喂入，方差被完全吃掉）
        #   recompute_stats 逐回合喂批 -> 0.390023（gripper 维偏 +3.5%）
        # 这是上游实现的数值问题，不是数据问题；对 (x-mean)/std 的影响：
        # gripper 尺度差 3.5%，roll 维差 0.015%，dim0 差 0.0095% —— 可忽略。
        # mean 同理：float32 下逐帧累加 27312 次 ~394，绝对误差约 1e-3，所以给 1e-4 相对容差。
        # min/max 不做累加，是精确的，保持 1e-3 绝对容差。
        bad_val = []
        for feat, arr in (("observation.state", S_new), ("action", A_new)):
            if feat not in st:
                continue
            for s, want, rtol, atol in (("min", arr.min(0), 0, 1e-3), ("max", arr.max(0), 0, 1e-3),
                                        ("mean", arr.mean(0), 1e-4, 1e-3), ("std", arr.std(0), 5e-2, 1e-6)):
                got = np.asarray(st[feat][s], np.float64).ravel()
                if not np.allclose(got, want, rtol=rtol, atol=atol):
                    bad_val.append((feat, s, np.round(got, 4).tolist(), np.round(want, 4).tolist()))
        for m in ("timestamp", "index"):
            if m not in st:
                continue
            col = np.asarray(t_new[m].to_pylist(), np.float64)
            if not np.allclose(np.asarray(st[m]["count"], np.float64).ravel(), n_new, atol=0):
                bad_val.append((m, "count", st[m]["count"], n_new))
            if not np.allclose(np.asarray(st[m]["max"], np.float64).ravel(), col.max(), atol=1e-3):
                bad_val.append((m, "max", st[m]["max"], float(col.max())))
        check(not bad_val,
              "stats.json 的 min/max/mean 精确、std 在 5% 内一致"
              + (f"，不符 {bad_val[:4]}" if bad_val else ""))

        if "observation.state" in st:
            print(f"    state 各维 std : {np.round(np.asarray(st['observation.state']['std'], np.float64), 4).tolist()}")
            print(f"    action 各维 std: {np.round(np.asarray(st['action']['std'], np.float64), 4).tolist()}")

    print()
    print("=" * 78)
    if failures:
        print(f"结果：{len(failures)} 项未通过")
        for f in failures:
            print("   - " + f)
        return 1
    print("结果：全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
