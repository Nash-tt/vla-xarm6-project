#!/usr/bin/env python3
"""只保留扫描段：把数据集里"动作语义不正常"的接近段裁掉，生成一个新数据集。

问题
----
原数据集每个回合一开头都有一段"接近段"：机械臂朝一个几百毫米外的目标点慢速移动，
这段里 `action` 冻结在那个远端目标上、不跟随 state。ep0 第 300 帧：

    state y =   1.924 mm
    action y = -229.240 mm        |action - state| = 231 mm

全库统计能看出污染有多大 —— `action` 的 y 维中位数是 -156.9mm（落在冻结值区间），
roll 维最小值是 32.31°，而 state 的 roll 只在 [163.97, 193.04] 之间。

扫描段里 `action` 才是正常的"当前位置指令"（|a-s| 中位 ~0.17mm，略微领先实测姿态）。
裁掉接近段 = 不再教模型"输出一个几百毫米外的点"。

判据
----
每个回合独立定边界 b，取下面两个条件的**较晚者**：

  1. 95% 规则：第一个 i，使其后 >= keep_frac 比例的帧满足 |a[:3]-s[:3]| < thr_mm
  2. 解冻点：最后一个 |a[:3]-s[:3]| > unfreeze_mm 的位置 + 1
     （防 95% 规则漏掉尾巴上零星的大偏差）

保留 [b, n) 这些帧。

为什么可以不重编码视频
----------------------
视频是**共享 mp4**：多个回合共用同一个文件，靠每回合的
`videos/<key>/from_timestamp` 定位；data parquet 的 `timestamp` 每回合从 0 重新计。
读取端 `dataset_reader._query_videos()` 取的是：

    from_timestamp + timestamp

所以只要
  * data parquet 只留边界后的帧，`frame_index` 从 0 重编号，`timestamp = frame_index/fps`
  * episodes 里 `from_timestamp_new = from_timestamp_old + b/fps`

偏移后的绝对时间戳就和原数据集几乎逐位相同（差 ~1e-6 s，而 tolerance_s = 1e-4），
mp4 一个字节都不用动。

`to_timestamp` 不用改：它等于 `from_timestamp + length/fps`，
而 `from_old + (b + (n-b))/fps == from_old + n/fps == to_timestamp_old`。

顺带一提，官方 `recompute_stats(skip_image_video=False)` 其实**从不重算图像统计**
（`dataset_tools.py` 里只遍历 numeric_keys，源码有 TODO）。所以这里也原样保留
逐回合的 image/video 统计列，与官方行为一致；读取路径 `load_episodes()` 本来
就把 `stats/` 开头的列全部丢掉，训练不受影响。

用法
----
    python trim_scan_segments.py --src <原数据集> --dst <新数据集> --dry-run
    python trim_scan_segments.py --src <原数据集> --dst <新数据集>

原数据集全程只读。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# 官方统计函数，保证逐回合 stats 列的语义（RunningQuantileStats / DEFAULT_QUANTILES）一致
# lerobot 源码路径；用 pip 装的 lerobot 时这行是空操作，指到源码树可覆盖它
sys.path.insert(0, os.environ.get("LEROBOT_SRC", "/opt/lerobot/src"))
from lerobot.datasets.compute_stats import compute_episode_stats  # noqa: E402

NUMERIC_DTYPES = {"float32", "float64", "int64", "int32", "bool"}


def rot_dist(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """姿态偏差（取三个角里最大的那个），**按角度环绕**算。

    必须环绕：pi0 那版数据集的 roll 是展开的（163~193），smolvla 那版是 wrap 的
    （-180~180）。同一批动作在两版里 roll 差 360°，直接相减会得到 359.8° 的假偏差。
    换算到 [-180, 180) 之后，两版给出完全相同的偏差。
    """
    d = action[:, 3:6] - state[:, 3:6]
    d = (d + 180.0) % 360.0 - 180.0
    return np.abs(d).max(axis=1)


# --------------------------------------------------------------------------- #
# roll 角度约定（--unwrap-roll）
# --------------------------------------------------------------------------- #
def _circular_mean_deg(v: np.ndarray) -> float:
    """角度均值。普通算术均值对 wrap 数据没有意义（±180 两侧会被平均到 0 附近）。"""
    r = np.radians(v)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def unwrap_roll_inplace(
    cols: list[np.ndarray], epi: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> dict:
    """把 roll 列（第 3 维）解成连续且各回合落在同一支上的表示。

    为什么需要：同一个物理姿态 185°，展开表示写成 185，wrap 表示写成 -175。
    对回归型策略这是实打实的伤害 ——
      · 归一化：LeRobot 默认 MEAN_STD，除以 std。wrap 版 roll 的 std 是 172°，
        展开版只有 7.8°，同一个信号被压掉 22 倍；而且均值 23° 落在两簇中间的空档，
        是个数据里根本不存在的值。
      · 回合内跳变：物理上连续跨过 180° 的那一回合，相邻两帧的目标值差 359°。
    两版数据（pi0 / smolvla）本就是同一批轨迹，roll 只差 360 的整数倍，
    所以解 wrap 不改变任何物理含义，只是把它们对齐到同一支上。

    做法：① 每回合内部 np.unwrap 解连续；② 每回合整体平移 360 的整数倍，
    使其中位数最接近全局圆均值。这样结果与"哪一版"无关。
    """
    r_col = 3
    flat = np.concatenate([c[:, r_col] for c in cols])
    circ_ref = _circular_mean_deg(flat)
    info = {"circular_mean": circ_ref, "per_episode_shift": {}}
    for c in cols:
        for e in range(len(starts)):
            s0, s1 = int(starts[e]), int(ends[e])
            seg = c[s0:s1, r_col]
            u = np.degrees(np.unwrap(np.radians(seg)))
            shift = 360.0 * np.round((circ_ref - float(np.median(u))) / 360.0)
            c[s0:s1, r_col] = u + shift
            if shift:
                info["per_episode_shift"][e] = float(shift)
    return info


# --------------------------------------------------------------------------- #
# 边界
# --------------------------------------------------------------------------- #
def compute_boundary(
    state: np.ndarray,
    action: np.ndarray,
    thr_mm: float,
    thr_deg: float,
    win: int,
    tol: float,
    unfreeze_mm: float,
    max_trim_frac: float,
) -> tuple[int, dict]:
    """返回 (boundary, 诊断信息)。boundary 之前的帧被丢掉。

    主判据是**窗口密度**：看污染是否"成片"，而不是单帧。

        d_pos[t] = ||a[t,:3] - s[t,:3]||                    位置偏差，mm
        d_rot[t] = max_i |a[t,3+i] - s[t,3+i]|              姿态偏差，deg
        bad[t]   = (d_pos[t] >= thr_mm) or (d_rot[t] >= thr_deg)
        roll[t]  = bad 在窗口 [t-win+1, t] 里的比例
        T        = 最后一个 roll[t] > tol 的 t
        boundary = T + 1

    姿态维必须一起看：接近段冻结的是**整个末端位姿**，有时位置差得不远而姿态差很多。
    但姿态的孤立偏差和位置的孤立偏差性质不同 —— 位置上有 1~3mm 的过冲是正常跟踪误差，
    姿态上差 100 多度只可能是"命令还指着另一个目标"。所以：

      * 密度判据（本函数）负责切成片的污染块
      * **孤立帧**不在这里切（切了会把整段扫描陪葬），而是由 find_anomalies() 单独列出来，
        写进 trim_report.json，让人决定怎么处理

    再和另外两个更粗的判据取较晚者，作为兜底：
      * 95% 规则：第一个 i，使其后 >= 1-tol 比例的帧满足 bad 为假
      * 解冻点：最后一个 d_pos > unfreeze_mm 的帧 + 1
    """
    d_pos = np.linalg.norm(action[:, :3] - state[:, :3], axis=1)  # (n,) mm
    d_rot = rot_dist(action, state)  # (n,) deg，已按角度环绕
    n = len(d_pos)
    if n == 0:
        raise ValueError("空回合")

    bad = ((d_pos >= thr_mm) | (d_rot >= thr_deg)).astype(np.float64)

    # 滚动比例：roll[t] = mean(bad[t-win+1 .. t])
    w = max(1, min(int(win), n))
    csum = np.concatenate([[0.0], np.cumsum(bad)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - w + 1)
    roll = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)

    hits = np.flatnonzero(roll > tol)
    dense = int(hits[-1]) + 1 if hits.size else 0

    # 兜底 1：95% 规则
    ok = ~(bad.astype(bool))
    cok = np.cumsum(ok[::-1])[::-1].astype(np.float64)
    suffix_frac = cok / np.arange(n, 0, -1)
    h95 = np.flatnonzero(suffix_frac >= (1.0 - tol))
    rule95 = int(h95[0]) if h95.size else n

    # 兜底 2：解冻点
    frozen = np.flatnonzero(d_pos > unfreeze_mm)
    unfreeze = int(frozen[-1]) + 1 if frozen.size else 0

    b = max(dense, rule95, unfreeze, 0)
    cap = max(1, int(n * max_trim_frac))
    b = min(b, cap, n - 1)  # 至少留 1 帧，且最多裁掉 max_trim_frac
    b = max(0, b)

    diag = {
        "n": n,
        "dense": dense,
        "rule95": rule95,
        "unfreeze": unfreeze,
        "biting": "dense" if b == dense else ("rule95" if b == rule95 else "unfreeze"),
        "d_median_head": float(np.median(d_pos[: max(1, n // 10)])),
        "d_median_keep": float(np.median(d_pos[b:])) if b < n else float("nan"),
        "frac_bad_keep": float(np.mean(bad[b:] > 0)) if b < n else 0.0,
        "roll_max_keep": float(np.nanmax(roll[b:])) if b < n else 0.0,
        "d_pos_max_keep": float(d_pos[b:].max()) if b < n else 0.0,
        "d_rot_max_keep": float(d_rot[b:].max()) if b < n else 0.0,
    }
    return b, diag


def find_anomalies(
    state: np.ndarray,
    action: np.ndarray,
    start: int,
    hard_mm: float,
    hard_deg: float,
    max_list: int = 50,
) -> list[dict]:
    """列出保留区里**孤立**的异常帧（位置或姿态偏差离谱的单帧）。

    这些帧密度判据故意不切（切了要陪葬整段），单独报出来供人判断。
    """
    d_pos = np.linalg.norm(action[:, :3] - state[:, :3], axis=1)
    d_rot = rot_dist(action, state)
    sel = np.flatnonzero((d_pos >= hard_mm) | (d_rot >= hard_deg))
    out = []
    for k in sel[:max_list]:
        out.append(
            {
                "frame_index_new": int(start + k),
                "frame_index_in_episode": int(k),
                "state": [round(float(v), 4) for v in state[k]],
                "action": [round(float(v), 4) for v in action[k]],
                "d_pos_mm": round(float(d_pos[k]), 4),
                "d_rot_deg": round(float(d_rot[k]), 4),
            }
        )
    if sel.size > max_list:
        out.append({"note": f"还有 {int(sel.size) - max_list} 帧未列出"})
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="裁掉接近段、只保留扫描段，生成新数据集（元数据裁剪，不重编码视频）"
    )
    ap.add_argument("--src", required=True, type=Path, help="原数据集根目录（只读）")
    ap.add_argument("--dst", required=True, type=Path, help="新数据集根目录")
    ap.add_argument("--thr-mm", type=float, default=1.0, help="位置偏差判定阈值，mm（默认 1.0）")
    ap.add_argument(
        "--thr-deg", type=float, default=1.0, help="姿态偏差判定阈值，度（默认 1.0）"
    )
    ap.add_argument(
        "--hard-mm",
        type=float,
        default=2.0,
        help="孤立异常帧的位置报警线，mm（默认 2.0；只报告不裁剪）",
    )
    ap.add_argument(
        "--hard-deg",
        type=float,
        default=2.0,
        help="孤立异常帧的姿态报警线，度（默认 2.0；只报告不裁剪）",
    )
    ap.add_argument(
        "--win-s", type=float, default=1.0, help="窗口密度判据的窗长，秒（默认 1.0）"
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.2,
        help="容忍的污染密度：任意窗内超标帧占比 <= tol 就算干净（默认 0.2）",
    )
    ap.add_argument(
        "--unfreeze-mm",
        type=float,
        default=50.0,
        help="解冻点阈值：最后一个 |a-s| 超过此值的位置之后才算干净（默认 50.0）",
    )
    ap.add_argument(
        "--trim-rot-anomalies",
        action="store_true",
        help=(
            "把'孤立但姿态偏差离谱'的帧也切掉：边界推到最后一个 |姿态偏差| >= hard_deg 的帧之后。"
            "代价是该回合尾部若干帧陪葬（孤立帧不能单独删，删了视频时间戳就对不上了）。"
            "默认关闭，只报告。"
        ),
    )
    ap.add_argument(
        "--max-trim-frac",
        type=float,
        default=0.85,
        help="单回合最多裁掉的比例，防止判据跑飞（默认 0.85）",
    )
    ap.add_argument(
        "--unwrap-roll",
        action="store_true",
        help=(
            "把每回合的 roll 解成连续、且各回合落在同一支上的表示（展开到同一 360 分支）。"
            "源数据 roll 若是 wrap 到 ±180 的，归一化尺度会被放大 ~22 倍，且有回合会在中间跳 360°；"
            "打开本项后输出的 state/action 的 roll 与已展开的那版数据集口径一致。"
            "边界判据用环绕口径，所以本项不影响裁剪结果。"
        ),
    )
    ap.add_argument("--link-videos", action="store_true", help="视频用硬链接而不是复制（省空间）")
    ap.add_argument("--dry-run", action="store_true", help="只打印边界，不写任何文件")
    ap.add_argument("--overwrite", action="store_true", help="目标已存在时先删掉（谨慎）")
    args = ap.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    if not (src / "meta" / "info.json").is_file():
        print(f"[err] {src} 不像 LeRobot 数据集（没有 meta/info.json）", file=sys.stderr)
        return 2
    if dst == src:
        print("[err] --dst 不能和 --src 相同", file=sys.stderr)
        return 2
    if dst.exists() and not args.dry_run:
        if not args.overwrite:
            print(f"[err] {dst} 已存在；加 --overwrite 会先删除它", file=sys.stderr)
            return 2
        print(f"[warn] 删除已存在的 {dst}")
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    features = info["features"]

    print(f"源数据集 : {src}")
    print(f"  fps={fps}  total_episodes={info['total_episodes']}  total_frames={info['total_frames']}")

    # ---------------- 读 data ----------------
    data_files = sorted((src / "data").glob("*/*.parquet"))
    if len(data_files) != 1:
        print(f"[warn] data 下有 {len(data_files)} 个 parquet，本脚本按单文件处理")
    data_path = data_files[0]
    tbl = pq.read_table(data_path)
    n_all = tbl.num_rows
    S = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float64)
    A = np.asarray(tbl["action"].to_pylist(), dtype=np.float64)
    epi = np.asarray(tbl["episode_index"].to_pylist(), dtype=np.int64)
    print(f"data     : {data_path.relative_to(src)}  rows={n_all}")

    n_ep = int(info["total_episodes"])
    if epi.max() + 1 != n_ep:
        print(f"[warn] episode_index 最大值 {epi.max()} + 1 != total_episodes {n_ep}")

    # ---------------- 算边界 ----------------
    print()
    print("=" * 100)
    print(
        f"定边界（每回合取 窗口密度 / 95%规则 / 解冻点 的较晚者；"
        f"thr={args.thr_mm}mm  win={args.win_s}s  tol={args.tol}  unfreeze={args.unfreeze_mm}mm）"
    )
    print("=" * 100)
    win_frames = max(1, int(round(args.win_s * fps)))
    print(
        f"{'ep':>3} {'总帧':>6} {'窗口密度':>8} {'95%规则':>8} {'解冻点':>7} {'边界':>6} {'保留':>6} "
        f"{'保留%':>7} {'谁生效':>8} {'裁后|a-s|中位':>13} {'超标%':>7} {'窗内峰值':>9}"
    )

    starts = np.searchsorted(epi, np.arange(n_ep), side="left")
    ends = np.searchsorted(epi, np.arange(n_ep), side="right")

    boundaries: list[int] = []
    diags: list[dict] = []
    anomalies: dict[int, list[dict]] = {}
    for e in range(n_ep):
        s0, s1 = int(starts[e]), int(ends[e])
        b, diag = compute_boundary(
            S[s0:s1],
            A[s0:s1],
            args.thr_mm,
            args.thr_deg,
            win_frames,
            args.tol,
            args.unfreeze_mm,
            args.max_trim_frac,
        )
        boundaries.append(b)
        diags.append(diag)
        anomalies[e] = find_anomalies(
            S[s0 + b : s1], A[s0 + b : s1], 0, args.hard_mm, args.hard_deg
        )
        if args.trim_rot_anomalies and anomalies[e]:
            rot_bad = [
                a for a in anomalies[e]
                if "note" not in a and a["d_rot_deg"] >= args.hard_deg
            ]
            if rot_bad:
                last = max(a["frame_index_in_episode"] for a in rot_bad)
                new_b = min(b + last + 1, diag["n"] - 1)
                if new_b > b:
                    print(
                        f"      (--trim-rot-anomalies) ep{e}: 边界 {b} -> {new_b}"
                        f"，再弃 {new_b - b} 帧以切掉姿态异常"
                    )
                    b = new_b
                    boundaries[-1] = b
                    diags[-1]["trimmed_anomalies_to"] = b
                    anomalies[e] = find_anomalies(
                        S[s0 + b : s1], A[s0 + b : s1], 0, args.hard_mm, args.hard_deg
                    )
        kept = diag["n"] - b
        print(
            f"{e:>3} {diag['n']:>6} {diag['dense']:>8} {diag['rule95']:>8} {diag['unfreeze']:>7} "
            f"{b:>6} {kept:>6} {100.0 * kept / diag['n']:>6.1f}% {diag['biting']:>8} "
            f"{diag['d_median_keep']:>13.4f} {100.0 * diag['frac_bad_keep']:>6.2f}% "
            f"{diag['roll_max_keep']:>9.3f}"
        )

    keep_mask = np.zeros(n_all, dtype=bool)
    for e in range(n_ep):
        s0, s1 = int(starts[e]), int(ends[e])
        keep_mask[s0 + boundaries[e] : s1] = True

    n_new = int(keep_mask.sum())
    n_old = n_all
    print()
    print(
        f"合计: {n_old} -> {n_new} 帧 ({100.0 * n_new / n_old:.1f}%)   "
        f"边界: 中位 {int(np.median(boundaries))}  最小 {min(boundaries)}  最大 {max(boundaries)}"
    )
    kept_per_ep = np.array([diags[e]["n"] - boundaries[e] for e in range(n_ep)])
    print(f"      每回合保留: 中位 {int(np.median(kept_per_ep))}  最少 {kept_per_ep.min()}  最多 {kept_per_ep.max()}")
    kept_d = np.concatenate(
        [np.linalg.norm(A[int(starts[e]) + boundaries[e] : int(ends[e]), :3]
                        - S[int(starts[e]) + boundaries[e] : int(ends[e]), :3], axis=1)
         for e in range(n_ep)]
    )
    print(
        f"      裁后 |a-s|: 中位 {np.median(kept_d):.4f}mm  p99 {np.percentile(kept_d, 99):.4f}mm  "
        f"最大 {kept_d.max():.4f}mm   超 {args.thr_mm}mm 占比 {100.0 * np.mean(kept_d >= args.thr_mm):.2f}%"
    )
    print(
        "      生效判据分布: "
        + "  ".join(
            f"{k}={sum(1 for x in diags if x['biting'] == k)}"
            for k in ("dense", "rule95", "unfreeze")
        )
    )

    # ---------------- 可选：统一 roll 的角度约定 ----------------
    unwrap_info = None
    if args.unwrap_roll:
        S = S.copy()
        A = A.copy()
        roll_before = np.concatenate([S[:, 3], A[:, 3]])
        unwrap_info = unwrap_roll_inplace([S, A], epi, starts, ends)
        print()
        print(f"roll 角度约定：解 wrap（全局圆均值 {unwrap_info['circular_mean']:.3f}°）")
        print(
            f"  state roll std {np.std(roll_before[:len(S)]):7.3f}° -> {S[:, 3].std():7.3f}°"
            f"   范围 [{S[:, 3].min():.3f}, {S[:, 3].max():.3f}]"
        )
        print(
            f"  action roll std {np.std(roll_before[len(S):]):7.3f}° -> {A[:, 3].std():7.3f}°"
            f"   范围 [{A[:, 3].min():.3f}, {A[:, 3].max():.3f}]"
        )
        shifts = unwrap_info["per_episode_shift"]
        if shifts:
            print(f"  平移了 360° 整数倍的回合: {shifts}")
        else:
            print("  各回合本就在同一支上，无需平移")

    n_anom = sum(len(v) for v in anomalies.values())
    print()
    print(f"孤立异常帧（|位置偏差| >= {args.hard_mm}mm 或 |姿态偏差| >= {args.hard_deg}°）：共 {n_anom} 帧")
    print("  密度判据按设计不切这种孤立帧（切了会把整段扫描陪葬）。它们只被报告，不被裁剪。")
    if n_anom:
        for e, lst in anomalies.items():
            for a in lst:
                if "note" in a:
                    continue
                print(
                    f"    ep{e:<3} 帧{a['frame_index_in_episode']:>5}  "
                    f"位置偏差 {a['d_pos_mm']:8.4f}mm  姿态偏差 {a['d_rot_deg']:8.4f}°"
                )
                if a["d_rot_deg"] >= 10.0:
                    print(f"          state ={a['state']}")
                    print(f"          action={a['action']}")
    else:
        print("    （无）")

    if args.dry_run:
        print("\n[dry-run] 没有写任何文件。")
        return 0

    # ---------------- 写新数据集 ----------------
    print()
    print(f"写入 {dst} ...")
    t0 = time.time()
    for sub in ("data/chunk-000", "meta/episodes/chunk-000", "videos"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    # 1) data parquet
    keep_tbl = tbl.filter(pa.array(keep_mask))
    cum = np.concatenate([[0], np.cumsum(kept_per_ep)]).astype(np.int64)
    ts_new = np.zeros(n_new, dtype=np.float64)
    fi_new = np.zeros(n_new, dtype=np.int64)
    ep_new = np.zeros(n_new, dtype=np.int64)
    for e in range(n_ep):
        nk = int(kept_per_ep[e])
        off = int(cum[e])
        ts_new[off : off + nk] = np.arange(nk) / fps
        fi_new[off : off + nk] = np.arange(nk)
        ep_new[off : off + nk] = e
    idx_new = np.arange(n_new, dtype=np.int64)

    names = keep_tbl.schema.names
    out = keep_tbl
    out = out.set_column(names.index("timestamp"), "timestamp", pa.array(ts_new.astype(np.float32), pa.float32()))
    out = out.set_column(names.index("frame_index"), "frame_index", pa.array(fi_new, pa.int64()))
    out = out.set_column(names.index("episode_index"), "episode_index", pa.array(ep_new, pa.int64()))
    out = out.set_column(names.index("index"), "index", pa.array(idx_new, pa.int64()))

    # 解 wrap 后的 state/action（若启用）—— 必须写进 data，否则统计与数据对不上
    roll_src: dict[str, np.ndarray] = {}
    if args.unwrap_roll:
        for f, full in (("observation.state", S), ("action", A)):
            keep_vals = full[keep_mask]
            orig_ty = keep_tbl.schema.field(f).type
            out = out.set_column(names.index(f), f, pa.array(keep_vals.tolist(), orig_ty))
            roll_src[f] = full

    pq.write_table(out, dst / "data/chunk-000/file-000.parquet", compression="snappy")
    print(f"  data/chunk-000/file-000.parquet   {n_new} 行")

    # 2) episodes parquet
    ep_tbl = pq.read_table(src / "meta/episodes/chunk-000/file-000.parquet")
    ep_names = ep_tbl.schema.names
    cols = {n: ep_tbl[n].to_pylist() for n in ep_names}

    new_lengths = kept_per_ep.astype(np.int64)
    video_keys = [k for k in features if features[k]["dtype"] == "video"]
    numeric_feats = {k: v for k, v in features.items() if v["dtype"] in NUMERIC_DTYPES}

    # 逐回合重算 numeric 统计
    stat_names = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    stat_cols = {(f, s): [None] * n_ep for f in numeric_feats for s in stat_names}

    for e in range(n_ep):
        s0 = int(starts[e]) + boundaries[e]
        s1 = int(ends[e])
        epi_slice = slice(s0, s1)
        ep_data: dict[str, np.ndarray] = {}
        for f in numeric_feats:
            if f in roll_src:  # 已解 wrap：统计必须跟着走新值
                arr = np.asarray(roll_src[f][s0:s1])
            else:
                vals = tbl[f].to_pylist()[epi_slice]
                arr = np.asarray(vals)
            if arr.ndim > 1:
                ep_data[f] = arr.astype(np.float64)
            else:
                ep_data[f] = arr.astype(np.float64)
        stats = compute_episode_stats(ep_data, numeric_feats)
        for f, sd in stats.items():
            for s, val in sd.items():
                stat_cols[(f, s)][e] = np.asarray(val).ravel()

    new_cols: dict[str, pa.Array] = {}
    new_cols["length"] = pa.array(new_lengths, pa.int64())
    new_cols["dataset_from_index"] = pa.array(cum[:-1].astype(np.int64), pa.int64())
    new_cols["dataset_to_index"] = pa.array(cum[1:].astype(np.int64), pa.int64())

    for vk in video_keys:
        old_from = np.asarray(cols[f"videos/{vk}/from_timestamp"], dtype=np.float64)
        new_from = old_from + np.asarray(boundaries, dtype=np.float64) / fps
        new_cols[f"videos/{vk}/from_timestamp"] = pa.array(new_from.tolist(), pa.float64())
        # to_timestamp 保持不变（= from_old + n_old/fps）

    for (f, s), vals in stat_cols.items():
        colname = f"stats/{f}/{s}"
        if colname not in ep_names:
            print(f"  [warn] 原 episodes 里没有列 {colname}，跳过")
            continue
        want = ep_tbl.schema.field(colname).type
        # 整数特征的 min/max/分位数在 parquet 里是 list<int64>，浮点特征是 list<double>，
        # 逐列取元素类型再转，避免 pyarrow 拒绝写入。
        elem = want.value_type.to_pandas_dtype()
        new_cols[colname] = pa.array(
            [np.asarray(v).ravel().astype(elem, copy=False) for v in vals], want
        )

    out_ep = ep_tbl
    for cn, arr in new_cols.items():
        out_ep = out_ep.set_column(ep_names.index(cn), cn, arr)
    out_ep = out_ep.cast(ep_tbl.schema)  # 保证类型/元数据与原表一致
    pq.write_table(out_ep, dst / "meta/episodes/chunk-000/file-000.parquet", compression="snappy")
    print(f"  meta/episodes/chunk-000/file-000.parquet   {n_ep} 行")

    # 3) info.json —— 只改 total_frames，用文本替换保住原有 CRLF/缩进
    raw = (src / "meta" / "info.json").read_text(encoding="utf-8")
    new_raw, nsub = re.subn(
        r'("total_frames"\s*:\s*)\d+', r"\g<1>" + str(n_new), raw, count=1
    )
    if nsub != 1:
        raise RuntimeError("info.json 里没找到 total_frames，格式可能变了")
    (dst / "meta" / "info.json").write_text(new_raw, encoding="utf-8")
    print(f"  meta/info.json                     total_frames {info['total_frames']} -> {n_new}")

    # 4) tasks.parquet 原样复制
    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    print("  meta/tasks.parquet                 原样复制")

    # 5) videos
    nvid = 0
    for vf in sorted((src / "videos").rglob("*")):
        if not vf.is_file():
            continue
        rel = vf.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.link_videos:
            import os

            os.link(vf, target)
        else:
            shutil.copy2(vf, target)
        nvid += 1
    mode = "硬链接" if args.link_videos else "复制"
    print(f"  videos/                            {nvid} 个文件（{mode}）")

    # 6) 报告
    report = {
        "src": str(src),
        "dst": str(dst),
        "fps": fps,
        "rule": {
            "thr_mm": args.thr_mm,
            "thr_deg": args.thr_deg,
            "win_s": args.win_s,
            "win_frames": win_frames,
            "tol": args.tol,
            "unfreeze_mm": args.unfreeze_mm,
            "max_trim_frac": args.max_trim_frac,
            "hard_mm": args.hard_mm,
            "hard_deg": args.hard_deg,
            "definition": (
                "bad[t] = (||a[t,:3]-s[t,:3]|| >= thr_mm) or (max_i |a[t,3+i]-s[t,3+i]| >= thr_deg); "
                "boundary = max("
                "最后一个『win 帧窗口内 bad 占比 > tol』的帧 + 1, "
                "第一个使其后 >=1-tol 比例的帧 bad 为假的帧, "
                "最后一个 ||a[:3]-s[:3]|| > unfreeze_mm 的帧 + 1)"
            ),
        },
        "frames": {"old": n_old, "new": n_new, "kept_frac": n_new / n_old},
        "isolated_anomalies": {
            "threshold_mm": args.hard_mm,
            "threshold_deg": args.hard_deg,
            "total": n_anom,
            "note": (
                "密度判据故意不切这些孤立帧（切掉会把整段扫描陪葬）。"
                "它们留在数据集里，影响约为 n_anom/总帧数。"
            ),
            "per_episode": {str(e): lst for e, lst in anomalies.items() if lst},
        },
        "episodes": [
            {
                "episode_index": e,
                "n_old": diags[e]["n"],
                "boundary": boundaries[e],
                "n_new": int(kept_per_ep[e]),
                "dense": diags[e]["dense"],
                "rule95": diags[e]["rule95"],
                "unfreeze": diags[e]["unfreeze"],
                "biting": diags[e]["biting"],
                "d_median_keep_mm": diags[e]["d_median_keep"],
                "frac_bad_keep": diags[e]["frac_bad_keep"],
                "roll_max_keep": diags[e]["roll_max_keep"],
                "d_pos_max_keep_mm": diags[e]["d_pos_max_keep"],
                "d_rot_max_keep_deg": diags[e]["d_rot_max_keep"],
                "isolated_anomalies": len(anomalies[e]),
            }
            for e in range(n_ep)
        ],
        "notes": [
            "视频未重编码；靠每回合 from_timestamp 平移定位，绝对时间戳与原数据集一致。",
            "逐回合 image/video 统计列保留原值（与官方 recompute_stats 行为一致，读取路径也不消费）。",
            "逐回合 numeric 统计列（state/action/timestamp/frame_index/episode_index/index/task_index）已重算。",
            "数据集级 meta/stats.json 需另跑官方 recompute_stats 重算。",
        ]
        + (
            [
                "roll 已解 wrap（--unwrap-roll）：每回合内部解连续，再整体平移 360 的整数倍对齐同一支；"
                "物理含义不变，只是与展开版数据集口径一致。逐回合统计列已按新值重算。"
            ]
            if args.unwrap_roll
            else [
                "roll 保持源数据集原有角度约定（未加 --unwrap-roll）。"
                "若源数据 roll 是 wrap 到 ±180 的，MEAN_STD 归一化尺度会被放大，见 --unwrap-roll 说明。"
            ]
        ),
    }
    if unwrap_info is not None:
        report["roll_unwrap"] = unwrap_info
    (dst / "trim_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("  trim_report.json                   边界与统计报告")

    print(f"\n完成，用时 {time.time() - t0:.1f}s")
    print("下一步：重算 meta/stats.json（官方 recompute_stats），然后回读验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
