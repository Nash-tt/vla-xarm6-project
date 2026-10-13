#!/usr/bin/env python
"""把数据集里的一小段导成一个自包含的 `.npz` 回放包，供**另一台主机**发过来做远程推理测试。

为什么不直接把 1.8G 数据集搬过去：对面只要 `grpcio + numpy`，不需要 LeRobot、不需要
数据集、不需要解码视频。这个包就是那段回放的"录像带"。

── 关键设计：图像预先缩到 224，且是**逐位等价**的 ──────────────────────────
包里的图不是原图，是已经过 `resize_robot_observation_image()` 缩到 224×224 的结果
（就是服务端自己会调的那个函数，这里直接 import 复用它）。

为什么这不算"降质"：服务端收到图之后还会再缩一次到 policy 的 224×224。而 224→224
的 bilinear（align_corners=False, scale=1）映射到原像素中心、权重恰为 1，实测是
**精确恒等**（见下）。所以：

    全分辨率路径：uint8 720×1280 --resize--> uint8 224×224 --/255--> 模型
    本包路径    ：uint8 224×224  --resize(恒等)--> uint8 224×224 --/255--> 模型

两条路给模型的张量逐位相同，MAE 可与本地 replay_client.py 的结果直接对比。
而带宽：2.96MB/帧 → 0.30MB/帧，降 10 倍。

想更省可以用 `--img-size 128`，但那时服务端要 128→224 上采样，与训练时的
720→224 不再是同一个算子，MAE 会偏，只能当连通性测试看。

用法：
    python make_replay_bundle.py --episode 0 --out replay_bundle.npz
    # 然后 scp replay_bundle.npz + remote/ 到另一台主机
"""

import argparse
import glob
import os
import json
import time

for _v in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from lerobot.async_inference.helpers import resize_robot_observation_image  # noqa: E402

ROOT = "/data/vla/vla_batch_20261007_pi0_6d_rollunwrap"
REPO = "local/vla_batch_20261007_pi0_6d_rollunwrap"

# 与 replay_client.py / 训练时逐位一致的维序
DIMS = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]

# 数据集内的名字 -> 客户端要用的相机名。
# 注意方向：数据集叫 rgb / ultrasound，但**发出去的键必须是 policy 的键**
# base_0_rgb / left_wrist_0_rgb（helpers.py:166 直接索引，rename_map 救不了）。
DS_CAM_BASE, CAM_BASE = "observation.images.rgb", "base_0_rgb"
DS_CAM_WRIST, CAM_WRIST = "observation.images.ultrasound", "left_wrist_0_rgb"


def to_hwc_uint8_resized(img_chw: torch.Tensor, size: int) -> np.ndarray:
    """(3,H,W) float[0,1] -> (size,size,3) uint8，走的正是服务端那条 resize。

    先按 replay_client 的做法量化成 uint8 HWC（真机相机的格式），再交给
    lerobot 的 resize 函数 —— 不自己写 interpolate，免得和它有任何偏差。
    """
    hwc = img_chw.permute(1, 2, 0).numpy()
    u8 = np.ascontiguousarray((hwc * 255.0).round().clip(0, 255).astype(np.uint8))
    resized = resize_robot_observation_image(torch.from_numpy(u8), (3, size, size))
    return resized.permute(1, 2, 0).contiguous().numpy()


def pick_moving_frame(A: np.ndarray, n_frames: int) -> int:
    """挑一帧"在动"的起点，避开起停段 —— 与 measure_inference/replay_client 同策略。"""
    spd = np.linalg.norm(np.diff(A, axis=0), axis=1)
    moving = spd > np.percentile(spd[spd > 0], 40)
    room = n_frames + 1
    ok = np.where(moving & (np.arange(len(spd)) + room < len(A)))[0]
    if len(ok) == 0:
        raise SystemExit("这条 episode 里找不到「在动且后面还剩够帧数」的起点")
    return int(ok[len(ok) // 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--repo-id", default=REPO)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=-1, help="-1 = 自动挑一帧正在运动的")
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--img-size", type=int, default=224,
                    help="存进包里的图像边长。224=与服务端逐位等价（推荐）；调小只省带宽，会引入偏差")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="replay_bundle.npz")
    args = ap.parse_args()

    print("=" * 74)
    print("导出回放包")
    print("=" * 74)

    print("\n【1】读数据集")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    ep = ds.meta.episodes[args.episode]
    ep_from = int(ep["dataset_from_index"])
    ep_len = int(ep["length"])
    task = str(ep["tasks"][0])
    print(f"  episode {args.episode}: {ep_len} 帧, 全局起点 {ep_from}")
    print(f"  任务串: {task!r}")

    print("\n【2】读 GT action（parquet，很便宜）")
    files = sorted(glob.glob(f"{args.root}/data/*/*.parquet"))
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    sub = df[df.episode_index == args.episode].sort_values("frame_index")
    A = np.stack(sub["action"].values).astype(np.float64)

    if args.start_frame < 0:
        start = pick_moving_frame(A, args.n_frames)
        print(f"  自动挑起点: episode 内第 {start} 帧")
    else:
        start = args.start_frame
        if start + args.n_frames > ep_len:
            raise SystemExit(f"起点 {start} + {args.n_frames} 帧 超出了 {ep_len} 帧")

    seg = A[start : start + args.n_frames]
    roll = seg[:, 3]
    print(f"  回放区间: episode 内 {start} -> {start + args.n_frames}"
          f"  ({args.n_frames} 帧, {args.n_frames / args.fps:.1f}s @{args.fps}fps)")
    print(f"  roll 区间 [{roll.min():.2f}, {roll.max():.2f}] "
          f"{'OK（未跨 ±180，解卷绕正确）' if (roll.min() > 90 or roll.max() < -90) else '!! 跨了 ±180'}")
    print(f"  x 区间 [{seg[:, 0].min():.2f}, {seg[:, 0].max():.2f}] mm")

    print(f"\n【3】解码 {args.n_frames} 帧图像并缩到 {args.img_size}×{args.img_size}")
    t0 = time.perf_counter()
    rgb = np.empty((args.n_frames, args.img_size, args.img_size, 3), dtype=np.uint8)
    us = np.empty((args.n_frames, args.img_size, args.img_size, 3), dtype=np.uint8)
    states = np.empty((args.n_frames, 6), dtype=np.float32)
    for k in range(args.n_frames):
        fr = ds[ep_from + start + k]
        states[k] = fr["observation.state"].numpy().astype(np.float32)
        rgb[k] = to_hwc_uint8_resized(fr[DS_CAM_BASE], args.img_size)
        us[k] = to_hwc_uint8_resized(fr[DS_CAM_WRIST], args.img_size)
        if (k + 1) % 50 == 0:
            print(f"    {k + 1}/{args.n_frames}  ({time.perf_counter() - t0:.0f}s)")
    print(f"  解码完成，耗时 {time.perf_counter() - t0:.1f}s")

    meta = {
        "episode": args.episode,
        "start_frame": start,
        "n_frames": args.n_frames,
        "img_size": args.img_size,
        "fps": args.fps,
        "dims": DIMS,
        "cam_base": CAM_BASE,
        "cam_wrist": CAM_WRIST,
        "task": task,
        "ckpt": args.ckpt,
        "source_root": args.root,
        "made_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"\n【4】写 {args.out}")
    np.savez_compressed(
        args.out,
        state=states,
        action_gt=seg.astype(np.float32),
        rgb=rgb,
        us=us,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    size_mb = os.path.getsize(args.out) / 1e6
    per_frame = size_mb / args.n_frames
    print(f"  {size_mb:.1f} MB（{per_frame * 1e6:.0f} B/帧）")
    print(f"  传输预估: 走 Tailscale 按 20Mbps 算约 {size_mb * 8 / 20:.0f}s")
    print(f"\n  单帧线上体积 ≈ {2 * args.img_size * args.img_size * 3 / 1e6:.2f} MB(图) "
          f"+ 小量 state → 15Hz 需要 {15 * 2 * args.img_size ** 2 * 3 * 8 / 1e6:.0f} Mbps")
    print("\n  下一步：将回放包与 remote/ 目录复制到测试主机，操作见 docs/06-环境与操作指南.md")


if __name__ == "__main__":
    main()
