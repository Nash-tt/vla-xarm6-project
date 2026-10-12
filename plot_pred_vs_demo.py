"""
离线对比：模型预测动作 vs 数据集示教动作（两图）

图1 (pred_vs_demo.png)：逐帧对比。每个采样帧喂【真实观测】，策略输出 50 步 chunk，
                        取 chunk 第 0 步作为"模型在该帧会下发的动作"，与示教对比。
图2 (chunk_vs_future.png)：某锚点处，完整 50 步预测 chunk vs 真实未来 50 步
                        ——用于判断 chunk 内部是否平滑、是否跟得住。

用法:
    python plot_pred_vs_demo.py --episode 0 --stride 10
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体 ----
for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors

CKPT = "smolvla_ultrasound_20k_tol1ms/checkpoints/020000/pretrained_model"
REPO = "local/vla_batch_20261007_smolvla_6d_no_depth"
DS_ROOT = "/data/vla/vla_batch_20261007_smolvla_6d_no_depth"
FPS = 15.0
DIMS = ["x (mm)", "y (mm)", "z (mm)", "roll (°)", "pitch (°)", "yaw (°)"]


def ep_range(ds, episode):
    for row in ds.meta.episodes:
        if int(row["episode_index"]) == episode:
            return int(row["dataset_from_index"]), int(row["dataset_to_index"])
    raise ValueError(f"episode {episode} 不存在")


def build_batch(sample):
    b = {k: sample[k].unsqueeze(0) for k in
         ("observation.images.rgb", "observation.images.ultrasound", "observation.state")}
    b["task"] = [sample["task"]]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载数据集 / 策略 ...")
    ds = LeRobotDataset(REPO, root=DS_ROOT)
    policy = SmolVLAPolicy.from_pretrained(CKPT).eval().to(dev)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=CKPT)
    chunk_size = policy.config.n_action_steps

    f0, f1 = ep_range(ds, args.episode)
    idxs = list(range(f0, f1, args.stride))
    print(f"episode {args.episode}: 帧 {f0}..{f1}, 采样 {len(idxs)} 点 (stride={args.stride})")

    gt, pred = [], []
    anchor = idxs[len(idxs) // 2]          # 锚点（大致在扫描段）
    anchor_chunk = None
    for n, gi in enumerate(idxs):
        s = ds[gi]
        gt.append(s["action"].numpy())
        obs = pre(build_batch(s))
        with torch.inference_mode():
            chunk = post(policy.predict_action_chunk(obs))[0].float().cpu().numpy()
        pred.append(chunk[0])
        if gi == anchor:
            anchor_chunk = chunk
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(idxs)}")

    gt = np.array(gt); pred = np.array(pred)
    t = np.arange(len(gt)) * args.stride / FPS
    err = np.abs(pred - gt)

    print("\n=== 逐维度绝对误差 (mean / p95 / max) ===")
    for i, d in enumerate(DIMS):
        print(f"  {d:11s} mean={err[:,i].mean():8.3f}  p95={np.percentile(err[:,i],95):8.3f}  max={err[:,i].max():8.3f}")

    # ================= 图1：逐帧对比 =================
    fig, axes = plt.subplots(7, 1, figsize=(13, 20))
    for i in range(6):
        ax = axes[i]
        ax.plot(t, gt[:, i], color="0.45", lw=2.4, label="示教 (ground truth)", zorder=1)
        ax.plot(t, pred[:, i], color="crimson", lw=1.1, alpha=0.9, label="预测 (chunk 第0步)", zorder=2)
        ax.set_ylabel(DIMS[i]); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
            ax.set_title(f"模型预测 vs 示教动作 — episode {args.episode}（训练集内，stride={args.stride}）")
    ax = axes[6]
    for i in range(6):
        ax.plot(t, err[:, i], lw=1.0, label=DIMS[i])
    ax.set_ylabel("绝对误差"); ax.set_xlabel("时间 (s)"); ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8)
    ax.set_title("逐维度绝对误差")
    plt.tight_layout(); plt.savefig("pred_vs_demo.png", dpi=110)
    print("图1 已保存: pred_vs_demo.png")

    # ================= 图2：完整 chunk vs 真实未来 =================
    if anchor_chunk is not None:
        fut = np.array([ds[anchor + j]["action"].numpy() for j in range(len(anchor_chunk))])
        tc = np.arange(len(anchor_chunk)) / FPS
        fig2, ax2 = plt.subplots(6, 1, figsize=(13, 17))
        for i in range(6):
            a = ax2[i]
            a.plot(tc, fut[:, i], color="0.45", lw=2.4, label="真实未来轨迹", marker="o", ms=3, zorder=1)
            a.plot(tc, anchor_chunk[:, i], color="crimson", lw=1.4, label="预测 chunk", marker="x", ms=3, zorder=2)
            a.set_ylabel(DIMS[i]); a.grid(alpha=0.3)
            if i == 0:
                a.legend(loc="best")
                a.set_title(f"某锚点(t={anchor/FPS:.1f}s) 一次推理的完整 {len(anchor_chunk)} 步 chunk vs 真实未来")
        ax2[-1].set_xlabel("未来时间 (s)")
        plt.tight_layout(); plt.savefig("chunk_vs_future.png", dpi=110)
        print("图2 已保存: chunk_vs_future.png")

    # ================= 噪声诊断 =================
    print("\n=== 稳定性诊断：示教几乎不变的轴 vs 预测抖动 ===")
    for i, d in enumerate(DIMS):
        gt_std = gt[:, i].std()
        res_std = (pred[:, i] - gt[:, i]).std()
        flag = "  ← 示教≈静止但预测抖动大" if (gt_std < 1.0 and res_std > gt_std * 2) else ""
        print(f"  {d:11s} 示教std={gt_std:8.3f}   残差std={res_std:8.3f}{flag}")


if __name__ == "__main__":
    main()
