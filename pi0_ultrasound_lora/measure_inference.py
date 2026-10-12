#!/usr/bin/env python
"""实测 pi0-LoRA 的推理显存 / 延迟 / 动作误差, 并和未微调的 pi0_base 对比。

训练日志里的 10.63 GiB 是「batch=8 + 反向 + 优化器」的数字, 跟部署时的推理完全是
两回事。这里量部署真正关心的量, 并且拿 pi0_base 原模型做对照。

── 两个加载上的坑 (都踩过) ──────────────────────────────────────────────
1. `PI0Policy.from_pretrained(ckpt)` 对 PEFT checkpoint 是**错的**。
   它只找 model.safetensors, 而 LoRA checkpoint 里只有 adapter_model.safetensors。
   找不到时它不报错, 只打印一句 "Returning model without loading pretrained weights",
   然后返回一个**随机初始化的模型** —— 静默给出垃圾结果 (实测 MAE 52.7, 真实是 1.6)。
   正确入口是 policies/factory.py:565 的 make_policy(), 这里复刻那三行:
       读 adapter_config.json -> 拿 base_model_name_or_path -> 加载底座 -> 叠 adapter

2. config 必须取自 checkpoint, 且必须用**基类** PreTrainedConfig 读。
   - 取自 pi0_base 的话 output_features.action.shape 是 32 而不是 6, 归一化映射也不同;
   - ckpt 的 config.json 带 "type":"pi0", 直接调 PI0Config.from_pretrained 会因为
     draccus 解不了这个判别字段而报错 (PI0Config 本身不是 ChoiceRegistry)。
   - 还要先 import lerobot.policies.pi0.configuration_pi0 注册子类, 否则同样找不到。

用法:
    python measure_inference.py [--ckpt DIR] [--reps 20] [--episodes 6]
                                [--samples 5] [--seed 0] [--no-base]

pi0 是 flow matching, 每次 predict_action_chunk 都现抽一组噪声, 所以结果本身是随机的。
脚本同时给两个口径, 别混用:

  单次MAE  —— 抽一次就执行整个 chunk, **这就是部署时会拿到的误差**, 引用这个
  均值MAE  —— 平均多次抽样, 噪声互相抵消, 只用于横向对比, 实机拿不到, 偏乐观

(没固定种子时, 同一 checkpoint 单次抽样曾两轮测出 1.641 / 2.091 —— 全是采样噪声。)
"""

import argparse
import gc
import glob
import os
import time

for _v in ("ALL_PROXY", "all_proxy"):  # socks:// scheme httpx 不认; 离线也不需要网络
    os.environ.pop(_v, None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from peft import PeftConfig, PeftModel  # noqa: E402

import lerobot.policies.pi0.configuration_pi0  # noqa: E402,F401  注册 'pi0' 子类
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: E402

ROOT = "/data/vla/vla_batch_20261007_pi0_6d_rollunwrap"
REPO = "local/vla_batch_20261007_pi0_6d_rollunwrap"
DEFAULT_CKPT = "pi0_ultrasound_lora/run1/checkpoints/012250/pretrained_model"
GIB = 2**30
DIMS = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]


def memline(tag):
    torch.cuda.synchronize()
    print(
        f"    {tag:<20s} 当前 {torch.cuda.memory_allocated() / GIB:5.2f} GiB"
        f"   保留 {torch.cuda.memory_reserved() / GIB:5.2f} GiB"
        f"   峰值 {torch.cuda.max_memory_allocated() / GIB:5.2f} GiB"
    )


def load_lora(ckpt):
    """复刻 make_policy() 的 PEFT 分支 (factory.py:565-589)。"""
    peft_cfg = PeftConfig.from_pretrained(ckpt)
    base = peft_cfg.base_model_name_or_path
    if not base:
        raise SystemExit("adapter_config.json 里没有 base_model_name_or_path")
    cfg = PreTrainedConfig.from_pretrained(ckpt)  # 基类: ckpt 的 config.json 带 type 字段
    policy = PI0Policy.from_pretrained(base, config=cfg)
    policy = PeftModel.from_pretrained(policy, ckpt, config=peft_cfg, is_trainable=False)
    return policy.get_base_model(), cfg, base


def load_base(base_path, cfg):
    """同样用 ckpt 的 config 加载底座 —— 否则 action 维度是 32, 没法直接对比。"""
    return PI0Policy.from_pretrained(base_path, config=cfg)


def pick_frames(eps, data, n_ep, chunk):
    """每条 episode 挑一个「在动」的帧, 且后面还剩至少 chunk 帧。"""
    out = []
    for ep_i in range(n_ep):
        sub = data[data.episode_index == ep_i].sort_values("frame_index")
        A = np.stack(sub["action"].values)
        f0 = sub["frame_index"].values
        spd = np.linalg.norm(np.diff(A, axis=0), axis=1)
        moving = spd > np.percentile(spd[spd > 0], 40)
        ok = np.where(moving & (np.arange(len(spd)) + chunk + 1 < len(A)))[0]
        if len(ok) == 0:
            continue
        j = int(ok[len(ok) // 2])  # 取中间, 避开起停段
        out.append((ep_i, int(f0[j]), int(eps.iloc[ep_i]["dataset_from_index"]) + int(f0[j]), A, j))
    return out


SPAN_MIN = 0.05  # 50 步行程低于此值时, 「误差/行程」比值无意义 (除以 ~0)


def lora_off(inner):
    """把所有 lora_B 置零 —— 等价于「LoRA 增量没训练过」, 但保留 adapter 里
    modules_to_save 那些**被完整训练过**的层 (动作投影等)。

    用来分解收益来源: 有多少来自 LoRA 增量, 有多少来自重训的投影层。
    PEFT 初始化时 B 恒为 0, 所以置零 = 精确回到「未微调」的那个 LoRA 状态。
    """
    saved = []
    for n, p in inner.named_parameters():
        if "lora_B" in n:
            saved.append((p, p.detach().clone()))
            p.data.zero_()
    return saved


def lora_restore(saved):
    for p, v in saved:
        p.data.copy_(v)


def mae_table(label, inner, pre, post, frames, ds, chunk, samples=1, seed0=0):
    """对每条 episode 算 50 步 chunk 的 MAE, 并给出「照抄当前动作」基线。

    pi0 是 flow matching: 每次 predict_action_chunk 都**现抽高斯噪声**, 所以单次
    结果本身就是随机的。必须固定种子 + 多次采样平均, 否则帧间差异分不清是模型
    的差异还是采样噪声。(之前没固定种子时, 同一 checkpoint 两轮测出 1.641 / 2.091。)
    """
    print(f"\n  {label}   (每帧采样 {samples} 次, seed={seed0})")
    print(f"  {'ep':>4s} {'帧':>7s} {'单次MAE':>10s} {'均值MAE':>9s} "
          f"{'照抄当前动作':>13s} {'单次/基线':>10s}")
    es, em, eb, dims_err, dims_span = [], [], [], [], []
    for ep_i, fi, g, A, j in frames:
        fr = ds[g]
        b = pre({
            "observation.images.rgb": fr["observation.images.rgb"],
            "observation.images.ultrasound": fr["observation.images.ultrasound"],
            "observation.state": fr["observation.state"],
            "task": fr["task"],
        })
        ps = []
        for s in range(samples):
            torch.manual_seed(seed0 + s)  # 同一 seed 序列给所有帧, 帧间才可比
            with torch.no_grad():
                ps.append(post(inner.predict_action_chunk(b))[0].float().cpu().numpy())
        stack = np.stack(ps)  # (samples, chunk, 6)
        gt = A[j : j + chunk]
        # 【部署口径】实机只抽一次噪声就执行整个 chunk, 这就是你会拿到的误差
        e_single = float(np.abs(stack - gt[None]).mean())
        # 【方差缩减口径】平均多次采样 —— 平均能抵消噪声, 所以比部署乐观
        e_mean = float(np.abs(stack.mean(axis=0) - gt).mean())
        e_b = np.abs(np.repeat(A[j][None], len(gt), axis=0) - gt).mean()
        es.append(e_single)
        em.append(e_mean)
        eb.append(e_b)
        dims_err.append(np.abs(stack - gt[None]).mean(axis=(0, 1)))  # 单次口径
        dims_span.append(gt.max(axis=0) - gt.min(axis=0))
        print(f"  {ep_i:>4d} {fi:>7d} {e_single:10.3f} {e_mean:9.3f} {e_b:13.3f} "
              f"{e_single / e_b:9.2f}x")

    es, em, eb = float(np.mean(es)), float(np.mean(em)), float(np.mean(eb))
    print(f"  {'平均':>4s} {'':>7s} {es:10.3f} {em:9.3f} {eb:13.3f} {es / eb:9.2f}x")
    print(f"  >> 引用部署数字请用「单次MAE」= {es:.3f}; 「均值MAE」{em:.3f} 是平均 "
          f"{samples} 次抽样的结果, 实机拿不到, 乐观了 {100 * (es - em) / em:.0f}%")
    if dims_err:
        de = np.mean(dims_err, axis=0)
        dsp = np.mean(dims_span, axis=0)
        print("    逐维 MAE (单次口径, 6 条平均):")
        for i, nm in enumerate(DIMS):
            if dsp[i] >= SPAN_MIN:
                ratio = f"{100 * de[i] / dsp[i]:5.1f}%"
            else:
                ratio = "  n/a"  # 该维几乎不动, 比值是除以 ~0, 无意义
            print(f"      {nm:<10s} {de[i]:8.3f}   该维 50 步平均行程 {dsp[i]:8.3f}   "
                  f"占比 {ratio}")
    return es, em, eb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--samples", type=int, default=5,
                    help="每帧重复采样次数 (pi0 每次调用都现抽噪声, 必须多次平均)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-base", action="store_true", help="跳过 pi0_base 对照")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("没有 CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"checkpoint : {args.ckpt}")

    # ---------- 加载 LoRA ----------
    print("\n【1】加载 (LoRA checkpoint)")
    t0 = time.perf_counter()
    inner, cfg, base_path = load_lora(args.ckpt)
    inner.eval()
    inner.to("cuda")
    t_load = time.perf_counter() - t0
    print(f"  底座     : {base_path}")
    print(f"  耗时     : {t_load:.1f} s    dtype: {next(inner.parameters()).dtype}")
    print(f"  总参数   : {sum(p.numel() for p in inner.parameters()) / 1e9:.4f} B")

    # adapter 是否真的注入了。别按名字比对 (peft 会改名, 容易假阴性);
    # 用 lora_B 的性质: PEFT 初始化时 B 恒为 0, 读进来非 0 就说明是训练写过的权重。
    from safetensors.torch import load_file

    ap_file = os.path.join(args.ckpt, "adapter_model.safetensors")
    adapter = load_file(ap_file)
    bn = [p.detach().float().norm().item() for n, p in inner.named_parameters() if "lora_B" in n]
    print(f"  adapter  : {sum(v.numel() for v in adapter.values()) / 1e6:.2f} M 参数, "
          f"{len(adapter)} 张量, {os.path.getsize(ap_file) / 1e6:.1f} MB")
    print(f"  已注入   : 模型内 lora_B {len(bn)} 个, 范数 [{min(bn):.3f}, {max(bn):.3f}], "
          f"全零 {sum(1 for x in bn if x == 0.0)} 个   <- 全零 = 没加载")
    memline("加载完")

    # ---------- 数据 ----------
    ds = LeRobotDataset(REPO, root=ROOT, video_backend="pyav")
    eps = pq.read_table(sorted(glob.glob(f"{ROOT}/meta/episodes/*/*.parquet"))[0]).to_pandas()
    data = pq.read_table(sorted(glob.glob(f"{ROOT}/data/*/*.parquet"))[0]).to_pandas()
    pre, post = make_pre_post_processors(inner.config, pretrained_path=args.ckpt)
    chunk = inner.config.n_action_steps
    frames = pick_frames(eps, data, min(args.episodes, len(eps)), chunk)

    s0 = ds[frames[0][2]]
    print(f"\n【2】输入 (episode 0 第 {frames[0][1]} 帧, 运动中)")
    for k in ("observation.images.rgb", "observation.images.ultrasound", "observation.state"):
        print(f"  {k:<32s} {tuple(s0[k].shape)}  {s0[k].dtype}")
    print(f"  task  {s0['task']!r}")
    batch = pre({
        "observation.images.rgb": s0["observation.images.rgb"],
        "observation.images.ultrasound": s0["observation.images.ultrasound"],
        "observation.state": s0["observation.state"],
        "task": s0["task"],
    })

    # ---------- 显存 + 延迟 ----------
    def timeit(n):
        with torch.no_grad():
            for _ in range(3):
                inner.predict_action_chunk(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            for _ in range(n):
                inner.predict_action_chunk(batch)
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / n

    print("\n【3】推理显存与延迟")
    inner.model.gradient_checkpointing_disable()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        t = time.perf_counter()
        inner.predict_action_chunk(batch)
        torch.cuda.synchronize()
    cold = time.perf_counter() - t
    steady = timeit(args.reps)
    peak = torch.cuda.max_memory_allocated() / GIB
    print(f"    冷启动首帧     : {cold * 1e3:7.1f} ms   (含 cudnn autotune)")
    print(f"    稳态 {args.reps:2d} 次均值 : {steady * 1e3:7.1f} ms   ({1 / steady:.2f} Hz)")
    memline("推理峰值")
    gc_on = timeit(args.reps)
    inner.model.gradient_checkpointing_enable()
    print(f"    gradient_checkpointing=True 时 {gc_on * 1e3:.1f} ms "
          f"(推理无影响: 那条分支只在 training 下生效)")
    inner.model.gradient_checkpointing_disable()
    print(f"\n    >> 峰值 {peak:.2f} GiB, 每 chunk {steady * 1e3:.0f} ms "
          f"= {chunk} 步共 {chunk / 15:.2f} s 轨迹 -> 实时余量 {chunk / 15 / steady:.1f}x")

    # ---------- 动作误差 ----------
    print(f"\n【4】训练集动作 MAE (拟合度, 绝不是泛化数字)")
    e_lora, e_lora_avg, b_lora = mae_table("LoRA 微调后", inner, pre, post, frames, ds, chunk,
                                           args.samples, args.seed)

    # 消融: 关掉 LoRA 增量, 只留 adapter 里被完整训练过的层 (modules_to_save)。
    # 用来回答「收益里多少是 LoRA 挣的, 多少是重训投影层挣的」。
    print("\n【5】消融: 把 lora_B 全置零 (保留训练过的投影层)")
    saved = lora_off(inner)
    e_off, _, _ = mae_table("LoRA 关掉 (B=0)", inner, pre, post, frames, ds, chunk,
                            args.samples, args.seed)
    lora_restore(saved)
    print(f"\n  >> LoRA 增量本身的贡献: {e_off:.3f} -> {e_lora:.3f} "
          f"({100 * (e_lora - e_off) / e_off:+.1f}%)")

    if not args.no_base:
        print("\n【6】对照: 未微调的 pi0_base")
        del inner
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        inner_b = load_base(base_path, cfg)
        inner_b.eval()
        inner_b.to("cuda")
        print(f"  加载耗时 {time.perf_counter() - t0:.1f} s")
        e_base, _, b_base = mae_table("pi0_base (无 adapter)", inner_b, pre, post, frames,
                                      ds, chunk, args.samples, args.seed)
        print(f"\n  >> 微调总收益: 单次口径 MAE {e_base:.3f} -> {e_lora:.3f} "
              f"({100 * (e_lora - e_base) / e_base:+.1f}%)")

    print("\n【提醒】两条都是训练集上的数字。这个 run 的 eval_split=0.0, "
          "28 条 episode 全进了训练, 所以这里测不出泛化。")


if __name__ == "__main__":
    main()
