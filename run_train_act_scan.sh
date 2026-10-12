#!/usr/bin/env bash
# ACT 训练 —— xArm6 超声扫描 (TCP 6D 位姿控制), 只留扫描段的数据集
#
# 用法:
#   ./run_train_act_scan.sh              # 正式训练 (20k 步, bs=24, chunk=50)
#   SMOKE=1 ./run_train_act_scan.sh      # 冒烟测试 (20 步, bs=4)
#
# 可覆盖的环境变量: LEROBOT_BIN / STEPS / BATCH_SIZE / NUM_WORKERS /
#                   SAVE_FREQ / OUTPUT_DIR / JOB_NAME / CHUNK_SIZE
set -euo pipefail

# 绝对路径指向 conda 的 lerobot 环境, 不依赖"当前激活的环境"
# (2026-10-11 环境事故: 装 openpi 时把依赖灌进了 lerobot 环境)
LEROBOT_BIN="${LEROBOT_BIN:-/opt/conda/envs/lerobot/bin/lerobot-train}"

# 只留扫描段的副本, 且 roll 已解 wrap (见 tools/trim_scan_segments.py --unwrap-roll)
# 为什么必须用解 wrap 这版: ACT 的 normalization_mapping 是 STATE/ACTION 全 MEAN_STD,
#   即除以 std。wrap 版 roll 的 std=172, 真实物理区间 [163.967, 193.039] 被压进
#   z ∈ [0.819, 0.988] 且全挤在 +0.9 附近, 等于常数, roll 学不到;
#   解 wrap 后 std=7.7661, 同一区间映射到 z ∈ [-1.442, +2.272], 跨度 3.71。
#   与 smolvla 版解 wrap 后的数值逐位相同, 两版可混用。
DATASET_ROOT="${DATASET_ROOT:-/data/vla/vla_batch_20261007_pi0_6d_rollunwrap_scan}"
DATASET_REPO_ID=local/vla_batch_20261007_pi0_6d_rollunwrap_scan

# ACT 没有可用的预训练权重 —— 从随机初始化训练, 不像 pi0/smolvla 需要 --policy.path。
# (Hub 上那些 lerobot/act_* 是 ALOHA 特定机器人的, 套不到 6 维 TCP 控制上)
# resnet18 的 ImageNet 骨干权重会由 torchvision 自动下载, 缓存在
# ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth。本脚本会 unset 代理并置
# HF_HUB_OFFLINE=1, 所以**首次运行前**该文件必须已在缓存里, 否则加载骨干时失败。

# chunk_size=50 与真机客户端的 --actions-per-chunk 50 对齐。
# n_action_steps 必须 <= chunk_size (configuration_act.py:143 会校验), 默认值 100
# 不改的话直接报错。取满 50 = 不做时间集成, 整块开环执行。
CHUNK_SIZE="${CHUNK_SIZE:-50}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS="${STEPS:-20}"; BATCH_SIZE="${BATCH_SIZE:-4}"; NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-20}"
  OUTPUT_DIR="${OUTPUT_DIR:-act_scan_smoke}"
  JOB_NAME="${JOB_NAME:-act_scan_smoke}"
else
  # batch_size=24 是实测上限: 原生 720x1280 输入下 3090 24GB 上
  #   B=16 → 11.3 GiB / 389 ms/步 ; B=24 → 16.8 GiB / 577 ms/步 ; B=32 → OOM
  # (影像塔出 23x40=920 个 token, 显存几乎全在这里; chunk 50→100 只差 0.4 GiB)
  STEPS="${STEPS:-20000}"; BATCH_SIZE="${BATCH_SIZE:-24}"; NUM_WORKERS="${NUM_WORKERS:-8}"
  SAVE_FREQ="${SAVE_FREQ:-2000}"
  OUTPUT_DIR="${OUTPUT_DIR:-act_ultrasound_scan_20k}"
  JOB_NAME="${JOB_NAME:-act_ultrasound_scan}"
fi

# 显存碎片整理; 代理必须去掉 (ALL_PROXY 的 socks:// scheme 会让 httpx 直接报错)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1

[[ -d "$DATASET_ROOT" ]] || { echo "找不到数据集: $DATASET_ROOT" >&2; exit 1; }
[[ -x "$LEROBOT_BIN" ]]  || { echo "找不到 lerobot-train: $LEROBOT_BIN" >&2; exit 1; }

echo "policy    : ACT (从零训练, 51.6M 参数, resnet18 骨干)"
echo "dataset   : $DATASET_ROOT"
echo "steps     : $STEPS   batch_size: $BATCH_SIZE   workers: $NUM_WORKERS"
echo "chunk     : chunk_size=$CHUNK_SIZE  n_action_steps=$CHUNK_SIZE"
echo "output    : $OUTPUT_DIR"
echo

# 说明:
#   --policy.type=act  —— 不用 --policy.path (没有预训练权重可加载)
#
#   不用 --rename_map  —— 这是 ACT 与 pi0 最省事的一处差别。pi0 的 rename_map 是因为
#     pi0_base 的 config.json 把图像槽位名写死成 base_0_rgb / left_wrist_0_rgb,
#     数据集里的 observation.images.* 对不上, 必须在传送层改名。ACT 从零训练,
#     特征是 make_policy 用数据集 meta 现推的 (factory.py:530-531), 实测得到:
#         observation.images.rgb        VISUAL (3, 720, 1280)
#         observation.images.ultrasound VISUAL (3, 256, 256)
#         observation.state             STATE  (6,)
#         action                        ACTION (6,)
#     即数据集自己的名字直接就是策略的特征名, 无需干预。
#
#   同样地, 不要传 --policy.input_features=null / output_features=null。
#     ACTConfig 两者的默认值本就是空 dict, 而 factory.py:531 只在为空时才用数据集
#     推导 —— 现状已经是对的, 显式传反而多一处出错的地方 (pi0 那边传 null 会清掉
#     固定槽位并直接报 "All image features are missing from the batch", 别照搬)。
#
#   tolerance_s=1e-3  —— 与另两个脚本同源, 绕开 LeRobot 的 float32 边界 bug:
#     decode_video_frames_pyav 里 `torch.tensor(timestamps)` 把 float64 降成 float32,
#     而 float32 在 t≈1277s 处的 ULP = 1.22e-4 > 默认的 1e-4。裁剪后的数据集实测
#     最大偏差 3.052e-05 (0/27312 帧超 1e-4), 默认值其实过得去; 保留 1e-3 是留 33 倍
#     余量, 零代价 —— 容差只影响校验, 不影响选帧 (解码器总是取最近帧)。
#
#   optimizer_lr / weight_decay / lr_backbone  —— 全部取 ACTConfig 默认
#     (1e-5 / 1e-4 / 1e-5)。这套默认值来自原论文的 ALOHA 配置 (batch 8);
#     这里 batch 24 更大, 学习率不变是保守的一侧, 没有放大。
#
#   use_amp  —— 未开启 (ACTConfig 默认 False)。开着大概率能省掉近一半步时,
#     但 VAE 分支的 log-prob 项在 AMP 下容易出 NaN, 要开请先冒烟确认 loss 不发散。
#
#   时间集成 (temporal_ensembling)  —— ACT 是这套代码里唯一支持它的策略:
#     传 --policy.temporal_ensemble_coeff=<如 0.01>, 此时 n_action_steps 必须为 1
#     (configuration_act.py:138 会校验)。部署侧要想拿到集成收益, 客户端也得改成
#     每步都请求。当前默认整块开环, 与真机客户端的行为一致。
exec "$LEROBOT_BIN" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --tolerance_s=1e-3 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.n_action_steps="$CHUNK_SIZE" \
  --policy.push_to_hub=false \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq=50 \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=false
