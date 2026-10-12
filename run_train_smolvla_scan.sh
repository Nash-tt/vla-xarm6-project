#!/usr/bin/env bash
# SmolVLA 微调 —— xArm6 超声扫描 (TCP 6D 位姿控制), 只留扫描段的数据集
#
# 用法:
#   ./run_train_smolvla_scan.sh              # 正式训练 (20k 步, bs=32)
#   SMOKE=1 ./run_train_smolvla_scan.sh      # 冒烟测试 (20 步, bs=2)
#
# 可覆盖的环境变量: LEROBOT_BIN / WEIGHTS / STEPS / BATCH_SIZE / NUM_WORKERS /
#                   SAVE_FREQ / OUTPUT_DIR / JOB_NAME
set -euo pipefail

# 绝对路径指向 conda 的 lerobot 环境, 不依赖"当前激活的环境"
# (2026-10-11 环境事故: 装 openpi 时把依赖灌进了 lerobot 环境)
LEROBOT_BIN="${LEROBOT_BIN:-/opt/conda/envs/lerobot/bin/lerobot-train}"

# 只留扫描段的副本, 且 roll 已解 wrap (见 tools/trim_scan_segments.py --unwrap-roll)
# 为什么必须用解 wrap 这版:
#   原数据集 roll wrap 到 ±180, std=172.06; 解 wrap 后 std=7.83。
#   本配置的 normalization_mapping 是 ACTION: MEAN_STD —— 除以 std。
#   wrap 版会把整个物理 roll 区间 [163.967, 193.039] 压进 z ∈ [0.8193, 0.9883],
#   跨度只有 0.169 个标准差且全挤在 z≈+0.9, 等于模型看到常数, roll 根本学不到。
#   解 wrap 版同一区间映射到 z ∈ [-1.4424, +2.2719], 跨度 3.71。
DATASET_ROOT="${DATASET_ROOT:-/data/vla/vla_batch_20261007_smolvla_6d_no_depth_rollunwrap_scan}"
DATASET_REPO_ID=local/vla_batch_20261007_smolvla_6d_no_depth_rollunwrap_scan

# smolvla_base 的本地 snapshot (HF 缓存里的快照目录)
WEIGHTS="${WEIGHTS:-lerobot/smolvla_base}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS="${STEPS:-20}"; BATCH_SIZE="${BATCH_SIZE:-2}"; NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-20}"
  OUTPUT_DIR="${OUTPUT_DIR:-smolvla_scan_smoke}"
  JOB_NAME="${JOB_NAME:-smolvla_scan_smoke}"
else
  STEPS="${STEPS:-20000}"; BATCH_SIZE="${BATCH_SIZE:-32}"; NUM_WORKERS="${NUM_WORKERS:-8}"
  SAVE_FREQ="${SAVE_FREQ:-2000}"
  OUTPUT_DIR="${OUTPUT_DIR:-smolvla_ultrasound_scan_20k}"
  JOB_NAME="${JOB_NAME:-smolvla_ultrasound_scan}"
fi

# 显存碎片整理; 代理必须去掉 (ALL_PROXY 的 socks:// scheme 会让 httpx 直接报错)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1

[[ -d "$WEIGHTS" ]]      || { echo "找不到 smolvla_base 权重目录: $WEIGHTS" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "找不到数据集: $DATASET_ROOT" >&2; exit 1; }
[[ -x "$LEROBOT_BIN" ]]  || { echo "找不到 lerobot-train: $LEROBOT_BIN" >&2; exit 1; }

echo "policy    : smolvla_base (全量微调, 非 LoRA)"
echo "weights   : $WEIGHTS"
echo "dataset   : $DATASET_ROOT"
echo "steps     : $STEPS   batch_size: $BATCH_SIZE   workers: $NUM_WORKERS"
echo "output    : $OUTPUT_DIR"
echo

# 说明:
#   tolerance_s=1e-3  —— 不能去掉, 原因不是数据脏而是 float32 精度:
#     decode_video_frames_pyav 里 `torch.tensor(timestamps)` 把 float64 降成 float32,
#     而 float32 在 t≈1277s 处的 ULP = 1.22e-4 > 默认的 1e-4; 超声视频长 3270s,
#     ULP 到末尾是 2.44e-4。判定是严格小于, 偏差恰好等于 1e-4 也算失败。
#     全量数据集 7 帧踩中 (0.014%)。裁剪后的数据集实测最大偏差 3.052e-05
#     (check_timestamps.py, 0/27312 帧超 1e-4), 默认值其实也过得去,
#     但那 7 帧是踩格点边界的偶发事件, 保留 1e-3 留 33 倍余量, 零代价 ——
#     容差只影响校验, 不影响选帧 (解码器总是取最近帧)。
#
#   --policy.use_amp=true  —— smolvla 走 AMP 而不是 pi0 LoRA 那套 bfloat16 +
#                             gradient_checkpointing, 与上次 20k run 保持一致。
#
#   input_features/output_features 传 null  —— smolvla 用数据集推导的特征,
#     与 pi0 不同 (pi0_base 的 config 已声明固定槽位, 那边绝不能传 null)。
exec "$LEROBOT_BIN" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --tolerance_s=1e-3 \
  --policy.path="$WEIGHTS" \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq=50 \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=false
