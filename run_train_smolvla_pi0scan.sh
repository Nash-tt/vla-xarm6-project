#!/usr/bin/env bash
# SmolVLA 微调（第二组）—— 用 pi0 那版数据集重训一轮，做数据集口径的对照
#
# 与 run_train_smolvla_scan.sh 的唯一差别是数据集：
#   run_train_smolvla_scan.sh  → vla_batch_20261007_smolvla_6d_no_depth_rollunwrap_scan
#   本脚本                     → vla_batch_20261007_pi0_6d_rollunwrap_scan
#
# 两者结构完全相同（实测：27312 帧 / 28 回合 / 15 fps / 同样 4 个 feature /
# 同样的 state 名，解 wrap 后数值逐位相同），**唯一差别是 task 字符串**：
#   smolvla 版  60 字符
#   pi0 版      214 字符（"Move from the configured capture-start pose to the
#               first taught scan point using a constant-height approach,
#               descend vertically, then follow the fitted trajectory while
#               collecting synchronized RGB-D and ultrasound."）
# SmolVLA 是三个策略里**唯一真的读语言**的（pi0 结构里有但不吃，ACT 根本没语言塔），
# 所以这一组是"指令措辞对策略有无影响"的对照实验。
# 其余超参与 RUN_TAG 之前的 20k run 完全一致，便于直接对比。
#
# 用法:
#   ./run_train_smolvla_pi0scan.sh           # 正式训练 (20k 步, bs=32)
#   SMOKE=1 ./run_train_smolvla_pi0scan.sh   # 冒烟测试 (20 步, bs=2)
#
# 可覆盖的环境变量: LEROBOT_BIN / WEIGHTS / STEPS / BATCH_SIZE / NUM_WORKERS /
#                   SAVE_FREQ / OUTPUT_DIR / JOB_NAME
set -euo pipefail

LEROBOT_BIN="${LEROBOT_BIN:-/opt/conda/envs/lerobot/bin/lerobot-train}"

DATASET_ROOT="${DATASET_ROOT:-/data/vla/vla_batch_20261007_pi0_6d_rollunwrap_scan}"
DATASET_REPO_ID=local/vla_batch_20261007_pi0_6d_rollunwrap_scan

WEIGHTS="${WEIGHTS:-lerobot/smolvla_base}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS="${STEPS:-20}"; BATCH_SIZE="${BATCH_SIZE:-2}"; NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-20}"
  OUTPUT_DIR="${OUTPUT_DIR:-smolvla_pi0scan_smoke}"
  JOB_NAME="${JOB_NAME:-smolvla_pi0scan_smoke}"
else
  STEPS="${STEPS:-20000}"; BATCH_SIZE="${BATCH_SIZE:-32}"; NUM_WORKERS="${NUM_WORKERS:-8}"
  SAVE_FREQ="${SAVE_FREQ:-2000}"
  OUTPUT_DIR="${OUTPUT_DIR:-smolvla_pi0scan_20k}"
  JOB_NAME="${JOB_NAME:-smolvla_pi0scan}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1

# 权重既可以是本地 snapshot 目录，也可以是 HF repo id（org/name）——后者在联网时由
# from_pretrained 自己去拉。但这个脚本置了 HF_HUB_OFFLINE=1，走 repo id 时该权重必须
# 已在 HF 缓存里，否则报的是网络错误，很难查。所以这里两种写法都放行，只拦明显写错的。
if [[ -d "$WEIGHTS" ]]; then
  :
elif [[ "$WEIGHTS" == */* && "$WEIGHTS" != /* && "$WEIGHTS" != ./* ]]; then
  echo "smolvla_base: 权重按 HF repo id 处理 —— $WEIGHTS（离线模式下须已在缓存里）"
else
  echo "smolvla_base: 权重既不是已存在的目录，也不像 HF repo id（org/name）: $WEIGHTS" >&2
  exit 1
fi
[[ -d "$DATASET_ROOT" ]] || { echo "找不到数据集: $DATASET_ROOT" >&2; exit 1; }
[[ -x "$LEROBOT_BIN" ]]  || { echo "找不到 lerobot-train: $LEROBOT_BIN" >&2; exit 1; }

echo "policy    : smolvla_base (全量微调, 非 LoRA)  —— 数据集换成 pi0 版"
echo "weights   : $WEIGHTS"
echo "dataset   : $DATASET_ROOT"
echo "steps     : $STEPS   batch_size: $BATCH_SIZE   workers: $NUM_WORKERS"
echo "output    : $OUTPUT_DIR"
echo

# 说明:
#   tolerance_s=1e-3  —— 同另两个脚本, 绕开 decode_video_frames_pyav 的 float32 边界
#     (float64 被 torch.tensor 降成 float32, t≈1277s 处 ULP=1.22e-4 > 默认 1e-4)。
#     本数据集实测最大偏差 3.052e-05, 默认值其实过得去; 1e-3 留 33 倍余量, 零代价。
#
#   --policy.input_features=null / output_features=null  —— **本脚本必须传**, 与 pi0 相反。
#     smolvla 的特征是从数据集 meta 推导的; pi0_base 的 config 已声明固定槽位,
#     那边传 null 会直接报 "All image features are missing from the batch"。
#     ACT 也不传 (ACTConfig 两者默认就是空 dict, 显式传反而多一处出错点)。
#     三个策略三种写法, 别互相照搬。
#
#   --policy.use_amp=true  —— smolvla 走 AMP, 与上一轮 20k run 保持一致,
#     这样两组之间只差数据集这一个变量。
#
#   rename_map  —— smolvla 不需要 (实测为 {})。
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
