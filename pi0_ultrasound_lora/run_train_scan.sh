#!/usr/bin/env bash
# π0 (LoRA) 微调（第二组）—— 换用**裁剪 + 解 wrap** 的数据集
#
# 与 run_train.sh 的唯一差别是数据集和一个新的输出目录，超参与 run1 完全一致，
# 使"数据集口径"成为两组之间唯一的变量：
#
#   run_train.sh (run1) → vla_batch_20261007_pi0_6d_rollunwrap       49053 帧（未裁）
#   本脚本     (run2)  → vla_batch_20261007_pi0_6d_rollunwrap_scan  27312 帧（裁后）
#
# 两份数据集的 roll 口径**本来就相同**（都是展开口径：state roll 区间
# [163.967, 193.039]、按回合对齐到同一 360 分支），差别只在：
#   * run1 的 49053 帧里约 44% 是接近段 —— action 冻结在几百毫米外的目标上，
#     位置偏差中位 231mm，roll 维甚至出现 32.31° 这种物理上不存在的值
#   * run2 把接近段裁掉了，位置偏差 >= 10mm 的帧为 0
# 即 run1 有相当一部分 token 在教模型"输出一个几百毫米外的点"。
#
# ⚠️ 输出目录是新的 run2_scan，**不要指到 run1**，那会覆盖已部署的 checkpoint
#    （serve.py 的 DEFAULT_CKPT 指向 run1/checkpoints/012250）。
#
# 用法:
#   WEIGHTS=<pi0_base snapshot 目录> ./run_train_scan.sh   # 正式训练
#   SMOKE=1 ./run_train_scan.sh                            # 冒烟测试 (20 步, bs=1)
#
# 可覆盖的环境变量: LEROBOT_BIN / WEIGHTS / STEPS / BATCH_SIZE / NUM_WORKERS /
#                   SAVE_FREQ / OUTPUT_DIR / JOB_NAME
set -euo pipefail

LEROBOT_BIN="${LEROBOT_BIN:-/opt/conda/envs/lerobot/bin/lerobot-train}"

DATASET_ROOT="${DATASET_ROOT:-/data/vla/vla_batch_20261007_pi0_6d_rollunwrap_scan}"
DATASET_REPO_ID=local/vla_batch_20261007_pi0_6d_rollunwrap_scan

WEIGHTS="${WEIGHTS:-lerobot/pi0_base}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS="${STEPS:-20}"; BATCH_SIZE="${BATCH_SIZE:-1}"; NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-1000}"
  OUTPUT_DIR="${OUTPUT_DIR:-pi0_ultrasound_lora/scan_smoke}"
  JOB_NAME="${JOB_NAME:-pi0_ultrasound_scan_smoke}"
else
  # batch_size=8 与 run1 实测一致（run1 启动时覆盖了脚本默认的 4）。
  # 不改成别的值 —— 这一组要做的是数据集对照，动 batch 就说不清了。
  STEPS="${STEPS:-24500}"; BATCH_SIZE="${BATCH_SIZE:-8}"; NUM_WORKERS="${NUM_WORKERS:-8}"
  SAVE_FREQ="${SAVE_FREQ:-500}"
  OUTPUT_DIR="${OUTPUT_DIR:-pi0_ultrasound_lora/run2_scan}"
  JOB_NAME="${JOB_NAME:-pi0_ultrasound_scan}"
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
  echo "pi0_base: 权重按 HF repo id 处理 —— $WEIGHTS（离线模式下须已在缓存里）"
else
  echo "pi0_base: 权重既不是已存在的目录，也不像 HF repo id（org/name）: $WEIGHTS" >&2
  exit 1
fi
[[ -d "$DATASET_ROOT" ]] || { echo "找不到数据集: $DATASET_ROOT" >&2; exit 1; }
[[ -x "$LEROBOT_BIN" ]]  || { echo "找不到 lerobot-train: $LEROBOT_BIN" >&2; exit 1; }
[[ "$OUTPUT_DIR" != *"/run1"* ]] || { echo "拒绝写入 run1（已部署的 checkpoint）" >&2; exit 1; }

echo "policy    : pi0 + LoRA (r=16, alpha=16)  —— 数据集换成裁剪版"
echo "weights   : $WEIGHTS"
echo "dataset   : $DATASET_ROOT"
echo "steps     : $STEPS   batch_size: $BATCH_SIZE   workers: $NUM_WORKERS"
echo "output    : $OUTPUT_DIR"
echo

# 说明（与 run_train.sh 相同的地方不再重复，只标差异）:
#
#   tolerance_s=1e-3  —— 理由同 run1, 但那里的 7 帧踩格点是在**未裁**数据集的
#     49053 帧上统计的。裁剪后的 27312 帧实测最大偏差 3.052e-05
#     (0/27312 帧超 1e-4)，默认 1e-4 其实也过得去。保留 1e-3 是为了与 run1
#     保持同一容差，且零代价 —— 容差只影响校验，不影响选帧。
#
#   【不要传 --policy.input_features / --policy.output_features / --policy.empty_cameras】
#     pi0_base 的 config.json 已声明固定槽位 (base_0_rgb / left_wrist_0_rgb /
#     right_wrist_0_rgb + state(32) + action(32))。传 null 清空后会改用数据集的名字，
#     _preprocess_images 在 batch 里一个图都找不到，直接报
#     "All image features are missing from the batch"。（实测踩过）
#     注意 smolvla 那边必须传 null —— 三个策略三种写法，别照搬。
#
#   --rename_map  —— 必须, 两台相机映射到 pi0_base 声明的前两个图像槽。
#     ACT/SmolVLA 都不需要这一项。
#
#   --peft.*  —— LoRA r=16/alpha=16, full_training_modules 让新本体的投影层全量训练。
#     LoRA 的 ckpt 里是 adapter_model.safetensors, 部署时必须走 serve.py 的 PEFT 分支
#     （上游 from_pretrained 只找 model.safetensors, 找不到会静默返回随机模型）。
exec "$LEROBOT_BIN" \
  --tolerance_s=1e-3 \
  --policy.path="$WEIGHTS" \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_decay_lr=1e-5 \
  --rename_map='{"observation.images.rgb":"observation.images.base_0_rgb","observation.images.ultrasound":"observation.images.left_wrist_0_rgb"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16 \
  --peft.full_training_modules='["state_proj","action_in_proj","action_out_proj","action_time_mlp_in","action_time_mlp_out"]' \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --log_freq=10 \
  --save_freq="$SAVE_FREQ" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=false
