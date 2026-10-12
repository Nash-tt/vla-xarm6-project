#!/usr/bin/env bash
# π0 (LoRA) 微调 —— xArm6 超声扫描 (TCP 6D 位姿控制)
#
# 用法:
#   WEIGHTS=<pi0_base snapshot 目录> ./run_train.sh              # 正式训练
#   SMOKE=1    ./run_train.sh                                    # 冒烟测试 (20 步, bs=1)
#
# 可覆盖的环境变量: LEROBOT_BIN / WEIGHTS / STEPS / BATCH_SIZE / NUM_WORKERS / SAVE_FREQ / OUTPUT_DIR / JOB_NAME
set -euo pipefail

# 默认指向 conda 的 lerobot 环境 (2026-10-11 环境事故后重建的那个)。
# 用绝对路径而不是"当前激活的环境" —— 之前就是因为依赖激活状态, 装 openpi 时
# 把依赖灌进了 lerobot 环境, 毁了 torch/transformers/lerobot 三件套。
LEROBOT_BIN="${LEROBOT_BIN:-/opt/conda/envs/lerobot/bin/lerobot-train}"

# roll 已解卷绕的副本 (见 unwrap_roll.py / recompute_stats.py)
DATASET_ROOT="${DATASET_ROOT:-/data/vla/vla_batch_20261007_pi0_6d_rollunwrap}"
DATASET_REPO_ID=local/vla_batch_20261007_pi0_6d_rollunwrap

# pi0_base 的本地 snapshot 路径（HF 缓存里的快照目录）
WEIGHTS="${WEIGHTS:-lerobot/pi0_base}"

# 冒烟测试把一切压到最小，只验证「显存 + 流程」跑得通
if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS="${STEPS:-20}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-1000}"
  OUTPUT_DIR="${OUTPUT_DIR:-pi0_ultrasound_lora/smoke}"
  JOB_NAME="${JOB_NAME:-pi0_ultrasound_smoke}"
else
  STEPS="${STEPS:-24500}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  NUM_WORKERS="${NUM_WORKERS:-8}"
  SAVE_FREQ="${SAVE_FREQ:-500}"
  OUTPUT_DIR="${OUTPUT_DIR:-pi0_ultrasound_lora/run1}"
  JOB_NAME="${JOB_NAME:-pi0_ultrasound_lora}"
fi

# 显存碎片整理；代理必须去掉（ALL_PROXY 的 socks:// scheme 会让 httpx 直接报错）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1

[[ -d "$WEIGHTS" ]] || { echo "找不到 pi0_base 权重目录: $WEIGHTS" >&2; exit 1; }
[[ -d "$DATASET_ROOT" ]] || { echo "找不到数据集: $DATASET_ROOT" >&2; exit 1; }
# 环境坏了的话这条会先炸, 而不是等 exec 时报一句看不懂的错
[[ -x "$LEROBOT_BIN" ]] || { echo "找不到 lerobot-train: $LEROBOT_BIN" >&2; exit 1; }

echo "policy    : pi0 + LoRA (r=16, alpha=16)"
echo "weights   : $WEIGHTS"
echo "dataset   : $DATASET_ROOT"
echo "steps     : $STEPS   batch_size: $BATCH_SIZE   workers: $NUM_WORKERS"
echo "output    : $OUTPUT_DIR"
echo

# 说明:
#   optimizer_lr / scheduler_decay_lr  —— LoRA 惯例, 相比全量微调放大 ~4-10x (pi0 全量默认 2.5e-5)
#   rename_map                         —— 两台相机映射到 pi0_base 声明的前两个图像槽
#
#   【不要传 --policy.input_features/--policy.output_features/--policy.empty_cameras】
#   pi0_base 的 config.json 已经声明好了: 3 个图像槽 (base_0_rgb / left_wrist_0_rgb /
#   right_wrist_0_rgb) + state(32) + action(32)。而 make_policy 里是这么写的:
#       if not cfg.input_features:
#           cfg.input_features = {从数据集推导}
#   一旦传 null 把 input_features 清空, 就会用数据集的名字 (observation.images.rgb 等)
#   覆盖掉 pi0 的槽位名, 于是 _preprocess_images 在 batch 里一个图都找不到, 直接报
#   "All image features are missing from the batch"。(实测踩过)
#
#   第三个图像槽不需要 empty_cameras: _preprocess_images 会把 batch 中缺失的图像键
#   自动补成 mask=0 的占位图 (SigLIP 用 -1 填充), right_wrist_0_rgb 天然被 mask 掉。
#   state/action 的 6 维会在模型内部 pad 到 max_state_dim=32, 也不需要干预。
#
#   peft.*                             —— 默认只挂 action expert 的 q/v_proj + 投影层;
#                                         full_training_modules 让新本体的投影层全量训练
#
#   tolerance_s=1e-3                   —— 绕开 LeRobot 的 float32 边界 bug。
#     默认 1e-4, step 455 时崩在:
#       FrameTimestampError: (tensor([0.0001]) > tolerance_s=0.0001)
#       video: videos/observation.images.ultrasound/chunk-000/file-000.mp4
#     但数据本身没问题 —— 用 check_timestamps.py 扫全量: 真实偏差最大 7.1e-6 s。
#     真正的原因是精度: decode_video_frames_pyav 里 `torch.tensor(timestamps)` 把
#     float64 时间戳降成 float32, 而 float32 在 t≈1277s 处的 ULP = 1.22e-4 > 1e-4。
#     两个本该差 5e-6 的数被量化到相邻浮点格点, 算出来正好是 1 个 ULP。
#     判定还是严格小于 (`is_within_tol = min_ < tolerance_s`), 所以恰好卡死。
#     全量 49053 帧里有 7 帧踩中 (0.014%), 都在这条 3270s 长的超声视频上。
#     RGB 拆成 10 个文件、每个最长 366s, ULP 只有 3.05e-5, 所以从不触发。
#     1e-3 是实测最大偏差的 8 倍, 又只占一帧间隔 (66.7ms) 的 1.5%, 不可能选错帧。
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
