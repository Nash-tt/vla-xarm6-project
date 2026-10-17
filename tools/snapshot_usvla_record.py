"""从 OpenPI/US-VLA 实验产物导出白名单摘要，不复制路径、原始日志或凭据。"""
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def evidence(path):
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def snapshot(root, experiment_id):
    root = Path(root)
    runs = []
    for name in ("official_6000d_smoke_bs8_E01", "official_6000d_public_E01",
                 "official_6000d_smoke_bs16_E01"):
        directory = root / name
        record = load(directory / "resolved_config.json")
        cfg = record["config"]
        result = load(directory / "result.json")
        fields = ("status", "completed_steps", "last_logged_progress_steps", "resumable_checkpoint",
                  "elapsed_seconds_including_initialization_and_checkpoint", "finished_at", "stopped_at",
                  "sampled_peak_memory_MiB", "smoke_checkpoint_removed_for_disk_space")
        item = {
            "run_name": name,
            "started_at": record["started_at"],
            "configuration": {k: cfg[k] for k in ("batch_size", "num_train_steps", "seed", "num_workers",
                "model", "optimizer", "lr_schedule", "ema_decay", "save_interval", "keep_period", "log_interval")},
            "result": {k: result[k] for k in fields if k in result},
            "packages": record["packages"],
            "source_sha256": record["source_sha256"],
            "evidence": [evidence(directory / file) for file in ("resolved_config.json", "result.json", "train.log")],
        }
        summary = directory / "throughput_summary.json"
        if summary.exists():
            item["throughput"] = load(summary)
            item["evidence"].append(evidence(summary))
        runs.append(item)
    profiles = []
    for bs in (8, 16):
        path = root / f"memory_bs{bs}_E01" / "memory_report.json"
        raw = load(path)
        keys = ("batch_size", "status", "parameter_bytes_by_module", "trainable_parameter_bytes",
                "frozen_parameter_bytes", "optimizer_state_bytes", "total_train_state_bytes", "batch_bytes",
                "ema_bytes", "compiled_memory_analysis", "synchronized_step_seconds", "last_metrics")
        profile = {k: raw[k] for k in keys if k in raw}
        profile["allocator_after_step5"] = raw["snapshots"]["step_5"]["device_memory_stats"]
        profile["live_array_bytes_after_step5"] = raw["snapshots"]["step_5"]["live_jax_array_bytes"]
        profile["evidence"] = evidence(path)
        profiles.append(profile)
    return {
        "experiment_id": experiment_id,
        "scope": "官方公开数据适配、冒烟训练、吞吐对比与内存剖析；未完成正式训练和独立评估",
        "upstream_repository": "https://github.com/VMVLab/US-VLA",
        "upstream_commit": "dde576ae9992d33231f3de925ff44a1fc818ab00",
        "local_changes": "USFM 权重映射与结构兼容修正；实际训练源码哈希见各 run",
        "dataset": {"id": "usvla/US", "revision": "afa73d512b7a980b55e84536b47ae08c059b0697",
                    "episodes": 120, "frames": 29665, "fps": 30, "independent_validation": False},
        "gpu": {"name": "NVIDIA RTX 6000D", "memory_total_MiB": 85651, "driver": "580.119.02"},
        "runs": runs,
        "memory_profiles": profiles,
        "limits": ["最后日志进度不代表精确终止更新数", "未保存的正式训练不可恢复",
                   "nvidia-smi 进程占用、allocator 缓冲峰值和显存池容量不是同一指标",
                   "未在 RTX 5090 实测；未证明完整训练最低显存", "无独立评估或论文指标复现结论"],
    }
