#!/usr/bin/env python3
"""从 LeRobot checkpoint 和训练日志导出有限字段的实验快照，不运行训练。"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # 不用 last：它可能指向旧恢复点。优先选最大的数字 checkpoint。
    configs = list(args.run.glob('checkpoints/[0-9]*/pretrained_model/train_config.json'))
    configs = [p for p in configs if p.parent.parent.name.isdigit()]
    if not configs:
        parser.error('没有找到数字 checkpoint 中的 train_config.json')
    config = max(configs, key=lambda p: int(p.parent.parent.name))
    data = json.loads(config.read_text())
    policy = data.get('policy', {})
    keys = ['type', 'chunk_size', 'n_action_steps', 'dtype', 'use_amp',
            'gradient_checkpointing', 'freeze_vision_encoder', 'train_expert_only',
            'train_state_proj', 'pretrained_backbone_weights', 'normalization_mapping']
    source = policy.get('pretrained_path') or ''
    # 只导出底座 revision，避免将机器上的完整缓存路径写入仓库。
    revision = source.split('/snapshots/', 1)[-1] if '/snapshots/' in source else policy.get('pretrained_revision')
    result = {
        'experiment_id': args.id,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_name': args.run.name,
        'config_source': str(config.relative_to(args.run)),
        'config_sha256': digest(config),
        'checkpoint_step': int(config.parent.parent.name),
        'configured_steps': data.get('steps'),
        'batch_size': data.get('batch_size'),
        'num_workers': data.get('num_workers'),
        'seed': data.get('seed'),
        'dataset_repo_id': data.get('dataset', {}).get('repo_id'),
        'base_revision': revision,
        'policy': {k: policy[k] for k in keys if k in policy},
        'optimizer': data.get('optimizer'),
        'scheduler': data.get('scheduler'),
        'peft': data.get('peft'),
        'rename_map': data.get('rename_map'),
        'status': 'checkpoint_available_log_not_provided',
    }
    adapter = config.parent / 'adapter_config.json'
    if adapter.exists():
        a = json.loads(adapter.read_text())
        result['adapter'] = {k: a.get(k) for k in
                             ['peft_type', 'r', 'lora_alpha', 'lora_dropout', 'target_modules', 'modules_to_save']}
    if args.log:
        # 读取单次快照；运行中的日志可能继续增长，所有结果仅代表读取时刻。
        raw = args.log.read_bytes()
        content = raw.decode('utf-8', errors='replace')
        progress = re.findall(r'(\d+)/(\d+)\s+\[([^<\]]+)', content)
        memory = [float(v) for v in re.findall(r'mem_gb:([\d.]+)', content)]
        metrics = re.findall(r'loss:([\d.eE+-]+).*?updt_s:([\d.]+).*?mem_gb:([\d.]+)', content)
        result['log_source'] = args.log.name
        result['log_sha256'] = hashlib.sha256(raw).hexdigest()
        ended = 'End of training' in content
        failed = 'Traceback (most recent call last)' in content
        result['status'] = 'ended_marker_present' if ended else ('error_marker_present' if failed else 'no_end_marker')
        if progress:
            step, total, elapsed = progress[-1]
            result['last_log_progress'] = {'step': int(step), 'total': int(total), 'elapsed': elapsed.strip()}
        result['max_logged_cuda_allocated_gib'] = max(memory) if memory else None
        result['memory_scope'] = 'LeRobot mem_gb: torch.cuda.max_memory_allocated / 1024**3; 非整卡占用'
        if metrics:
            result['last_logged_metrics'] = dict(zip(['loss', 'update_seconds', 'cuda_allocated_gib'], map(float, metrics[-1])))
        for label in ['num_learnable_params', 'num_total_params']:
            matches = re.findall(label + r'=(\d+)', content)
            result[label] = int(matches[-1]) if matches else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
