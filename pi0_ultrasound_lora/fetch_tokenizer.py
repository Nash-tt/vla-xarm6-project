#!/usr/bin/env python
"""预下载 pi0 需要的 PaliGemma tokenizer, 让训练可以全程离线跑。

pi0_base 的 checkpoint 里不带 tokenizer 文件, 但它的
policy_preprocessor.json 声明了 tokenizer_name="google/paligemma-3b-pt-224"。
该仓库是 gated 的, 需要: 1) 已登录 HF  2) 已在网页上接受 Gemma 许可。

只拉 tokenizer 相关的几个小文件, 不下 3B 权重。

用法: python fetch_tokenizer.py
"""

import os
import sys

# ALL_PROXY 的 socks:// scheme httpx 不认, 会直接报 Unknown scheme;
# 但 HTTP(S)_PROXY 必须保留, 去掉就只能裸连 (直连 HF 不通)。只删 ALL_PROXY。
for var in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(var, None)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet 协议在本机代理下会静默卡死

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

REPO = "google/paligemma-3b-pt-224"
PATTERNS = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
]

try:
    who = HfApi().whoami()
    print(f"已登录: {who.get('name')} ({who.get('email', 'n/a')})")
except Exception as e:
    print(f"HF 未登录或 token 失效: {e}", file=sys.stderr)
    print("先跑 `hf auth login --force`", file=sys.stderr)
    raise SystemExit(1)

try:
    path = snapshot_download(REPO, allow_patterns=PATTERNS)
except Exception as e:
    print(f"下载失败: {type(e).__name__}: {e}", file=sys.stderr)
    print("多半是还没在 https://huggingface.co/google/paligemma-3b-pt-224 接受 Gemma 许可", file=sys.stderr)
    raise SystemExit(1)

print(f"\n已缓存到: {path}")
for f in sorted(os.listdir(path)):
    print(f"  {f}  ({os.path.getsize(os.path.join(path, f)) / 1e6:.1f} MB)")

# 真加载一次, 确认离线可用
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(REPO)
print(f"\n离线加载 OK, vocab_size={tok.vocab_size}")

# 顺便量一下数据集里当前的任务描述占多少 token (max_length=48)
# 用法: python fetch_tokenizer.py [dataset_root]
if len(sys.argv) > 1:
    import pyarrow.parquet as pq

    df = pq.read_table(os.path.join(sys.argv[1], "meta/tasks.parquet")).to_pandas()
    for task in df.index:
        n = len(tok(task)["input_ids"])
        flag = "  ← 会被截断!" if n > 48 else ""
        print(f"  {n:3d} token: {task!r}{flag}")
