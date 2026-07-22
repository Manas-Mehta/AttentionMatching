#!/usr/bin/env python3
"""
Confirm the model and all RULER configs are cached, loadable OFFLINE, and correct.
Run on the login node AFTER slurm/prestage.sh (or anywhere the cache is visible).

    python experiments/verify_assets.py

Forces HF_HUB_OFFLINE=1 so it proves a compute node will succeed.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", f"/scratch/{os.environ.get('USER','')}/hf_cache")

import sys
from collections import Counter

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CONFIGS = ["4096", "8192", "16384"]
EXPECTED_TASKS = {
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
}
# task -> (start row, expected answer style) for the 4096 config, from earlier probing
BLOCKS_4K = [
    (0, "niah_single_1"), (500, "niah_multikey_2"), (1000, "fwe"), (1500, "qa_1"),
    (2000, "niah_single_3"), (2500, "niah_multikey_1"), (3000, "niah_single_2"),
    (3500, "cwe"), (4000, "qa_2"), (4500, "vt"), (5000, "niah_multivalue"),
    (5500, "niah_multikey_3"), (6000, "niah_multiquery"),
]

ok = True

def check(cond, msg):
    global ok
    print(("  OK  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        ok = False

print(f"HF_HOME = {os.environ['HF_HOME']}")
print(f"HF_HUB_OFFLINE = {os.environ['HF_HUB_OFFLINE']}\n")

# ---- model ----
print(f"[model] {MODEL}")
try:
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    check(cfg.num_hidden_layers == 36, f"num_hidden_layers = {cfg.num_hidden_layers} (want 36)")
    check(cfg.num_attention_heads == 32, f"num_attention_heads = {cfg.num_attention_heads} (want 32)")
    check(cfg.num_key_value_heads == 8, f"num_key_value_heads = {cfg.num_key_value_heads} (want 8)")
    check(cfg.max_position_embeddings >= 16384,
          f"max_position_embeddings = {cfg.max_position_embeddings} (>=16384 so 16k needs no YaRN)")
    check(tok is not None, "tokenizer loads offline")
    # weight shards present
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, local_files_only=True)
    shards = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    check(len(shards) >= 1, f"{len(shards)} safetensors shard(s) on disk at {path}")
except Exception as e:
    check(False, f"model load raised {type(e).__name__}: {e}")

# ---- datasets ----
print("\n[datasets] simonjegou/ruler")
try:
    from datasets import load_dataset
    for c in CONFIGS:
        ds = load_dataset("simonjegou/ruler", c, split="test")
        tasks = Counter(ds["task"])
        n = len(ds)
        check(n == 6500, f"config {c}: {n} rows (want 6500)")
        check(set(tasks) == EXPECTED_TASKS,
              f"config {c}: {len(tasks)} tasks {'== expected 13' if set(tasks)==EXPECTED_TASKS else 'MISMATCH '+str(set(tasks)^EXPECTED_TASKS)}")
        check(all(v == 500 for v in tasks.values()),
              f"config {c}: every task has 500 rows ({sorted(set(tasks.values()))})")
    # block layout only guaranteed for 4096
    ds4 = load_dataset("simonjegou/ruler", "4096", split="test")
    good = all(ds4[start]["task"] == name for start, name in BLOCKS_4K)
    check(good, "config 4096: task block layout matches expected offsets")
    # spot-check the UUID tasks really contain UUIDs
    import re
    uu = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    nm3 = ds4[5500]
    ns1 = ds4[0]
    check(bool(uu.search(nm3["answer"][0])), f"nm3 answer is a UUID: {nm3['answer'][0]}")
    check(not uu.search(ns1["answer"][0]), f"ns1 answer is NOT a UUID: {ns1['answer'][0]}")
except Exception as e:
    check(False, f"dataset load raised {type(e).__name__}: {e}")

print("\n" + ("ALL CHECKS PASSED — safe to sbatch." if ok else "SOME CHECKS FAILED — see above."))
sys.exit(0 if ok else 1)
