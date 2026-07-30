#!/usr/bin/env python3
"""
Build the key-length sweep dataset for the "UUID Bit" experiment.

Isolation design: take the REAL ruler niah_single_3 4k documents (simonjegou/ruler,
config 4096) and change ONLY the key string. Preamble, essay haystack, needle
sentence, keyword, question, and answer_prefix stay byte-identical -- the single
independent variable across the sweep is the number of random bits in the key.

A random lowercase hex char is exactly 4 bits (uniform over 0-9a-f), so an N-bit
key is N/4 hex chars. The 122-bit UUID anchor is NOT regenerated here (reuse the
existing niah_single_3 runs).

Output: official/data/keylen/keylen_{bits}.jsonl  (one row per document), with the
same schema RULER rows use, so the loader's keylen branch returns them unchanged.

Data source, in order of preference:
  1. local HuggingFace cache  (works on the cluster; HF_DATASETS_OFFLINE=1)
  2. HuggingFace datasets-server HTTP API  (works on a laptop with internet)
Both return simonjegou/ruler rows in the same order, so the first N niah_single_3
rows are the same 50 documents the eval runner uses (--start-article 0).
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

HEX = "0123456789abcdef"
DATASET = "simonjegou/ruler"
CONFIG = "4096"
TASK = "niah_single_3"


def load_ns3_rows(n):
    """First n niah_single_3 rows, HF cache if available else datasets-server API."""
    try:
        from datasets import load_dataset  # cluster path (cached, offline-ok)
        ds = load_dataset(DATASET, CONFIG, split="test")
        rows = [r for r in ds if r["task"] == TASK][:n]
        if len(rows) >= n:
            print(f"[source] local HuggingFace cache: {len(rows)} rows")
            return rows
    except Exception as e:
        print(f"[source] HF datasets lib unavailable ({e.__class__.__name__}); using HTTP API")

    rows, off = [], 0
    while len(rows) < n:
        want = min(50, n - len(rows))
        url = (
            "https://datasets-server.huggingface.co/filter"
            f"?dataset={DATASET}&config={CONFIG}&split=test"
            "&where=%22task%22%3D%27" + TASK + "%27"
            f"&offset={off}&length={want}"
        )
        with urllib.request.urlopen(url) as fh:
            d = json.load(fh)
        batch = [r["row"] for r in d["rows"]]
        if not batch:
            break
        rows += batch
        off += len(batch)
    print(f"[source] datasets-server HTTP API: {len(rows)} rows")
    return rows[:n]


def make_key(rng, bits):
    return "".join(rng.choice(HEX) for _ in range(bits // 4))


def fmt_uuid_dashes(hexstr):
    """32 hex chars -> canonical 8-4-4-4-12 UUID layout (dashes carry zero entropy)."""
    if len(hexstr) != 32:
        sys.exit(f"--dash needs 128-bit (32-hex) keys, got {len(hexstr)} hex chars")
    return f"{hexstr[:8]}-{hexstr[8:12]}-{hexstr[12:16]}-{hexstr[16:20]}-{hexstr[20:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="documents per bit-length")
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 32, 64, 96])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dash", action="store_true",
                    help="format keys as dashed UUIDs (8-4-4-4-12); requires --bits 128. "
                         "Same seed+bits as the bare run reuses the identical keys, so bare-vs-dash "
                         "isolates the dash/tokenization effect alone. Writes keylen_{bits}dash.jsonl.")
    ap.add_argument("--outdir", default=None, help="default: <repo>/official/data/keylen")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir) if args.outdir else repo / "official" / "data" / "keylen"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_ns3_rows(args.n)
    if len(rows) < args.n:
        sys.exit(f"ERROR: only got {len(rows)} rows, needed {args.n}")

    for bits in args.bits:
        if bits % 4 != 0:
            sys.exit(f"ERROR: bits must be a multiple of 4 (got {bits})")
        rng = random.Random(args.seed * 10000 + bits)  # independent per length, reproducible
        suffix = "dash" if args.dash else ""
        out = []
        for r in rows:
            uuid = r["answer"][0]
            ctx = r["context"]
            if ctx.count(uuid) != 1:
                sys.exit(f"ERROR: expected UUID exactly once, found {ctx.count(uuid)}")
            key = make_key(rng, bits)                       # bare hex (same draw for bare & --dash)
            if args.dash:
                key = fmt_uuid_dashes(key)                  # dashes only reformat; entropy unchanged
            new_ctx = ctx.replace(uuid, key)
            assert key in new_ctx and uuid not in new_ctx
            out.append({
                "context": new_ctx,
                "task": f"keylen_{bits}{suffix}",
                "question": r["question"],       # keyed by the word -> unchanged
                "answer": [key],
                "answer_prefix": r["answer_prefix"],
                "max_new_tokens": r["max_new_tokens"],
            })
        fp = outdir / f"keylen_{bits}{suffix}.jsonl"
        with open(fp, "w") as fh:
            for o in out:
                fh.write(json.dumps(o) + "\n")
        label = f"{bits} bits, dashed UUID layout" if args.dash else f"{bits} bits = {bits//4} hex chars"
        print(f"wrote {len(out):3d} rows -> {fp}   ({label})")


if __name__ == "__main__":
    main()
