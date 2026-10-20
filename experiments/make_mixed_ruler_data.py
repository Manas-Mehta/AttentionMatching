#!/usr/bin/env python3
"""
Combined-task documents built on RULER's own documents (mixedr_<len>.jsonl).

Same idea as make_mixed_data.py (several task types in one document, one compacted cache,
every type scored separately with OUR exact-set scoring), but every piece that RULER has is
taken from RULER (simonjegou/ruler, the copy the AM paper's RULER runs use):

  haystack     RULER's own niah_single_3 context at this length, row i, with RULER's needle
               sentence removed: the Paul Graham essay text in RULER's order and spacing
  instruction  RULER's niah template (plural form) + RULER's vt opening sentence
  uuid         RULER needle sentence, RULER key (harvested from RULER's niah questions),
               RULER question ("... mentioned in the provided text?")
  multikey     as niah_multikey_1: 4 needles "One of the special magic numbers for K is: N.",
               7-digit values, RULER keys, RULER question; 2 of the 4 keys are asked
  chain        as vt: 1 chain, 4 hops (5 variables), 5 random capital letters per name,
               5-digit value, RULER's line text "VAR B = VAR A", inserted inline as in RULER,
               lines in order; RULER's question ("Find all variables that are assigned the
               value V in the text above") + ", directly or through a chain of assignments."
               (RULER shows this with a worked example, which we leave out), plus our chain_last
               and chain_hop questions
  cwe          RULER's header sentence and inline list format ("1. w 2. w ..."), words from
               RULER's cwe vocabulary, RULER's question wording for the top 5

Insertion: each block goes in at a random sentence break of the haystack (". " before a
capital letter), the way RULER inserts its needles; the cwe block is its own paragraph. The
haystack is cut from the end at a sentence break so the document is at most as long as
RULER's original document for that row (RULER also truncates its haystack to fit).

Kept from our 4k design, not RULER: the cwe list sizes (5 common words x 8, 20-60 rare
words x 2; a full RULER cwe list fills a whole document), and the extra chain questions.
Left out: RULER's worked examples (vt and cwe documents open with a solved example task),
and the 4k set's SimpleLogic block.

Determinism: document i draws its blocks from random.Random(seed * 1000003 + i) before
anything length-dependent, so the 4k and 8k sets get the same keys, values, names and words
for the same i; only the haystack and the positions differ.

Scoring fields are those of the 4k combined task (score_candidates = the document's own
strings of that kind; exact set match in qa_evaluator.chain_score_prediction).

Run on the torch login node (needs the cached simonjegou/ruler and the Qwen tokenizer):
    HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        python experiments/make_mixed_ruler_data.py --length 8192 --n 100
"""
import argparse
import json
import random
import re
import string
import sys
import uuid as uuidlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_mixed_data import Counter, seven_digit, QWEN  # noqa: E402

# RULER template text, as rendered in simonjegou/ruler (checked against the cached rows).
RULER_NIAH_PLURAL = ("Some special magic {v} are hidden within the following text. Make sure to "
                     "memorize it. I will quiz you about the {v} afterwards.")
RULER_VT_OPEN = "Memorize and track the chain(s) of variable assignment hidden in the following text."
RULER_CWE_HEADER = ("Below is a numbered list of words. In these words, some appear more often "
                    "than others. Memorize the ones that appear most often.")
INSTRUCTION = RULER_NIAH_PLURAL.format(v="uuids and numbers") + " " + RULER_VT_OPEN
NEEDLE_UUID = "One of the special magic uuids for {k} is: {v}."
NEEDLE_NUM = "One of the special magic numbers for {k} is: {v}."
Q_UUID = "What is the special magic uuid for {k} mentioned in the provided text?"
Q_NUM = "What is the special magic number for {k} mentioned in the provided text?"
# RULER's vt question relies on its worked example to show that the whole chain counts; without
# the example the model answers only the directly assigned variable (smoke, 1x, doc 0). The clause
# after the comma is the disambiguation our chain grids use (make_chain_data.make_question "all").
Q_VT = ("Question: Find all variables that are assigned the value {v} in the text above, directly "
        "or through a chain of assignments.")
Q_CWE = "Question: What are the {n} most common words in the above list?"
# ours (kept for continuity with the 4k combined task)
Q_LAST = ("The text above contains 1 chain(s) of variable assignments. Each chain begins with a "
          "variable assigned the value {v}, and each chain contains {n} variables in total. For "
          "each chain, report the LAST variable in that chain.")
Q_HOP = 'The text contains exactly one statement of the form "VAR X = VAR {a}". What is X?'
CHAIN_FORMAT = "Output only the variable name or names, comma separated, with no explanation and nothing else."

RULER_NEEDLE_RE = re.compile(r" One of the special magic uuids for [\w-]+ is: [0-9a-f-]{36}\.")
BREAK_RE = re.compile(r"(?<=[.!?]) (?=[A-Z\[])")
WORD_RE = re.compile(r"[a-z][a-z'-]*")


def ruler_rows(length, task):
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", str(length), split="test")
    return [r for r in ds if r["task"] == task]


def harvest(lengths):
    """RULER's own keys (niah questions) and cwe words, over every cached length."""
    keys, words = set(), set()
    for L in lengths:
        try:
            from datasets import load_dataset
            ds = load_dataset("simonjegou/ruler", str(L), split="test")
        except Exception as e:  # a length that is not cached
            print(f"[harvest] {L}: not available ({str(e)[:60]})")
            continue
        for r in ds:
            if r["task"].startswith("niah"):
                m = re.search(r"for ([a-z]+-[a-z]+) mentioned", r["question"])
                if m:
                    keys.add(m.group(1))
            elif r["task"] == "cwe":
                words.update(re.findall(r"\d+\. ([a-z]+)(?=\s|$)", r["context"]))
    return sorted(keys), sorted(words)


def strip_ruler(row):
    """RULER ns3 context -> (haystack body, RULER's removed needle)."""
    first, body = row["context"].split("\n", 1)
    assert first.startswith("A special magic uuid is hidden"), first[:80]
    found = RULER_NEEDLE_RE.findall(body)
    assert len(found) == 1, f"expected one RULER needle, got {len(found)}"
    body = body.replace(found[0], "", 1)
    assert "special magic" not in body
    return body.rstrip("\n"), found[0]


def draw_blocks(rng, keys_pool, words_pool, banned, args):
    """Every block's strings for one document, drawn before anything length-dependent."""
    keys = rng.sample(keys_pool, 1 + args.nm_pairs)
    numbers = set()
    u = str(uuidlib.UUID(int=rng.getrandbits(128), version=4))
    nm_vals = [seven_digit(rng, numbers) for _ in range(args.nm_pairs)]
    names = []
    while len(names) < args.hops + 1:
        nm = "".join(rng.choices(string.ascii_uppercase, k=5))
        if nm.lower() not in banned and nm not in names:
            names.append(nm)
    value = str(rng.randint(10000, 99999))
    n_rare = rng.randint(args.cwe_rare_min, args.cwe_rare_max)
    own = {w for k in keys for w in k.split("-")}          # no list word inside this doc's keys
    cw = rng.sample([w for w in words_pool if w not in own], args.cwe_common + n_rare)
    items = cw[:args.cwe_common] * args.cwe_freq_common + cw[args.cwe_common:] * args.cwe_freq_rare
    rng.shuffle(items)
    asked = rng.sample(range(args.nm_pairs), args.nm_asked)
    h = rng.randrange(args.hops)
    if h == args.hops - 1:       # never the final link (as in make_mixed_data.py)
        h = int(value) % (args.hops - 1)
    return dict(uuid_key=keys[0], uuid=u, nm_keys=keys[1:], nm_vals=nm_vals, asked=asked,
                names=names, value=value, cwe_common=cw[:args.cwe_common], cwe_all=cw,
                cwe_items=items, hop=h)


def block_texts(b):
    """(name, text, kind) per inserted unit; kind 'inline' = sentence, 'para' = own paragraph."""
    out = [("uuid", NEEDLE_UUID.format(k=b["uuid_key"], v=b["uuid"]), "inline")]
    for j, (k, v) in enumerate(zip(b["nm_keys"], b["nm_vals"])):
        out.append((f"multikey{j}", NEEDLE_NUM.format(k=k, v=v), "inline"))
    n = b["names"]
    chain = [f"VAR {n[0]} = {b['value']}"] + [f"VAR {n[j + 1]} = VAR {n[j]}" for j in range(len(n) - 1)]
    for j, ln in enumerate(chain):
        out.append((f"chain{j}", ln, "inline"))
    lst = " ".join(f"{i + 1}. {w}" for i, w in enumerate(b["cwe_items"]))
    out.append(("cwe", RULER_CWE_HEADER + "\n" + lst, "para"))
    return out


def assemble(body, breaks_used, units, pos_rng):
    """Insert units at sentence breaks of `body` (breaks_used = break offsets available).
    Chain lines keep their order. Returns the document body."""
    k = len(units)
    slots = sorted(pos_rng.sample(range(len(breaks_used)), k))
    order = list(range(k))
    pos_rng.shuffle(order)
    chain_ix = [i for i, u in enumerate(units) if u[0].startswith("chain")]
    chain_slots = sorted(order[i] for i in chain_ix)
    for i, s in zip(chain_ix, chain_slots):      # chain lines in reading order
        order[i] = s
    at = {breaks_used[slots[order[i]]]: units[i] for i in range(k)}
    pieces, last = [], 0
    for off in sorted(at):
        name, text, kind = at[off]
        pieces.append(body[last:off])
        pieces.append(" " + text if kind == "inline" else "\n" + text + "\n")
        last = off + (1 if kind == "para" else 0)   # a paragraph replaces the break's space
    pieces.append(body[last:])
    return "".join(pieces)


def build_doc(i, row, b, counter, args):
    body, ruler_needle = strip_ruler(row)
    target = counter(row["context"])               # RULER's own length for this row
    units = block_texts(b)
    breaks = [m.start() for m in BREAK_RE.finditer(body)]
    pos_seed = random.Random(args.seed * 1000003 + i).getrandbits(32) ^ args.length
    # keep the longest haystack prefix (cut at a sentence break) that fits RULER's length
    lo, hi, best = len(units) + 1, len(breaks), None
    while lo <= hi:
        mid = (lo + hi) // 2
        cut = breaks[mid - 1] + 1                     # keep text up to that break
        avail = [o for o in breaks[:mid - 1]]
        doc = INSTRUCTION + "\n" + assemble(body[:cut].rstrip(" "), avail, units,
                                            random.Random(pos_seed)) + "\n"
        if counter(doc) <= target:
            best, lo = (doc, mid), mid + 1
        else:
            hi = mid - 1
    if best is None:
        sys.exit(f"ERROR: doc {i}: blocks do not fit in {target} tokens")
    return best[0], target, ruler_needle


def questions(task, i, b, args):
    n = b["names"]
    qs = [dict(qtype="uuid", question=Q_UUID.format(k=b["uuid_key"]), answer=[b["uuid"]],
               score_candidates=[b["uuid"]], max_new_tokens=64)]
    for j in b["asked"]:
        qs.append(dict(qtype="multikey", question=Q_NUM.format(k=b["nm_keys"][j]),
                       answer=[b["nm_vals"][j]], score_candidates=list(b["nm_vals"]), max_new_tokens=24))
    qs.append(dict(qtype="chain_all", question=Q_VT.format(v=b["value"]), answer=list(n),
                   score_candidates=list(n), answer_format=CHAIN_FORMAT, max_new_tokens=48,
                   depth=len(n)))
    qs.append(dict(qtype="chain_last", question=Q_LAST.format(v=b["value"], n=len(n)), answer=[n[-1]],
                   score_candidates=list(n), max_new_tokens=24, depth=len(n)))
    h = b["hop"]
    qs.append(dict(qtype="chain_hop", question=Q_HOP.format(a=n[h]), answer=[n[h + 1]],
                   score_candidates=[x for x in n if x != n[h]], max_new_tokens=24, depth=1))
    qs.append(dict(qtype="cwe", question=Q_CWE.format(n=args.cwe_common), answer=list(b["cwe_common"]),
                   score_candidates=list(b["cwe_all"]), max_new_tokens=48))
    for qi, q in enumerate(qs):
        q["qid"] = f"{task}_{i}_q{qi}"
    return qs


def spans(doc, b, counter):
    blocks = {}
    for name, text, kind in block_texts(b):
        s = doc.index(text)
        e = s + len(text)
        ts, te = counter(doc[:s]), counter(doc[:e])
        blocks[name] = {"char_start": s, "char_end": e, "tok_start": ts, "tok_end": te, "tokens": te - ts}
    ch = [blocks.pop(f"chain{j}") for j in range(len(b["names"]))]
    blocks["chain"] = {"line_tok_start": [c["tok_start"] for c in ch],
                       "line_char_start": [c["char_start"] for c in ch],
                       "tokens": sum(c["tokens"] for c in ch)}
    n_tok = counter(doc)
    for v in blocks.values():
        v["share"] = v["tokens"] / n_tok
    return blocks, n_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=8192, help="RULER config: 4096, 8192, 16384, ...")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hops", type=int, default=4)            # RULER vt: num_hops 4 -> 5 variables
    ap.add_argument("--nm-pairs", type=int, default=4)        # RULER niah_multikey_1: 4 keys
    ap.add_argument("--nm-asked", type=int, default=2)
    ap.add_argument("--cwe-common", type=int, default=5)
    ap.add_argument("--cwe-freq-common", type=int, default=8)
    ap.add_argument("--cwe-rare-min", type=int, default=20)
    ap.add_argument("--cwe-rare-max", type=int, default=60)
    ap.add_argument("--cwe-freq-rare", type=int, default=2)
    ap.add_argument("--harvest-lengths", default="4096,8192,16384")
    ap.add_argument("--tokenizer", default=QWEN)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--task", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir) if args.outdir else repo / "official" / "data" / "mixed"
    outdir.mkdir(parents=True, exist_ok=True)
    task = args.task or f"mixedr_{args.length // 1024}k"
    counter = Counter(args.tokenizer, None)

    rows = ruler_rows(args.length, "niah_single_3")
    if len(rows) < args.n:
        sys.exit(f"ERROR: only {len(rows)} niah_single_3 rows at {args.length}")
    keys_pool, words_pool = harvest([int(x) for x in args.harvest_lengths.split(",")])
    # Strings that must not occur in the essay text. The filter uses the haystacks of EVERY
    # harvest length (not just this one), so the pools, and therefore every document's draws,
    # are identical whichever length is being built.
    hay = " ".join(strip_ruler(r)[0] for L in args.harvest_lengths.split(",")
                   for r in ruler_rows(int(L), "niah_single_3")[:args.n]).lower()
    hay_words = set(WORD_RE.findall(hay))
    # ...and every word of our own template sentences (instruction, cwe header, questions)
    templates = " ".join([INSTRUCTION, RULER_CWE_HEADER, NEEDLE_UUID, NEEDLE_NUM, Q_UUID, Q_NUM,
                          Q_VT, Q_CWE, Q_LAST, Q_HOP, CHAIN_FORMAT]).lower()
    hay_words |= set(WORD_RE.findall(templates))
    # the scorer matches whole words with \b, so "haven't" also contains the word "haven"
    hay_words |= set(re.findall(r"[a-z]+", hay + " " + templates))
    keys_pool = [k for k in keys_pool if k not in hay]
    key_words = {w for k in keys_pool for w in k.split("-")}
    words_pool = [w for w in words_pool if w not in hay_words and len(w) > 2]
    banned = hay_words | set(words_pool) | key_words
    print(f"[pools] {len(keys_pool)} RULER keys, {len(words_pool)} RULER cwe words "
          f"(absent from the essay text)")

    out, lens = [], []
    for i in range(args.n):
        rng = random.Random(args.seed * 1000003 + i)
        b = draw_blocks(rng, keys_pool, words_pool, banned, args)
        doc, target, ruler_needle = build_doc(i, rows[i], b, counter, args)
        blocks, n_tok = spans(doc, b, counter)
        lens.append((n_tok, target))
        meta = {"uuid_key": b["uuid_key"], "multikey": dict(zip(b["nm_keys"], b["nm_vals"])),
                "chain_names": [b["names"]], "chain_value": b["value"], "cwe_common": b["cwe_common"],
                "sizes": {"nm_pairs": args.nm_pairs, "cwe_rare": len(b["cwe_all"]) - args.cwe_common},
                "ruler": {"config": args.length, "task": "niah_single_3", "row": i,
                          "row_tokens": target, "removed_needle": ruler_needle.strip()}}
        out.append({"context": doc, "task": task, "questions": questions(task, i, b, args),
                    "doc_index": i, "ctx_target": target, "ctx_tokens": n_tok, "blocks": blocks,
                    "meta": meta, "seed": args.seed, "forced_answer_suffix": "\nFinal answer:"})
    fp = outdir / f"{task}.jsonl"
    with open(fp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    gap = [t - n for n, t in lens]
    print(f"wrote {len(out)} rows -> {fp}")
    print(f"  tokens {min(n for n, _ in lens)}-{max(n for n, _ in lens)}; "
          f"below RULER's row length by {min(gap)}-{max(gap)}")
    print(f"  questions per doc {len(out[0]['questions'])} ({sum(len(r['questions']) for r in out)} total)")


if __name__ == "__main__":
    main()
