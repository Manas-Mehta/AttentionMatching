#!/usr/bin/env python3
"""
Build the standalone logic datasets: ONE logic problem per ~4k-token document, padded with
the same prose filler as the mixed set (make_mixed_data.py). No other task in the document.

    logic_pw_4k   one ProofWriter theory (Tafjord et al. 2021, data V2020.12.3), closed-world
                  (CWA) depth-5 test set, unchanged: its facts and rules in the original order,
                  and its own ~20 True/False questions with their original labels and depths.
    logic_sl_4k   one SimpleLogic block (the original Paradox label-priority sampler, vendored
                  in make_mixed_data.py), built exactly as the mixed set's logic block; every
                  predicate in the block asked.

Placement as in the mixed set: the problem is one contiguous block, one sentence per line, at
a random slot between filler sentences. The prompt states the closed-world reading ("anything
that cannot be derived counts as false"), which both datasets' labels use.

Guards:
  * no ProofWriter entity name (the 18 the data uses) occurs in the filler; for SimpleLogic,
    no predicate and not "Alice" outside the block (the mixed set's contamination check);
  * 3 ProofWriter test theories whose labels an independent solver does not reproduce are
    skipped (ids in PW_SKIP; experiments/check_logic_data.py is that solver).

    python experiments/make_logic_data.py --dataset pw --n 50 \
        --proofwriter /path/to/proofwriter-dataset-V2020.12.3 \
        --tokenizer-json /path/to/Qwen3-4B-Instruct-2507/tokenizer.json
    python experiments/make_logic_data.py --dataset sl --n 50 --tokenizer-json ...
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_chain_data as mc  # noqa: E402  (prose pool)
import make_mixed_data as mm  # noqa: E402  (tokenizer counter, SimpleLogic, placement)

# All entity names in ProofWriter's synthetic theories.
PW_ENTITIES = ["Anne", "Bob", "Charlie", "Dave", "Erin", "Fiona", "Gary", "Harry",
               "bald eagle", "bear", "cat", "cow", "dog", "lion", "mouse", "rabbit",
               "squirrel", "tiger"]
# Theories (CWA depth-5 test) whose labels check_logic_data.py's solver does not reproduce
# (`check_logic_data.py --scan-proofwriter`). All three have negated rule conditions, where a
# plain fixpoint and ProofWriter's stratified negation can differ; skipped, not fixed.
PW_SKIP = {"AttNeg-CWA-D5-1186", "AttNeg-CWA-D5-189", "RelNeg-CWA-D5-45"}

# Closed-world reading, as ProofWriter's CWA labels use it: an underivable fact is false, so
# "X is not Y" is true exactly when "X is Y" cannot be derived. The first wording stated only
# the first half; the model then marked underivable negated statements False (job 19387294:
# 0.11 on those 200 questions vs 0.95-0.99 on the other kinds).
PW_QUESTION = ('Based only on the facts and rules in the text, is the following statement '
               'true? "{s}" Use the closed-world rule: any fact that cannot be derived from '
               'the facts and rules is false. So a statement saying that something is not the '
               'case is true exactly when the positive fact cannot be derived. '
               'Answer True or False.')


def word_re(w):
    return re.compile(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", re.I)


def pw_parts(rec):
    """One ProofWriter theory as a block + its own questions, all fields as released."""
    sents = [t["text"] for t in rec["triples"].values()] + [r["text"] for r in rec["rules"].values()]
    if " ".join(sents) != rec["theory"]:
        sys.exit(f"ERROR: {rec['id']}: facts + rules do not reproduce the theory text")
    text = "Facts and rules:\n" + "\n".join(sents)
    qs = []
    for q in rec["questions"].values():
        label = q["answer"] is True or str(q["answer"]) == "True"
        qs.append(dict(qtype="logic", logic_kind=q["strategy"], statement=q["question"],
                       label=label, depth=int(q["QDep"]), representation=q["representation"],
                       question=PW_QUESTION.format(s=q["question"]),
                       answer=["True" if label else "False"], score_candidates=["True", "False"],
                       max_new_tokens=8))
    meta = {"source": "ProofWriter V2020.12.3 CWA/depth-5/meta-test.jsonl", "id": rec["id"],
            "maxD": rec["maxD"], "NFact": rec["NFact"], "NRule": rec["NRule"],
            "triples": {k: v["representation"] for k, v in rec["triples"].items()},
            "rules": {k: v["representation"] for k, v in rec["rules"].items()}}
    return text, qs, meta


def sl_parts(rng, logic_pool, args):
    """One SimpleLogic block, built and asked exactly as in make_mixed_data.build_doc_parts."""
    n_pred = rng.randint(args.logic_preds_min, args.logic_preds_max)
    text, lq, meta = mm.build_logic(rng, logic_pool, n_pred, 0)
    qs = []
    for kind, p, label, d in lq:
        qs.append(dict(qtype="logic", logic_kind=kind, predicate=p, label=label, depth=d,
                       question=(f"Based only on the facts and rules about {mm.ENTITY} in the text, "
                                 f"is {mm.ENTITY} {p}? Answer True if it follows from them, "
                                 f"otherwise answer False."),
                       answer=["True" if label else "False"], score_candidates=["True", "False"],
                       max_new_tokens=8))
    meta["sizes"] = {"logic_preds": n_pred}
    return text, qs, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["pw", "sl"], required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proofwriter", default=None, help="proofwriter-dataset-V2020.12.3 directory")
    ap.add_argument("--pw-skip", default=None, help="file with theory ids to skip (one per line)")
    ap.add_argument("--logic-preds-min", type=int, default=15)
    ap.add_argument("--logic-preds-max", type=int, default=30)
    ap.add_argument("--tokenizer", default=mm.QWEN)
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--outdir", default=None, help="default: <repo>/official/data/logic")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir) if args.outdir else repo / "official" / "data" / "logic"
    outdir.mkdir(parents=True, exist_ok=True)
    counter = mm.Counter(args.tokenizer, args.tokenizer_json)
    prose_pool = mc.build_prose_pool(repo / "official" / "data" / "chain" / "prose_pool.json")
    ctx_tag = f"{args.ctx // 1024}k" if args.ctx % 1024 == 0 else str(args.ctx)
    task = f"logic_{args.dataset}_{ctx_tag}"

    if args.dataset == "pw":
        if not args.proofwriter:
            sys.exit("ERROR: --proofwriter is required for --dataset pw")
        skip = set(PW_SKIP)
        if args.pw_skip:
            skip |= {l.strip() for l in open(args.pw_skip) if l.strip()}
        fp = Path(args.proofwriter) / "CWA" / "depth-5" / "meta-test.jsonl"
        recs = [json.loads(l) for l in open(fp)]
        recs = [r for r in recs if r["id"] not in skip]
        picked = random.Random(args.seed).sample(recs, args.n)
        regs = [word_re(e) for e in PW_ENTITIES]
        before = len(prose_pool)
        prose_pool = [s for s in prose_pool if not any(r.search(s) for r in regs)]
        print(f"[pw] {len(recs)} theories after skipping {len(skip)}; filler {len(prose_pool)}/{before} "
              f"sentences kept (no ProofWriter entity name)")
    else:
        blob = " ".join(prose_pool).lower()
        words = mc.usable_words(prose_pool)
        vocab = [w.strip() for w in mm.LOGIC_VOCAB_FILE.read_text().splitlines() if w.strip()]
        logic_pool = [w for w in vocab if w not in blob and w not in words]
        print(f"[sl] {len(logic_pool)} of {len(vocab)} adjectives usable (absent from the filler)")

    rows, lengths, retries = [], [], 0
    for i in range(args.n):
        for attempt in range(50):
            rng = random.Random(args.seed * 1000003 + i + 7919 * attempt)
            if args.dataset == "pw":
                text, qs, meta = pw_parts(picked[i])
            else:
                text, qs, meta = sl_parts(rng, logic_pool, args)
            built = mm.build_document(rng, counter, args.ctx, [[text]], prose_pool)
            if built is None:
                retries += 1
                continue
            doc, n_tok, n_fill = built
            outside = doc.replace(text, "")
            if args.dataset == "pw":
                bad = [e for e in PW_ENTITIES if word_re(e).search(outside)]
            else:
                bad = mm.contamination(doc, {}, set(meta["preds"]), text)
            if not bad:
                break
            retries += 1
        else:
            sys.exit(f"ERROR: document {i} could not be built in 50 attempts")

        s = doc.index(text)
        e = s + len(text)
        ts, te = counter(doc[:s]), counter(doc[:e])
        blocks = {"logic": {"char_start": s, "char_end": e, "tok_start": ts, "tok_end": te,
                            "tokens": te - ts, "share": (te - ts) / n_tok}}
        for qi, q in enumerate(qs):
            q["qid"] = f"{task}_{i}_q{qi}"
        lengths.append(n_tok)
        rows.append({"context": doc, "task": task, "questions": qs, "doc_index": i,
                     "ctx_target": args.ctx, "ctx_tokens": n_tok, "n_filler_units": n_fill,
                     "blocks": blocks, "meta": meta, "seed": args.seed,
                     "forced_answer_suffix": "\nFinal answer:"})

    fp = outdir / f"{task}.jsonl"
    with open(fp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    nq = [len(r["questions"]) for r in rows]
    share = [r["blocks"]["logic"]["share"] for r in rows]
    print(f"\nwrote {len(rows)} rows -> {fp}")
    print(f"  context   {min(lengths)}-{max(lengths)} tokens (target {args.ctx})")
    print(f"  logic     {min(share):.1%}-{max(share):.1%} of tokens (mean {sum(share) / len(share):.1%})")
    print(f"  questions {min(nq)}-{max(nq)} per document ({sum(nq)} total)")
    print(f"  retries   {retries}")


if __name__ == "__main__":
    main()
