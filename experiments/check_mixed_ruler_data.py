#!/usr/bin/env python3
"""
Independent check of mixedr_*.jsonl (make_mixed_ruler_data.py), from the document text alone.

Per document:
  1. haystack   remove our inserted blocks and the instruction line; what is left must be a
                prefix of RULER's own niah_single_3 haystack for that row (RULER's needle
                removed), cut at a sentence end. --ruler checks this against the cached rows.
  2. uniqueness every planted string occurs exactly as often as planted, and nowhere else
                (uuid, 7-digit numbers, chain value, chain names, list words).
  3. answers    re-derived by parsing the text: needles -> key/value map; VAR lines -> follow
                the chain from its value; the numbered list -> counts, top 5 with a margin.
  4. questions  every answer is in its candidate set; chain_hop candidates exclude the asked
                name; qids unique.
With two files (--pair), also checks that document i has the same blocks in both.

    python experiments/check_mixed_ruler_data.py official/data/mixed/mixedr_8k.jsonl [--ruler 8192]
    python experiments/check_mixed_ruler_data.py A.jsonl --pair B.jsonl
"""
import argparse
import collections
import json
import re
import sys

NEEDLE_RE = re.compile(r"One of the special magic (uuids|numbers) for ([a-z]+-[a-z]+) is: ([0-9a-f-]{36}|\d{7})\.")
VAR_RE = re.compile(r"VAR ([A-Z]{5}) = (?:VAR ([A-Z]{5})|(\d{5}))")
LIST_RE = re.compile(r"\n(Below is a numbered list of words\.[^\n]*)\n([^\n]*)\n")


def count_word(text, w):
    return len(re.findall(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", text, flags=re.I))


def check_doc(r, ruler_body=None):
    errs = []
    doc, m = r["context"], r["meta"]
    instr, body = doc.split("\n", 1)
    if not instr.startswith("Some special magic uuids and numbers are hidden"):
        errs.append("instruction line")
    # --- 3. answers from the text
    needles = NEEDLE_RE.findall(doc)
    kv = {k: v for _, k, v in needles}
    if len(needles) != 1 + m["sizes"]["nm_pairs"] or len(kv) != len(needles):
        errs.append(f"needles: {len(needles)}")
    vars_ = VAR_RE.findall(doc)
    nxt = {a: b for b, a, _ in vars_ if a}           # VAR b = VAR a : a -> b
    start = [b for b, a, v in vars_ if v]
    chain = start[:]
    while chain and chain[-1] in nxt:
        chain.append(nxt[chain[-1]])
    lm = LIST_RE.findall(doc)
    if len(lm) != 1:
        errs.append(f"cwe blocks: {len(lm)}")
        return errs
    items = re.findall(r"(\d+)\. ([a-z]+)", lm[0][1])
    if [int(n) for n, _ in items] != list(range(1, len(items) + 1)):
        errs.append("cwe numbering")
    cnt = collections.Counter(w for _, w in items)
    ranked = cnt.most_common()
    top5 = {w for w, _ in ranked[:5]}
    if len(ranked) > 5 and ranked[4][1] <= ranked[5][1]:
        errs.append("cwe top-5 tie")
    for q in r["questions"]:
        t, gold = q["qtype"], q["answer"]
        if t == "uuid":
            k = re.search(r"uuid for ([a-z]+-[a-z]+) mentioned", q["question"]).group(1)
            want = [kv.get(k)]
        elif t == "multikey":
            k = re.search(r"number for ([a-z]+-[a-z]+) mentioned", q["question"]).group(1)
            want = [kv.get(k)]
        elif t == "chain_all":
            v = re.search(r"value (\d{5})", q["question"]).group(1)
            want = chain if start and [x for b, a, x in vars_ if x] == [v] else None
        elif t == "chain_last":
            want = chain[-1:]
        elif t == "chain_hop":
            a = re.search(r'VAR X = VAR ([A-Z]{5})"', q["question"]).group(1)
            want = [nxt.get(a)]
            if a in q["score_candidates"]:
                errs.append("hop candidates include the asked name")
        elif t == "cwe":
            want = sorted(top5)
            gold = sorted(gold)
        else:
            errs.append(f"unknown qtype {t}")
            continue
        if want != gold:
            errs.append(f"{t}: text gives {want}, gold {gold}")
        if not set(x.lower() for x in q["answer"]) <= set(x.lower() for x in q["score_candidates"]):
            errs.append(f"{t}: answer not in candidates")
    if len(chain) != 5:
        errs.append(f"chain length {len(chain)}")
    # --- 2. uniqueness: planted strings only where planted
    expect = {}
    for _, k, v in needles:
        expect[v] = 1
    expect[m["chain_value"]] = 1
    for j, nm in enumerate(m["chain_names"][0]):
        expect[nm] = 2 if j < len(m["chain_names"][0]) - 1 else 1
    for w, c in cnt.items():
        expect[w] = c
    for s, c in expect.items():
        got = count_word(doc, s)
        if got != c:
            errs.append(f"'{s}' occurs {got}x, planted {c}x")
    # --- 1. haystack = RULER's text with our blocks removed
    if ruler_body is not None:
        b = body.rstrip("\n")
        b = b.replace("\n" + lm[0][0] + "\n" + lm[0][1] + "\n", " ")
        for _, k, v in needles:
            kind = "uuids" if "-" in v else "numbers"
            b = b.replace(f" One of the special magic {kind} for {k} is: {v}.", "", 1)
        for bb, a, v in vars_:
            b = b.replace(f" VAR {bb} = " + (f"VAR {a}" if a else v), "", 1)
        if not ruler_body.startswith(b):
            i = next((x for x in range(min(len(b), len(ruler_body))) if b[x] != ruler_body[x]), None)
            errs.append(f"haystack differs from RULER at char {i}: {b[max(0,(i or 0)-40):(i or 0)+40]!r}")
        elif not re.search(r"[.!?]$", b):
            errs.append("haystack not cut at a sentence end")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ruler", type=int, default=None, help="RULER config to compare the haystack with")
    ap.add_argument("--pair", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path)]
    bodies = None
    if a.ruler:
        sys.path.insert(0, "experiments")
        from make_mixed_ruler_data import ruler_rows, strip_ruler
        rr = ruler_rows(a.ruler, "niah_single_3")
        bodies = [strip_ruler(rr[r["meta"]["ruler"]["row"]])[0] for r in rows]
    bad = 0
    qids = [q["qid"] for r in rows for q in r["questions"]]
    if len(qids) != len(set(qids)):
        print("duplicate qids"); bad += 1
    for i, r in enumerate(rows):
        e = check_doc(r, bodies[i] if bodies else None)
        if e:
            bad += 1
            print(f"doc {r['doc_index']}: " + "; ".join(e[:4]))
    over = [r["doc_index"] for r in rows if r["ctx_tokens"] > r["meta"]["ruler"]["row_tokens"]]
    print(f"{len(rows)} docs, {len(qids)} questions, {bad} with errors, {len(over)} longer than RULER's row"
          + ("" if bodies else "  (haystack not compared: pass --ruler)"))
    if a.pair:
        other = [json.loads(l) for l in open(a.pair)]
        keyf = lambda r: json.dumps({k: v for k, v in r["meta"].items() if k != "ruler"}, sort_keys=True)
        same = sum(keyf(x) == keyf(y) and [q["answer"] for q in x["questions"]] == [q["answer"] for q in y["questions"]]
                   for x, y in zip(rows, other))
        print(f"paired with {a.pair}: {same}/{len(rows)} documents have identical blocks and answers")
        bad += len(rows) - same
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
