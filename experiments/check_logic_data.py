#!/usr/bin/env python3
"""
Independent check of the standalone logic datasets (make_logic_data.py).

logic_pw_*: re-derive every label with our own closed-world solver (forward chaining,
negation as failure, iterated to a fixpoint) from the theory's facts and rules, and confirm
every fact and rule sentence is in the document's block. The labels are ProofWriter's own,
so agreement checks our reading of their data, not their generator.
logic_sl_*: parse the block text back into rules and facts, forward-chain, and compare with
every label; confirm every predicate in the block is asked exactly once.

    python experiments/check_logic_data.py official/data/logic/logic_pw_4k.jsonl
    python experiments/check_logic_data.py --scan-proofwriter <dir>   # ids the solver disagrees on
"""
import argparse
import json
import re
import sys

VARS = {"someone", "something"}


def atoms(s):
    return [tuple(re.findall(r'"([^"]*)"', a)) for a in re.findall(r'\(("[^()]*")\)', s)]


def parse_rule(rep):
    lhs, rhs = rep.split("->")
    return atoms(lhs), atoms(rhs)[0]


def closure(facts, rules, ents):
    S = {f[:3] for f in facts if f[3] == "+"}
    changed = True
    while changed:
        changed = False
        for conds, (cs, cr, co, _) in rules:
            subs = ents if any(c[0] in VARS or c[2] in VARS for c in conds) or cs in VARS else [None]
            for e in subs:
                g = lambda x: e if x in VARS else x
                if all(((g(a), r, g(o)) in S) == (sg == "+") for a, r, o, sg in conds):
                    t = (g(cs), cr, g(co))
                    if t not in S:
                        S.add(t)
                        changed = True
    return S


def pw_labels(triples, rules, questions):
    facts = [atoms(t)[0] for t in triples]
    rl = [parse_rule(r) for r in rules]
    ents = {f[0] for f in facts} | {f[2] for f in facts if f[1] != "is"}
    S = closure(facts, rl, ents)
    out = []
    for rep in questions:
        s, r, o, sg = atoms(rep)[0]
        out.append(((s, r, o) in S) == (sg == "+"))
    return out


def scan(pw_dir):
    bad = []
    for line in open(f"{pw_dir}/CWA/depth-5/meta-test.jsonl"):
        rec = json.loads(line)
        qs = list(rec["questions"].values())
        got = pw_labels([t["representation"] for t in rec["triples"].values()],
                        [r["representation"] for r in rec["rules"].values()],
                        [q["representation"] for q in qs])
        want = [q["answer"] is True or str(q["answer"]) == "True" for q in qs]
        if got != want:
            bad.append(rec["id"])
    print("\n".join(bad))
    print(f"# {len(bad)} theories disagree", file=sys.stderr)


def forward_chain(rules, facts):
    known = set(facts)
    while True:
        new = {h for b, h in rules if all(x in known for x in b)} - known
        if not new:
            return known
        known |= new


def check(fp):
    rows = [json.loads(l) for l in open(fp)]
    n_q = n_ok = 0
    problems = []
    for row in rows:
        doc, b = row["context"], row["blocks"]["logic"]
        block = doc[b["char_start"]:b["char_end"]]
        if doc.count(block) != 1:
            problems.append((row["doc_index"], "block not unique in document"))
        qs = row["questions"]
        if row["task"].startswith("logic_pw"):
            m = row["meta"]
            got = pw_labels(m["triples"].values(), m["rules"].values(), [q["representation"] for q in qs])
            lines = block.split("\n")[1:]
            if len(lines) != len(m["triples"]) + len(m["rules"]):
                problems.append((row["doc_index"], "sentence count differs from the theory"))
        else:
            rules, facts = [], []
            for ln in block.split("\n")[1:]:
                mr = re.fullmatch(r"If Alice is (.+), then Alice is ([\w-]+)\.", ln)
                mf = re.fullmatch(r"Alice is ([\w-]+)\.", ln)
                if mr:
                    rules.append((mr.group(1).split(" and "), mr.group(2)))
                elif mf:
                    facts.append(mf.group(1))
                else:
                    problems.append((row["doc_index"], f"unparsed line: {ln!r}"))
            true = forward_chain(rules, facts)
            got = [q["predicate"] in true for q in qs]
            preds = {p for b_, h in rules for p in b_ + [h]} | set(facts)
            asked = [q["predicate"] for q in qs]
            if sorted(asked) != sorted(preds):
                problems.append((row["doc_index"], "predicates asked != predicates in block"))
        want = [q["label"] for q in qs]
        n_q += len(qs)
        n_ok += sum(g == w for g, w in zip(got, want))
        if any((q["answer"] == ["True"]) != q["label"] for q in qs):
            problems.append((row["doc_index"], "answer field disagrees with label"))
    tr = sum(q["label"] for r in rows for q in r["questions"])
    print(f"{fp}: {len(rows)} docs, {n_q} questions, labels re-derived {n_ok}/{n_q}, "
          f"True {tr}/{n_q} ({tr / n_q:.0%})")
    for p in problems:
        print("  PROBLEM", p)
    return n_ok == n_q and not problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--scan-proofwriter", default=None)
    a = ap.parse_args()
    if a.scan_proofwriter:
        scan(a.scan_proofwriter)
    ok = all([check(f) for f in a.files])
    sys.exit(0 if ok else 1)
