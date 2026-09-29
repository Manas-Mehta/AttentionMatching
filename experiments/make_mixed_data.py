#!/usr/bin/env python3
"""
Build the mixed-task dataset for the Trading-Memory-for-Compute grid (findings.md §B8, v2).

One document = shared prose filler + five task blocks, and ~25-40 questions that all run
against ONE compacted cache. Every question is labelled by type, and every task is
analysed separately: per type, how much CoT buys back under compression, and how the query
set used to build the cache shifts that.

    block      text in the document                            questions         varies per doc
    uuid       "One of the special magic uuids for K is: U."    1                 -
    multikey   2-6 x "... special magic number for K is: N."    2                 # of pairs
    chain      VAR a = 60494 ... VAR e = VAR d  (depth 5)       last + 1 hop      -
    cwe        numbered word list, 5 words repeated 8x          1                 # of rare words
    logic      SimpleLogic facts + rules about Alice            1 per predicate   # of predicates

Block sizes vary per document (§B8b) so that a method's advantage on a task can be read
at every block share, rather than confounded with how much space the task took. Each
block's token span and share of the document is stored in `blocks`.

Logic block = the ORIGINAL SimpleLogic label-priority sampler (Zhang et al. 2022, "On the
Paradox of Learning to Reason from Data"), vendored verbatim below from
github.com/joshuacnf/paradox-learning2reason (sample/sample.py), with their 150-adjective
vocabulary (official/data/mixed/logic_vocab.txt). Label-priority, not rule-priority,
because its labels are balanced (measured: 49% True vs 73%). One question per predicate that
appears in the block, labelled by forward chaining; False questions carry the paper's
backward-chaining depth. Only surface change: sentences read "If Alice is A and B, then
Alice is C." / "Alice is A." instead of the paper's "If A and B, then C." / "Alice A.".
Known property of this sampler, reported rather than engineered away: most rules can be
deleted without changing any answer (measured 14%-necessary on their own data).

Guards (as in make_chain_data.py):
  * every name, key, value, list word and predicate occurs ONLY in its own block, so the
    substring scorer cannot be fooled and the filler holds no decoys;
  * deterministic assembly, so the length search measures the emitted document.

Every question carries its own `score_candidates` (the strings the scorer counts as a
guess) and `max_new_tokens`, so the loader needs no per-task knowledge.

Tokenizer: pass --tokenizer-json (a local tokenizer.json, needs only the `tokenizers`
package) or run where transformers + the HF cache exist (cluster login node).

    python experiments/make_mixed_data.py --ctx 4096 --n 100 \
        --tokenizer-json /path/to/Qwen3-4B-Instruct-2507/tokenizer.json
"""
import argparse
import json
import random
import re
import sys
import uuid as uuidlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_chain_data as mc  # noqa: E402  (prose pool, word list, chain question)

QWEN = "Qwen/Qwen3-4B-Instruct-2507"

# Generic on purpose: the context must not tell a self-study curriculum what gets asked.
PREAMBLE = "Read the following text carefully. You will be asked questions about it."

ENTITY = "Alice"

# 150-adjective vocabulary, copied verbatim from github.com/joshuacnf/paradox-learning2reason
# (sample/vocab.txt) into official/data/mixed/logic_vocab.txt.
LOGIC_VOCAB_FILE = Path(__file__).resolve().parents[1] / "official" / "data" / "mixed" / "logic_vocab.txt"


# --------------------------------------------------------------------------- tokenizer
class Counter:
    def __init__(self, model, tokenizer_json):
        if tokenizer_json:
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(tokenizer_json)
            self._n = lambda s: len(t.encode(s, add_special_tokens=False).ids)
        else:
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self._n = lambda s: len(t(s, add_special_tokens=False).input_ids)

    def __call__(self, text):
        return self._n(text)


# --------------------------------------------------------------------------- SimpleLogic
# BEGIN verbatim from paradox-learning2reason/sample/sample.py (uses the global `random`).
def sample_label_priority(preds):
    preds_ = preds[:]
    random.shuffle(preds_)
    pred_num = len(preds)

    graph_depth = random.randint(1, pred_num // 2)
    width = pred_num // graph_depth

    preds_0 = preds_[:pred_num % graph_depth]
    preds_ = preds_[pred_num % graph_depth:]

    rules = []
    levels = []

    prev_level = [[x, random.randint(0, 1)] for x in preds_[:width]]
    if graph_depth > 1:
        prev_level[0][1], prev_level[1][1] = 0, 1
    else:
        prev_level[0][1], prev_level[1][1], prev_level[2][1], prev_level[3][1] = 0, 1, 0, 1
    preds_ = preds_[width:]
    levels.append(prev_level)

    # phase_1
    for d in range(0, graph_depth - 1):
        level = [[x, random.randint(0, 1)] for x in preds_[:width]]
        preds_ = preds_[width:]
        if len(preds_0) != 0:
            level.append((preds_0[0], random.randint(0, 1)))
            preds_0 = preds_0[1:]
        level[0][1], level[1][1] = 0, 1

        for node in level:
            lit, label = node[0], node[1]
            head_cand = [x[0] for x in prev_level if x[1] == label]
            head_num = random.randint(1, min(3, len(head_cand)))
            head = random.sample(head_cand, head_num)
            rules.append((head, lit))

        levels.append(level)
        prev_level = level

    # phase_2
    rule_num = random.randint(0 * pred_num, 3 * pred_num)
    nodes = [x for y in levels for x in y]
    neg_nodes = [x for x in nodes if x[1] == 0]
    rule_cnt = 0
    while rule_cnt < rule_num:
        tail_node = random.sample(nodes, 1)[0]

        tail = tail_node[0]
        head_cand = [x for x in nodes if x[0] != tail]
        while True:
            head_num = random.randint(1, min(3, len(head_cand)))
            head_nodes = None
            head_nodes = random.sample(head_cand, head_num)
            if not (all([x[1] == 1 for x in head_nodes]) and tail_node[1] == 0):
                break
        head = [x[0] for x in head_nodes]
        rules.append((head, tail))
        rule_cnt += 1

        # if all predicates in the head and tail of a rule are True,
        # we add one extra rule where its head and tail are both False
        # to balance the number of True/Positive predicates in all rules
        if all(x[1] == 1 for x in head_nodes):
            neg_tail = random.sample(neg_nodes, 1)[0][0]
            neg_head_cand = [x for x in neg_nodes if x[0] != neg_tail]
            neg_head_num = random.randint(1, min(3, len(neg_head_cand)))
            neg_head_nodes = random.sample(neg_head_cand, neg_head_num)
            neg_head = [x[0] for x in neg_head_nodes]
            rules.append((neg_head, neg_tail))
            rule_cnt += 1

    facts = [x[0] for x in levels[0] if x[1] == 1]

    query = random.sample([x[0] for x in nodes], 1)[0]

    return rules, facts, query


def forward_chain(rules, facts):
    res = {}
    for fact in facts:
        res[fact] = 0

    depth = 1
    prev_len = 0
    while len(res) > prev_len:
        new_facts = []
        for rule in rules:
            head, tail = rule
            if all([lit in res for lit in head]):
                new_facts.append(tail)
        prev_len = len(res)
        for fact in new_facts:
            if fact not in res:
                res[fact] = depth
        depth += 1

    return res


def backward_chain_(u, depth, rules, facts, max_depth, ances):
    INF = 100000000
    if u in facts:
        return INF
    if u in ances or depth == max_depth:
        return depth

    res = depth
    for rule in [x for x in rules if x[1] == u]:
        head, _ = rule
        tmp = INF
        for lit in head:
            ances.add(u)
            tmp = min(tmp, backward_chain_(lit,
                depth + 1, rules, facts, max_depth, ances))
            ances.remove(u)
        res = max(res, tmp)
    return res


def backward_chain(query, rules, facts, max_depth):
    return backward_chain_(query, 0, rules, facts, max_depth, set())
# END verbatim.


def build_logic(rng, preds_pool, n_pred, n_redundant):
    """One label-priority example, every predicate in the text queried.

    Mirrors the paper's process_example: shuffle rule bodies, rules and facts, then label by
    forward chaining. (In the paper's code a rule is (head, tail) = (body, conclusion).)
    """
    random.seed(rng.getrandbits(64))                 # the vendored sampler uses global random
    preds = rng.sample(preds_pool, n_pred)
    rules, facts, _ = sample_label_priority(preds)
    rules = [(list(b), h) for b, h in rules]
    [random.shuffle(b) for b, _ in rules]
    random.shuffle(rules)
    random.shuffle(facts)

    closure = forward_chain(rules, facts)
    # optional knob (default 0, not in the original): also state some derived facts
    redundant = []
    if n_redundant:
        cand = [p for p, d in closure.items() if d >= 2]
        redundant = rng.sample(cand, min(n_redundant, len(cand)))
    stated = facts + redundant
    closure = forward_chain(rules, stated)

    lines = [f"Facts and rules about {ENTITY}:"]
    for body, head in rules:
        lines.append(f"If {ENTITY} is {' and '.join(body)}, then {ENTITY} is {head}.")
    for f in stated:
        lines.append(f"{ENTITY} is {f}.")
    text = "\n".join(lines)

    in_text = [p for p in preds if re.search(r"(?<![\w-])" + re.escape(p) + r"(?![\w-])", text)]
    queries = []
    for p in in_text:
        if p in closure:
            if p in facts:
                # the original sampler sometimes states a fact that the rules also derive
                others = forward_chain(rules, [x for x in stated if x != p])
                kind = "stated_derivable" if p in others else "stated_base"
            else:
                kind = "redundant" if p in redundant else "derived"
            queries.append((kind, p, True, closure[p]))
        else:
            queries.append(("false", p, False, backward_chain(p, rules, stated, 7)))
    rng.shuffle(queries)
    meta = {"preds": preds, "rules": [[b, h] for b, h in rules], "facts": facts,
            "redundant_facts": redundant, "closure_depth": closure,
            "n_preds_not_in_text": len(preds) - len(in_text)}
    return text, queries, meta


# --------------------------------------------------------------------------- other blocks
def build_cwe(rng, words, n_common, freq_common, n_rare, freq_rare):
    common = words[:n_common]
    rare = words[n_common:n_common + n_rare]
    items = common * freq_common + rare * freq_rare
    rng.shuffle(items)
    lines = ["Below is a numbered list of words."]
    lines += [f"{i + 1}. {w}" for i, w in enumerate(items)]
    return "\n".join(lines), common, common + rare


def seven_digit(rng, taken):
    while True:
        n = str(rng.randint(1_000_000, 9_999_999))
        if n not in taken:
            taken.add(n)
            return n


def build_doc_parts(rng, words, logic_pool, args):
    """All blocks + questions for one document.

    Returns (units, unit_names, questions, meta, owners, logic_preds). A unit is a list of
    lines: one line = one contiguous slot, several lines (the chain) = one slot per line.
    """
    nm_pairs = rng.randint(args.nm_pairs_min, args.nm_pairs_max)
    cwe_rare = rng.randint(args.cwe_rare_min, args.cwe_rare_max)
    n_pred = rng.randint(args.logic_preds_min, args.logic_preds_max)

    names = rng.sample(words, 2 + nm_pairs + (args.hops + 1) + args.cwe_common + cwe_rare)
    it = iter(names)
    take = lambda k: [next(it) for _ in range(k)]

    units, unit_names, qs, owners = [], [], [], {}   # owners: string -> expected count
    numbers = set()

    # uuid needle
    k = "-".join(take(2))
    u = str(uuidlib.UUID(int=rng.getrandbits(128), version=4))
    units.append([f"One of the special magic uuids for {k} is: {u}."])
    unit_names.append("uuid")
    owners[k] = 1
    owners[u] = 1
    qs.append(dict(qtype="uuid", question=f"What is the special magic uuid for {k} mentioned in the text?",
                   answer=[u], score_candidates=[u], max_new_tokens=64))

    # multi-key needles
    keys = take(nm_pairs)
    vals = [seven_digit(rng, numbers) for _ in keys]
    for j, (kk, v) in enumerate(zip(keys, vals)):
        units.append([f"One of the special magic numbers for {kk} is: {v}."])
        unit_names.append(f"multikey{j}")
        owners[kk] = 1
        owners[v] = 1
    for j in rng.sample(range(nm_pairs), args.nm_asked):
        qs.append(dict(qtype="multikey",
                       question=f"What is the special magic number for {keys[j]} mentioned in the text?",
                       answer=[vals[j]], score_candidates=list(vals), max_new_tokens=24))

    # variable chain (one chain; lines stay in order, spread across the document)
    chain_names = take(args.hops + 1)
    value = str(rng.randint(10000, 99999))
    clines = [f"VAR {chain_names[0]} = {value}"] + [
        f"VAR {chain_names[j + 1]} = VAR {chain_names[j]}" for j in range(args.hops)]
    units.append(clines)
    unit_names.append("chain")
    for j, nm in enumerate(chain_names):
        owners[nm] = 2 if j < args.hops else 1
    owners[value] = 1
    q, ans, _ = mc.make_question(value, [chain_names], args.hops, 1, "last")
    qs.append(dict(qtype="chain_last", question=q, answer=ans,
                   score_candidates=list(chain_names), max_new_tokens=24, depth=args.hops + 1))
    h = rng.randrange(args.hops)
    if h == args.hops - 1:
        # never the final link, so the hop answer is never the chain-last answer. Remapped
        # from the already-drawn value, so the random stream (and every document) is unchanged.
        h = int(value) % (args.hops - 1)
    qs.append(dict(qtype="chain_hop",
                   question=(f"The text contains exactly one statement of the form "
                             f"\"VAR X = VAR {chain_names[h]}\". What is X?"),
                   answer=[chain_names[h + 1]],
                   score_candidates=[n for n in chain_names if n != chain_names[h]],
                   max_new_tokens=24, depth=1))

    # common-word extraction
    cwe_words = take(args.cwe_common + cwe_rare)
    cwe_text, common, cwe_all = build_cwe(rng, cwe_words, args.cwe_common, args.cwe_freq_common,
                                          cwe_rare, args.cwe_freq_rare)
    units.append([cwe_text])
    unit_names.append("cwe")
    for w in common:
        owners[w] = args.cwe_freq_common
    for w in cwe_all[args.cwe_common:]:
        owners[w] = args.cwe_freq_rare
    qs.append(dict(qtype="cwe",
                   question=(f"In the numbered list of words in the text, which {args.cwe_common} "
                             f"words appear most often? List exactly {args.cwe_common} words."),
                   answer=list(common), score_candidates=list(cwe_all), max_new_tokens=48))

    # logic
    ltext, lq, lmeta = build_logic(rng, logic_pool, n_pred, args.logic_redundant)
    units.append([ltext])
    unit_names.append("logic")
    for kind, p, label, d in lq:
        qs.append(dict(qtype="logic", logic_kind=kind, predicate=p, label=label, depth=d,
                       # "False" = cannot be derived (the paper's label), stated explicitly
                       # because a prompted model was never trained on that convention.
                       question=(f"Based only on the facts and rules about {ENTITY} in the text, "
                                 f"is {ENTITY} {p}? Answer True if it follows from them, "
                                 f"otherwise answer False."),
                       answer=["True" if label else "False"], score_candidates=["True", "False"],
                       max_new_tokens=8))
    meta = {"logic": lmeta, "chain_names": [chain_names], "chain_value": value,
            "multikey": dict(zip(keys, vals)), "uuid_key": k, "cwe_common": common,
            "sizes": {"nm_pairs": nm_pairs, "cwe_rare": cwe_rare, "logic_preds": n_pred}}
    return units, unit_names, qs, meta, owners, set(lmeta["preds"])


# --------------------------------------------------------------------------- document
def build_document(rng, counter, ctx_tokens, units, prose_pool):
    """Place the block units at random slots between filler sentences.

    A unit with several lines (the chain) takes one slot per line, in order, so the chain
    is spread across the document. Every other unit is one contiguous slot.
    Returns None if the blocks alone do not fit in ctx_tokens.
    """
    lines_flat = [ln for u in units for ln in u]
    rng.shuffle(lines_flat)
    ch = next((u for u in units if len(u) > 1), None)
    if ch:                                   # restore chain order within its random slots
        pos = sorted(i for i, ln in enumerate(lines_flat) if ln in ch)
        for p, ln in zip(pos, ch):
            lines_flat[p] = ln

    slot_seed = rng.getrandbits(32)
    pool = random.Random(rng.getrandbits(32)).sample(prose_pool, len(prose_pool))

    def assemble(n_filler):
        n_filler = min(n_filler, len(pool))
        srng = random.Random(slot_seed)
        slots = sorted(srng.sample(range(n_filler + 1), len(lines_flat))) \
            if n_filler + 1 >= len(lines_flat) else [0] * len(lines_flat)
        body, li = [], 0
        for i in range(n_filler + 1):
            while li < len(slots) and slots[li] == i:
                body.append(lines_flat[li])
                li += 1
            if i < n_filler:
                body.append(pool[i])
        return PREAMBLE + "\n\n" + "\n".join(body) + "\n"

    if counter(assemble(len(lines_flat))) >= ctx_tokens:
        return None
    lo, hi = 0, len(pool)
    if counter(assemble(hi)) < ctx_tokens:
        sys.exit("ERROR: prose pool too small for the target context")
    while lo < hi:
        mid = (lo + hi) // 2
        if counter(assemble(mid)) < ctx_tokens:
            lo = mid + 1
        else:
            hi = mid
    n = max(0, lo - 1)
    doc = assemble(n)
    return doc, counter(doc), n


def contamination(doc, owners, logic_preds, logic_text):
    bad = []
    for s, want in owners.items():
        got = len(re.findall(r"(?<![\w-])" + re.escape(s) + r"(?![\w-])", doc))
        if got != want:
            bad.append((s, got, want))
    outside = doc.replace(logic_text, "")
    for p in list(logic_preds) + [ENTITY]:
        if re.search(r"(?<![\w-])" + re.escape(p) + r"(?![\w-])", outside):
            bad.append((p, "outside logic block", 0))
    return bad


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--nm-pairs-min", type=int, default=2)
    ap.add_argument("--nm-pairs-max", type=int, default=6)
    ap.add_argument("--nm-asked", type=int, default=2)
    ap.add_argument("--cwe-common", type=int, default=5)
    ap.add_argument("--cwe-freq-common", type=int, default=8)
    ap.add_argument("--cwe-rare-min", type=int, default=20)
    ap.add_argument("--cwe-rare-max", type=int, default=60)
    ap.add_argument("--cwe-freq-rare", type=int, default=2)
    ap.add_argument("--logic-preds-min", type=int, default=15)
    ap.add_argument("--logic-preds-max", type=int, default=30)
    ap.add_argument("--logic-redundant", type=int, default=0,
                    help="derived facts also stated (type-2 knob; 0 = the original sampler)")
    ap.add_argument("--tokenizer", default=QWEN)
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--outdir", default=None, help="default: <repo>/official/data/mixed")
    ap.add_argument("--task", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir) if args.outdir else repo / "official" / "data" / "mixed"
    outdir.mkdir(parents=True, exist_ok=True)
    counter = Counter(args.tokenizer, args.tokenizer_json)
    prose_pool = mc.build_prose_pool(repo / "official" / "data" / "chain" / "prose_pool.json")
    blob = " ".join(prose_pool).lower()
    words = mc.usable_words(prose_pool)
    vocab = [w.strip() for w in LOGIC_VOCAB_FILE.read_text().splitlines() if w.strip()]
    logic_pool = [w for w in vocab if w not in blob and w not in words]
    print(f"[logic] {len(logic_pool)} of {len(vocab)} adjectives usable (absent from the filler)")

    ctx_tag = f"{args.ctx // 1024}k" if args.ctx % 1024 == 0 else str(args.ctx)
    task = args.task or f"mixed_v2_{ctx_tag}"

    rows, lengths, retries = [], [], 0
    for i in range(args.n):
        for attempt in range(50):
            rng = random.Random(args.seed * 1000003 + i + 7919 * attempt)
            units, unit_names, qs, meta, owners, lpreds = build_doc_parts(rng, words, logic_pool, args)
            built = build_document(rng, counter, args.ctx, units, prose_pool)
            if built is None:                       # blocks too big for the context: redraw
                retries += 1
                continue
            doc, n_tok, n_fill = built
            if not contamination(doc, owners, lpreds, units[unit_names.index("logic")][0]):
                break
            retries += 1
        else:
            sys.exit(f"ERROR: document {i} could not be built in 50 attempts")

        blocks = {}
        for name, u in zip(unit_names, units):
            if len(u) > 1:                          # chain: record every line
                starts = [doc.index(ln) for ln in u]
                blocks[name] = {"line_tok_start": [counter(doc[:s]) for s in starts],
                                "tokens": sum(counter(ln) for ln in u)}
            else:
                s = doc.index(u[0])
                e = s + len(u[0])
                ts, te = counter(doc[:s]), counter(doc[:e])
                blocks[name] = {"char_start": s, "char_end": e, "tok_start": ts, "tok_end": te,
                                "tokens": te - ts}
            blocks[name]["share"] = blocks[name]["tokens"] / n_tok
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
    print(f"\nwrote {len(rows)} rows -> {fp}")
    print(f"  context   {min(lengths)}-{max(lengths)} tokens (target {args.ctx})")
    print(f"  questions {min(nq)}-{max(nq)} per document ({sum(nq)} total)")
    print(f"  retries   {retries}")


if __name__ == "__main__":
    main()
