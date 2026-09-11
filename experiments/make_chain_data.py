#!/usr/bin/env python3
"""
Build the entity-attribute chain dataset for Phase 1 ("Trading Memory for Compute").

This is NVIDIA RULER's ``variable_tracking`` task with the knobs the professor's
document needs, which the shipped HuggingFace copy (``simonjegou/ruler``) does not
expose.  Section 5 asks for "synthetic entity-attribute chains with depth set
independently of surface form"; section 2 asks for depth and breadth labelled
per instance; section 3 asks for needle entropy and haystack entropy varied
separately.  Four independent knobs, one task family:

    depth    --hops D      serial lookups: VAR b = VAR a = VAR ... = value
    breadth  --chains B    parallel chains that all start from the SAME value
    needle   --names       word  (low entropy) | rand5 (high entropy)
    haystack --haystack    prose (low) | repeat (near-zero) | random (high)

Both depth and breadth vary INSIDE this one family.  That is deliberate: if depth
only ever varied in one task and breadth only in another, both would be perfectly
predicted by task identity and the surface fit in section 2 could not separate their
coefficients.

Document layout (B=2, D=2 shown, haystack lines elided):

    Memorize and track the chain(s) of variable assignment hidden in the following text.
    ...
    VAR trumpet = 48312          <- chain 0 root
    ...
    VAR harbor  = VAR trumpet    <- chain 0 hop 1
    ...
    VAR candle  = 48312          <- chain 1 root, SAME value
    ...
    VAR ribbon  = VAR harbor     <- chain 0 hop 2  (last variable of chain 0)
    ...

Two question forms:

  ``last``  (default)  "follow each chain to its end, report the final variable"
            Answer length is B, independent of D.  Depth is pure serial work and
            does not inflate the output, so the two axes stay separable.
  ``all``   RULER's original "find all variables assigned the value V".
            Answer length is B*(D+1), i.e. depth and breadth are collinear in the
            output.  Kept only as a compatibility control.

Per-instance probes (section 5, "retained-but-latent labeling") are emitted with
every row: one probe for the roots plus one per hop per chain.  Each probe is a
single lookup against the SAME compacted cache.  All probes pass but the composed
question fails -> silent failure.  A probe fails -> storage failure.

Output: ``official/data/chain/{task}.jsonl``, one row per document, in the same
standardized shape ``load_ruler_data`` / ``load_keylen_data`` produce, plus the
extra fields the Phase 1 analysis needs (depth, breadth, probes, ...).

Run it on the cluster login node, where the ``am`` env has transformers and the
HuggingFace cache is warm:

    python experiments/make_chain_data.py --hops 4 --chains 1 \
        --names word --haystack prose --ctx 4096 --n 50

For a laptop preview without transformers, pass ``--approx-chars-per-token 3.9``;
the context length is then estimated rather than measured, so never use those
files for a real run.
"""
import argparse
import json
import random
import re
import string
import sys
import time
import urllib.request
from pathlib import Path

QWEN = "Qwen/Qwen3-4B-Instruct-2507"

PREAMBLE = "Memorize and track the chain(s) of variable assignment hidden in the following text."

# RULER's stock "noise" haystack: one sentence block repeated forever. Near-zero entropy.
REPEAT_SENTENCE = ("The grass is green. The sky is blue. The sun is yellow. "
                   "Here we go. There and back again.")

# Common English nouns for the low-entropy needle condition. Deliberately ordinary
# words: single tokens for Qwen, and nothing that looks like an identifier.
# apple/garden/hammer/and arrow/ember/raisin/were removed because they
# occur in the Paul Graham prose pool; an answer word that also appears in the
# haystack would break the substring scorer and give the model a decoy.
WORDS = """
anchor autumn bacon badge bamboo banner barrel basket beacon beetle
bishop blanket blossom bottle boulder bracket bridge bucket bundle burrow cabin
cactus camera candle canvas canyon carpet castle cattle cellar chapel cherry
chimney cinema circus clover cobalt collar column comet copper coral cotton
cradle crater crayon cricket crystal curtain cushion dagger daisy diamond dolphin
domino donkey dragon drawer eagle engine fabric falcon feather fern ferry
fiddle finger flame flask forest fountain fox furnace garlic gazelle
ginger glacier granite grape gravel guitar harbor harvest hazel helmet
hickory honey hornet island ivory jacket jasmine jungle kettle kitten ladder
lagoon lantern laurel lemon lentil lighthouse lilac lobster locker lotus lumber
magnet mango maple marble meadow melon meteor mirror mitten monsoon mosaic
mountain mushroom mustard nectar needle nickel noodle nutmeg oasis olive onion
orchard orchid ostrich otter oyster paddle palace pantry parade parrot pasture
peach pebble pelican pencil pepper pewter pigeon pillar pillow pistol planet
plaster plateau plum pocket pollen poplar portrait pottery prairie pretzel prism
pumpkin pyramid quarry quartz rabbit radish rafter rattle raven ribbon
river robin rocket rooster rosemary ruby saddle saffron sapphire satchel scarlet
sequoia shadow shovel shrimp silver socket sparrow spinach sponge spruce squirrel
stadium stallion statue stirrup summit sunset swallow sycamore syrup tangerine
teapot temple thimble thistle thunder tiger timber toffee tomato topaz torch
tower trellis trophy trumpet tulip tundra turnip turtle valley vanilla velvet
vessel village vinegar violet vulture walnut walrus whisker willow window winter
wizard wombat yarrow zebra zinnia
""".split()


# --------------------------------------------------------------------------- tokenizer
class Counter:
    """Token counter. Real Qwen tokenizer, or a chars/token estimate for previews."""

    def __init__(self, model, approx):
        self.approx = approx
        self.tok = None
        if approx is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def __call__(self, text):
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=False).input_ids)
        return int(len(text) / self.approx)

    def random_token_text(self, rng, n_tokens):
        """Decoded random vocabulary tokens: the maximum-entropy haystack."""
        if self.tok is None:
            raise SystemExit(
                "--haystack random needs the real tokenizer; drop --approx-chars-per-token")
        vocab = self.tok.vocab_size
        ids = [rng.randrange(vocab) for _ in range(n_tokens)]
        text = self.tok.decode(ids, skip_special_tokens=True)
        return " ".join(text.split())


# --------------------------------------------------------------------------- prose pool
NEEDLE_RE = re.compile(r"One of the special magic uuids for \S+ is: [0-9a-f\-]+\.\s*")


def _ruler_ns3_rows(n):
    """niah_single_3 rows at 16k: local HF cache first (cluster, offline-ok), then the HTTP API.

    16k rather than 4k because the contexts are four times longer for the same number
    of requests. RULER draws every niah_single_3 context from the same essay text, so
    the unique-sentence count saturates around 590 whatever you pull; that is ~18k
    tokens of prose, enough that a 4k document samples about a quarter of it.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("simonjegou/ruler", "16384", split="test")
        rows = [r for r in ds if r["task"] == "niah_single_3"][:n]
        if rows:
            print(f"[prose] local HuggingFace cache: {len(rows)} niah_single_3 rows")
            return rows
    except Exception as e:
        print(f"[prose] HF datasets lib unavailable ({e.__class__.__name__}); using HTTP API")

    rows, off = [], 2000                     # niah_single_3 block starts at row 2000
    while len(rows) < n and off < 2500:
        want = min(10, n - len(rows))
        url = ("https://datasets-server.huggingface.co/rows?dataset=simonjegou%2Fruler"
               f"&config=16384&split=test&offset={off}&length={want}")
        with urllib.request.urlopen(url) as fh:
            d = json.load(fh)
        batch = [r["row"] for r in d["rows"] if r["row"]["task"] == "niah_single_3"]
        if not batch:
            break
        rows += batch
        off += want
        time.sleep(0.3)
    print(f"[prose] datasets-server HTTP API: {len(rows)} rows")
    return rows


def build_prose_pool(cache_fp, n_rows=200):
    """Paul Graham essay sentences, taken from RULER's own niah_single_3 contexts.

    RULER builds its essay haystack from a scrape of paulgraham.com that is not in
    the repo (the json file there is a git-lfs pointer). The shipped niah_single_3
    contexts are that same essay text with a preamble line and one needle sentence
    added, so stripping those two things back out recovers the haystack exactly,
    with no extra download and no new dependency.
    """
    cache_fp = Path(cache_fp)
    if cache_fp.exists():
        sents = json.loads(cache_fp.read_text())
        print(f"[prose] cached pool: {len(sents)} sentences <- {cache_fp}")
        return sents

    sents, seen = [], set()
    for r in _ruler_ns3_rows(n_rows):
        body = r["context"].split("\n", 1)[1]          # drop the uuid preamble line
        body = NEEDLE_RE.sub("", body)                 # drop the needle sentence
        assert r["answer"][0] not in body, "needle survived the strip"
        for s in re.split(r"(?<=[.!?])\s+", body):
            s = " ".join(s.split())
            if 40 <= len(s) <= 400 and s not in seen:
                seen.add(s)
                sents.append(s)
    if len(sents) < 50:
        sys.exit(f"ERROR: prose pool too small ({len(sents)} sentences)")
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    cache_fp.write_text(json.dumps(sents))
    print(f"[prose] built pool: {len(sents)} sentences -> {cache_fp}")
    return sents


# --------------------------------------------------------------------------- chains
def usable_words(prose_pool):
    """WORDS minus anything that occurs anywhere in the haystack text.

    The scorer is a substring match, so a variable name that also appears in the
    filler would score as found whatever the model said, and would act as a decoy
    during the lookup. Checked as a plain substring, not a word boundary, so that
    'arrow' inside 'narrower' is caught too.
    """
    if not prose_pool:
        return list(WORDS)
    blob = " ".join(prose_pool).lower()
    keep = [w for w in WORDS if w not in blob]
    if len(keep) < len(WORDS):
        print(f"[names] dropped {len(WORDS) - len(keep)} words that occur in the haystack; "
              f"{len(keep)} usable")
    return keep


def make_names(rng, k, style, words=None):
    """k distinct variable names. 'word' = low needle entropy, 'rand5' = high."""
    if style == "word":
        words = words if words is not None else WORDS
        if k > len(words):
            sys.exit(f"ERROR: need {k} names but only {len(words)} usable words")
        return rng.sample(words, k)
    out, seen = [], set()
    while len(out) < k:
        n = "".join(rng.choices(string.ascii_uppercase, k=5))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_chains(rng, hops, chains, name_style, words=None):
    """B chains of D hops that all start from the same value.

    Returns (value, names[b][j], lines[b][j]).  names[b][0] is the root variable of
    chain b (the one assigned the literal value); names[b][hops] is its last.
    """
    value = str(rng.randint(10000, 99999))
    flat = make_names(rng, chains * (hops + 1), name_style, words)
    names, lines = [], []
    for b in range(chains):
        ns = flat[b * (hops + 1):(b + 1) * (hops + 1)]
        ls = [f"VAR {ns[0]} = {value}"]
        for j in range(hops):
            ls.append(f"VAR {ns[j + 1]} = VAR {ns[j]}")
        names.append(ns)
        lines.append(ls)
    return value, names, lines


def make_probes(value, names, hops, chains):
    """One probe per constituent lookup (section 5's retained-but-latent labelling).

    Probe 0 covers every root at once, because with B chains sharing one value the
    'who is assigned V' lookup genuinely has B answers. Every later probe is a single
    link and has exactly one answer.
    """
    # Wording matters here. "Which variable is assigned the value of VAR stallion?"
    # is read transitively -- every downstream variable also holds stallion's value --
    # and the model answers with the whole rest of the chain, which is not wrong.
    # Naming the literal statement form makes the single-link reading the only one.
    n_root = chains
    probes = [{
        "kind": "root",
        "chain": -1,
        "step": 0,
        "question": (f"The text contains exactly {n_root} statement(s) of the form "
                     f"\"VAR X = {value}\". List the variable(s) X."),
        "answer": [names[b][0] for b in range(chains)],
    }]
    for b in range(chains):
        for j in range(hops):
            probes.append({
                "kind": "hop",
                "chain": b,
                "step": j + 1,
                "question": (f"The text contains exactly one statement of the form "
                             f"\"VAR X = VAR {names[b][j]}\". What is X?"),
                "answer": [names[b][j + 1]],
            })
    return probes


def make_question(value, names, hops, chains, form):
    if form == "all":
        n_v = chains * (hops + 1)
        q = (f"Find all variables that are assigned the value {value} in the text above, "
             f"directly or through a chain of assignments.")
        ans = [n for ns in names for n in ns]
        return q, ans, n_v
    q = (f"The text above contains {chains} chain(s) of variable assignments. Each chain begins "
         f"with a variable assigned the value {value}, and each chain contains {hops + 1} "
         f"variables in total. For each chain, report the LAST variable in that chain.")
    ans = [ns[hops] for ns in names]
    return q, ans, chains


# --------------------------------------------------------------------------- document
def build_document(rng, counter, ctx_tokens, lines_flat, haystack, prose_pool):
    """Interleave the assignment lines into filler, sized to hit ctx_tokens.

    Within-chain order is preserved (so a chain always reads forwards, as in RULER);
    across chains the lines interleave at random positions.

    The filler pool and the slot positions are drawn ONCE, up front, so that
    assemble(n) is a pure function of n. The binary search below would otherwise
    measure one document and emit a different one.
    """
    slot_seed = rng.getrandbits(32)
    pool = []

    def grow(n):
        while len(pool) < n:
            need = n - len(pool)
            if haystack == "repeat":
                pool.extend([REPEAT_SENTENCE] * need)
            elif haystack == "prose":
                pool.extend(rng.sample(prose_pool, min(len(prose_pool), need)))
            else:                       # ~25 tokens per unit, comparable to a sentence
                pool.extend(counter.random_token_text(rng, 25) for _ in range(need))

    def assemble(n_filler):
        grow(n_filler)
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

    lo, hi = len(lines_flat), max(32, len(lines_flat))
    while counter(assemble(hi)) < ctx_tokens and hi < 200000:
        lo, hi = hi, hi * 2
    while lo < hi:                                   # smallest filler count that reaches ctx
        mid = (lo + hi) // 2
        if counter(assemble(mid)) < ctx_tokens:
            lo = mid + 1
        else:
            hi = mid
    n = max(len(lines_flat), lo - 1)                 # stay just under the target
    doc = assemble(n)
    return doc, counter(doc), n


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hops", type=int, default=4, help="chain length D (depth knob)")
    ap.add_argument("--chains", type=int, default=1, help="chains sharing the value B (breadth knob)")
    ap.add_argument("--names", choices=["word", "rand5"], default="word",
                    help="needle entropy: common word vs 5 random capitals")
    ap.add_argument("--haystack", choices=["prose", "repeat", "random"], default="prose",
                    help="haystack entropy: essay prose | RULER's repeated sentence | random tokens")
    ap.add_argument("--question", choices=["last", "all"], default="last")
    ap.add_argument("--ctx", type=int, default=4096, help="target context length in tokens")
    ap.add_argument("--n", type=int, default=50, help="documents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=QWEN)
    ap.add_argument("--approx-chars-per-token", type=float, default=None,
                    help="preview only: estimate length instead of tokenizing")
    ap.add_argument("--outdir", default=None, help="default: <repo>/official/data/chain")
    ap.add_argument("--prose-pool", default=None,
                    help="sentence-pool cache; default <repo>/official/data/chain/prose_pool.json. "
                         "Kept out of --outdir so a throwaway outdir does not re-scrape it.")
    ap.add_argument("--task", default=None, help="override the generated task name")
    ap.add_argument("--print-sample", type=int, default=0,
                    help="print N full prompts to stdout instead of a summary")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir) if args.outdir else repo / "official" / "data" / "chain"
    outdir.mkdir(parents=True, exist_ok=True)

    counter = Counter(args.tokenizer, args.approx_chars_per_token)
    pool_fp = Path(args.prose_pool) if args.prose_pool else repo / "official" / "data" / "chain" / "prose_pool.json"
    prose_pool = build_prose_pool(pool_fp) if args.haystack == "prose" else None
    words = usable_words(prose_pool)

    ctx_tag = f"{args.ctx // 1024}k" if args.ctx % 1024 == 0 else str(args.ctx)
    task = args.task or (f"chain_d{args.hops}_b{args.chains}_"
                         f"{args.names}_{args.haystack}_{ctx_tag}_{args.question}")

    def contamination(doc, names):
        """Names must occur ONLY in their own assignment lines.

        Name j of a chain is written twice, once where it is assigned and once where
        it is read by the next link; the last name of a chain is written once. Any
        other occurrence means the haystack happens to contain the name, which would
        both decoy the lookup and fool the substring scorer.
        """
        bad = []
        for ns in names:
            for j, nm in enumerate(ns):
                want = 2 if j < len(ns) - 1 else 1
                got = doc.count(nm)
                if got != want:
                    bad.append((nm, got, want))
        return bad

    rows, lengths, retries = [], [], 0
    for i in range(args.n):
        for attempt in range(25):
            rng = random.Random(args.seed * 1000003 + i + 7919 * attempt)
            value, names, lines = build_chains(rng, args.hops, args.chains, args.names, words)
            # interleave the chains, but keep each chain reading forwards
            by_chain = {b: iter(lines[b]) for b in range(args.chains)}
            picks = [b for b in range(args.chains) for _ in range(args.hops + 1)]
            rng.shuffle(picks)
            seq = [next(by_chain[b]) for b in picks]

            doc, n_tok, n_filler = build_document(rng, counter, args.ctx, seq,
                                                  args.haystack, prose_pool)
            if doc.count(value) != args.chains:              # the value must be unambiguous too
                retries += 1
                continue
            bad = contamination(doc, names)
            if not bad:
                break
            retries += 1
        else:
            sys.exit(f"ERROR: document {i} still contaminated after 25 attempts: {bad}")

        question, answer, n_expected = make_question(value, names, args.hops,
                                                     args.chains, args.question)
        probes = make_probes(value, names, args.hops, args.chains)
        lengths.append(n_tok)
        rows.append({
            "context": doc,
            "question": question,
            "answer": answer,
            "answer_prefix": "",                 # the harness owns the prefix in Phase 1
            "forced_answer_suffix": "\nFinal answer:",
            "max_new_tokens": 32 + 12 * n_expected,
            "task": task,
            # -- labels the Phase 1 surface fit needs, per instance --
            "depth": args.hops + 1,              # sequential lookups to reach a chain's end
            "breadth": args.chains,              # spans that must be held at once
            "hops": args.hops,
            "name_style": args.names,
            "haystack": args.haystack,
            "question_form": args.question,
            "ctx_target": args.ctx,
            "ctx_tokens": n_tok,
            "n_filler_units": n_filler,
            "value": value,
            "chain_names": names,
            "probes": probes,
            "doc_index": i,
            "seed": args.seed,
        })

    fp = outdir / f"{task}.jsonl"
    with open(fp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n_probes = len(rows[0]["probes"])
    mode = "ESTIMATED" if args.approx_chars_per_token else "tokenized"
    print(f"\nwrote {len(rows)} rows -> {fp}")
    print(f"  task      {task}")
    print(f"  depth {rows[0]['depth']}   breadth {rows[0]['breadth']}   "
          f"probes/doc {n_probes}   answers/doc {len(rows[0]['answer'])}")
    print(f"  context   {min(lengths)}-{max(lengths)} tokens ({mode}, target {args.ctx})")
    print(f"  retries   {retries} (documents redrawn for a name or value collision)")

    for i in range(args.print_sample):
        r = rows[i]
        print("\n" + "=" * 78)
        print(f"DOCUMENT {i}   depth={r['depth']} breadth={r['breadth']} "
              f"names={r['name_style']} haystack={r['haystack']} tokens={r['ctx_tokens']}")
        print("=" * 78)
        print(r["context"][:1200])
        print(f"   [... {r['ctx_tokens']} tokens total ...]")
        print(r["context"][-400:])
        print("-" * 78)
        print("QUESTION:", r["question"])
        print("ANSWER:  ", r["answer"])
        print("CHAINS:  ", r["chain_names"], " value =", r["value"])
        print("-" * 78)
        for p in r["probes"]:
            print(f"  probe[{p['kind']}:{p['chain']}:{p['step']}] {p['question']}  ->  {p['answer']}")


if __name__ == "__main__":
    main()
