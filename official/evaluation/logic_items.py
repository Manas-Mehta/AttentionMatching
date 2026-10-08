"""
Scoring for the ProofWriter list questions (experiments/make_logic_data.py, logic_pwl_*).

Two question types, both answered as a list of short sentences:

  logic_list  "List every fact about Dave ..."  Gold = every positive fact with that subject
              in the closed-world closure of the theory, each tagged stated / derived and
              with its derivation depth. Scored per fact.
  logic_copy  "Copy out the facts and rules ..."  Gold = the theory's sentences, each tagged
              with its role (needed / redundant / never fires). Scored per sentence. A
              measuring question: it shows what the cache still holds.

The evaluator only needs the scalar (`item_score`). The breakdowns (`list_breakdown`,
`copy_breakdown`) are recomputed offline from the saved answer text and the dataset file.

Facts are parsed, not string-matched: "Dave is big and red", "big, red" (no subject) and
"Dave is big." all give the same facts. Clauses that contain a negation are skipped
(negated facts are never gold), and clauses about another entity are ignored.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

ATTRS = {"big", "blue", "cold", "furry", "green", "kind", "nice", "quiet", "red", "rough",
         "round", "smart", "white", "young"}
VERBS = {"chase": "chases", "chases": "chases", "eat": "eats", "eats": "eats",
         "like": "likes", "likes": "likes", "need": "needs", "needs": "needs",
         "see": "sees", "sees": "sees", "visit": "visits", "visits": "visits"}
ENTITIES = ["bald eagle", "anne", "bob", "charlie", "dave", "erin", "fiona", "gary", "harry",
            "bear", "cat", "cow", "dog", "lion", "mouse", "rabbit", "squirrel", "tiger"]
_ENT = "|".join(ENTITIES)
_LEAD = re.compile(rf"^\W*(?:\d+[.)]\s*)?\W*(?:the\s+)?({_ENT})\b(.*)$")
_REL = re.compile(rf"\b({'|'.join(VERBS)})\s+(?:the\s+)?({_ENT})\b")
_NEG = re.compile(r"\bnot\b|n't\b|\bno\b|\bnever\b")
_SPLIT = re.compile(r"[\n.;]+")
# inside a line, a new clause starts at ", <entity>" or "and <entity> is/<verb>"
_SUB = re.compile(rf",\s*(?=(?:the\s+)?(?:{_ENT})\b)|\band\s+(?=(?:the\s+)?(?:{_ENT})\s+(?:is|{'|'.join(VERBS)})\b)")

Fact = Tuple[str, str, str]
LOGIC_ITEM_QTYPES = ("logic_list", "logic_copy")


def parse_facts(text: str, subject: str) -> Set[Fact]:
    """Positive facts about `subject` (lower-case entity) asserted in `text`."""
    subject = subject.lower()
    out = set()
    clauses = [c for line in _SPLIT.split(text.lower()) for c in _SUB.split(line)]
    for clause in clauses:
        if not clause.strip() or _NEG.search(clause):
            continue
        m = _LEAD.match(clause)
        if m:
            if m.group(1) != subject:
                continue  # a fact about someone else
            rest = m.group(2)
        else:
            rest = clause  # bare list ("big, red") or a subject-less phrase ("eats the cat")
        for v, o in _REL.findall(rest):
            out.add((subject, VERBS[v], o))
        rest = _REL.sub(" ", rest)
        for w in re.findall(r"[a-z]+", rest):
            if w in ATTRS:
                out.add((subject, "is", w))
    return out


def _fact(it: Dict) -> Fact:
    return tuple(x.lower() for x in it["fact"])


def _norm(s: str) -> str:
    s = re.sub(r"^\W*(?:\d+[.)]\s*)?", "", s.lower())
    return " ".join(re.findall(r"[a-z]+", s))


def copy_found(text: str, items: List[Dict]) -> Tuple[List[bool], int]:
    """Per gold sentence: copied (exact after normalising)? Plus the number of extra lines."""
    lines = {_norm(l) for l in text.split("\n") if _norm(l)}
    gold = [_norm(it["text"]) for it in items]
    return [g in lines for g in gold], len(lines - set(gold))


def item_score(pred: str, q: Dict) -> float:
    """Scalar for the evaluator: |found & gold| / max(|predicted|, |gold|) (cwe-style)."""
    items = q.get("logic_items") or []
    if q.get("qtype") == "logic_copy":
        found, extra = copy_found(pred, items)
        return sum(found) / max(sum(found) + extra, len(items), 1)
    gold = {_fact(it) for it in items}
    pred_f = parse_facts(pred, q["logic_subject"])
    return len(pred_f & gold) / max(len(pred_f), len(gold), 1)


def list_breakdown(pred: str, items: List[Dict], subject: str) -> Dict:
    """Counts for one logic_list answer. `wrong` = predicted facts not in the closure."""
    pred_f = parse_facts(pred, subject)
    gold = {_fact(it): it for it in items}
    stated = {f for f, it in gold.items() if it["stated"]}
    out = {"stated_gold": len(stated), "stated_found": len(pred_f & stated),
           "derived_gold": len(gold) - len(stated),
           "derived_found": sum(1 for f in pred_f if f in gold and not gold[f]["stated"]),
           "wrong": len(pred_f - set(gold)), "by_depth": {}}
    for f, it in gold.items():
        if not it["stated"]:
            d = out["by_depth"].setdefault(it["depth"], [0, 0])
            d[0] += f in pred_f
            d[1] += 1
    return out


def copy_breakdown(pred: str, items: List[Dict]) -> Dict:
    """Per role ("fact:needed", "rule:never", ...): [copied, total]; plus extra lines."""
    found, extra = copy_found(pred, items)
    out: Dict = {"extra": extra}
    for ok, it in zip(found, items):
        k = f"{it['kind']}:{it['role']}"
        out.setdefault(k, [0, 0])
        out[k][0] += ok
        out[k][1] += 1
    return out
