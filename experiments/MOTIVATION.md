# Why nm3/ns3 — the motivating dissociation

Source: **"End-to-End Context Compression at Scale"** (Li et al., NYU/UMD/Princeton/Columbia,
arXiv 2606.09659, Jun 2026) — the LCLM paper. Tables 7 (RULER 4k), 8 (8k), 9 (16k)
are the **only published per-task RULER breakdown for Attention Matching**. The AM
paper itself (Table 11) reports only an aggregate over all 13 tasks.

All LCLM numbers use decoder **`Qwen3-4B-Instruct-2507`** for every method.

## The dissociation

RULER **16k, 16x compression** (Table 9). Full-KV baseline in brackets:

| Method | ns3 | nm3 | cwe | fwe | AVG |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 (full KV) | 100.00 | 99.80 | 58.04 | 98.93 | 93.74 |
| **AM-Fast** | **0.20** | **0.00** | **55.44** | 94.67 | 37.17 |
| **AM-Slow** | **1.20** | **0.00** | 52.28 | 93.33 | 42.02 |
| KVzip | 91.60 | 18.40 | 3.18 | 88.87 | 57.97 |
| KVzipFast | 42.00 | 44.00 | 3.58 | 73.40 | 47.86 |
| LCLM 0.6b-4b (Mean) | 57.40 | 30.00 | 24.78 | 98.73 | 65.91 |
| ExpAttn | 1.60 | 0.00 | 8.52 | 71.07 | 50.67 |

Same picture at **4k, 16x** (Table 7):

| Method | ns3 | nm3 | cwe | AVG |
|---|---:|---:|---:|---:|
| full KV | 100.00 | 99.80 | 97.50 | 94.41 |
| AM-Fast | 7.80 | 3.80 | **76.82** | 53.09 |
| AM-Slow | 31.60 | 9.40 | 74.52 | 69.21 |
| KVzip | 96.00 | 41.00 | 21.80 | 62.73 |
| LCLM (Mean) | 54.60 | 48.60 | 80.54 | 75.06 |

Two things stand out:

1. **AM is the worst method on ns3/nm3** — literally 0.00 on nm3 at 16k, while KVzip
   holds 91.60 on ns3.
2. **AM is by far the best method on cwe** — at 16k it retains 55.44 of a 58.04
   ceiling (~95% of baseline) while KVzip collapses to 3.18 (~5%). AM is not merely
   better here, it is the *only* method that preserves the task.

This is a dissociation, not a quality gap. Something about AM trades exactly one
capability for the other.

## The UUID hypothesis — CONFIRMED at the data level

Pulled real examples from `simonjegou/ruler` (config 4096):

| Task | Query key | Answer |
|---|---|---|
| `niah_single_1` | word (`wandering-age`) | `8090293` — 7-digit number |
| `niah_single_2` | word (`abject-antler`) | `3678638` — 7-digit number |
| `niah_multikey_1` | word (`fair-sprout`) | `9375710` — 7-digit number |
| **`niah_single_3`** | word (`amused-quart`) | **`1ff49b78-8946-4e85-b59c-de66bacfb3d0` — UUID** |
| **`niah_multikey_3`** | **UUID** (`455ac762-…`) | **`1ee34dc4-3b04-4a8b-805b-fdc94e7ed827` — UUID** |
| `cwe` | — | list of common English words (aggregation) |

**The two tasks where AM collapses are exactly the two UUID tasks.** And nm3 — where
AM scores 0.00 — is doubly hard: the model must *match* a UUID key and then *emit* a
UUID value. A 7-digit number is ~3-4 tokens; a UUID is ~15-20 low-frequency hex
fragments that must all be reproduced verbatim and in order.

## Proposed mechanism

AM's objective (paper Eqs. 1-2) matches, per KV-head, (i) the block's attention
**output** and (ii) its attention **mass**, averaged over generic reference queries.
Values are then refit in closed form by least squares, `C_v = (XᵀX)⁻¹XᵀY`.

Least squares minimizes *average* error over reference queries. That is precisely
the right objective for aggregate/distributional tasks and precisely the wrong one
for reproducing a single rare exact string:

- **cwe/fwe are mass-statistics problems.** "Which words appear most often" is
  literally a question about attention mass. AM optimizes that directly — hence it
  is the only method that survives.
- **UUID retrieval needs a long run of consecutive verbatim tokens.** At 16x you
  retain 1/16 of keys, chosen *question-agnostically* (compaction happens before the
  query is known, and reference queries come from repeat-prefill). A needle UUID has
  no reason to attract attention at compaction time, so its ~18-token span is very
  unlikely to survive selection intact — and even if the keys survive, the refit
  values are a projection that blurs exactly the fine detail needed.

Also worth ruling in/out: **β pruning.** HighestAttnKeys clamps `β ∈ [-3,3]`; OMP
discards keys with `β < -7`. A needle key with very negative β contributes ~nothing
regardless of `C_v`.

## A SECOND, ratio-independent failure mode (16k)

Do not conflate this with the UUID effect. AM-Fast on **ns1** — which contains no
UUIDs — across every length x ratio cell:

| Context | 16x | 8x | 4x |
|---|---:|---:|---:|
| 4k  | 100.00 | 100.00 | 100.00 |
| 8k  | 100.00 | 97.60 | 99.40 |
| **16k** | **52.05** | **51.20** | **60.00** |

Broken at *all three ratios*, including 4x. AM-Slow behaves identically (61.00 / 53.80
/ 54.40). Every other method at 16k/4x gets ns1 = 100.00 (ExpAttn, KVzip, KVzipFast)
or ~99.6 (LCLM).

**vt** (variable tracking) shows the same signature: 95.16 / 96.60 / 98.75 at 4k, fine
at 8k, then **54.92 / 55.24 / 63.56** at 16k against a 99.52 ceiling.

Contrast the two patterns directly:

| | ns3 / nm3 (UUID) @16k | ns1 / vt @16k |
|---|---|---|
| 16x | 0.20 / 0.00 | 52.05 / 54.92 |
| 8x | 10.40 / 13.40 | 51.20 / 55.24 |
| 4x | **80.20 / 91.20** | **60.00 / 63.56** |
| Pattern | **recovers** with budget | **flat** — budget doesn't help |
| Reading | capacity / selection | implementation or conditioning |

A ratio-independent failure is not a capacity problem — giving AM 4x the budget
changes nothing, so it is not "too few keys retained."

**Hypothesis: ill-conditioning on homogeneous contexts.** ns1's haystack is a single
sentence repeated; vt is a chain of near-identical assignments. Both produce many
near-duplicate keys. Near-duplicate columns make `XᵀX` near-singular, so the
least-squares refit `C_v = (XᵀX)⁻¹XᵀY` and OMP's greedy selection both degrade — and
this gets worse with length (4x more near-identical keys at 16k than 4k) but is
*independent of how many you keep*. This would also explain LCLM's note that AM "fails
at 512K tokens due to numerical instability in the linear solver."

Competing explanation: **chunking**. At 16k with default `--chunk-size 4096` the
context splits into 4 independently-compacted chunks. But 8k would already be 2 chunks
and 8k is fine, so chunking alone does not obviously explain a cliff between 8k and
16k. Testable either way (P4/P5 below).

## Testable predictions

- **P1** — needle-token perplexity under the compacted cache is catastrophic for
  ns3/nm3, mild for ns1/ns2. (This is TODO item 3.)
- **P2** — oracle-retain the needle span in `C_k`; if ns3/nm3 recover, the failure is
  **key selection**. If they stay broken, it's **value fitting**. This cleanly splits
  the two candidate mechanisms.
- **P3** — AM's *own* intrinsic loss (attention-output MSE, mass error) should look
  **normal** on ns3/nm3 despite 0% accuracy. If intrinsic loss is fine while the task
  is at zero, the objective is *misaligned*, not under-optimized. This directly
  answers the "is there a clear reason for the dropoff" key question.
- **P4** — disable chunking at 16k; if ns1 recovers from 52.05, the length dropoff is
  a chunking artifact separable from the UUID effect.
- **P5** — log the condition number of `XᵀX` (and OMP residual decay) per head on
  ns1/vt vs ns3/qa1 at 4k vs 16k. If conditioning blows up specifically on the
  repetitive-context tasks at 16k, that confirms the ill-conditioning story and
  predicts a cheap fix (ridge / rank-truncation in the solver). Note the AM paper
  says it *tried* ℓ2 regularisation `(XᵀX + λI)⁻¹` and found it hurt on QuALITY —
  but QuALITY is 5-8k prose, exactly the regime where conditioning is fine. It may
  well help at 16k on homogeneous contexts. That would be a clean contribution.

## Open questions about the LCLM numbers

- **What exactly is "AM-Slow"?** LCLM states (§6.1) they do *not* apply self-study to
  RULER, so AM-Slow is most likely AM-OMP with repeat-prefill queries — not the
  paper's strongest `self-study` config. Worth confirming against
  [github.com/LeonLixyz/LCLM](https://github.com/LeonLixyz/LCLM).
- **Nobody has tested self-study on ns3/nm3.** Both papers skipped it on RULER for
  cost reasons. If self-study queries happen to cover needle-like content, this is
  cheap headroom and an easy contribution.
- Did LCLM use chunked compaction at 16k, and with what chunk size? Not stated.
