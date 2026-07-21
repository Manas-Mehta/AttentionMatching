# RULER dataset layout (`simonjegou/ruler`)

Verified against the HF datasets-server on 2026-07-21, config `4096`, split `test`.

**6500 rows = 13 tasks x 500 contiguous examples.** Tasks are *not* interleaved —
each task occupies one solid 500-row block. The upstream `scripts/v2/qwen-ruler.sh`
exploits this: its SLURM array steps `--start-article` by 500 with `--n-articles 50`,
so **each array element is one task, first 50 examples**.

That is also how the paper's RULER numbers were produced — 50 samples per task,
not all 500.

| Block | `--start-article` | Task |
| ---: | ---: | --- |
| 0  | 0    | `niah_single_1` |
| 1  | 500  | `niah_multikey_2` |
| 2  | 1000 | `fwe` |
| 3  | 1500 | `qa_1` |
| 4  | 2000 | `niah_single_3`  <- **ns3** |
| 5  | 2500 | `niah_multikey_1` |
| 6  | 3000 | `niah_single_2` |
| 7  | 3500 | `cwe` |
| 8  | 4000 | `qa_2` |
| 9  | 4500 | `vt` |
| 10 | 5000 | `niah_multivalue` |
| 11 | 5500 | `niah_multikey_3` <- **nm3, our main target** |
| 12 | 6000 | `niah_multiquery` |

## Two ways to select a task

**(a) By name** — `load_ruler_data` accepts a task filter parsed from the dataset name:

```bash
--dataset-name ruler_4k_niah_multikey_3 --n-articles 100 --start-article 0
```

Clearer and length-independent. Note the block table above assumes the 4096
config; row ordering is not guaranteed identical across `8192` / `16384`, so
prefer name-based filtering when sweeping context length.

**(b) By offset** — matches the paper's scripts exactly, useful for reproduction:

```bash
--dataset-name ruler_4k --n-articles 50 --start-article 5500
```

## Known upstream bug

In `official/evaluation/datasets.py`, `load_ruler_data`:

```python
rows = [r for r in rows if r['task'] == task_filter]
if not rows:
    available_tasks = sorted(set(r['task'] for r in rows))   # rows is already empty
    raise ValueError(f"No examples for task '{task_filter}'. Available: {available_tasks}")
```

A typo'd task name reports `Available: []` instead of the real task list. Harmless
but confusing — worth patching since we will use task filters heavily.

## Grouping caveat

`load_ruler_data` groups rows by `hash(context)`. RULER contexts are essentially
unique per example, so **1 article ≈ 1 question** and `--n-articles 100` ≈ 100 samples.
Worth asserting in our runner rather than assuming, since Python's `hash()` on
`str` is randomized per process (`PYTHONHASHSEED`) — grouping is stable *within*
a run but article ordering could shift *between* runs. Set `PYTHONHASHSEED=0`
for reproducibility.
