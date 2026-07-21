# AttentionMatching

Experiments on **Attention Matching (AM)** for fast KV-cache compaction, extending
[Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284)
(Zweiger, Fu, Guo, Kim — ICML 2026).

Primary focus: **RULER** tasks (especially `niah_multikey_3`) across newer model
families, and understanding *why* AM degrades on retrieval-style tasks while
holding up on aggregation tasks (CWE/FWE).

## Layout

| Path | Purpose |
| --- | --- |
| `official/` | Vendored copy of [adamzweiger/compaction](https://github.com/adamzweiger/compaction) @ `2020f6d`. We patch this freely; changes are committed **here only** and never sent upstream. |
| `experiments/` | Our own runners, model-support patches, and analysis code. |
| `slurm/` | HPC batch scripts. |
| `env/` | Environment setup (conda / pip pins). |
| `results/` | Run outputs — **gitignored**. |

`TODO.md` is the working goal list and is intentionally **gitignored**.

### Why `official/` is a plain vendored copy

Commit `dd99b04` is an unmodified snapshot of upstream. Every change we make
shows up as a clean diff against it:

```bash
git log --oneline -- official/     # everything we changed since vendoring
```

To see upstream's own newer work, compare against a fresh clone — we do not
track them as a git remote.

## Running

All upstream entry points expect to run from the `official/` directory:

```bash
cd official
python -m evaluation.run_qa_evaluation \
  --dataset-name ruler_4k_niah_multikey_3 \
  --model-name Qwen/Qwen3-4B \
  --methods original no_context highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy \
  --target-size 0.05 \
  --query-config repeat \
  --algorithm-config best
```

## Citation

```bibtex
@misc{zweiger2026fastkvcompactionattention,
      title={Fast {KV} Compaction via {Attention Matching}},
      author={Adam Zweiger and Xinghong Fu and Han Guo and Yoon Kim},
      year={2026},
      eprint={2602.16284},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.16284},
}
```

Upstream code is under its original license — see [`official/LICENSE`](official/LICENSE).
