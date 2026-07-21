# Reproduction plan — Stage 1

**Goal:** reproduce LCLM Tables 7 and 9 for AM-Fast on `Qwen/Qwen3-4B-Instruct-2507`,
on a 4-task subset, then use that as the platform for the diagnostics.

**Target numbers** (LCLM, AM-Fast, 100 samples/task should land within a few points):

| Task | 4k @16x | 16k @16x | 16k @4x | Why this task |
|---|---:|---:|---:|---|
| `niah_single_3` (ns3) | 7.80 | 0.20 | 80.20 | UUID, ratio-dependent |
| `niah_multikey_3` (nm3) | 3.80 | 0.00 | 91.20 | UUID x2, the worst cell |
| `cwe` | 76.82 | 55.44 | 79.80 | AM's win — mass task |
| `niah_single_1` (ns1) | 100.00 | 52.05 | 60.00 | control; ratio-*independent* break |

Full-KV baselines to confirm first: 4k → 100/99.80/97.50/100, 16k → 100/99.80/58.04/100.

---

## A. Environment (HPC: Torch cluster)

- [ ] A1. Confirm cluster basics — scheduler (Slurm?), GPU types/partitions, module
      system, whether internet egress is available on compute nodes
- [ ] A2. Create conda env `am` with py3.12; install pinned
      `torch==2.8.0 transformers==4.57.1 vllm==0.11.0 accelerate==1.12.0 datasets==4.4.1`
      (from `official/requirements.txt`)
- [ ] A3. Verify CUDA build matches cluster drivers (`torch.cuda.is_available()`,
      `torch.version.cuda`, `nvidia-smi`)
- [ ] A4. Set `HF_HOME` / `HF_HUB_CACHE` to scratch, **not** home (quota)
- [ ] A5. `huggingface-cli login` if needed; confirm no gated-repo issues
- [ ] A6. Freeze working env to `env/environment.yml` + `env/pip-freeze.txt`, commit

## B. Data prep

- [ ] B1. Pre-download `simonjegou/ruler` configs `4096` and `16384` to scratch cache
      (compute nodes may be offline — do this on a login node)
- [ ] B2. Pre-download `Qwen/Qwen3-4B-Instruct-2507` weights to scratch cache
- [ ] B3. Sanity-check row counts and task blocks match `RULER_LAYOUT.md`
      (13 x 500 = 6500 for 4096; **verify separately for 16384**, ordering is not
      guaranteed to match)
- [ ] B4. Decide sampling: `--dataset-name ruler_4k_niah_multikey_3 --n-articles 100`.
      Set `PYTHONHASHSEED=0` (loader groups contexts by `hash(context)`)

## C. Code prep

- [ ] C1. Patch the `load_ruler_data` task-filter bug (reports `Available: []`) —
      small, and we rely on task filtering heavily
- [ ] C2. Add an assertion that `--dataset-name ruler_*_<task>` actually yields the
      expected task and sample count, so a silent empty filter can't pass
- [ ] C3. Confirm `Qwen3-4B-Instruct-2507` hits the `"qwen" in model_name.lower()`
      branch in `official/evaluation/utils.py:159` → uses the patched `Qwen3ForCausalLM`.
      **Critical:** the `else` branch silently falls back to `AutoModelForCausalLM`,
      where β is never applied and AM degrades without error. Add a hard failure
      instead of a warning.
- [ ] C4. Map ratios: LCLM's 16x/8x/4x = `--target-size 0.0625 / 0.125 / 0.25`
- [ ] C5. Write `experiments/run_ruler.py` (or a thin shell wrapper) that sweeps
      task x length x ratio and writes one JSON per cell into `results/`
- [ ] C6. Write `experiments/aggregate.py` → produces our version of Tables 7/9
      side-by-side with LCLM's published numbers and a delta column

## D. Smoke tests (cheap → expensive)

- [ ] D1. `python -m examples.qa_demo --model Qwen/Qwen3-4B-Instruct-2507 --target-size 0.1`
      — confirms the whole pipeline on one short article
- [ ] D2. RULER 4k, ns1, **5 samples**, `--methods original` only → expect ~100%.
      Validates data loading + prompt format + scoring before any compaction
- [ ] D3. Same 5 samples, `--methods no_context` → expect ~0%. Confirms the scorer
      isn't accidentally passing
- [ ] D4. RULER 4k, ns1, 5 samples, AM-Fast @16x → expect ~100%
- [ ] D5. RULER 4k, nm3, 5 samples, AM-Fast @16x → expect ~0-5%. **This is the
      real signal**: if nm3 is not near-zero, our config differs from LCLM's
- [ ] D6. Record wall-clock per sample to size the full jobs

## E. Full Stage-1 runs

- [ ] E1. 4k x {16x} x {ns3, nm3, cwe, ns1} @ 100 samples + full-KV baseline
- [ ] E2. 16k x {16x, 4x} x same 4 tasks @ 100 samples + full-KV baseline
      (4x included specifically to test the ratio-independence of the ns1 break)
- [ ] E3. Compare against the target table above; investigate any cell off by >10 pts
- [ ] E4. Log wall-clock per task for the benchmarking TODO

## F. Open config questions to resolve before E
- [ ] F1. Did LCLM use **chunked** compaction at 16k, and at what chunk size? Not
      stated in the paper — check `github.com/LeonLixyz/LCLM`
- [ ] F2. What exactly is **AM-Slow**? Probably AM-OMP w/ repeat-prefill (they say
      they don't use self-study on RULER) — confirm in their code
- [ ] F3. Did they use the **nonuniform head budget** (`optimized_agnostic.json`) or
      uniform? Materially changes results
- [ ] F4. Is their "16x" measured over article tokens only or the whole sequence?
      (`--ignore-article-indices` toggles this in our repo)

---

## Info needed from Manas about the HPC

1. Scheduler + a working example job script (partition, GPU type, time limits, account)
2. GPU model and count per node — H100/H200/A100? 4B model at 16k should fit on one
   80GB card comfortably, but compaction peak memory is the thing to watch
3. Whether compute nodes have internet (decides if all downloads must be pre-staged)
4. Scratch/work path and quota, for `HF_HOME` and `results/`
5. Whether conda/mamba is available or if it's module-based / container-based (Apptainer?)
