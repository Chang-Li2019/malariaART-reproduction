# `mutagenesis_rf/` — the 56-gene deep mutational scan

These seven files are shipped **byte-identical to the versions that produced the
results in `results/mutagenesis_56genes/`**. They were independently audited
(see `results/mutagenesis_56genes/INDEPENDENT_AUDIT.md`, verdict: no correctness
bugs), so they are deliberately not refactored — editing them would invalidate
that audit.

> **Data note:** running the scan needs the per-isolate sequences and clinical
> ART-R phenotypes, which are **not** included in this repository — see *Data
> availability* in the top-level `README.md`. The shipped scan outputs in
> `results/mutagenesis_56genes/` do not require them.

The one consequence: they carry absolute paths to the original machine. To run
them elsewhere, edit the three constants at the top of `rf_pipeline.py`:

```python
ROOT = Path("/mnt/Work1/changli/MalariaGen")   # -> your checkout
PLM  = ROOT / "malariaPLM"                      # -> where rf_models_final lives
DATA = ROOT / "malariaART" / "data"             # -> where all_seqs/ lives
```

`MALARIAGEN_DATA_DIR` is also respected as an environment variable. The rest of
the release resolves paths through `config.yaml`; this subdirectory is the
exception, by design.

## What each file does

| File | Role |
|---|---|
| `rf_pipeline.py` | Shared library: lazy ESM-3 loader, `full_embedding` / `pool_to` / `featurize` (self-detecting pooling), `load_labels` (>=5 h clearance, temporal split), model IO |
| `train_rf_all.py` | Phase 0 — CPU-only retrain of one `RandomForestClassifier(n_estimators=100, random_state=42)` + `StandardScaler` per gene from cached features; writes `{gene}_rf.pkl` + `training_summary.csv` |
| `rf_mutagenesis.py` | `MutagenesisRF` — single-gene DMS in three modes: `scan_exact`, `scan_windowed` (approximate), `scan_chunkreuse` (exact, for proteins > 1500 aa) |
| `run_all.py` | Resumable batch runner, shortest gene first; defers proteins above `--max_len` |
| `validate_windowed.py` | Pre-flight on K13: confirms pooling reproduces the cached features, and measures windowed-vs-exact agreement |
| `validate_chunkreuse.py` | Hard gate: proves chunk-reuse stitching is numerically exact against naive `get_LL` on a giant protein; exits non-zero on failure |
| `run_giants_queued.sh` | Waits for the small-gene run, runs the chunk-reuse gate, then launches the 21 large proteins |

## How the shipped results were produced

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALARIAGEN_DATA_DIR=/path/to/malariaART/data
PY=/path/to/envs/esm_env/bin/python

# Phase 0 — retrain all 56 RF models from cached features (CPU, minutes)
$PY train_rf_all.py --genes ../../data/raw/gene_lists/dms_genes_56.txt \
                    --out ../../results/rf_models_final --seed 42

# Phase 1 — the 35 proteins <= 1500 aa, exact scan (~27 GPU-hours)
$PY -u run_all.py --max_len 1500

# Phase 2 — the 21 proteins > 1500 aa, chunk-reuse (~6 GPU-days)
bash run_giants_queued.sh <PID_OF_PHASE_1>
```

Total: **1,725,713 mutations** across 56 genes (every position x 19 substitutions).

## Two things worth knowing

**The windowed mode is not usable.** `scan_windowed` was written as a 10-50x
speedup and then rejected: it correlates only r <= 0.55 with the exact scan.
`scan_chunkreuse` is the accepted optimisation — it recomputes only the tiling
windows containing the mutation and reuses the wild-type windows, which is
algebraically identical to a full re-embedding (validated to `max|delta| = 0.0` on
predicted probabilities). Use `exact` or `chunkreuse`, never `windowed`.

**GPU batching does not help.** A single ESM-3 forward pass already saturates an
RTX 3090 at ~8.7 sequences/s, and batching OOMs by batch size 12.

## Verifying the outputs

```bash
# chunk-reuse exactness gate (needs a GPU)
$PY validate_chunkreuse.py --gene PF3D7_1451200 --n_mut 20

# rankscore transform, all 56 genes (CPU, seconds)
$PY ../postprocess/make_rankscore.py --all --check

# literature markers vs DMS percentiles, with training-leakage control
$PY ../postprocess/literature_validation.py --check --report
```
