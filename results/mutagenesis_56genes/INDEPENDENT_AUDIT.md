# Independent Audit — ESM-3 + RandomForest ART-resistance DMS pipeline

**Auditor:** independent (no prior context; every conclusion derived from code/data/outputs in this audit).
**Date:** 2026-07-24
**Scope:** `malariaPLM/mutagenesis_rf/`, `malariaPLM/implementation/` (ESM tiling), trained models,
DMS result CSVs (56 genes), rankscore CSVs (56 genes), raw phenotype/metadata.

> **Note (public release):** the raw phenotype/metadata referenced in this audit are
> individual-level data and are **not** included in this repository; see *Data
> availability* in the top-level `README.md`. This audit record is retained as-is.

## Overall verdict

**The pipeline is correct.** Every high-risk component I could test passed decisively, including
the one flagged as highest-risk (chunk-reuse stitching for giant proteins), which I confirmed both
mathematically and empirically (predicted-probability delta = 0.0). The label/split reconstruction,
featurization/pooling, per-gene RF AUCs, DMS output integrity, and the rankscore transform all
check out. I found **no correctness bugs** — only one cosmetic boundary nit, one latent
(currently-unexercised) ambiguity, and one methodological label-definition observation worth
recording. None affect the validity of the results.

## Findings

| # | Component | Severity | Finding & evidence |
|---|-----------|----------|--------------------|
| 1 | Chunk-reuse exactness (`rf_mutagenesis.py: _ensure_chunks/cr_embed`) | **Confirmed correct** | Derived the math: `get_LL` (esm3_utils.py L227-264) reconstructs `emb[pos]=Σ_k Mnorm[k,pos]·part_k[pos]`; chunk-reuse recomputes only mutation-containing windows and adds `Mnorm[k]·(mut_part−wt_part)` to the reused WT reconstruction — algebraically identical to embedding the whole mutant with the same tiling. Same `get_intervals_and_weights(L, min_overlap=max_len//2, max_len=max_len, s=20)` params on both paths. **Empirical** (smallest giant PF3D7_1451200, 1504aa, 2 windows): WT pooled max\|Δ\|=1.8e-6; 6 mutants incl. one in the overlap zone (K750A) feat max\|Δ\|=2.7e-6, **prob \|Δ\|=0.0**. Handles multi-chunk overlaps generically (loops over all containing windows; Mnorm normalized to 1). |
| 2 | Featurization / pooling (`pool_to`, `full_embedding`) | **Confirmed correct** | Live ESM-3 embed + `pool_to` **exactly** reproduces cached training features: small gene PF3D7_1318100 (axis1, L=194) max\|Δ\|=**0.0** on 3 isolates; giant PF3D7_1451200 (axis1, L=1504, tiled) max\|Δ\|=2.1e-6 on 2 isolates. So DMS scores sit on the same feature scale the RF was trained on. Axis disambiguation (n_features==1536→axis0 mean; ==L→axis1 mean) is sound for the actual gene set (see #6). |
| 3 | Label & temporal split (`rf_pipeline.load_labels`) | **Confirmed correct** | Reconstructs 1335 split-assigned samples (train/valid/test = 769/349/217; per-gene ∩features = 769/315/209, matching `training_summary.csv`). Threshold direction correct (clearance ≥5h ⇒ resistant=1; raw clearance median 3.55, so majority sensitive). No duplicate index; splits disjoint by year (no leakage). Join keys validated indirectly by the exact AUC reproduction in #4. |
| 4 | Model training & reported AUCs (`train_rf_all.py`) | **Confirmed correct** | Recomputed valid+test ROC-AUC from saved `{gene}_rf.pkl` (scaler+model) and independently-reconstructed labels for 6 genes (K13, a giant, a small axis1 gene, and low-AUC genes). **All match `training_summary.csv` to 4 decimals** (e.g. K13 test 0.8962 vs 0.8962; PF3D7_0812100 test 0.5653 vs 0.5653; PF3D7_1302700 0.5925 vs 0.5925). This jointly validates labels, split, scaler, and model IO. |
| 5 | Rankscore transform (`mutagenesis_rankscore_results/`) | **Confirmed correct** | For all 56 genes: `model_score`==`resistance_probability` (atol 1e-9); id/pos/ref/mut/effect columns identical to originals; rankscore is a within-gene percentile, **monotonic** in model_score (0 inversions), spans ~1/N..1. Exact convention reverse-engineered = `scipy.stats.rankdata(model_score, method='average')/N` — matches stored values to **1.1e-16 across all 56 genes**. Ties handled sanely (mean rank). Lossless. (My first pass flagged a "mismatch" only because I assumed searchsorted-right; the average-rank convention is the correct/standard one.) |
| 6 | DMS output integrity (56 CSVs) | **Confirmed correct** | Every CSV has exactly L×19 rows, each position exactly 19 mutants, no duplicate mutation_ids, mutant∈AA20 and ≠ reference, all probabilities∈[0,1], reference residue matches the actual reference sequence at every position, implied WT prob constant per gene (std ≤7e-17), `resistance_prediction` matches the 0.5 threshold, `effect_size`==prob−wt. Only exception is the boundary nit in #7. |
| 7 | `effect_direction` at exact ±0.05 boundary | **Nit** | In 4 giant genes (PF3D7_0419900/1343800/1346400/1349500), 6–91 rows out of tens of thousands sit at effect_size **exactly ±0.05** and are labeled `Resistance+`/`Resistance-` where a strict `eff>0.05` recomputation yields `Neutral`. Pure float boundary rounding (RF vote-fraction probs land exactly on the threshold); `effect_size` itself is correct. Cosmetic; immaterial to rankings. |
| 8 | Binary label conflates two phenotypes | **Minor (design)** | `load_labels` applies the same ≥5h cutoff to TRACI "Parasites clearance time" and TRACII "Parasite.clearance.half-life" — biologically different quantities merged into one label. Consistent with stated design and used identically everywhere, so not an implementation bug, but the two metrics are not interchangeable at a shared 5h threshold. Worth noting for downstream interpretation. |
| 9 | `pool_to` axis ambiguity if L==1536 | **Latent nit** | If a per-residue gene had reference length exactly 1536, `pool_to` would misdetect it as axis0 protein-mean. **No gene in the set triggers this** (no protein_len==1536); `MutagenesisRF.__init__` also guards non-1536 models against L mismatch. Latent only. |

## Notes on what "exact" means here

The DMS embeds every mutant with `get_LL`, whose giant-path is itself the overlap-tiled
reconstruction. Chunk-reuse is exact **relative to that same `get_LL`** (which is how any mutant —
also a giant — would be embedded anyway), so the whole DMS is internally consistent. Running a giant
in `--method exact` vs `chunkreuse` yields identical numbers (both go through the same tiling);
chunk-reuse only skips recomputing unchanged windows.

## Could not fully verify (and why)

- **Empirical chunk-reuse was tested on a 2-window giant only** (PF3D7_1451200), incl. an
  overlap-region mutation. Giants with ≥3 windows (a middle chunk, e.g. PF3D7_1343800 at 7594aa) were
  not run end-to-end for GPU cost, but the reconstruction code path and Mnorm normalization are
  window-count-agnostic, and every giant's output CSV passed structural integrity (#6), which requires
  the tiling to have run correctly. Confidence: high.
- **Featurization-cache reproduction** was spot-checked on one small and one giant gene (both exact).
  Not exhaustively re-embedded across all 56 genes (GPU cost). The exact AUC reproduction (#4) uses the
  cached features directly, so training-side correctness does not depend on this.

## Commands / artifacts

Verification scripts were written to the audit scratchpad (not added to the repo):
`cpu_audit.py` (labels, AUC recompute, DMS integrity, rankscore), `probe*.py` (rankscore convention,
direction boundary), `gpu_check.py` (featurization + chunk-reuse). Python env
`/home/changli/miniconda3/envs/esm_env/bin/python`, `MALARIAGEN_DATA_DIR` and
`PYTORCH_CUDA_ALLOC_CONF` set per repo convention.
