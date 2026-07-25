# malariaART — reproduction package

Protein language model prediction of artemisinin partial resistance (ART-R) in
*Plasmodium falciparum*, plus a 56-gene deep mutational scan.

This folder reproduces the numbers behind the manuscript, and says plainly which
ones it cannot. Start with `verify/`, read `CONFLICTS.md`, then go deeper if needed.

```bash
python verify/verify_manuscript_claims.py
```

Seconds to run, no GPU, no network. Current result: **43 PASS, 8 FAIL, 2 MISSING**.
The 8 FAILs are manuscript/data discrepancies, not script errors — each is
documented in **[CONFLICTS.md](CONFLICTS.md)**. Read that file before citing anything.

---

## What is here

```
config.yaml            every path and analysis parameter, in one place
CONFLICTS.md           manuscript vs data discrepancies (read this)
PROVENANCE.md          figure/table -> source file -> script, one row each
verify/                Tier 0: recompute every manuscript claim
code/                  the pipeline, 00 -> 06, plus shared modules
data/raw/              inputs: phenotypes, cohort, sequences, reference, gene lists
data/published_tables/ the supplementary tables as published (.xlsx + CSV exports)
data/figure_source/    one tidy table per figure panel  <- give these to whoever draws the figures
results/               trained models, the 56-gene DMS, permutation outputs
environment/           conda + pip specifications
```

Total ~360 MB. The ESM-3 feature cache (33.4 GB) is deliberately **not** shipped —
see Tier 2 below.

### Code layout

The numbered scripts `00`–`06` are the pipeline. They share five small modules,
each with one responsibility:

| Module | Holds |
|---|---|
| `code/config.py` | `load_config()` — resolves every path in `config.yaml` |
| `code/cohort.py` | labels and the temporal train/valid/test split |
| `code/sequences.py` | the reference FASTA and per-gene isolate sequences |
| `code/tables.py` | readers for the published tables and gene lists |
| `code/models.py` | loading RF models and turning embeddings into predictions |

`code/mutagenesis_rf/` (the 56-gene scan) is shipped byte-identical to the
independently-audited version and is the one part not written to this package's
code style — see its own README.

---

## The four tiers

Pick the depth you need. Every number in the paper is reachable from Tier 0 or Tier 1.

### Tier 0 — verify (seconds, CPU)

Recomputes every numeric claim from the shipped published tables.

```bash
python verify/verify_manuscript_claims.py --csv verify/verification_report.csv
```

### Tier 1 — rebuild the analysis (minutes to hours, CPU)

```bash
python code/00_build_cohort.py                     # 1293 isolates, 769/315/209
python code/01_build_gene_panel.py                 # the 173 -> 171 -> 167 chain
python code/04_single_variant_gwas.py --genes PF3D7_0709000
python code/06_build_figure_source.py              # regenerate data/figure_source/
python code/postprocess/make_rankscore.py --all --check
python code/postprocess/literature_validation.py --check --report
```

Model retraining (`code/03_train_classifiers.py`) also lives in this tier but needs
features, so run Tier 2 for at least one gene first.

### Tier 2 — re-derive features from sequence (GPU)

The ESM-3 feature cache is 33.4 GB for the full panel and is not shipped. Regenerate
it from the included isolate sequences:

```bash
# one gene, minutes — validates the whole chain cheaply
python code/02_extract_esm3_features.py --genes PF3D7_1343700 --out features
python code/03_train_classifiers.py --genes PF3D7_1343700 --features features
# expect: valid auROC 0.8796, test auROC 0.8962

# the full panel, GPU-hours to days
python code/02_extract_esm3_features.py --all --out features
```

Needs the `esm_env` environment and a CUDA GPU (developed on an RTX 3090). ESM-3 is
used strictly as a **frozen** feature extractor — no gradients, no fine-tuning.

### Tier 3 — the 56-gene deep mutational scan (GPU-weeks; outputs shipped)

1,725,713 mutations (every position × 19 substitutions) across 56 genes. The finished
scan is in `results/mutagenesis_56genes/`, so nobody has to rerun it. The code is in
`code/mutagenesis_rf/` — shipped byte-identical to the audited version, with its own
[README](code/mutagenesis_rf/README.md) and an independent audit in
`results/mutagenesis_56genes/INDEPENDENT_AUDIT.md`.

---

## Environment

```bash
conda env create -f environment/environment.yml   # creates esm_env
conda activate esm_env
```

Developed with torch 2.5.1 + CUDA, esm 3.1.2, scikit-learn 1.6.0, numpy 1.26.4,
pandas 2.1.4, scipy 1.15.0. Tier 0 and most of Tier 1 need only numpy/pandas/scipy/
scikit-learn/pyyaml. ESM-3 weights (~5 GB) download once from HuggingFace on first use.

`code/convert_supplementary_to_csv.py` is the only script needing `openpyxl`; its
outputs are already committed, so you do not need to run it.

---

## Data sources

| Input | Source |
|---|---|
| Genotypes, sample metadata | MalariaGEN Pf7, via `malariagen_data` |
| TRAC-I clearance times | Zhu et al. 2018, *Nat Commun* 9:5158, supplementary MOESM20 |
| TRAC-II clearance half-lives | Zhu et al. 2022, *Commun Biol* 5:274 |
| Reference proteome / CDS | PlasmoDB release 66, *P. falciparum* 3D7 |
| Protein language model | ESM-3 Open Small (`esm3_sm_open_v1`, d_model 1536), Hayes et al. 2025 *Science* |
| Evidence scores (Fig 3B/C) | Oberstaller et al. 2021, *IJP-DDR* 16:119–128, supplementary `mmc1.xlsx` (PMC8187163) |
| Transcriptomic benchmark | GuanLab/Predict-Malaria-ART-Resistance (LightGBM) |
| Parasite lines | BEI Resources MRA-1236 (Cam2, K13 C580Y) and MRA-1254 (Cam2_rev) |

---

## Known limits

Three things in the manuscript cannot be reproduced from any data that exists:

1. **Ring-stage survival values (Fig 5C)** — no raw or summarised numbers survive.
2. **ATP4 100-permutation result (Fig 5B)** — the pickle does not exist.
3. **Figure rendering** — the published figures were drawn outside this tree, so
   this package ships per-panel source data rather than plotting code.

And two published claims do not hold on recomputation: the auPR-gain count is 77/167
rather than 78/167, and K13's validation auROC is 0.8796 rather than "higher than
0.88". Full list in [CONFLICTS.md](CONFLICTS.md).

---

## Citation

If you use the 56-gene mutational scan, note that its rank scores reflect training
data as much as biology: WHO-validated K13 markers that appear in the training
isolates sit at a median 99.8th percentile, while the two that do not sit at a median
45th. Run `python code/postprocess/literature_validation.py --report` for the
leakage-controlled breakdown before drawing conclusions from a high rank score.
