# malariaART

Predicting artemisinin partial resistance (ART-R) in *Plasmodium falciparum*
directly from protein sequence, using **ESM-3 protein-language-model embeddings**.
This repository holds the code, inputs, and trained models for that method, plus a
56-gene deep mutational scan built on top of it.

---

## The method

For each parasite isolate we take the amino-acid sequence of a gene and ask a
protein language model what it "thinks" of that sequence, then let a simple
classifier map that representation to a resistance phenotype.

1. **Sequence.** Per-isolate genotypes (MalariaGEN Pf7) are translated against the
   PlasmoDB-66 3D7 reference to give one protein sequence per isolate per gene.
2. **Embedding.** ESM-3 Open Small (`esm3_sm_open_v1`, d_model 1536) is used as a
   **frozen feature extractor** — no gradients, no fine-tuning. Each residue gets a
   1536-d vector; a gene's isolate is summarised by mean-pooling over positions
   (a 1536-d protein vector).
3. **Classifier.** A per-gene model predicts ART-R from that vector. Four are
   trained — RandomForest, GradientBoosting, LogisticRegression, SVM — and the one
   with the best validation auROC is kept. RandomForest is the headline model.
4. **Phenotype.** ART-R is defined on clinical parasite clearance, independent of
   genotype: half-life **≥ 5 h** is resistant, `< 5 h` sensitive.
5. **Scope.** K13 is used to establish and benchmark the approach, which is then
   applied to 167 candidate genes and compared against single-variant predictors.
6. **Attribution.** To find which residue a gene's model relies on, we permute the
   amino acid at one position across isolates, **re-embed the changed sequence**, and
   re-score — the auROC drop is that position's importance. This prioritised
   **EXO E415G** and **ATP4 G1128R**, which were then tested by CRISPR–Cas9 editing.
7. **Mutational scan.** Separately, every possible substitution (position × 19
   residues) is scored across 56 genes — 1,725,713 mutations in all.

### Cohort and split

1,293 clinical isolates (2011–2018), split by year so the model is always tested on
the future:

| Split | Years | n | resistant / sensitive |
|---|---|---|---|
| train | ≤ 2014 | 769 | 254 / 515 |
| validation | 2015–2016 | 315 | 154 / 161 |
| test | ≥ 2017 | 209 | 150 / 59 |

K13 reaches auROC **0.88** (validation) and **0.90** (test); across the 167-gene
panel, 77 genes gain ≥ 0.4 auPR over their best single variant.

---

## The pipeline (`code/`)

Numbered scripts `00`–`06` run in order; each resolves its paths through
`config.yaml`.

| Script | Does |
|---|---|
| `00_build_cohort.py` | assemble the 1,293-isolate cohort, labels, and temporal split |
| `01_build_gene_panel.py` | the candidate-gene panel and its per-gene provenance |
| `02_extract_esm3_features.py` | embed isolate sequences with frozen ESM-3 (GPU) |
| `03_train_classifiers.py` | train RF / GB / LR / SVM per gene; keep best by validation auROC |
| `04_single_variant_gwas.py` | Fisher tests + per-variant auROC (the single-variant comparator) |
| `05_permutation_importance.py` | input-level permutation with re-embedding (EXO / ATP4) |
| `06_build_figure_source.py` | one tidy source-data table per figure panel |

Shared modules: `config.py` (paths), `cohort.py` (labels + split), `sequences.py`
(FASTA + isolate sequences), `tables.py` (published-table readers), `models.py`
(model IO + pooling). ESM-3 is wrapped in `code/implementation/`.
`code/mutagenesis_rf/` holds the 56-gene scan (shipped as run, with its own README);
`code/postprocess/` turns the scan into per-variant rank scores.

---

## What is in the repo

```
config.yaml            every path and analysis parameter
code/                  the pipeline (00-06), shared modules, ESM-3 wrapper, the DMS
data/raw/              reference proteome/CDS, gene lists
data/example/          5 example isolate sequences (2011-2013) — input-format illustration
data/published_tables/ the supplementary tables (.xlsx + CSV exports)
data/figure_source/    one CSV per figure panel
results/               trained per-gene models, permutation outputs, the 56-gene scan
environment/           conda + pip specifications
```

The ESM-3 feature cache is **not** shipped; regenerate it from isolate sequences
(obtained per **Data availability**) with `02_extract_esm3_features.py`.

---

## Running it

**CPU only** — everything except embedding. Six packages, no torch/esm/GPU, Python ≥ 3.10:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements-core.txt
python code/06_build_figure_source.py       # rebuild the figure tables from shipped results
```

**With a GPU** — to embed sequences or run the scan (adds `torch` + `esm`, CUDA GPU).
Isolate sequences are not shipped (see **Data availability**); these commands assume
you have placed per-gene sequences under `data/raw/sequences/`. The five sequences in
`data/example/` illustrate the expected format.

```bash
pip install -r environment/requirements-esm.txt
huggingface-cli login                        # ESM-3 Open Small is gated; ~5 GB, downloads once
python code/02_extract_esm3_features.py --genes PF3D7_1343700 --out features
python code/03_train_classifiers.py --genes PF3D7_1343700 --features features
```

One K13 gene is ~140 s on CPU; the full panel wants a GPU (developed on an RTX 3090).
Package versions are pinned to those that produced the results (numpy 1.26.4,
pandas 2.1.4, scipy 1.15.0, scikit-learn 1.6.0; torch 2.5.1 / CUDA 12.4).

---

## Data availability

Individual-level data are **not** included in this repository — specifically the
per-isolate protein sequences, the clinical parasite-clearance (ART-R) phenotypes,
and the per-sample metadata. They are available from their original sources (see
**Data sources** below): the clearance phenotypes from Zhu et al. 2018 (*Nat Commun*
9:5158, MOESM20) and Zhu et al. 2022 (*Commun Biol* 5:274), and genotypes / sample
metadata from MalariaGEN Pf7 via `malariagen_data`; per-isolate sequences are
reconstructed by translating Pf7 genotypes against the PlasmoDB-66 3D7 reference.

Consequently the cohort-building and sequence-consuming steps (`00_build_cohort`,
`02_extract_esm3_features`, `04_single_variant_gwas`) cannot be run end-to-end from
this repository alone. `data/example/` ships five isolate sequences (2011–2013, K13)
to illustrate the input format only — no phenotype is included. Everything downstream
of the embeddings — trained models, permutation outputs, figure tables, and the
aggregate supplementary tables — is shipped and runs as-is.

---

## Data sources

| Input | Source |
|---|---|
| Genotypes, sample metadata | MalariaGEN Pf7, via `malariagen_data` |
| TRAC-I clearance times | Zhu et al. 2018, *Nat Commun* 9:5158, MOESM20 |
| TRAC-II clearance half-lives | Zhu et al. 2022, *Commun Biol* 5:274 |
| Reference proteome / CDS | PlasmoDB release 66, *P. falciparum* 3D7 |
| Protein language model | ESM-3 Open Small, Hayes et al. 2025, *Science* |
| Evidence scores | Oberstaller et al. 2021, *IJP-DDR* 16:119–128 (PMC8187163) — **not redistributed here**; download the supplement into `data/raw/external/` |
| Transcriptomic benchmark | GuanLab/Predict-Malaria-ART-Resistance (LightGBM) |
| Parasite lines | BEI Resources MRA-1236 (Cam2, K13 C580Y), MRA-1254 (Cam2_rev) |

---

## Not in the repo

The figures are drawn outside this tree, so `data/figure_source/` ships per-panel
tables rather than plotting code. The ring-stage-survival measurements behind the
CRISPR panel are wet-lab data and are not included; `data/figure_source/fig5c_rsa.csv`
carries the clone/edit structure only.

---

## A caveat on the mutational-scan rank scores

The 56-gene scan's rank scores reflect training data as much as biology:
WHO-validated K13 markers that appear in the training isolates sit at a median
99.8th percentile, while the two that do not sit at a median 45th. Run
`python code/postprocess/literature_validation.py --report` for the
leakage-controlled breakdown before reading a high rank score as a signal.
