# Manuscript ↔ data discrepancies

Every item below was found by recomputing a manuscript claim from the data in this
release. Run `python verify/verify_manuscript_claims.py` to reproduce the whole
list; it currently reports **43 PASS, 8 FAIL, 2 MISSING**.

Nothing here has been "fixed" in the manuscript. This file records what the data
says so the authors can decide.

---

## 1. Results and Discussion describe a different model from the Methods

**Severity: high — the paper describes two mutually exclusive architectures.**

| Section | Description |
|---|---|
| Results | "our model incorporates parallel convolutional layers with kernel sizes of 3, 7, and 11 … followed by global pooling and a fully connected classifier" |
| Discussion | "We leverage transfer learning by **fine-tuning** a pretrained ESM3" |
| Methods | ESM-3 "used exclusively as a **frozen feature extractor**. No fine-tuning of ESM3 parameters was performed" + four sklearn classifiers |
| Supplementary Fig. S1 | draws the CNN: "ESM3 Embedding (1536D) → Linear Projection 128-D → CNN k=3 / k=7 / k=7 / k=11 → Global Avg Max Pool → Sigmoid" |

**What the plotted numbers actually are.** K13's values in Supplementary Table 4,
`auROC_PR.*.txt` and `snp_ml_comparison.csv` are identical to 17 significant digits
to the RandomForest row of the frozen-ESM-3 pipeline (valid auROC 0.879567637331613,
test 0.896214689265537). **Figures 1B, 2 and 4 are the frozen-ESM-3 + RandomForest
model**, matching the Methods, not the Results prose.

The CNN does exist and was trained (valid auROC 0.8800, test 0.8907 — within 0.006
of the RF), and it *is* the model behind Figure 1C and the Figure 1D predictions.
So the paper genuinely uses two models, but attributes all of them to the CNN.

Two further inconsistencies inside the CNN description:

* Supplementary Fig. S1 shows kernel sizes 3, 7, 7, 11 — the text says 3, 7, 11.
* The saved CNN configs mostly disagree with "3, 7, 11": `[5,9,13]`, `[3,5,7,9]`,
  `[3,7,9]`. The config actually used for the Fig 1C mutational scan is **`[3,7,9]`**.

---

## 2. The 167-gene panel cannot be derived as the Methods describe

**Severity: high — the central analysis set is not reproducible from the text.**

Methods: *"This curation yielded 171 candidates. Genes lacking a processable protein
sequence in PlasmoDB-66 (i.e., not present on chromosomes 1–14) were excluded,
resulting in 167 genes included in the final analysis."*

What the data shows (`python code/01_build_gene_panel.py`):

```
PlasmoDB "artemisinin resistance" query        173
  - not on chromosomes 1-14                      2   (PF3D7_API01300, PF3D7_MIT02300)
  = after chromosome filter                    171   <- the Methods' "171"
  - dropped, reason undocumented                23
  + added literature ART-R genes                19
  = analysed panel                             167
```

So the chromosome filter is the **173 → 171** step, not 171 → 167. The remaining
step is a swap: 23 ordinary chromosomal genes are dropped (ABC transporters, MSP1,
DHFR-TS, KIC4, KIC5, PI3K, QRP1, …) and 19 literature ART-R genes are added that
were never in the PlasmoDB query (RAD5, ARPS10, PI4K, VPS13, WD11, SENP2, KIC-pathway
members, …). Neither step appears in the Methods.

Separately, the number **171** traces to `malariaART/Analysis/results_all_genes.csv`
— a **retired ProteinBERT run** whose per-gene values differ from the published ones
(e.g. CRT train/valid/test 0.746/0.689/0.680 there vs 0.767/0.560/0.527 published).
Citing it alongside the ESM-3 analysis conflates two pipelines.

**Consequence:** `data/raw/gene_lists/analysed_panel_167.tsv` is shipped as primary
data. `data/raw/gene_lists/gene_panel_provenance.csv` gives a per-gene in/out reason.

---

## 3. Figure 5C mislabels the edited mutations

**Severity: high — confirmed by the corresponding author as a figure error.**

Figure 5C axis labels read **"EXO: V480L"** and **"ATP4: G128R"**. Everything else —
Results, Abstract, Methods, Supplementary Tables 8 and 9 — says **EXO E415G** and
**ATP4 G1128R**.

Settled by translating the Supplementary Table 9 HDR donor sequences against the
PlasmoDB-66 CDS (`check_crispr_donors` in the verification script):

| Gene | Donor spans | Amino-acid change | nt changes |
|---|---|---|---|
| EXO (PF3D7_1362500) | CDS codons 342–373 | **E415G** — the single nonsynonymous change | 12 (1 nonsyn + 11 synonymous shield) |
| ATP4 (PF3D7_1211900) | CDS codons 1059–1090 | **G1128R** | 20 (1 nonsyn + 19 synonymous shield) |

The constructs are E415G and G1128R. **The figure labels are wrong and must be
corrected to "EXO: E415G" and "ATP4: G1128R".** "G128R" is additionally a dropped
digit. V480L is the second-ranked EXO permutation variant (importance 0.0053 vs
E415G's 0.326), which is the likely source of the stale label.

Also inconsistent: Fig 5C shows `***`; the Results text says `p < 0.01`.

---

## 4. Supplementary Table 7 does not contain what it is cited for

**Severity: medium.**

* Table 7 is titled *"Permutation results for previously reported variants in EXO"*
  but contains **no EXO row and no ATP4 row**. Its six genes are exactly the Figure 4C
  set (MSH6, CRT, PF3D7_1119600, EXP1, GCN20, PRS), with RF test-set permutation columns.
* The Results cite *"Supplementary Table 7 and 8"* for the Figure 5A/B EXO/ATP4
  permutation analysis, but **Table 8 is the oligonucleotide table**.
* There are therefore two different "Table 8"s in the supplement (numbered `8` and
  `S8`), plus an `S9`. The Fig 5A/B supplementary table is missing entirely.

---

## 5. The permutation analysis used a different split and a different model

**Severity: medium — affects how Figure 5A/B should be described.**

* **Split.** Methods: positions were shuffled *"across all test-set samples"*. The
  code that produced the published figure (`malariaART/Analysis/figures.ipynb`,
  cell 20) evaluates on the **validation** split.
* **Model.** Figure 5A/B was produced with the retired **ProteinBERT** Keras models,
  not the frozen-ESM-3 models used for every other figure.

`code/05_permutation_importance.py` uses the validation split (a named `SPLIT`
constant, matching the published figure) and re-derives the analysis on the ESM-3
models rather than the retired ProteinBERT ones.

---

## 6. Numeric claims that do not hold

| Claim | Location | Recomputed | Status |
|---|---|---|---|
| "78/167 genes showed an increase of at least 0.4 in auPR" | Results, Fig 4B | **77/167** (46.1 %) | off by one; "over 46 %" still holds |
| "ROC-AUC higher than 0.88 in both the validation and test sets" | Results, Fig 1B | validation auROC = **0.8796** | rounds to 0.88 but is not above it |
| "both auROC and auPR > 0.9" on later years | Discussion | test auROC = **0.8962** (auPR 0.9257 does exceed 0.9) | auROC claim fails |
| Africa accuracy 0.82 | Results, Fig 1D | **0.828** (24/29) | rounds to 0.83 |
| "A small group of genes (CRT, MSH6, PF3D7_119600, GCN20, EXP1, PRS) showed higher auROC with the single-variant approach" | Results, Fig 4C | those six are exactly the **top six by margin**, but **56** genes have single-variant auROC > DL auROC | gene list correct; "a small group" is imprecise |
| Methods: 1,294 isolates | Methods | **1293** (1294 counts the header row) | Results' 1293 is correct |

Claims that **do** reproduce exactly: the 769/315/209 split; CRT 0.77/0.56/0.53 with
test auPR 0.75; WD11 0.48/0.78/0.72/0.71/0.82; ATP4 rank 54; SE Asia 0.93 and CDC 0.85;
Fig 3A Wilcoxon p < 0.05; genomic Spearman ρ ≥ 0.36; the 0.27–0.42 frequency range;
BTB/POZ + propeller enrichment in the K13 scan.

---

## 7. The cohort membership list is not derivable from the stated criteria

**Severity: medium.**

Methods: samples were filtered to those that *"passed MalariaGEN's quality control
criteria and had experimentally measured parasite clearance times"*.

Applying exactly that filter gives **1230** isolates (742 / 295 / 193). The published
cohort is **1293** (769 / 315 / 209) and corresponds precisely to the sample list in
`ARTR_metadata.csv` — which includes **101 samples marked `QC pass == False`**.

So the cohort, like the gene panel, has to be shipped as data rather than derived.
`cohort.load_labels()` intersects with that list and documents why.

---

## 8. Smaller items

* **Two ranking criteria.** Methods say genes were ranked by "validation-set
  (auROC+auPRC)/2" in one paragraph and by "validation-set auROC" two paragraphs
  later. `code/03_train_classifiers.py` ranks by validation auROC (the criterion
  that reproduces the published order).
* **Truncated gene ID.** The sentence "…including PF3D7_1346400, PF3D7_1349500, K13,
  RAD5, and PF3D7_" ends mid-identifier in the .docx — a broken EndNote field ate it.
  The intended gene is **PF3D7_1318300** (rank 5 in the test set; EXO is rank 6).
* **Malformed gene ID.** "PF3D7_119600" is not a valid identifier; it is
  **PF3D7_1119600** (a dropped digit).
* **Two transcriptomic benchmark numbers.** The Methods say the LightGBM values were
  "retrieved from pre-computed output files", but `transcriptomics_comparison.ipynb`
  **re-trains** the model locally, giving different numbers. Pick one. That notebook
  also has a real bug: cell 6 scores on unstandardised `valid_ds[exp_cols]` /
  `test_ds[exp_cols]` while the model was fit on scaled features.
* **The transcriptomic comparison is much weaker on validation than on test.** The
  new Results sentence says "sequence-derived predictive signals from multiple
  individual genes exceeded the performance of the previously published whole-genome
  transcriptional profile model (vertical line in Fig. 2)". Counting genes that beat
  the GuanLab reference: **test auROC 40/167 (24 %)** but **validation auROC only
  4/167 (2 %)**. A reader looking at Fig 2A will see almost every gene to the left of
  the line. Worth qualifying which panel the claim refers to.
* **Stale threshold file.** `malariaPLM/mutation_results.txt` used classification
  threshold 0.3402 while Supplementary Table 3 used 0.3387; 11 of 37 calls differ.
  Regenerating Fig 1D from that file will not give 0.93 / 0.82 / 0.85.
* **Label definition.** TRAC-I contributes *parasite clearance time* and TRAC-II
  contributes *clearance half-life*; the pipeline applies the same ≥ 5 h cutoff to
  both. Consistent throughout, so not an implementation bug, but the two quantities
  are not interchangeable at a shared threshold.
* **Feature convention is not uniform.** The Methods describe one pooling rule
  (mean over the embedding dimension → a vector of length = protein length). In
  practice the cached features are inconsistent across genes: K13 is 1536-dim
  (pooled over sequence positions) while others are L-dim. Each model's
  `n_features` identifies which; `models.pool_embedding` self-detects.
* **A prior audit in the tree contains an error.** `Methods_Rewritten.md` states
  that ESM-3 Open Small is 1152-dimensional. It is **1536** (`esm/pretrained.py:104`);
  1152 is ESMC-600M. Do not propagate that correction.

---

## 9. Data that does not exist anywhere

| Item | Consequence |
|---|---|
| **Ring-stage survival values (Fig 5C)** | No raw or summarised RSA numbers, and no plotting script, exist in the source tree. Fig 5C survives only as a flattened vector image. `data/figure_source/fig5c_rsa.csv` ships the clone/edit structure with the corrected labels, awaiting the measurements. |
| **ATP4 100-permutation result (Fig 5B)** | `PF3D7_1211900_permut100.pkl` does not exist; only a 30-permutation dictionary (G1128R importance 0.176, baseline 0.680). Note this contradicts the Results claim that EXO's baseline auROC exceeded ATP4's — on the 30-permutation baselines it is the other way round (0.662 vs 0.680). The claim *is* supported by the model performance tables (0.865 vs 0.688). |
| **Raw Sanger `.ab1` traces** | Only chromatogram JPEGs embedded in Supplementary Figs S2–S4. |
| **Plotting code for Figs 1–5** | Made in R/Excel outside the tree; there are zero `.R` files in the repository. This release ships per-panel source data instead. |

Recovered during this work: the **Oberstaller 2021 evidence scores** behind Fig 3B/C
were absent from the tree. They are now included
(`data/raw/external/oberstaller_evidence_scores.csv`, from PMC8187163 `mmc1.xlsx`),
and the ρ ≥ 0.36 claim reproduces.
