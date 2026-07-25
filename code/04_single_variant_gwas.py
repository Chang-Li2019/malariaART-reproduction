"""Tier 1 — the single-variant (GWAS-style) benchmark the PLM is compared against.

For each gene: align every isolate's protein sequence to the 3D7 reference, find
positions with a non-reference allele, and for each (position, alternate residue)
run Fisher's exact test and score the variant as a standalone predictor on each
split. The variant with the highest test auROC represents the gene.

CPU only, minutes — this step needs sequences, not ESM-3 features.

    python code/04_single_variant_gwas.py --genes PF3D7_0709000

CRT (PF3D7_0709000) should select 326_N>S. The odds ratio is computed on the
test split, which is what reproduces the published table (Supplementary Table 5):
whole-cohort odds ratios differ by an order of magnitude. See CONFLICTS.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score, roc_auc_score

from cohort import load_labels
from config import load_config, Config
from sequences import load_gene_sequences, read_fasta
from tables import read_gene_panel

SPLITS = ("train", "valid", "test")
ODDS_RATIO_SPLIT = "test"  # matches the published Supplementary Table 5


def align_sequences(sequences: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Aligned residue matrix (samples x reference positions), padded/truncated to L."""
    length = len(reference)
    rows = []
    sample_ids = []
    for sample_id, sequence in zip(sequences["sample_id"], sequences["sequence"]):
        if not isinstance(sequence, str):
            continue
        padded = sequence[:length].ljust(length, "-")
        rows.append(np.frombuffer(padded.encode("ascii", "replace"), dtype="S1"))
        sample_ids.append(sample_id)
    matrix = np.vstack(rows) if rows else np.empty((0, length), "S1")
    return pd.DataFrame(matrix, index=sample_ids)


def list_variant_alleles(matrix: pd.DataFrame, reference: str) -> list[tuple[int, str]]:
    """Every (1-based position, alternate residue) seen at least once."""
    alleles = []
    for column_index in range(matrix.shape[1]):
        residues = {value.decode() for value in matrix.iloc[:, column_index].unique()}
        alternates = residues - {reference[column_index], "-", "X", "*"}
        for alternate in sorted(alternates):
            alleles.append((column_index + 1, alternate))
    return alleles


def fisher_test(present: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Odds ratio and p-value for variant presence against resistance."""
    resistant_with = int(((present == 1) & (labels == 1)).sum())
    sensitive_with = int(((present == 1) & (labels == 0)).sum())
    resistant_without = int(((present == 0) & (labels == 1)).sum())
    sensitive_without = int(((present == 0) & (labels == 0)).sum())
    table = [[resistant_with, sensitive_with], [resistant_without, sensitive_without]]
    return fisher_exact(table)


def variant_auc(present: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """auROC and auPRC treating variant presence as the prediction score."""
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(labels, present), average_precision_score(labels, present)


def case_control_frequencies(present: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Allele frequency overall, in resistant cases and sensitive controls."""
    n_case = int((labels == 1).sum())
    n_control = int((labels == 0).sum())
    return {
        "n": len(labels),
        "freq": float(present.mean()) if len(labels) else float("nan"),
        "case_n": n_case,
        "case_freq": float(present[labels == 1].mean()) if n_case else float("nan"),
        "ctrl_n": n_control,
        "ctrl_freq": float(present[labels == 0].mean()) if n_control else float("nan"),
    }


def score_variant(matrix: pd.DataFrame, labels: pd.DataFrame, reference: str,
                  position: int, alternate: str) -> dict:
    """One variant's per-split AUCs, frequencies, and test-split odds ratio."""
    present = (matrix.iloc[:, position - 1] == alternate.encode()).to_numpy().astype(int)
    record = {
        "position": position,
        "ref": reference[position - 1],
        "alt": alternate,
        "variant": f"{position}_{reference[position - 1]}>{alternate}",
    }
    for split in SPLITS:
        mask = (labels["split"] == split).to_numpy()
        split_labels = labels["binary_label"].to_numpy()[mask]
        auc, aupr = variant_auc(present[mask], split_labels)
        record[f"{split}_auc"] = auc
        record[f"{split}_aupr"] = aupr
        for key, value in case_control_frequencies(present[mask], split_labels).items():
            record[f"{split}_{key}"] = value

    odds_mask = (labels["split"] == ODDS_RATIO_SPLIT).to_numpy()
    odds_ratio, p_value = fisher_test(present[odds_mask], labels["binary_label"].to_numpy()[odds_mask])
    record["odds_ratio"] = odds_ratio
    record["p_value"] = p_value
    return record


def scan_gene(config: Config, gene_id: str, reference: dict[str, str], labels: pd.DataFrame) -> pd.DataFrame:
    """Score every variant in one gene across all splits."""
    reference_sequence = reference[gene_id]
    sequences = load_gene_sequences(config, gene_id)
    sequences = sequences[sequences["sample_id"].isin(labels.index)]
    matrix = align_sequences(sequences, reference_sequence)
    if matrix.empty:
        raise ValueError("no cohort sequences")
    gene_labels = labels.loc[matrix.index]
    rows = [score_variant(matrix, gene_labels, reference_sequence, position, alternate)
            for position, alternate in list_variant_alleles(matrix, reference_sequence)]
    return pd.DataFrame(rows)


def select_top_variant(scan: pd.DataFrame) -> pd.Series | None:
    """The gene's representative variant: highest test-set auROC."""
    scorable = scan.dropna(subset=["test_auc"])
    if scorable.empty:
        return None
    return scorable.loc[scorable["test_auc"].idxmax()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--genes", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    labels = load_labels(config).dropna(subset=["split"])
    reference = read_fasta(config.reference_proteins)
    genes = list(read_gene_panel(config)["Gene ID"]) if args.all else (args.genes or [])
    if not genes:
        parser.error("give --genes GENE [GENE ...] or --all")

    rows = []
    for position, gene_id in enumerate(genes, 1):
        scan = scan_gene(config, gene_id, reference, labels)
        top = select_top_variant(scan)
        if top is None:
            print(f"[{position}/{len(genes)}] {gene_id}  no scorable variant ({len(scan)} tested)")
            continue
        row = {"gene": gene_id, "top_snp": top["variant"], "n_variants_tested": len(scan),
               "odds_ratio": top["odds_ratio"], "p_value": top["p_value"]}
        for split in SPLITS:
            row[f"snp_{split}_auc"] = top[f"{split}_auc"]
            row[f"snp_{split}_aupr"] = top[f"{split}_aupr"]
        rows.append(row)
        print(f"[{position}/{len(genes)}] {gene_id}  top={top['variant']:<12} "
              f"test auROC {top['test_auc']:.4f}  OR {top['odds_ratio']:.3g}  ({len(scan)} variants)")

    if rows and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
