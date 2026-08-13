"""Tier 2 — input-level permutation variant importance (the Fig 6A/B analysis).

This mirrors the published method (malariaART/src/permutation.py): for each
variable amino-acid position, the residue identity is shuffled across isolates,
the modified sequence is RE-EMBEDDED with frozen ESM-3, re-pooled, and re-scored
by the trained gene-specific RandomForest. Importance is the mean auROC drop over
`n_permutations` shuffles; a position is significant when its permuted 95% CI lies
entirely below the unperturbed baseline.

    python code/05_permutation_importance.py --gene PF3D7_1362500

Why re-embed rather than shuffle a cached feature column: ESM-3 is contextual, so
a single substitution is smeared across many embedding dimensions. Shuffling one
frozen feature column therefore shows ~0 importance even for a driver variant,
because the RandomForest recovers it from correlated columns. Permuting the input
and re-embedding is the only faithful analog of the published ProteinBERT figure,
and it reproduces EXO E415G (~0.32 drop) and ATP4 G1128R (~0.14 drop).

Requires the ESM-3 environment and a GPU-class machine (re-embedding thousands of
modified sequences). Targets the protein-mean (1536-dim) EXO and ATP4 models
behind Fig 6A/B.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cohort import load_labels
from config import load_config, Config
from models import load_rf_model, predict_resistance
from sequences import load_gene_sequences, read_fasta

SPLIT = "test"  # the held-out split the Methods describe ("across all test-set samples")


def load_esm_model():
    """Construct the frozen ESM-3 client from the shipped implementation module."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "implementation"))
    from esm3_utils import ESM3Utils

    return ESM3Utils({"sequence": True, "return_embeddings": True}, max_len=1500)


def pooled_embedding(esm_model, sequence: str, cache: dict) -> np.ndarray:
    """Protein-level ESM-3 vector (mean over sequence positions), cached by sequence."""
    if sequence not in cache:
        matrix = np.asarray(esm_model.get_LL(sequence, return_embeddings=True), dtype=np.float32)
        cache[sequence] = matrix.mean(axis=0)
    return cache[sequence]


def modified_sequence(raw: str, position: int, residue: str) -> str:
    """The sequence with reference position (1-based) set to a given residue."""
    index = position - 1
    if index >= len(raw):
        return raw
    return raw[:index] + residue + raw[index + 1:]


def list_variant_positions(sequences: pd.DataFrame, reference: str) -> dict[int, str]:
    """1-based positions carrying more than one residue, mapped to a variant label."""
    length = len(reference)
    residue_grid = np.array([list(str(s)[:length].ljust(length, "-")) for s in sequences["sequence"]])
    positions = {}
    for column_index in range(length):
        residues = set(np.unique(residue_grid[:, column_index]))
        alternates = sorted(residues - {reference[column_index], "-", "X", "*"})
        if alternates:
            positions[column_index + 1] = f"{reference[column_index]}{column_index + 1}{alternates[0]}"
    return positions


def load_split_sequences(config: Config, gene_id: str, labels: pd.DataFrame) -> dict:
    """Raw protein sequence per split isolate, stop-codon sequences dropped."""
    split_ids = set(labels.index[labels["split"] == SPLIT])
    isolates = load_gene_sequences(config, gene_id)
    isolates = isolates[isolates["sample_id"].isin(split_ids)].drop_duplicates("sample_id")
    sequences = {}
    for sample_id, sequence in zip(isolates["sample_id"], isolates["sequence"]):
        if isinstance(sequence, str) and "*" not in sequence:
            sequences[sample_id] = sequence
    return sequences


def score(bundle: dict, sample_ids: list, embeddings: dict, target: np.ndarray) -> float:
    """auROC of the RandomForest over a set of pooled embedding vectors."""
    matrix = np.vstack([embeddings[s] for s in sample_ids])
    return roc_auc_score(target, predict_resistance(bundle, matrix))


def permute_position(esm_model, bundle: dict, sample_ids: list, raw_sequences: dict,
                     target: np.ndarray, baseline: float, position: int, label: str,
                     n_permutations: int, seed: int, cache: dict) -> dict:
    """Shuffle the residue at one position across isolates, re-embed, re-score."""
    residues = [raw_sequences[s][position - 1] if len(raw_sequences[s]) >= position else "-"
                for s in sample_ids]
    if len(set(residues)) < 2:
        return {}

    per_value = {(s, residue): pooled_embedding(esm_model, modified_sequence(raw_sequences[s], position, residue), cache)
                 for s in sample_ids for residue in set(residues)}

    generator = np.random.default_rng(seed)
    scores = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = generator.permutation(residues)
        matrix = np.vstack([per_value[(s, shuffled[k])] for k, s in enumerate(sample_ids)])
        scores[i] = roc_auc_score(target, predict_resistance(bundle, matrix))

    return {"variant": label, "position": position, "baseline_auc": baseline,
            "perm_auc_mean": float(scores.mean()),
            "ci_low": float(np.percentile(scores, 2.5)),
            "ci_high": float(np.percentile(scores, 97.5)),
            "importance": float(baseline - scores.mean()),
            "significant": bool(np.percentile(scores, 97.5) < baseline)}


def permutation_importance(config: Config, gene_id: str, esm_model) -> pd.DataFrame:
    """Mean auROC drop per variant position from input-level permutation + re-embedding."""
    bundle = load_rf_model(config, gene_id)
    if bundle["n_features"] != 1536:
        raise ValueError(f"{gene_id} is not a protein-mean (1536-dim) model; Fig 6A/B covers EXO and ATP4 only")

    labels = load_labels(config)
    raw_sequences = load_split_sequences(config, gene_id, labels)
    sample_ids = list(raw_sequences)
    target = labels.loc[sample_ids, "binary_label"].to_numpy()

    cache: dict = {}
    embeddings = {s: pooled_embedding(esm_model, raw_sequences[s], cache) for s in sample_ids}
    baseline = score(bundle, sample_ids, embeddings, target)

    reference = read_fasta(config.reference_proteins)[gene_id]
    valid_df = pd.DataFrame({"sequence": [raw_sequences[s] for s in sample_ids]})
    positions = list_variant_positions(valid_df, reference)

    rows = [permute_position(esm_model, bundle, sample_ids, raw_sequences, target, baseline,
                             position, label, config.n_permutations, config.random_seed, cache)
            for position, label in positions.items()]
    result = pd.DataFrame([r for r in rows if r])
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--gene", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    esm_model = load_esm_model()
    importance = permutation_importance(config, args.gene, esm_model)

    baseline = importance["baseline_auc"].iloc[0]
    print(f"{args.gene}  split={SPLIT}  baseline auROC = {baseline:.4f}  "
          f"({len(importance)} variant positions, {config.n_permutations} permutations)\n")
    columns = ["variant", "position", "importance", "perm_auc_mean", "ci_low", "ci_high", "significant"]
    print(importance[columns].head(15).to_string(index=False))

    out_path = Path(args.out) if args.out else config.permutation / f"{args.gene}_input_permutation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
