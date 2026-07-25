"""Tier 1 — permutation-based variant importance (the Fig 5A/B analysis).

For a gene, shuffle each variable residue position across samples, re-score the
model, and record the auROC drop. Importance is the mean drop over 100 shuffles;
a 95% CI comes from the percentile spread. A position is significant when its CI
lies entirely below the unperturbed baseline.

    python code/05_permutation_importance.py --gene PF3D7_1362500

Expected for EXO: E415G importance ~0.33 against a baseline auROC ~0.816, every
other position below 0.01.

Two things this makes explicit (see CONFLICTS.md): the published figure evaluated
the VALIDATION split (the Methods say test), and it used the retired ProteinBERT
model. This script uses the paper's final ESM-3 + RF model on the validation
split, and needs a feature cache (run 02_extract_esm3_features.py first).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cohort import load_labels
from config import load_config, Config
from models import load_rf_model, predict_resistance
from sequences import load_gene_sequences, read_fasta

SPLIT = "valid"  # the split the published Fig 5A/B was actually computed on


def load_split_features(feature_dir: Path, gene_id: str) -> dict:
    """Cached features for one gene, keyed by sample id."""
    with open(feature_dir / f"{gene_id}_features.pkl", "rb") as handle:
        return pickle.load(handle)


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


def score_model(model_bundle: dict, feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    """auROC of the model on a feature matrix."""
    return roc_auc_score(labels, predict_resistance(model_bundle, feature_matrix))


def permute_one_position(feature_matrix: np.ndarray, column_index: int,
                         generator: np.random.Generator) -> np.ndarray:
    """A copy of the matrix with one column shuffled across samples."""
    permuted = feature_matrix.copy()
    permuted[:, column_index] = generator.permutation(permuted[:, column_index])
    return permuted


def permutation_importance(config: Config, gene_id: str, feature_dir: Path) -> pd.DataFrame:
    """Mean auROC drop per variant position, with a 95% CI from the permuted scores.

    Only exact for per-residue pooling (n_features != 1536), where a residue
    position maps to one feature column; protein-mean genes are reported with
    exact=False and no importance.
    """
    features = load_split_features(feature_dir, gene_id)
    labels = load_labels(config)
    sequences = load_gene_sequences(config, gene_id)
    sequences = sequences[sequences["sample_id"].isin(labels.index[labels["split"] == SPLIT])]
    sequences = sequences.drop_duplicates("sample_id").set_index("sample_id")

    sample_ids = [s for s in sequences.index if s in features]
    if len(sample_ids) < 10:
        raise ValueError(f"only {len(sample_ids)} samples with features in split '{SPLIT}'")

    target = labels.loc[sample_ids, "binary_label"].to_numpy()
    feature_matrix = np.vstack([np.asarray(features[s]["embeddings"], np.float32).ravel() for s in sample_ids])
    model_bundle = load_rf_model(config, gene_id)
    baseline = score_model(model_bundle, feature_matrix, target)

    reference = read_fasta(config.reference_proteins)[gene_id]
    positions = list_variant_positions(sequences.loc[sample_ids], reference)
    per_residue = model_bundle["n_features"] != 1536
    generator = np.random.default_rng(config.random_seed)

    rows = [score_position(model_bundle, feature_matrix, target, baseline, position, label,
                           per_residue, config.n_permutations, generator)
            for position, label in positions.items()]
    return pd.DataFrame(rows)


def score_position(model_bundle: dict, feature_matrix: np.ndarray, target: np.ndarray,
                   baseline: float, position: int, label: str, per_residue: bool,
                   n_permutations: int, generator: np.random.Generator) -> dict:
    """Permutation statistics for a single variant position."""
    column_index = position - 1
    if not per_residue or column_index >= feature_matrix.shape[1]:
        return {"position": position, "variant": label, "baseline_auc": baseline,
                "perm_auc_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "importance": np.nan, "exact": False}
    scores = np.array([
        score_model(model_bundle, permute_one_position(feature_matrix, column_index, generator), target)
        for _ in range(n_permutations)
    ])
    return {"position": position, "variant": label, "baseline_auc": baseline,
            "perm_auc_mean": float(scores.mean()), "ci_low": float(np.percentile(scores, 2.5)),
            "ci_high": float(np.percentile(scores, 97.5)),
            "importance": float(baseline - scores.mean()), "exact": True}


def add_significance(importance: pd.DataFrame) -> pd.DataFrame:
    """Flag positions whose permuted 95% CI lies below the baseline, sorted by importance."""
    importance = importance.copy()
    importance["significant"] = importance["ci_high"] < importance["baseline_auc"]
    return importance.sort_values("importance", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--gene", required=True)
    parser.add_argument("--features")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    feature_dir = Path(args.features).resolve() if args.features else config.feature_cache
    if not (feature_dir / f"{args.gene}_features.pkl").exists():
        print(f"ERROR: no cached features for {args.gene} in {feature_dir}")
        print(f"       run: python code/02_extract_esm3_features.py --genes {args.gene}")
        return 2

    importance = add_significance(permutation_importance(config, args.gene, feature_dir))
    baseline = importance["baseline_auc"].iloc[0]
    print(f"{args.gene}  split={SPLIT}  baseline auROC = {baseline:.4f}  "
          f"({len(importance)} variant positions, {config.n_permutations} permutations)\n")
    columns = ["variant", "position", "importance", "perm_auc_mean", "ci_low", "ci_high", "significant"]
    print(importance[columns].head(15).to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
