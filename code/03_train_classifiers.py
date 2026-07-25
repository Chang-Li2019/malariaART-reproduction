"""Tier 1 — train the four ART-R classifiers per gene on frozen ESM-3 features.

Random Forest, Gradient Boosting, Logistic Regression, SVM. LR and SVM see
standardised features. The best model per gene is chosen on validation auROC.

Needs a feature cache: run code/02_extract_esm3_features.py first.

    python code/03_train_classifiers.py --genes PF3D7_1343700 PF3D7_1362500

Expected: K13 valid auROC 0.8796 / test 0.8962; EXO 0.8249 / 0.8654;
ATP4 0.6908 / 0.6880.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cohort import load_labels
from config import load_config, Config
from tables import read_gene_panel

SPLITS = ("train", "valid", "test")
SCALED_CLASSIFIERS = ("lr", "svm")


def load_features(feature_dir: Path, gene_id: str) -> dict:
    """Cached ESM-3 features for one gene, keyed by sample id."""
    with open(feature_dir / f"{gene_id}_features.pkl", "rb") as handle:
        return pickle.load(handle)


def build_split_matrix(features: dict, labels: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and label vector for one temporal split."""
    sample_ids = [s for s in labels.index[labels["split"] == split] if s in features]
    if not sample_ids:
        return np.empty((0, 0)), np.empty(0)
    rows = [np.asarray(features[s]["embeddings"], dtype=np.float32).ravel() for s in sample_ids]
    return np.vstack(rows), labels.loc[sample_ids, "binary_label"].to_numpy()


def make_classifier(name: str, config: Config):
    """One of the four classifiers, with the published settings."""
    seed = config.random_seed
    if name == "rf":
        return RandomForestClassifier(n_estimators=config.rf_n_estimators, random_state=seed)
    if name == "gb":
        return GradientBoostingClassifier(random_state=seed)
    if name == "lr":
        return LogisticRegression(max_iter=5000, random_state=seed)
    return SVC(probability=True, random_state=seed)


def evaluate_split(model, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """auROC and auPRC, or (nan, nan) when a split has a single class."""
    if len(features) == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    scores = model.predict_proba(features)[:, 1]
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def train_one_classifier(name: str, config: Config, scaler: StandardScaler,
                         splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Fit one classifier on train and score it on every split."""
    train_features, train_labels = splits["train"]
    model = make_classifier(name, config)
    scale = name in SCALED_CLASSIFIERS
    model.fit(scaler.transform(train_features) if scale else train_features, train_labels)

    result = {}
    for split in SPLITS:
        features, labels = splits[split]
        matrix = scaler.transform(features) if scale and len(features) else features
        auc, aupr = evaluate_split(model, matrix, labels)
        result[f"{split}_auc"] = auc
        result[f"{split}_aupr"] = aupr
    return result


def train_all_classifiers(features: dict, labels: pd.DataFrame, config: Config) -> dict[str, dict]:
    """Fit all four classifiers and score them on every split."""
    splits = {split: build_split_matrix(features, labels, split) for split in SPLITS}
    train_features, train_labels = splits["train"]
    if len(train_features) == 0 or len(np.unique(train_labels)) < 2:
        raise ValueError("training split is empty or single-class")
    scaler = StandardScaler().fit(train_features)
    return {name: train_one_classifier(name, config, scaler, splits) for name in ("rf", "gb", "lr", "svm")}


def pick_best_classifier(results: dict[str, dict]) -> str:
    """The classifier with the highest validation auROC."""
    def valid_auc(name: str) -> float:
        score = results[name]["valid_auc"]
        return -1.0 if np.isnan(score) else score

    return max(results, key=valid_auc)


def performance_row(gene_id: str, results: dict[str, dict], best: str) -> dict:
    """Flatten every classifier's scores into one row, with the best model's scores copied out."""
    row = {"gene": gene_id, "best_model": best}
    for name, scores in results.items():
        for metric, value in scores.items():
            row[f"{name}_{metric}"] = value
    row.update(results[best])
    return row


def train_genes(config: Config, feature_dir: Path, labels, genes: list[str]) -> list[dict]:
    """Train and score every gene that has cached features."""
    rows = []
    for position, gene_id in enumerate(genes, 1):
        if not (feature_dir / f"{gene_id}_features.pkl").exists():
            print(f"[{position}/{len(genes)}] {gene_id}  SKIP (no cached features)")
            continue
        results = train_all_classifiers(load_features(feature_dir, gene_id), labels, config)
        best = pick_best_classifier(results)
        rows.append(performance_row(gene_id, results, best))
        print(f"[{position}/{len(genes)}] {gene_id}  best={best}  "
              f"valid auROC {results[best]['valid_auc']:.4f}  test auROC {results[best]['test_auc']:.4f}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--genes", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--features", help="feature cache dir")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    feature_dir = Path(args.features).resolve() if args.features else config.feature_cache
    genes = list(read_gene_panel(config)["Gene ID"]) if args.all else (args.genes or [])
    if not genes:
        parser.error("give --genes GENE [GENE ...] or --all")
    if not feature_dir.exists():
        print(f"ERROR: feature cache not found at {feature_dir}")
        print(f"       run: python code/02_extract_esm3_features.py --genes {' '.join(genes[:3])}")
        return 2

    rows = train_genes(config, feature_dir, load_labels(config), genes)
    if rows and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
