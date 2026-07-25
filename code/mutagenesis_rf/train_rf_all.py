#!/usr/bin/env python
"""
Phase 0: (re)train a per-gene RandomForest for every gene in top50.gene.txt from the
cached ESM-3 features, into one clean folder (rf_models_final/).

CPU-only: reads pre-computed features from artgene_results/{gene}_features.pkl, so no
ESM inference is needed here. Each saved bundle records the pooling convention
(n_features == 1536 -> protein-mean/axis0 ; == protein length -> per-residue/axis1)
so the mutagenesis step can featurize mutants consistently.
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import rf_pipeline as rp

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def build_split_arrays(feat_pkl: Path, labels: pd.DataFrame):
    with open(feat_pkl, "rb") as fh:
        feats = pickle.load(fh)
    buckets = {"train": ([], []), "valid": ([], []), "test": ([], [])}
    for sid, v in feats.items():
        if sid not in labels.index:
            continue
        sp = labels.loc[sid, "split"]
        if sp not in buckets:
            continue
        buckets[sp][0].append(np.asarray(v["embeddings"], dtype=np.float32))
        buckets[sp][1].append(int(labels.loc[sid, "binary_label"]))
    out = {}
    for sp, (X, y) in buckets.items():
        out[sp] = (np.vstack(X) if X else np.empty((0, 0)), np.asarray(y))
    return out


def train_one(gene: str, labels: pd.DataFrame, out_dir: Path, seed: int = 42):
    feat_pkl = rp.ARTGENE_FEATURES / f"{gene}_features.pkl"
    if not feat_pkl.exists():
        return {"gene": gene, "status": "no_features"}

    arrays = build_split_arrays(feat_pkl, labels)
    Xtr, ytr = arrays["train"]
    Xva, yva = arrays["valid"]
    Xte, yte = arrays["test"]
    if Xtr.shape[0] == 0 or len(np.unique(ytr)) < 2:
        return {"gene": gene, "status": "insufficient_train",
                "n_train": int(Xtr.shape[0])}

    n_features = Xtr.shape[1]
    plen = len(rp.reference_sequence(gene))
    pooling = "axis0_1536" if n_features == 1536 else (
        "axis1_perres" if n_features == plen else "unknown")

    scaler = StandardScaler().fit(Xtr)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)

    def auc_aupr(X, y):
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            return (np.nan, np.nan)
        p = clf.predict_proba(scaler.transform(X))[:, 1]
        return roc_auc_score(y, p), average_precision_score(y, p)

    va_auc, va_aupr = auc_aupr(Xva, yva)
    te_auc, te_aupr = auc_aupr(Xte, yte)

    bundle = {
        "gene_id": gene, "model": clf, "scaler": scaler,
        "n_features": int(n_features), "protein_len": int(plen),
        "pooling": pooling, "seed": seed,
        "n_train": int(Xtr.shape[0]), "n_valid": int(Xva.shape[0]),
        "n_test": int(Xte.shape[0]),
        "valid_auc": float(va_auc) if va_auc == va_auc else None,
        "test_auc": float(te_auc) if te_auc == te_auc else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{gene}_rf.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    return {"gene": gene, "status": "ok", "n_features": n_features,
            "protein_len": plen, "pooling": pooling,
            "n_train": int(Xtr.shape[0]), "n_valid": int(Xva.shape[0]),
            "n_test": int(Xte.shape[0]),
            "valid_auc": round(va_auc, 4) if va_auc == va_auc else None,
            "test_auc": round(te_auc, 4) if te_auc == te_auc else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", default=str(rp.ROOT / "top50.gene.txt"))
    ap.add_argument("--out", default=str(rp.RF_MODELS_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    genes = [g.strip() for g in open(args.genes) if g.strip()]
    genes = list(dict.fromkeys(genes))  # dedupe, keep order
    out_dir = Path(args.out)
    labels = rp.load_labels()
    print(f"Loaded labels for {labels['split'].notna().sum()} samples "
          f"(train/valid/test = "
          f"{(labels['split']=='train').sum()}/"
          f"{(labels['split']=='valid').sum()}/"
          f"{(labels['split']=='test').sum()})")

    rows = []
    for i, g in enumerate(genes, 1):
        res = train_one(g, labels, out_dir, seed=args.seed)
        rows.append(res)
        msg = res.get("status")
        extra = ""
        if msg == "ok":
            extra = (f" nfeat={res['n_features']} {res['pooling']} "
                     f"n(tr/va/te)={res['n_train']}/{res['n_valid']}/{res['n_test']} "
                     f"valid_auc={res['valid_auc']} test_auc={res['test_auc']}")
        print(f"[{i:2d}/{len(genes)}] {g:16s} {msg}{extra}")

    summary = pd.DataFrame(rows)
    summ_path = out_dir / "training_summary.csv"
    summary.to_csv(summ_path, index=False)
    ok = (summary["status"] == "ok").sum()
    print(f"\nTrained {ok}/{len(genes)} genes -> {out_dir}")
    print(f"Summary: {summ_path}")


if __name__ == "__main__":
    main()
