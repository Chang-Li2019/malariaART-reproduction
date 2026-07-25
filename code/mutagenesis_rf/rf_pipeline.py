#!/usr/bin/env python
"""
Shared utilities for ESM-3 + RandomForest mutagenesis across the top-50 genes.

Key facts (see also the malariaPLM RF/ESM architecture notes):
  * Model = ESM-3 (ESM3_OPEN_SMALL, hidden dim 1536) embedding -> per-gene sklearn RF.
  * ESM3Utils.get_LL(seq, return_embeddings=True) -> full (L, 1536) matrix.
  * Cached training features are INCONSISTENT across genes: some genes were pooled
    with axis=0 -> (1536,) protein-mean, others with axis=1 -> (L,) per-residue.
    Each gene's saved RF `n_features_in_` matches ITS own cache. So to featurize any
    sequence for a given model we pool to MATCH that model's n_features (self-detecting).
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/Work1/changli/MalariaGen")
PLM = ROOT / "malariaPLM"
DATA = ROOT / "malariaART" / "data"
ARTGENE_FEATURES = PLM / "artgene_results"        # {gene}_features.pkl (cached ESM features)
ALL_SEQS = DATA / "all_seqs"                       # {gene}.csv.gz : sample_id, sequence
RF_MODELS_DIR = PLM / "rf_models_final"            # output of Phase 0

os.environ.setdefault("MALARIAGEN_DATA_DIR", str(DATA))
sys.path.insert(0, str(PLM / "implementation"))
sys.path.insert(0, str(PLM / "src"))

AA = list("ACDEFGHIKLMNPQRSTVWY")
ART_THRESHOLD = 5  # parasite clearance >= 5h -> resistant (binary_label = 1)


# --------------------------------------------------------------------------- #
# Reference sequences
# --------------------------------------------------------------------------- #
_PROTEIN_DB = None


def protein_db():
    global _PROTEIN_DB
    if _PROTEIN_DB is None:
        from sequence_utils import protein_db as _db
        _PROTEIN_DB = _db
    return _PROTEIN_DB


def reference_sequence(gene: str) -> str:
    return protein_db()[gene]["seq"]


# --------------------------------------------------------------------------- #
# ESM-3
# --------------------------------------------------------------------------- #
_ESM = None


def get_esm(max_len: int = 1500):
    """Lazily construct one shared ESM3Utils client."""
    global _ESM
    if _ESM is None:
        from esm3_utils import ESM3Utils
        _ESM = ESM3Utils({"sequence": True, "return_embeddings": True}, max_len=max_len)
    return _ESM


def full_embedding(esm, seq: str) -> np.ndarray:
    """Return the full per-residue ESM-3 embedding matrix, shape (L, 1536)."""
    return np.asarray(esm.get_LL(seq, return_embeddings=True), dtype=np.float32)


def pool_to(n_features: int, full_emb: np.ndarray) -> np.ndarray:
    """Pool an (L, D) matrix to the 1-D feature vector expected by a model."""
    L, D = full_emb.shape
    if n_features == D:          # protein-mean (axis=0) -> (D,) == (1536,)
        return full_emb.mean(axis=0)
    if n_features == L:          # per-residue (axis=1) -> (L,)
        return full_emb.mean(axis=1)
    raise ValueError(
        f"model expects n_features={n_features}, but embedding is L={L}, D={D}; "
        "cannot determine pooling axis"
    )


def featurize(esm, seq: str, n_features: int) -> np.ndarray:
    return pool_to(n_features, full_embedding(esm, seq))


# --------------------------------------------------------------------------- #
# Phenotype labels + temporal split
# --------------------------------------------------------------------------- #
def load_labels() -> pd.DataFrame:
    """
    Reconstruct the binary ART-resistance labels and the temporal train/valid/test
    split used by the original pipeline, keyed by sample_id.

    binary_label = 1 if parasite clearance (TRAC1) / half-life (TRAC2) >= 5h.
    split: year <= 2014 -> train ; 2015-2016 -> valid ; >= 2017 -> test.
    """
    meta = pd.read_csv(DATA / "metadata.csv", index_col=0)
    year = meta.drop_duplicates(subset="Sample").set_index("Sample")["Year"]

    t1 = pd.read_csv(DATA / "41467_2018_7588_MOESM20_ESM.csv")
    t1 = t1.dropna(subset=["Parasites clearance time"])
    lab1 = pd.Series(
        (t1["Parasites clearance time"] >= ART_THRESHOLD).astype(int).values,
        index=t1["SampleID.Pf3k"].astype(str),
    )

    t2 = pd.read_csv(DATA / "tracII_phenotype.txt", sep="\t")
    t2 = t2.dropna(subset=["Parasite.clearance.half-life"])
    lab2 = pd.Series(
        (t2["Parasite.clearance.half-life"] >= ART_THRESHOLD).astype(int).values,
        index=t2["Genome.data.ID"].astype(str),
    )

    lab = pd.concat([lab1, lab2])
    lab = lab[~lab.index.duplicated(keep="first")]
    lab = lab[lab.index != "RCN07860"]  # excluded in the original pipeline

    df = pd.DataFrame({"binary_label": lab})
    df["year"] = year.reindex(df.index)

    def split_of(y):
        if pd.isna(y):
            return None
        if y <= 2014:
            return "train"
        if y in (2015, 2016):
            return "valid"
        return "test"

    df["split"] = df["year"].map(split_of)
    return df


# --------------------------------------------------------------------------- #
# Model IO
# --------------------------------------------------------------------------- #
def load_rf(gene: str, models_dir: Path = RF_MODELS_DIR):
    """Load a Phase-0 RF bundle: dict with model, scaler, n_features, pooling, ..."""
    with open(Path(models_dir) / f"{gene}_rf.pkl", "rb") as fh:
        return pickle.load(fh)


def predict_proba(bundle, feats: np.ndarray) -> np.ndarray:
    """feats: (n, n_features) raw (unscaled). Returns P(resistant)."""
    X = bundle["scaler"].transform(feats)
    return bundle["model"].predict_proba(X)[:, 1]
