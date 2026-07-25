"""Load per-gene RandomForest models and turn embeddings into predictions."""

import pickle

import numpy as np

from config import Config


def load_rf_model(config: Config, gene_id: str) -> dict:
    """Load a per-gene RF bundle: model, scaler, n_features, protein_len, pooling."""
    with open(config.rf_models / f"{gene_id}_rf.pkl", "rb") as handle:
        return pickle.load(handle)


def pool_embedding(n_features: int, embedding: np.ndarray) -> np.ndarray:
    """Pool an (length, 1536) ESM-3 embedding to the vector the model expects.

    The cached features are not uniform across genes: some were pooled over the
    embedding dimension (giving a length-L vector) and some over the sequence
    dimension (giving 1536). The model's n_features says which. See CONFLICTS.md.
    """
    length, dimension = embedding.shape
    if n_features == dimension:
        return embedding.mean(axis=0)
    if n_features == length:
        return embedding.mean(axis=1)
    raise ValueError(f"model expects {n_features} features but embedding is {length} x {dimension}")


def predict_resistance(model_bundle: dict, features: np.ndarray) -> np.ndarray:
    """P(resistant) for raw (unscaled) feature rows."""
    scaled = model_bundle["scaler"].transform(features)
    return model_bundle["model"].predict_proba(scaled)[:, 1]
