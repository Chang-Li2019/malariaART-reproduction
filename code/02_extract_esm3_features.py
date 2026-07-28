"""Tier 2 — regenerate the ESM-3 feature cache from isolate protein sequences.

Not shipped (33.4 GB for the full panel); everything in Tier 1 needs it, so run
this first. ESM-3 Open Small (d_model 1536) is used as a frozen feature
extractor, no fine-tuning.

    python code/02_extract_esm3_features.py --genes PF3D7_1343700      # minutes
    python code/02_extract_esm3_features.py --all                      # GPU-hours

Requires the esm_env environment. Runs on GPU when one is visible and falls back
to CPU otherwise (K13 is ~140s on CPU; the full panel wants a GPU). ESM-3 Open
Small is gated: authenticate once with `huggingface-cli login` before first use.
Genes whose output already exists are skipped.
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from cohort import load_labels
from config import load_config, Config
from models import load_rf_model
from sequences import load_gene_sequences, read_fasta
from tables import read_gene_panel

ESM3_MAX_CONTEXT = 1500  # proteins longer than this are embedded in overlapping windows


def load_esm3_model(max_context: int):
    """Construct the frozen ESM-3 client from the shipped implementation module."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "implementation"))
    from esm3_utils import ESM3Utils

    return ESM3Utils({"sequence": True, "return_embeddings": True}, max_len=max_context)


def embed_sequence(esm_model, sequence: str) -> np.ndarray:
    """Full per-residue embedding matrix, shape (length, 1536)."""
    return np.asarray(esm_model.get_LL(sequence, return_embeddings=True), dtype=np.float32)


def choose_pooling_axis(config: Config, gene_id: str) -> int:
    """Match the published model's pooling: 0 for 1536-dim genes, 1 otherwise."""
    model_path = config.rf_models / f"{gene_id}_rf.pkl"
    if not model_path.exists():
        return 1
    return 0 if load_rf_model(config, gene_id)["n_features"] == 1536 else 1


def extract_gene_features(config: Config, gene_id: str, reference: dict[str, str],
                          esm_model, output_dir: Path) -> str:
    """Embed every cohort isolate's sequence for one gene and cache the result."""
    output_path = output_dir / f"{gene_id}_features.pkl"
    if output_path.exists():
        return "skip (exists)"

    cohort_samples = set(load_labels(config).index)
    isolates = load_gene_sequences(config, gene_id)
    isolates = isolates[isolates["sample_id"].isin(cohort_samples)]
    if isolates.empty:
        return "skip (no cohort samples)"

    axis = choose_pooling_axis(config, gene_id)
    reference_features = None
    if gene_id in reference:
        reference_features = embed_sequence(esm_model, reference[gene_id]).mean(axis=axis)

    features: dict[str, dict] = {}
    for sample_id, sequence in zip(isolates["sample_id"], isolates["sequence"]):
        if not isinstance(sequence, str) or "*" in sequence:  # stop-codon sequences are excluded
            continue
        pooled = embed_sequence(esm_model, sequence).mean(axis=axis)
        record = {"embeddings": pooled}
        if reference_features is not None and reference_features.shape == pooled.shape:
            record["embedding_distance"] = float(np.linalg.norm(pooled - reference_features))
        features[sample_id] = record

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(features, handle)
    return f"{len(features)} samples, axis={axis}"


def select_genes(config: Config, args: argparse.Namespace) -> list[str]:
    """The genes to process, from --all or --genes."""
    if args.all:
        return list(read_gene_panel(config)["Gene ID"])
    return args.genes or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--genes", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", help="output dir (default: config feature_cache)")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    output_dir = Path(args.out).resolve() if args.out else config.feature_cache

    genes = select_genes(config, args)
    if not genes:
        parser.error("give --genes GENE [GENE ...] or --all")

    reference = read_fasta(config.reference_proteins)
    esm_model = load_esm3_model(ESM3_MAX_CONTEXT)
    print(f"writing features to {output_dir}")
    for position, gene_id in enumerate(genes, 1):
        start = time.time()
        status = extract_gene_features(config, gene_id, reference, esm_model, output_dir)
        print(f"[{position}/{len(genes)}] {gene_id}  {status}  ({time.time() - start:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
