"""Read protein sequences: the 3D7 reference FASTA and per-gene isolate sets."""

import gzip
from pathlib import Path

import pandas as pd

from config import Config


def read_fasta(fasta_path: Path) -> dict[str, str]:
    """Map gene id (record id up to the first dot) to its amino-acid sequence."""
    sequences: dict[str, str] = {}
    current_id = None
    chunks: list[str] = []
    for line in open(fasta_path):
        if line.startswith(">"):
            if current_id is not None:
                sequences[current_id] = "".join(chunks)
            current_id = line[1:].split()[0].split(".")[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if current_id is not None:
        sequences[current_id] = "".join(chunks)
    return sequences


def reference_protein(reference: dict[str, str], gene_id: str) -> str:
    """The reference protein sequence for one gene."""
    return reference[gene_id]


def load_gene_sequences(config: Config, gene_id: str) -> pd.DataFrame:
    """Per-isolate protein sequences for one gene (columns: sample_id, sequence)."""
    path = config.sequences_dir / f"{gene_id}.csv.gz"
    with gzip.open(path, "rt") as handle:
        return pd.read_csv(handle, header=None, names=["sample_id", "sequence"])
