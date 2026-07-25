"""Leakage-controlled check of DMS rank scores against literature ART-R markers.

Reconstruction of a step whose original script was never saved. For each marker
it reports the within-gene percentile in the DMS, the binary call, and how many
training isolates already carried that exact allele. A marker present in training
is remembered, not predicted, so headline agreement is optimistic; --report
splits the K13 markers into seen vs unseen.

    python code/postprocess/literature_validation.py --check --report
"""

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cohort import load_labels
from config import load_config, Config
from sequences import read_fasta

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# gene id -> (display label, marker class, list of substitutions). Sources: WHO
# ART-R marker list (K13), Miotto 2015 background loci, two wrong-drug controls.
MARKERS = {
    "PF3D7_1343700": ("K13", "validated",
                      ["C580Y", "R561H", "N458Y", "I543T", "Y493H", "P553L", "A675V",
                       "R539T", "R622I", "F446I", "P574L", "C469Y", "M476I"]),
    "PF3D7_1318100": ("fd", "ART-background", ["D193Y"]),
    "PF3D7_1460900": ("arps10", "ART-background", ["D128H", "V127M", "D128Y"]),
    "PF3D7_1447900": ("mdr2", "ART-background", ["T484I", "I492V"]),
    "PF3D7_0810800": ("dhps", "NEG-antifolate", ["A581G", "K540E", "S436A"]),
    "PF3D7_0523000": ("mdr1", "NEG-aminoq", ["Y184F", "N1042D", "D1246Y", "N86Y"]),
}


def flatten_markers() -> list[dict]:
    """One row per (gene, marker) with parsed reference/position/alternate."""
    rows = []
    for gene_id, (label, marker_class, substitutions) in MARKERS.items():
        for substitution in substitutions:
            rows.append({
                "gene_id": gene_id,
                "gene_label": label,
                "marker": substitution,
                "marker_class": marker_class,
                "position": int(substitution[1:-1]),
                "alt_aa": substitution[-1],
            })
    return rows


def marker_percentile(config: Config, gene_id: str, position: int, alternate: str) -> tuple[float, str] | None:
    """Within-gene percentile and call for one substitution, or None if absent."""
    path = config.rankscore_results / f"{gene_id}_mutagenesis_rankscore.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    hit = table[(table["position"] == position) & (table["mutant"] == alternate)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    call = "Resistant" if row["model_score"] >= 0.5 else "Sensitive"
    return round(float(row["rankscore"]) * 100, 1), call


def count_training_leakage(config: Config, gene_id: str, train_ids: set[str]) -> dict[tuple[int, str], int]:
    """How many training isolates carry each (position, alternate residue) in a gene."""
    reference = read_fasta(config.reference_proteins)[gene_id]
    counts: dict[tuple[int, str], int] = {}
    with gzip.open(config.sequences_dir / f"{gene_id}.csv.gz", "rt") as handle:
        sequences = pd.read_csv(handle, header=None, names=["sample_id", "sequence"])
    for sample_id, sequence in zip(sequences["sample_id"], sequences["sequence"]):
        if sample_id not in train_ids or not isinstance(sequence, str):
            continue
        for index, (reference_aa, sample_aa) in enumerate(zip(reference, sequence)):
            if reference_aa != sample_aa and sample_aa in STANDARD_AMINO_ACIDS:
                key = (index + 1, sample_aa)
                counts[key] = counts.get(key, 0) + 1
    return counts


def build_summary(config: Config) -> pd.DataFrame:
    """One row per marker: percentile, call, and training-set leakage."""
    labels = load_labels(config)
    train_ids = set(labels.index[labels["split"] == "train"])
    leakage_by_gene: dict[str, dict] = {}

    rows = []
    for marker in flatten_markers():
        gene_id = marker["gene_id"]
        percentile = marker_percentile(config, gene_id, marker["position"], marker["alt_aa"])
        if percentile is None:
            continue
        if gene_id not in leakage_by_gene:
            leakage_by_gene[gene_id] = count_training_leakage(config, gene_id, train_ids)
        n_train = leakage_by_gene[gene_id].get((marker["position"], marker["alt_aa"]), 0)
        rows.append({
            "gene": f"{marker['gene_label']} {gene_id}",
            "marker": marker["marker"],
            "class": marker["marker_class"],
            "percentile": percentile[0],
            "call": percentile[1],
            "n_train_isolates": n_train,
            "seen_in_training": n_train > 0,
        })
    summary = pd.DataFrame(rows)
    return summary.sort_values(["class", "percentile"], ascending=[True, False], kind="stable")


def print_leakage_report(summary: pd.DataFrame) -> None:
    """Print the seen-vs-unseen breakdown for the K13 validated markers."""
    k13 = summary[summary["class"] == "validated"]
    seen = k13[k13["seen_in_training"]]
    unseen = k13[~k13["seen_in_training"]]
    resistant = (k13["call"] == "Resistant").sum()
    print("\nK13 validated markers — leakage-controlled view")
    print(f"  all {len(k13):>2}: median percentile {k13['percentile'].median():.1f}, "
          f"{resistant}/{len(k13)} called Resistant")
    print(f"  seen in training     {len(seen):>2}: median {seen['percentile'].median():.1f}")
    print(f"  UNSEEN in training   {len(unseen):>2}: median {unseen['percentile'].median():.1f}")


def compare_to_shipped(config: Config, summary: pd.DataFrame) -> None:
    """Compare the rebuilt summary with the shipped one and print the deltas."""
    shipped_path = config.dms_results.parent / "literature_validation_summary.csv"
    shipped = pd.read_csv(shipped_path)
    merged = shipped.merge(summary, on=["gene", "marker"], suffixes=("_shipped", "_rebuilt"))
    percentile_delta = (merged["percentile_shipped"] - merged["percentile_rebuilt"]).abs().max()
    train_delta = (merged["n_train_isolates_shipped"] - merged["n_train_isolates_rebuilt"]).abs().max()
    calls_identical = (merged["call_shipped"] == merged["call_rebuilt"]).all()
    print(f"\ncheck vs shipped: {len(merged)}/{len(shipped)} markers matched; "
          f"max |percentile delta| = {percentile_delta:.2f}; "
          f"max |n_train delta| = {train_delta}; calls identical = {calls_identical}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--check", action="store_true", help="compare against the shipped summary")
    parser.add_argument("--report", action="store_true", help="print the leakage-controlled breakdown")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    summary = build_summary(config)
    print(summary.to_string(index=False))

    if args.report:
        print_leakage_report(summary)
    if args.check:
        compare_to_shipped(config, summary)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
