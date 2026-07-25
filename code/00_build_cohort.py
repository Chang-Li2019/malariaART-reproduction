"""Rebuild the analysed cohort: binary ART-R label and temporal split per isolate.

    python code/00_build_cohort.py --out cohort.csv

Reproduces the published 1293 isolates split 769 / 315 / 209.
"""

import argparse
from pathlib import Path

import pandas as pd

from cohort import load_labels
from config import load_config, Config

EXPECTED_SPLIT = {"train": 769, "valid": 315, "test": 209}


def build_cohort(config: Config) -> pd.DataFrame:
    """Labels and split joined to country/year metadata, indexed by sample."""
    labels = load_labels(config)
    metadata = pd.read_csv(config.cohort_metadata).drop_duplicates(subset="Sample").set_index("Sample")
    columns = ["Study", "Country", "Admin level 1", "Year", "Population", "QC pass"]
    present = [column for column in columns if column in metadata.columns]
    cohort = labels.join(metadata[present], how="left")
    cohort.index.name = "sample"
    return cohort


def summarise_by_phenotype(cohort: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts of sensitive and resistant isolates (the Fig 1A numbers)."""
    assigned = cohort.dropna(subset=["split"])
    counts = assigned.groupby(["split", "binary_label"]).size().unstack(fill_value=0)
    counts.columns = ["sensitive" if label == 0 else "resistant" for label in counts.columns]
    counts["total"] = counts.sum(axis=1)
    return counts.reindex(["train", "valid", "test"])


def check_split_counts(cohort: pd.DataFrame) -> bool:
    """Print the reconstructed split and report whether it matches the published one."""
    counts = cohort["split"].value_counts().to_dict()
    print(f"cohort n = {len(cohort)} (expected 1293)")
    all_match = len(cohort) == 1293
    for split, expected in EXPECTED_SPLIT.items():
        observed = counts.get(split, 0)
        matches = observed == expected
        all_match = all_match and matches
        print(f"  {split:<6} {observed:>4}  (expected {expected})  {'OK' if matches else 'MISMATCH'}")
    return all_match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--out", help="write the cohort table here")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    cohort = build_cohort(config)
    ok = check_split_counts(cohort)

    print("\nper-split phenotype counts:")
    print(summarise_by_phenotype(cohort).to_string())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(args.out)
        print(f"\nwrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
