"""Binary ART-R labels and the temporal train/valid/test split, per isolate.

label = 1 when parasite clearance is >= 5 h. Split by collection year:
<= 2014 train, 2015-2016 valid, >= 2017 test.

Two facts honoured here:
  * TRAC-I reports clearance *time* and TRAC-II reports clearance *half-life*.
    The original pipeline applies the same 5 h cutoff to both; kept for
    comparability.
  * The stated QC filter does not reproduce the published cohort (it yields
    1230, not 1293). The cohort is exactly the sample list in ARTR_metadata.csv,
    so load_labels intersects with it.
"""

import pandas as pd
from pathlib import Path

from config import Config


def read_trac1_labels(phenotype_path: Path, threshold_hours: int) -> pd.Series:
    """TRAC-I resistance labels keyed by sample id, from clearance time."""
    table = pd.read_csv(phenotype_path)
    table = table.dropna(subset=["Parasites clearance time"])
    resistant = (table["Parasites clearance time"] >= threshold_hours).astype(int)
    return pd.Series(resistant.values, index=table["SampleID.Pf3k"].astype(str))


def read_trac2_labels(phenotype_path: Path, threshold_hours: int) -> pd.Series:
    """TRAC-II resistance labels keyed by sample id, from clearance half-life."""
    table = pd.read_csv(phenotype_path, sep="\t")
    table = table.dropna(subset=["Parasite.clearance.half-life"])
    resistant = (table["Parasite.clearance.half-life"] >= threshold_hours).astype(int)
    return pd.Series(resistant.values, index=table["Genome.data.ID"].astype(str))


def read_collection_years(pf7_metadata_path: Path) -> pd.Series:
    """Collection year per sample id, from the Pf7 metadata."""
    metadata = pd.read_csv(pf7_metadata_path, low_memory=False)
    metadata = metadata.drop_duplicates(subset="Sample")
    return metadata.set_index("Sample")["Year"]


def assign_split(year: float, config: Config) -> str | None:
    """Map a collection year onto the temporal split, or None if unknown."""
    if pd.isna(year):
        return None
    if year <= config.split_train_max_year:
        return "train"
    if int(year) in config.split_valid_years:
        return "valid"
    if year >= config.split_test_min_year:
        return "test"
    return None


def load_labels(config: Config) -> pd.DataFrame:
    """Label + year + split per isolate, restricted to the published cohort."""
    threshold = config.clearance_threshold_hours
    labels = pd.concat([
        read_trac1_labels(config.phenotype_trac1, threshold),
        read_trac2_labels(config.phenotype_trac2, threshold),
    ])
    labels = labels[~labels.index.duplicated(keep="first")]
    labels = labels[~labels.index.isin(config.excluded_samples)]

    years = read_collection_years(config.pf7_metadata)
    frame = pd.DataFrame({"binary_label": labels})
    frame["year"] = years.reindex(frame.index)
    frame["split"] = [assign_split(year, config) for year in frame["year"]]

    cohort_samples = set(pd.read_csv(config.cohort_metadata)["Sample"])
    return frame[frame.index.isin(cohort_samples)]
