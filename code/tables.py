"""Readers for the shipped published tables and gene lists."""

import pandas as pd

from config import Config


def read_performance_table(config: Config, ranking: str) -> pd.DataFrame:
    """167-gene performance table, validation-ranked or test-ranked."""
    filename = "auROC_PR.VAL.txt" if ranking == "valid" else "auROC_PR.TEST.txt"
    table = pd.read_csv(config.published_source_tables / filename, sep="\t")
    return table.rename(columns={"X": "gene"})


def read_snp_ml_table(config: Config) -> pd.DataFrame:
    """DL vs single-variant comparison (Supplementary Table 5)."""
    return pd.read_csv(config.published_source_tables / "snp_ml_comparison.csv")


def read_supplementary_table(config: Config, number: int) -> pd.DataFrame:
    """One supplementary table from its CSV export (banner row already stripped)."""
    path = config.published_supplementary_csv / f"supp_table_{number}.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_gene_panel(config: Config) -> pd.DataFrame:
    """The 167 analysed genes with their symbols."""
    return pd.read_csv(config.gene_list_panel_167, sep="\t")


def read_plasmodb_query(config: Config) -> pd.DataFrame:
    """The 173 raw PlasmoDB 'artemisinin resistance' hits."""
    return pd.read_csv(config.gene_list_plasmodb_173, sep="\t")


def read_dms_gene_ids(config: Config) -> list[str]:
    """The 56 genes carried through the deep mutational scan."""
    return config.gene_list_dms_56.read_text().split()


def read_kucharski_gene_ids(config: Config) -> list[str]:
    """The 12 Kucharski-review ART-R genes."""
    gene_ids = []
    for line in config.gene_list_kucharski_12.read_text().splitlines():
        if line.strip():
            gene_ids.append(line.split("\t")[0].strip())
    return gene_ids


def gene_symbols(config: Config) -> dict[str, str]:
    """Gene id -> display symbol, falling back to the id when none is annotated."""
    panel = read_gene_panel(config)
    symbols = {}
    for _, row in panel.iterrows():
        symbol = str(row.get("Gene Name or Symbol", "")).strip()
        if symbol and symbol.lower() not in ("nan", "n/a"):
            symbols[row["Gene ID"]] = symbol
        else:
            symbols[row["Gene ID"]] = row["Gene ID"]
    return symbols


def read_transcriptomic_benchmark(config: Config) -> dict[str, float]:
    """Mean auROC/auPRC of the GuanLab transcriptomic LightGBM model, per split."""
    benchmark = {}
    for split in ("valid", "test"):
        table = pd.read_csv(config.published_source_tables / f"transcriptome_{split}_perf.csv")
        benchmark[f"{split}_auc"] = float(table["auroc"].mean())
        benchmark[f"{split}_aupr"] = float(table["auprc"].mean())
    return benchmark
