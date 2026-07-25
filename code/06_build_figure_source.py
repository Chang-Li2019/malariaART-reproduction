"""Emit one tidy source-data table per figure panel.

The manuscript figures are drawn outside this repository, so the release ships
the numbers rather than plotting code. Each output carries a '# provenance:'
header naming where its numbers came from. Panels whose data does not exist are
written as an empty table with a '# STATUS: MISSING' header.

    python code/06_build_figure_source.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ranksums, spearmanr

from cohort import load_labels
from config import load_config, Config
from tables import (gene_symbols, read_gene_panel, read_kucharski_gene_ids, read_performance_table,
                    read_snp_ml_table, read_supplementary_table, read_transcriptomic_benchmark)

K13_DOMAINS = [("N-terminal", 1, 228), ("Coiled-coil", 229, 348),
               ("BTB/POZ", 349, 442), ("Kelch propeller", 443, 726)]
FIG4C_GENES = ["PF3D7_0709000", "PF3D7_0505500", "PF3D7_1119600",
               "PF3D7_1121700", "PF3D7_1121600", "PF3D7_1213800"]
FIG5C_CLONES = [
    ("K13-revertant", "Control", "Control"), ("K13-revertant", "ATP4 G1128R", "E1"),
    ("K13-revertant", "ATP4 G1128R", "F12"), ("K13-revertant", "EXO E415G", "A4"),
    ("K13-revertant", "EXO E415G", "G2"), ("K13-C580Y", "Control", "Control"),
    ("K13-C580Y", "ATP4 G1128R", "D5"), ("K13-C580Y", "ATP4 G1128R", "G1A"),
    ("K13-C580Y", "EXO E415G", "E7"), ("K13-C580Y", "EXO E415G", "G1E"),
]


def write_csv(frame: pd.DataFrame, path: Path, provenance: str, status: str = "OK") -> None:
    """Write a figure-source table with its provenance (and status, if not OK)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"# provenance: {provenance}\n")
        if status != "OK":
            handle.write(f"# STATUS: {status}\n")
        frame.to_csv(handle, index=False)
    print(f"  {path.name:<34} {len(frame):>6} rows   {status}")


def resistance_label(text: str) -> str | None:
    """Map a phenotype cell to 'R', 'S', or None (unknown / not available)."""
    value = str(text).strip()
    if value == "" or value.startswith("Not Available"):
        return None
    if "Partial Resistance" in value:
        return "R"
    if "Sensitive" in value:
        return "S"
    return None


def k13_domain(position: int) -> str:
    """The K13 domain a residue position falls in."""
    for name, start, end in K13_DOMAINS:
        if start <= position <= end:
            return name
    return "unassigned"


def write_fig1a(config: Config, out_dir: Path) -> None:
    labels = load_labels(config).dropna(subset=["split"])
    counts = labels.groupby(["split", "binary_label"]).size().unstack(fill_value=0)
    counts.columns = ["sensitive" if label == 0 else "resistant" for label in counts.columns]
    counts = counts.reindex(["train", "valid", "test"]).reset_index()
    counts["period"] = ["Before 2015", "2015-2016", "2017-2018"]
    counts["total"] = counts["sensitive"] + counts["resistant"]
    ordered = counts[["split", "period", "sensitive", "resistant", "total"]]
    write_csv(ordered, out_dir / "fig1a_cohort_split.csv",
              "data/raw/metadata/ARTR_metadata.csv + TRAC-I/II phenotype tables via code/cohort.load_labels")


def write_fig1b(config: Config, out_dir: Path) -> None:
    k13 = read_performance_table(config, "valid").set_index("gene").loc["PF3D7_1343700"]
    rows = []
    for split, period in [("train", "Before 2015"), ("valid", "2015-2016"), ("test", "2017-2018")]:
        rows.append({"split": split, "period": period,
                     "auROC": float(k13[f"{split}_auc"]), "auPRC": float(k13[f"{split}_aupr"])})
    write_csv(pd.DataFrame(rows), out_dir / "fig1b_k13_performance.csv",
              "data/published_tables/source_tables/auROC_PR.VAL.txt (row PF3D7_1343700)")


def write_fig1c(config: Config, out_dir: Path) -> None:
    scan = read_supplementary_table(config, 2).rename(columns={"standardized_probability": "score"})
    scan["position"] = pd.to_numeric(scan["position"])
    scan["score"] = pd.to_numeric(scan["score"])
    scan["domain"] = [k13_domain(position) for position in scan["position"]]
    write_csv(scan[["mutation_id", "position", "reference", "mutant", "score", "domain"]],
              out_dir / "fig1c_k13_dms.csv",
              "Supplementary Table 2 (K13 DMS, standardized probabilities); domains annotated here")

    summary = scan.groupby("domain")["score"].agg(["mean", "median", "max", "size"]).reset_index()
    domains = pd.DataFrame(K13_DOMAINS, columns=["domain", "start", "end"])
    domains = domains.merge(summary, on="domain", how="left")
    write_csv(domains, out_dir / "fig1c_k13_domains.csv", "domain boundaries + per-domain score summary")


def cohort_accuracy(table: pd.DataFrame) -> pd.DataFrame:
    """Model accuracy on curated literature variants, per cohort group."""
    columns = list(table.columns)
    totals: dict[str, list[int]] = {}
    for _, row in table.iterrows():
        who, rsa, clinical, prediction, regions = (row[columns[i]] for i in (1, 2, 3, 4, 5))
        predicted = resistance_label(prediction)
        truth = resistance_label(clinical) or resistance_label(rsa)
        if predicted is None or truth is None:
            continue
        groups = ["All"]
        if "Southeast Asia" in str(regions):
            groups.append("SE Asia")
        if "Africa" in str(regions):
            groups.append("Africa")
        if str(who).strip() == "Validate":
            groups.append("CDC Validate")
        for group in groups:
            totals.setdefault(group, [0, 0])
            totals[group][1] += 1
            totals[group][0] += int(predicted == truth)

    rows = []
    for group, (correct, total) in totals.items():
        rows.append({"cohort": group, "n_correct": correct, "sample_size": total,
                     "accuracy": round(correct / total, 4)})
    order = ["All", "SE Asia", "Africa", "CDC Validate"]
    return pd.DataFrame(rows).set_index("cohort").reindex(order).reset_index()


def write_fig1d(config: Config, out_dir: Path) -> None:
    accuracy = cohort_accuracy(read_supplementary_table(config, 3))
    write_csv(accuracy, out_dir / "fig1d_cohort_accuracy.csv",
              "Supplementary Table 3; truth = clinical phenotype where available, else RSA")


def build_top50(config: Config, ranking: str, split: str) -> pd.DataFrame:
    """Top 50 genes for one split, ranked by mean(auROC, auPRC)."""
    symbols = gene_symbols(config)
    benchmark = read_transcriptomic_benchmark(config)
    table = read_performance_table(config, ranking).copy()
    table["gene_symbol"] = [symbols.get(gene, gene) for gene in table["gene"]]
    table["mean_score"] = table[[f"{split}_auc", f"{split}_aupr"]].mean(axis=1)
    table = table.sort_values("mean_score", ascending=False).head(50).reset_index(drop=True)
    table["rank"] = table.index + 1
    table["transcriptomic_auROC"] = benchmark[f"{split}_auc"]
    table["transcriptomic_auPRC"] = benchmark[f"{split}_aupr"]
    columns = ["rank", "gene", "gene_symbol", f"{split}_auc", f"{split}_aupr", "mean_score",
               "transcriptomic_auROC", "transcriptomic_auPRC", "Transcript.Product.Description"]
    return table[[column for column in columns if column in table.columns]]


def write_fig2(config: Config, out_dir: Path) -> None:
    for panel, ranking, split in [("fig2a_valid_top50", "valid", "valid"),
                                   ("fig2b_test_top50", "test", "test")]:
        table = build_top50(config, ranking, split)
        write_csv(table, out_dir / f"{panel}.csv",
                  f"auROC_PR.{ranking.upper()}.txt ranked by mean({split} auROC, {split} auPRC); "
                  f"transcriptomic reference = mean of GuanLab transcriptome_{split}_perf.csv")


def write_fig3a(config: Config, out_dir: Path) -> None:
    performance = read_performance_table(config, "valid").copy()
    kucharski = set(read_kucharski_gene_ids(config))
    symbols = gene_symbols(config)
    performance["group"] = ["Reported ART-R genes" if gene in kucharski else "Other candidate genes"
                            for gene in performance["gene"]]
    performance["gene_symbol"] = [symbols.get(gene, gene) for gene in performance["gene"]]
    write_csv(performance[["gene", "gene_symbol", "group", "valid_auc", "valid_aupr", "test_auc", "test_aupr"]],
              out_dir / "fig3a_evidence_groups.csv",
              "auROC_PR.VAL.txt grouped by data/raw/gene_lists/kucharski_artr_12.tsv")

    rows = [fig3a_statistic_row(performance, metric) for metric in
            ("valid_auc", "valid_aupr", "test_auc", "test_aupr")]
    write_csv(pd.DataFrame(rows), out_dir / "fig3a_statistics.csv",
              "Wilcoxon rank-sum (scipy.stats.ranksums)")


def fig3a_statistic_row(performance: pd.DataFrame, metric: str) -> dict:
    """Rank-sum test of one metric, reported vs other genes."""
    reported = performance.loc[performance["group"] == "Reported ART-R genes", metric]
    other = performance.loc[performance["group"] != "Reported ART-R genes", metric]
    statistic, p_value = ranksums(reported, other)
    return {"metric": metric, "n_reported": len(reported), "n_other": len(other),
            "median_reported": round(reported.median(), 4), "median_other": round(other.median(), 4),
            "statistic": round(statistic, 4), "p_value": p_value, "stars": significance_stars(p_value)}


def significance_stars(p_value: float) -> str:
    """Conventional significance stars for a p-value."""
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def load_evidence_scores(config: Config) -> pd.DataFrame:
    """Oberstaller per-gene evidence scores, numeric."""
    evidence = pd.read_csv(config.evidence_scores)
    for column in evidence.columns:
        if column.endswith("score"):
            evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    return evidence


def write_fig3bc(config: Config, out_dir: Path) -> None:
    if not config.evidence_scores.exists():
        empty = pd.DataFrame(columns=["gene", "genome_evidence_score", "valid_auc"])
        write_csv(empty, out_dir / "fig3bc_evidence_spearman.csv",
                  "Oberstaller 2021 evidence scores", "MISSING")
        return
    merged = read_performance_table(config, "valid").merge(
        load_evidence_scores(config), left_on="gene", right_on="gene_ID", how="inner")
    keep = ["gene", "genome_evidence_score", "transcriptome_evidence_score", "functional_score",
            "total_combined_score", "valid_auc", "valid_aupr", "test_auc", "test_aupr"]
    write_csv(merged[keep], out_dir / "fig3bc_evidence_spearman.csv",
              "auROC_PR.VAL.txt joined to Oberstaller et al. 2021 mmc1.xlsx Supplementary Table 1 "
              "(PMC8187163); 152 of 167 genes matched")

    rows = []
    score_types = [("genome_evidence_score", "genomic"), ("transcriptome_evidence_score", "transcriptomic"),
                   ("functional_score", "functional"), ("total_combined_score", "combined")]
    for score_column, label in score_types:
        for metric in ("valid_auc", "valid_aupr", "test_auc", "test_aupr"):
            rho, p_value = spearmanr(merged[score_column], merged[metric])
            rows.append({"evidence_type": label, "performance_metric": metric,
                         "n": len(merged), "spearman_rho": round(rho, 4), "p_value": p_value})
    write_csv(pd.DataFrame(rows), out_dir / "fig3bc_statistics.csv", "scipy.stats.spearmanr")


def write_fig4ab(config: Config, out_dir: Path) -> None:
    snp = read_snp_ml_table(config).copy()
    symbols = gene_symbols(config)
    snp["gene_symbol"] = [symbols.get(gene, gene) for gene in snp["Gene"]]
    snp["auroc_gain"] = snp["ml_test_auc"] - snp["snp_test_auc"]
    snp["aupr_gain"] = snp["ml_test_aupr"] - snp["snp_test_aupr"]
    snp["aupr_gain_ge_0.4"] = snp["aupr_gain"] >= 0.4
    columns = ["Gene", "gene_symbol", "top_snp", "odds_ratio", "p_value",
               "ml_test_auc", "snp_test_auc", "auroc_gain",
               "ml_test_aupr", "snp_test_aupr", "aupr_gain", "aupr_gain_ge_0.4"]
    write_csv(snp[columns].rename(columns={"Gene": "gene"}), out_dir / "fig4ab_dl_vs_snp.csv",
              "data/published_tables/source_tables/snp_ml_comparison.csv (Supplementary Table 5)")

    n_gain = int(snp["aupr_gain_ge_0.4"].sum())
    summary = pd.DataFrame([{"n_genes": len(snp), "n_aupr_gain_ge_0_4": n_gain,
                             "pct": round(100 * n_gain / len(snp), 1), "manuscript_states": 78,
                             "note": "manuscript says 78/167; the table gives 77/167"}])
    write_csv(summary, out_dir / "fig4b_summary.csv", "counted from snp_ml_comparison.csv")


def write_fig4c(config: Config, out_dir: Path) -> None:
    frequencies = read_supplementary_table(config, 6)
    for column in frequencies.columns:
        if column.endswith("freq") or column.endswith("_n") or column == "position":
            frequencies[column] = pd.to_numeric(frequencies[column], errors="coerce")
    snp = read_snp_ml_table(config).set_index("Gene")
    symbols = gene_symbols(config)

    rows = []
    for gene_id in FIG4C_GENES:
        if gene_id not in snp.index:
            continue
        position = int(str(snp.loc[gene_id, "top_snp"]).split("_")[0])
        variant_rows = frequencies[(frequencies["gene"] == gene_id) & (frequencies["position"] == position)]
        if variant_rows.empty:
            continue
        rows.extend(fig4c_rows(variant_rows.iloc[0], gene_id, symbols, str(snp.loc[gene_id, "top_snp"])))
    write_csv(pd.DataFrame(rows), out_dir / "fig4c_variant_freq.csv",
              "Supplementary Table 6; the 0.27-0.42 range quoted in the text is the TEST split")


def fig4c_rows(frequency_row: pd.Series, gene_id: str, symbols: dict[str, str], variant: str) -> list[dict]:
    """One case/control frequency row per split for a single variant."""
    rows = []
    for split in ("train", "val", "test"):
        case = frequency_row.get(f"{split}_case_freq")
        control = frequency_row.get(f"{split}_ctrl_freq")
        difference = (case - control) if pd.notna(case) and pd.notna(control) else np.nan
        rows.append({"gene": gene_id, "gene_symbol": symbols.get(gene_id, gene_id), "variant": variant,
                     "split": split, "case_freq": case, "ctrl_freq": control, "difference": difference})
    return rows


def write_fig5ab(config: Config, out_dir: Path) -> None:
    exo_path = config.permutation / "PF3D7_1362500_permut100.pkl"
    if not exo_path.exists():
        empty = pd.DataFrame(columns=["gene", "variant", "importance"])
        write_csv(empty, out_dir / "fig5ab_permutation.csv", "permutation pickles", "MISSING")
        return
    write_csv(exo_permutation_table(exo_path), out_dir / "fig5a_exo_permutation.csv",
              "results/permutation/PF3D7_1362500_permut100.pkl (100 permutations, ProteinBERT model)")
    empty = pd.DataFrame(columns=["gene", "gene_symbol", "variant", "importance", "baseline_auc"])
    write_csv(empty, out_dir / "fig5b_atp4_permutation.csv",
              "PF3D7_1211900_permut100.pkl — does not exist in the source tree", "MISSING")


def exo_permutation_table(pickle_path: Path) -> pd.DataFrame:
    """EXO per-variant permutation importance from the saved pickle."""
    with open(pickle_path, "rb") as handle:
        importances, _mean, variants, confidence = pickle.load(handle)
    importances = np.asarray(importances, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    names = list(variants.values())
    baseline = float(np.median(importances + confidence.mean(axis=0)))
    rows = []
    for index, name in enumerate(names):
        rows.append({"gene": "PF3D7_1362500", "gene_symbol": "EXO", "variant": name,
                     "importance": float(importances[index]),
                     "perm_auc_mean": float(confidence[:, index].mean()),
                     "ci_low": float(confidence[0, index]), "ci_high": float(confidence[1, index]),
                     "baseline_auc": baseline,
                     "significant": bool(confidence[1, index] < baseline)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def write_fig5c(config: Config, out_dir: Path) -> None:
    rows = []
    for background, edit, clone in FIG5C_CLONES:
        rows.append({"background": background, "edit": edit, "clone": clone,
                     "replicate": np.nan, "rsa_survival_pct": np.nan})
    provenance = ("clone/edit structure read from Fig5.pptx; survival values NOT PRESENT in the source "
                  "tree. NOTE: the figure labels these edits 'EXO V480L' and 'ATP4 G128R' — confirmed "
                  "mislabels; the correct edits (Supplementary Tables 8/9) are EXO E415G and ATP4 G1128R.")
    write_csv(pd.DataFrame(rows), out_dir / "fig5c_rsa.csv", provenance, "MISSING (values)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    out_dir = Path(args.out).resolve() if args.out else config.figure_source_out
    print(f"writing figure source tables to {out_dir}\n")

    write_fig1a(config, out_dir)
    write_fig1b(config, out_dir)
    write_fig1c(config, out_dir)
    write_fig1d(config, out_dir)
    write_fig2(config, out_dir)
    write_fig3a(config, out_dir)
    write_fig3bc(config, out_dir)
    write_fig4ab(config, out_dir)
    write_fig4c(config, out_dir)
    write_fig5ab(config, out_dir)
    write_fig5c(config, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
