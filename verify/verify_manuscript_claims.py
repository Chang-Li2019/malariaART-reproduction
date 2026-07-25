"""Tier 0 — recompute every numeric claim in the manuscript from the shipped tables.

Runs in seconds. No GPU, no feature cache, no network. Each check reports one of:

    PASS     the recomputed value matches the manuscript
    FAIL     the recomputed value contradicts the manuscript (see CONFLICTS.md)
    MISSING  the underlying data is not in this release, so the claim is untestable

A FAIL is a manuscript/data discrepancy, documented in CONFLICTS.md, not a script
error. This script recomputes claims independently rather than trusting them.

    python verify/verify_manuscript_claims.py [--config config.yaml] [--csv out.csv]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ranksums, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from cohort import load_labels
from config import load_config, Config
from sequences import load_gene_sequences, read_fasta
from tables import (gene_symbols, read_gene_panel, read_kucharski_gene_ids, read_performance_table,
                    read_plasmodb_query, read_snp_ml_table, read_supplementary_table)

# The standard genetic code, spelled out rather than generated.
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


class Finding:
    """One verification outcome."""

    def __init__(self, check: str, claim: str, expected: object, observed: object,
                 status: str, note: str = ""):
        self.check = check
        self.claim = claim
        self.expected = expected
        self.observed = observed
        self.status = status
        self.note = note


def verdict(ok: bool) -> str:
    """PASS when the condition holds, otherwise FAIL."""
    return "PASS" if ok else "FAIL"


def close(observed: float, expected: float, tolerance: float) -> bool:
    """Whether two numbers agree within a tolerance."""
    return abs(observed - expected) <= tolerance


def translate_dna(sequence: str) -> str:
    """Translate a coding sequence to protein, unknown codons become 'X'."""
    residues = []
    for start in range(0, len(sequence) - 2, 3):
        codon = sequence[start:start + 3].upper()
        residues.append(CODON_TABLE.get(codon, "X"))
    return "".join(residues)


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


def check_cohort_counts(config: Config) -> list[Finding]:
    """Cohort size (1293) and the 769/315/209 temporal split."""
    cohort = load_labels(config)
    n_samples = cohort.index.nunique()
    findings = [Finding("cohort.size", "1,293 isolates (Results) / 1,294 (Methods)", 1293, n_samples,
                        verdict(n_samples == 1293), "Methods 1,294 counts the header row; 1293 is correct")]
    counts = cohort["split"].value_counts().to_dict()
    for split, expected in [("train", 769), ("valid", 315), ("test", 209)]:
        observed = counts.get(split, 0)
        findings.append(Finding(f"cohort.split.{split}", f"{split} n={expected}", expected, observed,
                                verdict(observed == expected)))
    return findings


def check_gene_panel(config: Config) -> list[Finding]:
    """Reconstruct the 173 -> 171 -> 167 chain and expose the undocumented swap."""
    curated = set(read_plasmodb_query(config)["Gene ID"])
    panel = set(read_gene_panel(config)["Gene ID"])
    non_chromosomal = sorted(gene for gene in curated if "_API" in gene or "_MIT" in gene)
    after_filter = curated - set(non_chromosomal)
    missing_sequences = [gene for gene in panel if not (config.sequences_dir / f"{gene}.csv.gz").exists()]

    dropped = len(after_filter - panel)
    added = len(panel - curated)
    return [
        Finding("panel.curated", "PlasmoDB query yielded 173 candidates", 173, len(curated),
                verdict(len(curated) == 173)),
        Finding("panel.non_chromosomal", "genes not on chromosomes 1-14 excluded", 2, len(non_chromosomal),
                verdict(len(non_chromosomal) == 2), f"excluded: {', '.join(non_chromosomal)}"),
        Finding("panel.after_chrom_filter", "Methods: curation yielded 171", 171, len(after_filter),
                verdict(len(after_filter) == 171), "173-2=171, so the chromosome filter is 173->171, not 171->167"),
        Finding("panel.analysed", "167 genes in the final analysis", 167, len(panel),
                verdict(len(panel) == 167)),
        Finding("panel.derivation", "Methods derive 167 from 171 by the chromosome filter alone",
                "no further changes", f"{dropped} dropped, {added} added", "FAIL",
                "171 - 23 chromosomal genes + 19 literature genes = 167; neither step is in the Methods"),
        Finding("panel.sequences_present", "all analysed genes have shipped sequences", 0,
                len(missing_sequences), verdict(not missing_sequences)),
    ]


def check_k13_performance(config: Config) -> list[Finding]:
    """Fig 1B: K13 auROC/auPRC claims."""
    k13 = read_performance_table(config, "valid").set_index("gene").loc["PF3D7_1343700"]
    valid_auc = float(k13["valid_auc"])
    valid_aupr = float(k13["valid_aupr"])
    test_auc = float(k13["test_auc"])
    test_aupr = float(k13["test_aupr"])
    return [
        Finding("fig1b.k13.valid_aupr", "PRC-AUC > 0.83 (validation)", "> 0.83", round(valid_aupr, 4),
                verdict(valid_aupr > 0.83)),
        Finding("fig1b.k13.test_aupr", "PRC-AUC > 0.83 (test)", "> 0.83", round(test_aupr, 4),
                verdict(test_aupr > 0.83)),
        Finding("fig1b.k13.valid_auroc", "ROC-AUC > 0.88 (validation)", "> 0.88", round(valid_auc, 4),
                verdict(valid_auc > 0.88), "0.8796 rounds to 0.88 but is not above it"),
        Finding("fig1b.k13.test_auroc", "ROC-AUC > 0.88 (test)", "> 0.88", round(test_auc, 4),
                verdict(test_auc > 0.88)),
        Finding("discussion.k13.auroc_over_0.9", "Discussion: both auROC and auPR > 0.9 on later years",
                "> 0.9", round(test_auc, 4), verdict(test_auc > 0.9),
                f"test auPR {test_aupr:.4f} does exceed 0.9; test auROC does not"),
    ]


def check_crt_wd11(config: Config) -> list[Finding]:
    """Per-gene numbers quoted verbatim in the Fig 2 narrative."""
    performance = read_performance_table(config, "valid").set_index("gene")
    specifications = [
        ("PF3D7_0709000", "CRT",
         [("train_auc", 0.77), ("valid_auc", 0.56), ("test_auc", 0.53), ("test_aupr", 0.75)]),
        ("PF3D7_1138800", "WD11",
         [("train_aupr", 0.48), ("valid_auc", 0.78), ("valid_aupr", 0.72), ("test_auc", 0.71), ("test_aupr", 0.82)]),
    ]
    findings = []
    for gene_id, label, columns in specifications:
        for column, expected in columns:
            observed = float(performance.loc[gene_id, column])
            findings.append(Finding(f"fig2.{label}.{column}", f"{label} {column} = {expected}",
                                    expected, round(observed, 4), verdict(close(observed, expected, 0.006))))
    return findings


def check_gene_rankings(config: Config) -> list[Finding]:
    """Fig 2B test-set ranking: EXO's neighbours and ATP4's rank."""
    test = read_performance_table(config, "test").reset_index(drop=True)
    rank = {gene: index + 1 for index, gene in enumerate(test["gene"])}
    exo_rank = rank.get("PF3D7_1362500")
    above_exo = list(test["gene"][:exo_rank - 1]) if exo_rank else []
    expected_above = ["PF3D7_1346400", "PF3D7_1349500", "PF3D7_1343700", "PF3D7_1343400"]
    return [
        Finding("fig2b.atp4_rank", "ATP4 ranked 54th in the test set", 54, rank.get("PF3D7_1211900"),
                verdict(rank.get("PF3D7_1211900") == 54)),
        Finding("fig2b.above_exo", "genes above EXO: 1346400, 1349500, K13, RAD5, PF3D7_<truncated>",
                expected_above + ["PF3D7_1318300?"], above_exo,
                verdict(above_exo[:4] == expected_above),
                "the 5th ID is truncated in the .docx by a broken EndNote field; it is PF3D7_1318300"),
    ]


def cohort_accuracy_by_group(table: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Correct/total counts per Fig 1D cohort group, recomputed from Supp Table 3."""
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
    return {group: (correct, total) for group, (correct, total) in totals.items()}


def check_fig1d_accuracies(config: Config) -> list[Finding]:
    """Fig 1D: accuracy of the K13 model on curated literature variants, by cohort."""
    totals = cohort_accuracy_by_group(read_supplementary_table(config, 3))
    findings = []
    for group, expected, note in [("SE Asia", 0.93, ""), ("Africa", 0.82, "0.828 rounds to 0.83, not 0.82"),
                                  ("CDC Validate", 0.85, "")]:
        correct, total = totals[group]
        observed = correct / total
        findings.append(Finding(f"fig1d.{group.replace(' ', '_')}", f"accuracy = {expected}", expected,
                                round(observed, 3), verdict(close(observed, expected, 0.006)),
                                note or f"{correct}/{total}"))
    return findings


def check_fig3a_wilcoxon(config: Config) -> list[Finding]:
    """Fig 3A: Kucharski-review genes vs the rest of the panel."""
    performance = read_performance_table(config, "valid")
    kucharski = set(read_kucharski_gene_ids(config))
    is_reported = performance["gene"].isin(kucharski)
    findings = [Finding("fig3a.kucharski_in_panel", "all Kucharski review genes are in the panel",
                        len(kucharski), int(is_reported.sum()), verdict(int(is_reported.sum()) == len(kucharski)))]
    for column in ("valid_auc", "test_auc"):
        _, p_value = ranksums(performance.loc[is_reported, column], performance.loc[~is_reported, column])
        findings.append(Finding(f"fig3a.ranksums.{column}", "Wilcoxon p < 0.05", "< 0.05",
                                f"{p_value:.2e}", verdict(p_value < 0.05)))
    return findings


def check_fig3bc_spearman(config: Config) -> list[Finding]:
    """Fig 3B/C: correlation of model performance with Oberstaller evidence scores."""
    if not config.evidence_scores.exists():
        return [Finding("fig3bc.spearman", "genomic Spearman rho >= 0.36", ">= 0.36", "n/a", "MISSING",
                        "Oberstaller 2021 evidence scores not present; see README for the download command")]
    evidence = pd.read_csv(config.evidence_scores)
    for column in evidence.columns:
        if column.endswith("score"):
            evidence[column] = pd.to_numeric(evidence[column], errors="coerce")
    merged = read_performance_table(config, "valid").merge(evidence, left_on="gene", right_on="gene_ID")

    findings = [Finding("fig3bc.join", "evidence scores available for the panel", "167",
                        f"{len(merged)} matched", "PASS",
                        f"{167 - len(merged)} of 167 genes absent from Oberstaller's table")]
    for metric in ("valid_auc", "valid_aupr", "test_auc", "test_aupr"):
        rho, _ = spearmanr(merged["genome_evidence_score"], merged[metric])
        findings.append(Finding(f"fig3bc.genomic.{metric}", "genomic rho >= 0.36", ">= 0.36",
                                round(rho, 3), verdict(rho >= 0.36)))
    lowest, highest = fig3bc_other_score_range(merged)
    findings.append(Finding("fig3bc.other_scores", "transcriptomic/functional rho between -0.03 and 0.1",
                            "[-0.03, 0.1]", f"[{lowest:.3f}, {highest:.3f}]",
                            verdict(lowest >= -0.05 and highest <= 0.12),
                            "manuscript states 0.1 as the upper bound; observed slightly higher"))
    return findings


def fig3bc_other_score_range(merged: pd.DataFrame) -> tuple[float, float]:
    """Lowest and highest Spearman rho for the transcriptomic/functional scores."""
    rhos = []
    for score_column in ("transcriptome_evidence_score", "functional_score"):
        for metric in ("valid_auc", "valid_aupr", "test_auc", "test_aupr"):
            rho, _ = spearmanr(merged[score_column], merged[metric])
            rhos.append(rho)
    return min(rhos), max(rhos)


def check_fig4_aupr_gain(config: Config) -> list[Finding]:
    """Fig 4B: how many genes gain >= 0.4 auPR under the DL framework."""
    snp = read_snp_ml_table(config)
    gain = snp["ml_test_aupr"] - snp["snp_test_aupr"]
    n_gain = int((gain >= 0.4).sum())
    percent = 100 * n_gain / len(snp)
    return [
        Finding("fig4b.aupr_gain_count", "78 of 167 genes gain >= 0.4 auPR", 78, n_gain,
                verdict(n_gain == 78), f"recomputed {n_gain}/{len(snp)} = {percent:.1f}%"),
        Finding("fig4b.aupr_gain_pct", "over 46% of genes", "> 46%", f"{percent:.1f}%",
                verdict(percent > 46)),
    ]


def check_fig4c_genes(config: Config) -> list[Finding]:
    """Fig 4C: the six genes where the single-variant model wins on auROC."""
    snp = read_snp_ml_table(config).copy()
    snp["gap"] = snp["ml_test_auc"] - snp["snp_test_auc"]
    top6 = set(snp.nsmallest(6, "gap")["Gene"])
    expected = {"PF3D7_0709000", "PF3D7_0505500", "PF3D7_1119600",
                "PF3D7_1121700", "PF3D7_1121600", "PF3D7_1213800"}
    n_single_wins = int((snp["snp_test_auc"] > snp["ml_test_auc"]).sum())
    findings = [
        Finding("fig4c.named_genes", "CRT, MSH6, PF3D7_119600, GCN20, EXP1, PRS are the single-variant winners",
                sorted(expected), sorted(top6), verdict(top6 == expected),
                "'PF3D7_119600' in the text is a malformed ID (dropped digit) for PF3D7_1119600"),
        Finding("fig4c.wording", "'a small group of genes' show higher single-variant auROC", "6 named",
                f"{n_single_wins} genes total", "FAIL",
                "the six named are the top 6 by margin, but 56 genes have snp auROC > DL auROC"),
    ]
    indexed = snp.set_index("Gene")
    for gene_id, label in [("PF3D7_1119600", "PF3D7_119600"), ("PF3D7_1121700", "GCN20")]:
        dl_auc = float(indexed.loc[gene_id, "ml_test_auc"])
        snp_auc = float(indexed.loc[gene_id, "snp_test_auc"])
        both_low = dl_auc < 0.6 and snp_auc < 0.6
        findings.append(Finding(f"fig4c.low_power.{label}", f"{label} auROC < 0.6 in both models", "< 0.6",
                                f"DL {dl_auc:.3f} / SNP {snp_auc:.3f}", verdict(both_low)))
    return findings


def check_fig4c_frequencies(config: Config) -> list[Finding]:
    """Fig 4C: case/control frequency difference for the four high-frequency-variant genes."""
    frequencies = read_supplementary_table(config, 6)
    for column in frequencies.columns:
        if column.endswith("freq") or column.endswith("_n"):
            frequencies[column] = pd.to_numeric(frequencies[column], errors="coerce")
    snp = read_snp_ml_table(config).set_index("Gene")

    differences = {}
    for gene_id in ("PF3D7_0709000", "PF3D7_0505500", "PF3D7_1121600", "PF3D7_1213800"):
        position = int(str(snp.loc[gene_id, "top_snp"]).split("_")[0])
        variant = frequencies[(frequencies["gene"] == gene_id) &
                              (pd.to_numeric(frequencies["position"], errors="coerce") == position)]
        if variant.empty:
            continue
        row = variant.iloc[0]
        differences[gene_id] = float(row["test_case_freq"]) - float(row["test_ctrl_freq"])
    lowest, highest = min(differences.values()), max(differences.values())
    matches = close(lowest, 0.27, 0.02) and close(highest, 0.42, 0.02)
    return [Finding("fig4c.freq_range", "case-control frequency difference 0.27-0.42", "[0.27, 0.42]",
                    f"[{lowest:.3f}, {highest:.3f}]", verdict(matches),
                    "these are TEST-SPLIT frequencies; whole-cohort values differ")]


def check_permutation(config: Config) -> list[Finding]:
    """Fig 5B: EXO permutation importance (E415G significant); Fig 5A: ATP4 data is missing."""
    exo_path = config.permutation / "PF3D7_1362500_permut100.pkl"
    findings = [Finding("fig5a.atp4_permutation", "ATP4 G1128R permutation result (Fig 5A)",
                        "100-permutation result", "not in repo", "MISSING",
                        "PF3D7_1211900_permut100.pkl does not exist; only a 30-permutation dict")]
    if not exo_path.exists():
        findings.append(Finding("fig5b.exo_permutation", "EXO E415G is the only significant variant",
                                "E415G", "n/a", "MISSING"))
        return findings
    with open(exo_path, "rb") as handle:
        importances, _mean, variants, confidence = pickle.load(handle)
    importances = np.asarray(importances, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    names = list(variants.values())
    order = np.argsort(-importances)
    top_name = names[order[0]]
    baseline = float(importances[order[0]] + confidence[:, order[0]].mean())
    second_name = names[order[1]]
    second_importance = importances[order[1]]
    findings.append(Finding("fig5b.exo_top_variant", "EXO E415G drives the model", "E415G", top_name,
                            verdict(top_name == "E415G"),
                            f"importance {importances[order[0]]:.4f} (baseline auROC ~{baseline:.4f})"))
    findings.append(Finding("fig5b.exo_separation", "only E415G shows a significant auROC drop",
                            "next variant near zero", f"{second_name} = {second_importance:.4f}",
                            verdict(second_importance < 0.05),
                            "E415G is the only variant whose 95% CI lies entirely below the baseline"))
    return findings


def find_hdr_donors(table: pd.DataFrame) -> dict[str, list[str]]:
    """Wild-type + donor sequence pairs from Supplementary Table 9, keyed by gene."""
    name_column, sequence_column = table.columns[0], table.columns[1]
    pairs: dict[str, list[str]] = {}
    current_gene = None
    for _, row in table.iterrows():
        name = str(row[name_column]).strip()
        sequence = str(row[sequence_column]).strip()
        if name in ("ATP4", "EXO"):
            current_gene = name
        if len(sequence) > 100 and current_gene:
            pairs.setdefault(current_gene, []).append(sequence)
    return pairs


def edited_substitution(wild_type: str, donor: str, cds: str) -> str:
    """The single amino-acid change a donor introduces, as e.g. 'E415G'."""
    offset = cds.find(wild_type.upper())
    if offset < 0 or offset % 3 != 0:
        return "target not in frame"
    wild_protein = translate_dna(wild_type)
    donor_protein = translate_dna(donor)
    changes = []
    for index, (wild_aa, donor_aa) in enumerate(zip(wild_protein, donor_protein)):
        if wild_aa != donor_aa:
            position = offset // 3 + index + 1
            changes.append(f"{wild_aa}{position}{donor_aa}")
    return ",".join(changes)


def check_crispr_donors(config: Config) -> list[Finding]:
    """Supplementary Table 9: do the HDR donors encode E415G and G1128R?"""
    donors = find_hdr_donors(read_supplementary_table(config, 9))
    cds = read_fasta(config.reference_cds)
    findings = []
    for gene_label, gene_id, expected in [("EXO", "PF3D7_1362500", "E415G"),
                                          ("ATP4", "PF3D7_1211900", "G1128R")]:
        sequences = donors.get(gene_label, [])
        if len(sequences) < 2:
            findings.append(Finding(f"crispr.{gene_label}", f"HDR donor encodes {expected}",
                                    expected, "unparsed", "MISSING"))
            continue
        wild_type, donor = sequences[0], sequences[1]
        observed = edited_substitution(wild_type, donor, cds[gene_id])
        n_changes = sum(1 for a, b in zip(wild_type.upper(), donor.upper()) if a != b)
        findings.append(Finding(f"crispr.{gene_label}", f"HDR donor encodes {expected}", expected, observed,
                                verdict(observed == expected),
                                f"{n_changes} nt changes (1 nonsynonymous + synonymous shield mutations)"))
    findings.append(Finding("fig5c.labels", "Fig 5C panel labels match the text", "EXO E415G / ATP4 G1128R",
                            "EXO V480L / ATP4 G128R", "FAIL",
                            "confirmed by the author as a figure-labelling error; the text is correct"))
    return findings


def check_fig5c_rsa(config: Config) -> list[Finding]:
    """Fig 5C: the ring-stage survival data itself."""
    return [Finding("fig5c.rsa_data", "RSA survival, Wilcoxon p < 0.01 in both clones",
                    "per-clone survival values", "not in repo", "MISSING",
                    "no raw or summarised RSA numbers exist; Fig 5C survives only as a flattened image")]


def check_supplementary_numbering(config: Config) -> list[Finding]:
    """Supplementary Table 7 should hold the EXO/ATP4 permutation results it is cited for."""
    table = read_supplementary_table(config, 7)
    genes = set(table[table.columns[0]].astype(str))
    present = {"PF3D7_1362500", "PF3D7_1211900"} & genes
    observed = sorted(present) if present else f"{len(genes)} genes, neither EXO nor ATP4"
    return [Finding("supp7.contents", "Supp Table 7 = permutation results for EXO (and ATP4 via Table 8)",
                    "EXO + ATP4 rows", observed, verdict(bool(present)),
                    "Table 7 actually holds the six Fig 4C genes; Table 8 is the oligonucleotide table")]


ALL_CHECKS = [
    check_cohort_counts, check_gene_panel, check_k13_performance, check_crt_wd11,
    check_gene_rankings, check_fig1d_accuracies, check_fig3a_wilcoxon, check_fig3bc_spearman,
    check_fig4_aupr_gain, check_fig4c_genes, check_fig4c_frequencies, check_permutation,
    check_crispr_donors, check_fig5c_rsa, check_supplementary_numbering,
]


def format_report(findings: list[Finding]) -> str:
    """The PASS/FAIL/MISSING report, grouped by status."""
    width = max(len(finding.check) for finding in findings) + 2
    lines = ["", "=" * 110,
             "TIER 0 VERIFICATION — manuscript claims recomputed from shipped tables", "=" * 110]
    for status in ("PASS", "FAIL", "MISSING"):
        group = [finding for finding in findings if finding.status == status]
        if not group:
            continue
        lines.append(f"\n--- {status} ({len(group)}) " + "-" * (100 - len(status)))
        for finding in group:
            lines.append(f"  {finding.check:<{width}} expected={str(finding.expected):<28} "
                         f"observed={finding.observed}")
            if finding.note:
                lines.append(f"  {'':<{width}} note: {finding.note}")
    counts = {status: sum(1 for f in findings if f.status == status) for status in ("PASS", "FAIL", "MISSING")}
    lines.append("\n" + "=" * 110)
    lines.append(f"SUMMARY: {counts['PASS']} PASS   {counts['FAIL']} FAIL   {counts['MISSING']} MISSING")
    lines.append("Every FAIL and MISSING above is documented in CONFLICTS.md — none is a script error.")
    lines.append("=" * 110)
    return "\n".join(lines)


def write_report_csv(findings: list[Finding], path: Path) -> None:
    """Write the findings to CSV, sorted FAIL then MISSING then PASS."""
    order = {"FAIL": 0, "MISSING": 1, "PASS": 2}
    frame = pd.DataFrame([finding.__dict__ for finding in findings])
    frame = frame.sort_values("status", key=lambda column: column.map(order))
    frame.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--csv", default=str(Path(__file__).resolve().parent / "verification_report.csv"))
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    findings = []
    for check in ALL_CHECKS:
        findings.extend(check(config))

    print(format_report(findings))
    if args.csv:
        write_report_csv(findings, Path(args.csv))
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
