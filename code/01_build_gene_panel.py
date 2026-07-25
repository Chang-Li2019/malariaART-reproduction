"""Audit how the 167-gene candidate panel was assembled.

The Methods say the panel is 173 PlasmoDB hits minus non-chromosomal genes,
giving 167. The data disagrees: 173 - 2 non-chromosomal = 171, then 23
undocumented drops and 19 literature additions give 167. This script recomputes
that chain and writes a per-gene membership reason, because the panel cannot be
regenerated from the text and is shipped as data instead. See CONFLICTS.md.

    python code/01_build_gene_panel.py --out gene_panel_provenance.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from config import load_config, Config
from tables import read_gene_panel, read_kucharski_gene_ids, read_plasmodb_query


def is_chromosomal(gene_id: str) -> bool:
    """False for apicoplast/mitochondrial genes (not on chromosomes 1-14)."""
    return "_API" not in gene_id and "_MIT" not in gene_id


def has_shipped_sequence(config: Config, gene_id: str) -> bool:
    """Whether this release ships an isolate sequence file for the gene."""
    return (config.sequences_dir / f"{gene_id}.csv.gz").exists()


def classify_membership(gene_id: str, in_query: bool, in_panel: bool, in_kucharski: bool) -> str:
    """One human-readable reason why a gene is in or out of the analysed panel."""
    if not is_chromosomal(gene_id):
        return "excluded: not on chromosomes 1-14"
    if in_query and in_panel:
        return "retained from PlasmoDB query"
    if in_query and not in_panel:
        return "excluded: reason not documented in Methods"
    if in_kucharski:
        return "added: literature ART-R gene (Kucharski review)"
    return "added: literature ART-R gene (not in PlasmoDB query)"


def build_provenance_table(config: Config) -> pd.DataFrame:
    """One row per gene ever considered, with its membership reason."""
    query = read_plasmodb_query(config)
    curated = set(query["Gene ID"])
    panel = read_gene_panel(config)
    analysed = set(panel["Gene ID"])
    kucharski = set(read_kucharski_gene_ids(config))
    descriptions = query.set_index("Gene ID")["Product Description"].to_dict()
    symbols = panel.set_index("Gene ID")["Gene Name or Symbol"].to_dict()

    rows = []
    for gene_id in sorted(curated | analysed):
        in_query = gene_id in curated
        in_panel = gene_id in analysed
        in_kucharski = gene_id in kucharski
        rows.append({
            "gene_id": gene_id,
            "in_plasmodb_query": in_query,
            "chromosomal": is_chromosomal(gene_id),
            "in_kucharski_review": in_kucharski,
            "in_analysed_panel": in_panel,
            "sequence_shipped": has_shipped_sequence(config, gene_id),
            "membership_reason": classify_membership(gene_id, in_query, in_panel, in_kucharski),
            "gene_symbol": str(symbols.get(gene_id, "") or ""),
            "product_description": str(descriptions.get(gene_id, "") or ""),
        })
    return pd.DataFrame(rows)


def count_derivation_steps(config: Config) -> dict[str, int]:
    """The 173 -> 171 -> 167 counts."""
    curated = set(read_plasmodb_query(config)["Gene ID"])
    analysed = set(read_gene_panel(config)["Gene ID"])
    non_chromosomal = {gene_id for gene_id in curated if not is_chromosomal(gene_id)}
    after_filter = curated - non_chromosomal
    return {
        "plasmodb_query": len(curated),
        "non_chromosomal": len(non_chromosomal),
        "after_chromosome_filter": len(after_filter),
        "dropped_undocumented": len(after_filter - analysed),
        "added_from_literature": len(analysed - curated),
        "analysed": len(analysed),
    }


def print_derivation(counts: dict[str, int]) -> None:
    """Print the derivation chain in the same shape as the Methods claim."""
    print("gene panel derivation")
    print(f"  PlasmoDB 'artemisinin resistance' query      {counts['plasmodb_query']:>4}")
    print(f"  - not on chromosomes 1-14                    {counts['non_chromosomal']:>4}")
    print(f"  = after chromosome filter                    {counts['after_chromosome_filter']:>4}"
          "   <- the Methods' '171'")
    print(f"  - dropped, reason undocumented               {counts['dropped_undocumented']:>4}")
    print(f"  + added literature ART-R genes               {counts['added_from_literature']:>4}")
    print(f"  = analysed panel                             {counts['analysed']:>4}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--out")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    print_derivation(count_derivation_steps(config))

    provenance = build_provenance_table(config)
    print("\nmembership reasons:")
    print(provenance["membership_reason"].value_counts().to_string())

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        provenance.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
