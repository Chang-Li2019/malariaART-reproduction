"""Convert a deep-mutational-scan result into within-gene rank scores.

Reconstruction of a step whose original script was never saved. The transform,
reverse-engineered from the shipped outputs and matching them to ~1e-16:

    rankscore = scipy.stats.rankdata(model_score, method='average') / N

i.e. a within-gene percentile of the raw resistance probability. model_score is
a verbatim copy of resistance_probability from the DMS output.

    python code/postprocess/make_rankscore.py --gene PF3D7_1343700 --check
    python code/postprocess/make_rankscore.py --all --out-dir rankscore_rebuilt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_config, Config
from tables import read_dms_gene_ids

OUTPUT_COLUMNS = ["mutation_id", "position", "reference", "mutant",
                  "model_score", "rankscore", "effect_size", "effect_direction"]


def compute_rankscore(scores: np.ndarray) -> np.ndarray:
    """Within-gene percentile of each score (average ranks, normalised by N)."""
    return rankdata(scores, method="average") / len(scores)


def rebuild_gene(config: Config, gene_id: str) -> pd.DataFrame:
    """Rebuild the rankscore table for one gene from its DMS output."""
    dms = pd.read_csv(config.dms_results / f"{gene_id}_mutagenesis_results.csv")
    rebuilt = dms.rename(columns={"resistance_probability": "model_score"}).copy()
    rebuilt["rankscore"] = compute_rankscore(rebuilt["model_score"].to_numpy())
    return rebuilt[OUTPUT_COLUMNS]


def compare_to_shipped(config: Config, gene_id: str, rebuilt: pd.DataFrame) -> float:
    """Max absolute rankscore deviation from the shipped file for one gene."""
    shipped = pd.read_csv(config.rankscore_results / f"{gene_id}_mutagenesis_rankscore.csv")
    same_ids = (shipped["mutation_id"].values == rebuilt["mutation_id"].values).all()
    if len(shipped) != len(rebuilt) or not same_ids:
        return float("nan")
    return float(np.abs(shipped["rankscore"].to_numpy() - rebuilt["rankscore"].to_numpy()).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--gene")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir")
    parser.add_argument("--check", action="store_true", help="compare against the shipped rankscores")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    genes = read_dms_gene_ids(config) if args.all else ([args.gene] if args.gene else [])
    if not genes:
        parser.error("give --gene GENE or --all")

    output_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    worst = 0.0
    failures = []
    for position, gene_id in enumerate(genes, 1):
        rebuilt = rebuild_gene(config, gene_id)
        message = f"[{position}/{len(genes)}] {gene_id}  {len(rebuilt)} mutations"
        if args.check:
            delta = compare_to_shipped(config, gene_id, rebuilt)
            matches = not np.isnan(delta) and delta < 1e-12
            worst = max(worst, 0.0 if np.isnan(delta) else delta)
            if not matches:
                failures.append(gene_id)
            message += f"   max|delta| = {delta:.2e}  {'OK' if matches else 'MISMATCH'}"
        if output_dir:
            rebuilt.to_csv(output_dir / f"{gene_id}_mutagenesis_rankscore.csv", index=False)
        print(message)

    if args.check:
        print(f"\nworst deviation across {len(genes)} gene(s): {worst:.3e}")
        print("reconstruction MATCHES the shipped rankscores" if not failures
              else f"MISMATCH in: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
