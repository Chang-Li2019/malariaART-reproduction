#!/usr/bin/env python
"""
Batch runner: exact full-DMS RF resistance prediction across genes, shortest first.

Resumable: a gene is skipped if its results CSV already exists (each gene is written
atomically at the end of its scan). By default only proteins with length <= --max_len
are run (the tractable, single-ESM-chunk genes); the long "giant" genes are deferred.
"""
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

import rf_pipeline as rp
from rf_mutagenesis import MutagenesisRF

OUT_DIR = rp.PLM / "mutagenesis_rf_results"


def log(msg, fh=None):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", default=str(rp.ROOT / "top50.gene.txt"))
    ap.add_argument("--models_dir", default=str(rp.RF_MODELS_DIR))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--max_len", type=int, default=1500,
                    help="only run genes with protein length <= this (giants deferred)")
    ap.add_argument("--min_len", type=int, default=0)
    ap.add_argument("--method", choices=["exact", "chunkreuse"], default="exact",
                    help="chunkreuse = exact reuse of WT windows for long (>1500) genes")
    ap.add_argument("--force", action="store_true", help="re-run even if CSV exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "run_all.log", "a")

    genes = [g.strip() for g in open(args.genes) if g.strip()]
    genes = list(dict.fromkeys(genes))
    sized = [(g, len(rp.reference_sequence(g))) for g in genes]
    selected = sorted([(g, L) for g, L in sized
                       if args.min_len <= L <= args.max_len], key=lambda x: x[1])
    deferred = [(g, L) for g, L in sized if L > args.max_len]

    log(f"=== run_all start: {len(selected)} genes (len<= {args.max_len}), "
        f"{len(deferred)} giants deferred ===", logf)
    log(f"selected order (short->long): "
        f"{', '.join(f'{g}({L})' for g, L in selected[:8])} ...", logf)

    # warm up ESM once
    rp.get_esm()

    summary_rows = []
    for i, (gene, L) in enumerate(selected, 1):
        csv = out_dir / f"{gene}_mutagenesis_results.csv"
        if csv.exists() and not args.force:
            log(f"[{i}/{len(selected)}] {gene} (L={L}) already done -> skip", logf)
            continue
        try:
            t0 = time.time()
            m = MutagenesisRF(gene, models_dir=Path(args.models_dir))
            df = (m.scan_chunkreuse() if args.method == "chunkreuse" and L > m.esm.max_len
                  else m.scan_exact())
            dt = time.time() - t0
            df.to_csv(csv, index=False)
            info = {"gene": gene, "protein_len": L, "n_features": m.n_features,
                    "pooling": m.bundle["pooling"], "n_mutations": len(df),
                    "wt_probability": df.attrs["wt_probability"],
                    "n_resistant": int((df["resistance_prediction"] == "Resistant").sum()),
                    "seconds": round(dt, 1)}
            with open(out_dir / f"{gene}_analysis_info.json", "w") as ifh:
                json.dump(info, ifh, indent=2)
            summary_rows.append(info)
            log(f"[{i}/{len(selected)}] {gene} (L={L}) DONE {len(df)} muts "
                f"in {dt/60:.1f} min, wt_p={info['wt_probability']:.3f}, "
                f"resistant={info['n_resistant']}", logf)
        except Exception as e:
            log(f"[{i}/{len(selected)}] {gene} (L={L}) FAILED: {e}", logf)
            logf.write(traceback.format_exc() + "\n")
            logf.flush()

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "run_all_summary.csv", index=False)
    log(f"=== run_all finished. Deferred giants: "
        f"{', '.join(f'{g}({L})' for g, L in sorted(deferred, key=lambda x: x[1]))} ===",
        logf)
    logf.close()


if __name__ == "__main__":
    main()
