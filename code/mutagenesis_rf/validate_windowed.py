#!/usr/bin/env python
"""
Validation on K13 (PF3D7_1343700) before running the full 56-gene DMS.

Check 1 (featurization correctness): re-embed cached training-sample sequences and
   confirm our pooling reproduces the stored features the RF was trained on.
Check 2 (windowed vs exact): score a random sample of single mutations both ways and
   report agreement, so we know the windowed speedup is safe for the full run.
"""
import argparse
import pickle

import numpy as np
import pandas as pd

import rf_pipeline as rp
from rf_mutagenesis import MutagenesisRF


def check_featurization(gene, n=8):
    print(f"\n=== Check 1: featurization reproduces cached training features ({gene}) ===")
    esm = rp.get_esm()
    with open(rp.ARTGENE_FEATURES / f"{gene}_features.pkl", "rb") as fh:
        feats = pickle.load(fh)
    seqs = pd.read_csv(rp.ALL_SEQS / f"{gene}.csv.gz", header=None)
    seqmap = dict(zip(seqs[0].astype(str), seqs[1]))
    n_features = next(iter(feats.values()))["embeddings"].shape[0]
    print(f"cached feature dim = {n_features}")

    sample_ids = [s for s in feats if s in seqmap][:n]
    diffs = []
    for sid in sample_ids:
        cached = np.asarray(feats[sid]["embeddings"], dtype=np.float32)
        recomputed = rp.pool_to(n_features, rp.full_embedding(esm, seqmap[sid]))
        d = float(np.max(np.abs(cached - recomputed)))
        diffs.append(d)
        print(f"  {sid:12s} max|Δ| = {d:.3e}")
    print(f"  -> worst max|Δ| across {len(sample_ids)} samples: {max(diffs):.3e}")
    return max(diffs)


def check_windowed(gene, n_mut=40, windows=(15, 25, 40), seed=0):
    print(f"\n=== Check 2: windowed vs exact predictions ({gene}) ===")
    m = MutagenesisRF(gene)
    rng = np.random.default_rng(seed)
    all_muts = list(m.single_mutations())
    idx = rng.choice(len(all_muts), size=min(n_mut, len(all_muts)), replace=False)
    muts = [all_muts[i] for i in sorted(idx)]

    # exact
    exact = {}
    feats = np.array([m.feat_exact(m.mutant_seq(p, a)) for (p, r, a) in muts],
                     dtype=np.float32)
    ep = m.predict(feats)
    for (p, r, a), pr in zip(muts, ep):
        exact[(p, r, a)] = float(pr)

    print(f"WT resistance probability = {m.wt_probability():.4f}")
    print(f"{'window':>8} {'mean|Δp|':>10} {'max|Δp|':>10} {'pearson':>9} "
          f"{'sign_agree':>11}")
    results = {}
    wt_prob = m.wt_probability()
    for w in windows:
        wp = {}
        # reuse scan_windowed on just the selected positions/alts
        by_pos = {}
        for (p, r, a) in muts:
            by_pos.setdefault(p, []).append((r, a))
        wt_full = m.wt_full
        for p, alts in by_pos.items():
            lo = max(0, p - 1 - w); hi = min(m.L, p - 1 + w + 1)
            wt_sub = m.wt[lo:hi]
            e_wt = rp.full_embedding(m.esm, wt_sub)
            fl = []
            for (r, a) in alts:
                mut_sub = wt_sub[:p - 1 - lo] + a + wt_sub[p - lo:]
                e_mut = rp.full_embedding(m.esm, mut_sub)
                approx = wt_full.copy()
                approx[lo:hi] = wt_full[lo:hi] + (e_mut - e_wt)
                fl.append(m._pool(approx))
            for (r, a), pr in zip(alts, m.predict(fl)):
                wp[(p, r, a)] = float(pr)
        ea = np.array([exact[k] for k in muts])
        wa = np.array([wp[k] for k in muts])
        dp = np.abs(ea - wa)
        pear = float(np.corrcoef(ea, wa)[0, 1])
        sign = float(np.mean(np.sign(ea - wt_prob) == np.sign(wa - wt_prob)))
        print(f"{w:>8} {dp.mean():>10.4f} {dp.max():>10.4f} {pear:>9.4f} "
              f"{sign:>11.2%}")
        results[w] = dict(mean=dp.mean(), max=dp.max(), pearson=pear, sign=sign)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", default="PF3D7_1343700")
    ap.add_argument("--n_mut", type=int, default=40)
    args = ap.parse_args()
    d = check_featurization(args.gene)
    if d > 1e-2:
        print("  WARNING: featurization does not match cached features (Δ too large).")
    check_windowed(args.gene, n_mut=args.n_mut)


if __name__ == "__main__":
    main()
