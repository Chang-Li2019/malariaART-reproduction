#!/usr/bin/env python
"""
Verify chunk-reuse stitching is EXACT vs the naive full re-embedding (get_LL) on a
giant gene, for both the wild type and a sample of single mutants. Exits non-zero
if the agreement is not tight, so the giants run can gate on it.
"""
import argparse
import sys

import numpy as np

import rf_pipeline as rp
from rf_mutagenesis import MutagenesisRF

EMB_TOL = 1e-4   # max abs diff on the pooled feature vector
PROB_TOL = 1e-4  # max abs diff on predicted resistance probability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", default="PF3D7_1451200")  # smallest giant (1504aa, 2 chunks)
    ap.add_argument("--n_mut", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = MutagenesisRF(args.gene)
    print(f"{args.gene}: L={m.L}, n_features={m.n_features}, pooling={m.bundle['pooling']}")
    m._ensure_chunks()
    print(f"chunks: {len(m._ints)} windows of sizes {[len(i) for i in m._ints]}")

    # WT: reassembled (chunk path) vs naive get_LL
    wt_naive = rp.pool_to(m.n_features, rp.full_embedding(m.esm, m.wt))
    wt_cr = rp.pool_to(m.n_features, m._wt_full_cr)
    wt_d = float(np.max(np.abs(wt_naive - wt_cr)))
    print(f"WT   pooled feature max|Δ| = {wt_d:.3e}")

    # mutants
    rng = np.random.default_rng(args.seed)
    all_muts = list(m.single_mutations())
    idx = rng.choice(len(all_muts), size=min(args.n_mut, len(all_muts)), replace=False)
    muts = [all_muts[i] for i in idx]

    fe = np.array([m.feat_exact(m.mutant_seq(p, a)) for (p, r, a) in muts])
    fc = np.array([m.cr_feat(p, a) for (p, r, a) in muts])
    feat_d = float(np.max(np.abs(fe - fc)))
    pe = m.predict(fe)
    pc = m.predict(fc)
    prob_d = float(np.max(np.abs(pe - pc)))
    print(f"mutant pooled feature max|Δ| ({len(muts)} muts) = {feat_d:.3e}")
    print(f"mutant predicted prob   max|Δ|                  = {prob_d:.3e}")

    ok = wt_d < EMB_TOL and feat_d < EMB_TOL and prob_d < PROB_TOL
    print("RESULT:", "PASS ✅ chunk-reuse is exact" if ok else "FAIL ❌ mismatch too large")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
