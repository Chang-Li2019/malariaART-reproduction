#!/usr/bin/env python
"""
RF-based deep-mutational-scan resistance prediction for a single gene.

For each single amino-acid substitution we recompute the ESM-3 features of the
mutant protein and score it with the gene's Phase-0 RandomForest.

Two featurization modes:
  * exact    : embed the whole mutant protein (ground truth, expensive).
  * windowed : embed only a window around the mutated residue for both WT and
               mutant, add the local delta to the cached WT embedding, then pool.
               ~ (2w+1)/L cheaper per mutant; accuracy validated against exact.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import rf_pipeline as rp


class MutagenesisRF:
    def __init__(self, gene, models_dir=rp.RF_MODELS_DIR, max_len=1500):
        self.gene = gene
        self.bundle = rp.load_rf(gene, models_dir)
        self.n_features = self.bundle["n_features"]
        self.esm = rp.get_esm(max_len=max_len)
        self.wt = rp.reference_sequence(gene)
        self.L = len(self.wt)
        if self.n_features not in (1536, self.L):
            raise ValueError(
                f"{gene}: model n_features={self.n_features} matches neither 1536 "
                f"nor protein length {self.L}; reference sequence may differ from "
                "the one used at training.")
        self._wt_full = None  # lazy (L, 1536)

    # -- embeddings -------------------------------------------------------- #
    @property
    def wt_full(self):
        if self._wt_full is None:
            self._wt_full = rp.full_embedding(self.esm, self.wt)
        return self._wt_full

    def feat_exact(self, seq):
        return rp.pool_to(self.n_features, rp.full_embedding(self.esm, seq))

    def _pool(self, full):
        return rp.pool_to(self.n_features, full)

    # -- prediction -------------------------------------------------------- #
    def predict(self, feats):
        return rp.predict_proba(self.bundle, np.asarray(feats, dtype=np.float32))

    def wt_probability(self):
        return float(self.predict(self._pool(self.wt_full)[None, :])[0])

    # -- mutation enumeration --------------------------------------------- #
    def single_mutations(self, positions=None):
        positions = range(1, self.L + 1) if positions is None else positions
        for pos in positions:
            if pos < 1 or pos > self.L:
                continue
            ref = self.wt[pos - 1]
            for alt in rp.AA:
                if alt != ref:
                    yield pos, ref, alt

    def mutant_seq(self, pos, alt):
        return self.wt[:pos - 1] + alt + self.wt[pos:]

    # -- exact scan -------------------------------------------------------- #
    def scan_exact(self, positions=None):
        muts = list(self.single_mutations(positions))
        feats = np.empty((len(muts), self.n_features), dtype=np.float32)
        for i, (pos, ref, alt) in enumerate(tqdm(muts, desc=f"{self.gene} exact")):
            feats[i] = self.feat_exact(self.mutant_seq(pos, alt))
        probs = self.predict(feats)
        return self._assemble(muts, probs)

    # -- windowed scan ----------------------------------------------------- #
    def scan_windowed(self, positions=None, w=25):
        wt_full = self.wt_full
        wt_pooled = self._pool(wt_full)
        by_pos = {}
        for pos, ref, alt in self.single_mutations(positions):
            by_pos.setdefault(pos, []).append((ref, alt))

        results = []
        for pos in tqdm(sorted(by_pos), desc=f"{self.gene} windowed(w={w})"):
            lo = max(0, pos - 1 - w)
            hi = min(self.L, pos - 1 + w + 1)
            wt_sub = self.wt[lo:hi]
            e_wt_sub = rp.full_embedding(self.esm, wt_sub)  # (win, 1536), once per pos
            feats = []
            metas = []
            for ref, alt in by_pos[pos]:
                mut_sub = wt_sub[:pos - 1 - lo] + alt + wt_sub[pos - lo:]
                e_mut_sub = rp.full_embedding(self.esm, mut_sub)
                approx = wt_full.copy()
                approx[lo:hi] = wt_full[lo:hi] + (e_mut_sub - e_wt_sub)
                feats.append(self._pool(approx))
                metas.append((pos, ref, alt))
            probs = self.predict(feats)
            for (pos_, ref, alt), p in zip(metas, probs):
                results.append((pos_, ref, alt, float(p)))
        muts = [(p, r, a) for (p, r, a, _) in results]
        probs = np.array([p for (_, _, _, p) in results])
        return self._assemble(muts, probs)

    # -- chunk-reuse scan (exact, for L > max_len giants) ------------------ #
    # get_LL tiles a long protein into overlapping windows and reassembles:
    #   embeddings_full[pos] = sum_k M_norm[k, pos] * part_k[local]
    # A point mutation only changes the window(s) that contain it, so we recompute
    # only those windows and reuse the WT embedding for the rest. Exact, not an
    # approximation (identical subsequences give identical ESM-3 embeddings).
    def _ensure_chunks(self):
        if getattr(self, "_cr_ready", False):
            return
        if self.L <= self.esm.max_len:
            raise ValueError(
                f"{self.gene}: L={self.L} <= max_len={self.esm.max_len}; "
                "chunk-reuse is only for giants (use scan_exact).")
        from esm3_intervals import get_intervals_and_weights
        ints, _M, Mnorm = get_intervals_and_weights(
            self.L, min_overlap=self.esm.max_len // 2,
            max_len=self.esm.max_len, s=20)
        self._ints = [np.asarray(idx) for idx in ints]
        self._Mnorm = Mnorm
        self._wt_arr = np.array(list(self.wt))
        self._wt_parts = []
        self._chunks_of_pos = {}          # 0-based pos -> [(chunk_k, local_idx), ...]
        for k, idx in enumerate(self._ints):
            sub = "".join(self._wt_arr[idx])
            self._wt_parts.append(rp.full_embedding(self.esm, sub))  # (len(idx), D)
            for local, pos in enumerate(idx):
                self._chunks_of_pos.setdefault(int(pos), []).append((k, local))
        D = self._wt_parts[0].shape[1]
        full = np.zeros((self.L, D), dtype=np.float64)   # match get_LL dtype
        for k, idx in enumerate(self._ints):
            full[idx] += self._wt_parts[k] * self._Mnorm[k, idx][:, None]
        self._wt_full_cr = full
        self._cr_ready = True

    def cr_embed(self, pos1, alt):
        """Exact full (L, D) embedding of the single mutant via chunk-reuse."""
        p0 = pos1 - 1
        full = self._wt_full_cr.copy()
        for (k, local) in self._chunks_of_pos[p0]:
            idx = self._ints[k]
            sub = self._wt_arr[idx].copy()
            sub[local] = alt
            mut_part = rp.full_embedding(self.esm, "".join(sub))
            full[idx] += (mut_part - self._wt_parts[k]) * self._Mnorm[k, idx][:, None]
        return full

    def cr_feat(self, pos1, alt):
        return rp.pool_to(self.n_features, self.cr_embed(pos1, alt))

    def scan_chunkreuse(self, positions=None):
        self._ensure_chunks()
        muts = list(self.single_mutations(positions))
        feats = np.empty((len(muts), self.n_features), dtype=np.float32)
        for i, (pos, ref, alt) in enumerate(
                tqdm(muts, desc=f"{self.gene} chunkreuse")):
            feats[i] = self.cr_feat(pos, alt)
        probs = self.predict(feats)
        return self._assemble(muts, probs)

    # -- output ------------------------------------------------------------ #
    def _assemble(self, muts, probs):
        wt_prob = self.wt_probability()
        rows = []
        for (pos, ref, alt), p in zip(muts, probs):
            eff = float(p) - wt_prob
            rows.append({
                "mutation_id": f"{ref}{pos}{alt}",
                "position": pos, "reference": ref, "mutant": alt,
                "resistance_probability": float(p),
                "resistance_prediction": "Resistant" if p > 0.5 else "Sensitive",
                "effect_size": eff,
                "effect_direction": ("Resistance+" if eff > 0.05 else
                                     "Resistance-" if eff < -0.05 else "Neutral"),
            })
        df = pd.DataFrame(rows)
        df.attrs["wt_probability"] = wt_prob
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--mode", choices=["exact", "windowed"], default="exact")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--positions", default=None,
                    help="e.g. '1-50' or '10,25,50'; default = whole protein")
    ap.add_argument("--models_dir", default=str(rp.RF_MODELS_DIR))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    positions = None
    if args.positions:
        positions = []
        for tok in args.positions.split(","):
            if "-" in tok:
                a, b = tok.split("-")
                positions.extend(range(int(a), int(b) + 1))
            else:
                positions.append(int(tok))

    m = MutagenesisRF(args.gene, models_dir=Path(args.models_dir))
    t0 = time.time()
    if args.mode == "exact":
        df = m.scan_exact(positions)
    else:
        df = m.scan_windowed(positions, w=args.window)
    dt = time.time() - t0

    out_dir = Path(args.out_dir) if args.out_dir else (
        rp.PLM / "mutagenesis_rf_results" / f"mutagenesis_{args.gene}")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{args.gene}_mutagenesis_results.csv"
    df.to_csv(csv, index=False)
    info = {"gene": args.gene, "mode": args.mode, "window": args.window,
            "protein_len": m.L, "n_features": m.n_features,
            "pooling": m.bundle["pooling"], "n_mutations": len(df),
            "wt_probability": df.attrs["wt_probability"], "seconds": round(dt, 1)}
    with open(out_dir / f"{args.gene}_analysis_info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))
    print(f"Saved {len(df)} mutations -> {csv}")


if __name__ == "__main__":
    main()
