"""Confidence intervals on the headline structure numbers.

Step 8 reported recall of evolution-proven base pairs (63.8% finetuned vs
40.8% ViennaRNA) with no error bars. Pairs within one tRNA are not
independent -- they share a sequence, a fold and a set of species -- so the
honest unit of resampling is the tRNA, not the pair. This bootstraps the 22
genes with replacement and recomputes pooled recall each time, keeping the
comparisons paired so every predictor sees the same resampled genes.

    python scripts/step12_confidence.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/step8_validation_base.json")
    ap.add_argument("--ft", default="outputs/step8_validation_ft.json")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/step12_confidence.json")
    args = ap.parse_args()

    base = {g["gene"]: g for g in json.load(open(args.base))["genes"]}
    ft = {g["gene"]: g for g in json.load(open(args.ft))["genes"]}
    genes = sorted(set(base) & set(ft))
    assert all(base[g]["n_pairs"] == ft[g]["n_pairs"] for g in genes)

    cols = {
        "finetuned, tRNA alone": np.array([ft[g]["isolated"] for g in genes], float),
        "finetuned, in genome": np.array([ft[g]["genome"] for g in genes], float),
        "base, tRNA alone": np.array([base[g]["isolated"] for g in genes], float),
        "base, in genome": np.array([base[g]["genome"] for g in genes], float),
        "ViennaRNA": np.array([ft[g]["vienna"] for g in genes], float),
    }
    pairs = np.array([ft[g]["n_pairs"] for g in genes], float)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(genes), size=(args.n, len(genes)))
    denom = pairs[idx].sum(axis=1)
    boot = {k: v[idx].sum(axis=1) / denom for k, v in cols.items()}

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    print(f"{len(genes)} tRNAs, {int(pairs.sum())} evolution-proven pairs, "
          f"{args.n:,} gene-level bootstraps\n")
    print(f"{'predictor':24s} {'recall':>7s}   95% CI")
    out = {"n_genes": len(genes), "n_pairs": int(pairs.sum()), "recall": {}, "deltas": {}}
    for k, v in cols.items():
        point = float(v.sum() / pairs.sum())
        lo, hi = ci(boot[k])
        out["recall"][k] = {"point": point, "ci": [lo, hi]}
        print(f"{k:24s} {100*point:6.1f}%   [{100*lo:.1f}, {100*hi:.1f}]")

    comparisons = [
        ("finetuned, tRNA alone", "ViennaRNA"),
        ("finetuned, in genome", "ViennaRNA"),
        ("finetuned, tRNA alone", "base, tRNA alone"),
        ("finetuned, in genome", "base, in genome"),
        ("finetuned, tRNA alone", "finetuned, in genome"),
    ]
    print(f"\n{'difference':46s} {'points':>7s}   95% CI       genes won")
    for a, b in comparisons:
        d = boot[a] - boot[b]
        point = float((cols[a].sum() - cols[b].sum()) / pairs.sum())
        lo, hi = ci(d)
        wins = int((cols[a] > cols[b]).sum()); losses = int((cols[a] < cols[b]).sum())
        out["deltas"][f"{a} - {b}"] = {"point": point, "ci": [lo, hi],
                                       "genes_better": wins, "genes_worse": losses}
        flag = "" if lo > 0 or hi < 0 else "  (crosses zero)"
        print(f"{a + ' - ' + b:46s} {100*point:+6.1f}   [{100*lo:+.1f}, {100*hi:+.1f}]"
              f"   {wins}-{losses}{flag}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
