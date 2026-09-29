"""Does mitochondrial tRNA fragility track how long a species lives?

The hypothesis: an animal that must run its mitochondria for decades, on DNA
that mutates fast and is barely repaired, is under pressure to carry tRNAs
whose shape survives a random letter change. A mouse has three years of
damage to tolerate; a bowhead whale has two centuries.

Fragility here is the mean fraction of predicted base pairs destroyed by a
random point mutation, averaged over every base of every tRNA in a genome.

Three things decide whether any correlation means anything:

  body mass    large animals live longer for reasons unrelated to tRNAs
  GC content   G-C pairs are stronger, so GC-rich tRNAs are trivially stabler
  phylogeny    close relatives share both traits, so species are not
               independent observations -- the single easiest way to get a
               spurious result in comparative biology

Phylogeny is handled by centring both variables within taxonomic family (and
separately within order): only the deviation of a species from its relatives
is used, which throws away every between-family difference. Significance
comes from permutation inside those same groups, so the null respects the
tree structure too.

    python scripts/step11_longevity.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    d = math.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / d) if d else float("nan")


def rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def ols(y, X):
    """Least squares with an intercept. Returns coefficients and R^2."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return beta, 1 - ss_res / ss_tot if ss_tot else float("nan"), resid


def partial_corr(y, x, controls):
    """Correlation of y and x after regressing both on the controls."""
    _, _, ry = ols(np.asarray(y, float), np.column_stack(controls))
    _, _, rx = ols(np.asarray(x, float), np.column_stack(controls))
    return pearson(ry, rx)


def group_center(values, groups, min_size=2):
    """Deviation from the group mean, keeping only groups with >= min_size."""
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    keep, out = [], []
    for g, ii in idx.items():
        if len(ii) < min_size:
            continue
        m = np.mean([values[i] for i in ii])
        for i in ii:
            keep.append(i)
            out.append(values[i] - m)
    return np.array(keep), np.array(out)


def permutation_p(x, y, groups, observed, n=10000, seed=0):
    """Shuffle y inside each group, so the null keeps the phylogeny."""
    rng = np.random.default_rng(seed)
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    y = np.asarray(y, float)
    hits = 0
    for _ in range(n):
        yp = y.copy()
        for ii in idx.values():
            if len(ii) > 1:
                yp[ii] = rng.permutation(y[ii])
        if abs(pearson(x, yp)) >= abs(observed):
            hits += 1
    return (hits + 1) / (n + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragility", default="outputs/fragility.jsonl")
    ap.add_argument("--windows", default="data/processed/trna_windows.jsonl")
    ap.add_argument("--metric", default="fragility", choices=["fragility", "breaking"])
    ap.add_argument("--out", default="outputs/step11_longevity.json")
    args = ap.parse_args()

    meta = {}
    with open(args.windows) as fh:
        for line in fh:
            d = json.loads(line)
            meta[d["accession"]] = d

    rows = []
    with open(args.fragility) as fh:
        for line in fh:
            d = json.loads(line)
            m = meta.get(d["accession"])
            if not m or not m.get("body_mass_g"):
                continue
            rows.append({
                "organism": d["organism"], "accession": d["accession"],
                "fragility": d["fragility"], "breaking": d["breaking"],
                "gc": d["gc"], "lifespan": m["lifespan_years"],
                "mass": m["body_mass_g"], "class": m["class"],
                "order": m["order"], "family": m["family"],
                "quality": m["data_quality"], "n_trnas": d["n_trnas"],
            })

    print(f"{len(rows):,} species with fragility, lifespan and body mass")
    by_class = defaultdict(int)
    for r in rows:
        by_class[r["class"]] += 1
    print("by class:", dict(sorted(by_class.items(), key=lambda kv: -kv[1])[:8]))

    frag = np.array([r[args.metric] for r in rows])
    life = np.log10([r["lifespan"] for r in rows])
    mass = np.log10([max(r["mass"], 1e-3) for r in rows])
    gc = np.array([r["gc"] for r in rows])

    print(f"\nmetric: {args.metric}")
    print(f"  fragility {frag.min():.4f} - {frag.max():.4f}, mean {frag.mean():.4f}")
    print(f"  lifespan  {10**life.min():.1f} - {10**life.max():.0f} yrs")

    out = {"n": len(rows), "metric": args.metric}

    print("\n--- raw associations (species treated as independent) ---")
    for name, v in (("log lifespan", life), ("log body mass", mass), ("GC%", gc)):
        r, rho = pearson(frag, v), spearman(frag, v)
        print(f"  fragility vs {name:14s} r={r:+.3f}  rho={rho:+.3f}")
        out[f"raw_{name}"] = {"pearson": r, "spearman": rho}

    print("\n--- lifespan, controlling body mass and GC ---")
    pc = partial_corr(frag, life, [mass, gc])
    print(f"  partial r = {pc:+.3f}")
    out["partial_r_mass_gc"] = pc

    beta, r2, _ = ols(frag, np.column_stack([life, mass, gc]))
    print(f"  fragility = {beta[0]:+.4f} {beta[1]:+.5f}*logLife "
          f"{beta[2]:+.5f}*logMass {beta[3]:+.5f}*GC   R2={r2:.3f}")
    out["ols"] = {"intercept": beta[0], "log_lifespan": beta[1],
                  "log_mass": beta[2], "gc": beta[3], "r2": r2}

    print("\n--- phylogenetic control: deviation within taxonomic group ---")
    for level in ("family", "order", "class"):
        groups = [r[level] for r in rows]
        keep_f, cf = group_center(frag, groups)
        keep_l, cl = group_center(life, groups)
        assert (keep_f == keep_l).all()
        if len(cf) < 30:
            print(f"  {level:7s} too few paired species ({len(cf)})")
            continue
        r = pearson(cf, cl)
        n_groups = len({groups[i] for i in keep_f})
        p = permutation_p(cf, cl, [groups[i] for i in keep_f], r)
        print(f"  within {level:7s} n={len(cf):4d} in {n_groups:3d} groups   "
              f"r={r:+.3f}   permutation p={p:.4f}")
        out[f"within_{level}"] = {"n": int(len(cf)), "groups": n_groups,
                                  "r": r, "p": p}

    print("\n--- per class ---")
    for cls in sorted(by_class, key=lambda c: -by_class[c]):
        ii = [i for i, r in enumerate(rows) if r["class"] == cls]
        if len(ii) < 40:
            continue
        r = pearson(frag[ii], life[ii])
        pc_c = partial_corr(frag[ii], life[ii], [mass[ii], gc[ii]])
        print(f"  {cls:16s} n={len(ii):4d}  r={r:+.3f}  partial={pc_c:+.3f}")
        out[f"class_{cls}"] = {"n": len(ii), "r": r, "partial": pc_c}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": out, "species": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
