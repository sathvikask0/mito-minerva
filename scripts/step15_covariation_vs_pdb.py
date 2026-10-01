"""Grade the covariation reference itself against the lab structures.

Step 7 scored the model against base pairs inferred from covariation across
animal mitochondrial genomes. A model trained on those same genomes could, in
principle, just be reproducing that covariation signal. This checks the
reference against the 8 experimental structures from step 13, using the same
pooled precision/recall/F1 and whole-tRNA bootstrap as step 14, and compares
it with the predictors graded there.

No model is run; it reads committed files only.

    python scripts/step15_covariation_vs_pdb.py
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COV = os.path.join(ROOT, "data/processed/trna_covariation.json")
PDB = os.path.join(ROOT, "data/processed/pdb_trna_pairs.json")
STEP14 = os.path.join(ROOT, "outputs/step14_pdb_grading.json")
OUT = os.path.join(ROOT, "outputs/step15_covariation_vs_pdb.json")
OTHERS = ["Minerva finetuned, tRNA alone", "Minerva finetuned, in genome", "ViennaRNA",
          "Minerva base, tRNA alone"]


def prf(m):
    tp, npd, nt = m[..., 0].sum(-1), m[..., 1].sum(-1), m[..., 2].sum(-1)
    p = np.where(npd > 0, tp / np.maximum(npd, 1), 0.0)
    r = tp / nt
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return p, r, f


def main() -> int:
    cov = json.load(open(COV))["genes"]
    pdb = json.load(open(PDB))["genes"]
    per_gene = json.load(open(STEP14))["per_gene"]
    genes = sorted(pdb)

    rows, detail = [], {}
    for g in genes:
        assert cov[g]["human_seq"] == pdb[g]["human_seq"], g  # same 1-based coordinates
        truth = {tuple(sorted(p)) for p in pdb[g]["consensus"]}
        pred = {(q["i"], q["j"]) for q in cov[g]["pairs"]}
        rows.append([len(pred & truth), len(pred), len(truth)])
        detail[g] = {"tp": len(pred & truth), "n_pred": len(pred), "n_true": len(truth),
                     "false_pairs": sorted(pred - truth), "missed": sorted(truth - pred)}
    cov_c = np.array(rows, float)

    def counts(name):
        return np.array([[per_gene[g][name][k] for k in ("tp", "n_pred", "n_true")]
                         for g in genes], float)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(genes), size=(10000, len(genes)))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    out = {"genes": genes, "pooled": {}, "vs_covariation": {}, "per_gene": detail}
    print(f"{len(genes)} tRNAs, {int(cov_c[:, 2].sum())} experimental pairs\n")
    print(f"{'predictor':32s} {'precision':>9s} {'recall':>7s} {'F1':>6s}  F1 95% CI")
    for name, c in [("Covariation reference (step 7)", cov_c)] + [(n, counts(n)) for n in OTHERS]:
        p, r, f = (float(x) for x in prf(c))
        fci = ci(prf(c[idx])[2])
        out["pooled"][name] = {"precision": p, "recall": r, "f1": f, "ci_f1": fci}
        print(f"{name:32s} {100*p:8.1f}% {100*r:6.1f}% {f:6.3f}  [{fci[0]:.2f}, {fci[1]:.2f}]")

    print(f"\n{'F1 minus covariation reference':32s} {'point':>6s}  95% CI            tRNAs better-worse")
    for name in OTHERS:
        c = counts(name)
        d = prf(c[idx])[2] - prf(cov_c[idx])[2]
        pt = float(prf(c)[2] - prf(cov_c)[2])
        fa = [float(prf(c[k])[2]) for k in range(len(genes))]
        fb = [float(prf(cov_c[k])[2]) for k in range(len(genes))]
        w, l = sum(a > b for a, b in zip(fa, fb)), sum(a < b for a, b in zip(fa, fb))
        lo, hi = ci(d)
        flag = "" if lo > 0 or hi < 0 else "  (crosses zero)"
        print(f"{name:32s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {w}-{l}{flag}")
        out["vs_covariation"][name] = {"point": pt, "ci": [lo, hi], "better": w, "worse": l}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
