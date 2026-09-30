"""Grade every predictor against experimentally solved tRNA structures.

Step 13 read the secondary structure of 8 human mitochondrial tRNAs off
X-ray and cryo-EM models. Those pairs owe nothing to the 15,589 genomes the
model was finetuned on, which removes the circularity in the covariation
test. Because the structures are complete, predicted pairs that are not real
now count against a predictor, so precision and F1 are reported alongside
recall.

A Minerva pair is called where the base-pairing map is >= 0.5 (|i-j| >= 4).
ViennaRNA contributes its minimum-free-energy structure. Intervals come from
resampling whole tRNAs.

    python scripts/step14_grade_vs_pdb.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from step2_benchmark import Folder  # noqa: E402
from step8_validate_structure import vienna_pairs  # noqa: E402
from mitominerva.cloverleaf import revcomp  # noqa: E402
from mitominerva.loading import load_model  # noqa: E402
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito  # noqa: E402

THRESH, MIN_SEP = 0.5, 4


def map_pairs(mat):
    L = mat.shape[0]
    return {(i + 1, j + 1) for i in range(L) for j in range(i + MIN_SEP, L)
            if mat[i, j] >= THRESH}


def score(pred, truth):
    tp = len(pred & truth)
    return tp, len(pred), len(truth)


def prf(tp, npred, ntrue):
    r = tp / ntrue if ntrue else float("nan")
    p = tp / npred if npred else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="data/processed/pdb_trna_pairs.json")
    ap.add_argument("--adapter", default="outputs/adapters/lora-r8/adapter")
    ap.add_argument("--genome-base", default="outputs/full_window/base_pairing.npy")
    ap.add_argument("--genome-ft", default="outputs/full_window_ft/base_pairing.npy")
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="outputs/step14_pdb_grading.json")
    args = ap.parse_args()

    pdb = json.load(open(args.pdb))["genes"]
    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    feats = {f["name"]: f for f in m.other_features if f["type"] == "tRNA"}

    seqs, spans = {}, {}
    for g in pdb:
        f = feats[g]
        minus = f["strand"] == -1
        raw = rcrs[f["start"] - 1:f["end"]]
        seqs[g] = (revcomp(raw) if minus else raw).upper()
        assert seqs[g] == pdb[g]["human_seq"], g
        spans[g] = (m.token_span(f["start"], f["end"]), minus)

    preds = {g: {} for g in pdb}
    for tag, adapter, gpath in (("base", None, args.genome_base),
                                ("finetuned", args.adapter, args.genome_ft)):
        tok, model = load_model(args.model, args.device, "float32", adapter)
        fold = Folder(model, tok, args.device)
        gmap = np.load(gpath).astype(np.float32)
        for g in pdb:
            preds[g][f"Minerva {tag}, tRNA alone"] = map_pairs(fold(seqs[g]))
            (s0, s1), minus = spans[g]
            sub = gmap[s0:s1, s0:s1]
            if minus:
                sub = sub[::-1, ::-1]
            preds[g][f"Minerva {tag}, in genome"] = map_pairs(sub)
        del model
    for g in pdb:
        preds[g]["ViennaRNA"] = {tuple(sorted(p)) for p in vienna_pairs(seqs[g])
                                 if abs(p[1] - p[0]) >= MIN_SEP}

    names = ["Minerva finetuned, tRNA alone", "Minerva finetuned, in genome",
             "ViennaRNA", "Minerva base, tRNA alone", "Minerva base, in genome"]
    genes = sorted(pdb)
    counts = {n: np.array([score(preds[g][n], {tuple(p) for p in pdb[g]["consensus"]})
                           for g in genes], float) for n in names}

    print(f"{len(genes)} tRNAs, {int(counts[names[0]][:, 2].sum())} experimental pairs\n")
    print(f"{'gene':6s} {'pairs':>5s} | " + " | ".join(f"{n.replace('Minerva ', '')[:18]:>18s}" for n in names))
    for k, g in enumerate(genes):
        cells = []
        for n in names:
            tp, npred, ntrue = counts[n][k]
            p, r, f = prf(tp, npred, ntrue)
            cells.append(f"R{int(tp):2d}/{int(ntrue):<2d} P{100*p:3.0f}% F{f:.2f}")
        print(f"{g:6s} {int(counts[names[0]][k, 2]):5d} | " + " | ".join(f"{c:>18s}" for c in cells))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(genes), size=(args.n_boot, len(genes)))

    def pooled(c):
        return prf(c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum())

    def boot(c):
        tp, npd, nt = c[idx, 0].sum(1), c[idx, 1].sum(1), c[idx, 2].sum(1)
        r = tp / nt
        p = np.where(npd > 0, tp / np.maximum(npd, 1), 0)
        f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
        return p, r, f

    out = {"genes": genes, "pooled": {}, "deltas": {}, "per_gene": {}}
    for subset_name, keep in (("all 8", genes), ("without TRNS2", [g for g in genes if g != "TRNS2"])):
        ki = [genes.index(g) for g in keep]
        sub_idx = rng.integers(0, len(ki), size=(args.n_boot, len(ki)))
        print(f"\n=== pooled, {subset_name} ({len(ki)} tRNAs) ===")
        print(f"{'predictor':32s} {'precision':>18s} {'recall':>18s} {'F1':>18s}")
        bs = {}
        for n in names:
            c = counts[n][ki]
            p, r, f = pooled(c)
            tp, npd, nt = c[sub_idx, 0].sum(1), c[sub_idx, 1].sum(1), c[sub_idx, 2].sum(1)
            br = tp / nt
            bp = np.where(npd > 0, tp / np.maximum(npd, 1), 0)
            bf = np.where(bp + br > 0, 2 * bp * br / np.maximum(bp + br, 1e-12), 0)
            bs[n] = (bp, br, bf)
            ci = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))
            (pl, ph), (rl, rh), (fl, fh) = ci(bp), ci(br), ci(bf)
            print(f"{n:32s} {100*p:5.1f} [{100*pl:4.1f},{100*ph:5.1f}] "
                  f"{100*r:5.1f} [{100*rl:4.1f},{100*rh:5.1f}] "
                  f"{f:.3f} [{fl:.2f},{fh:.2f}]")
            out["pooled"][f"{subset_name} | {n}"] = {
                "precision": p, "recall": r, "f1": f,
                "ci_precision": [pl, ph], "ci_recall": [rl, rh], "ci_f1": [fl, fh]}
        print(f"\n{'F1 difference':52s} {'point':>6s}  95% CI        tRNAs better-worse")
        for a, b in (("Minerva finetuned, tRNA alone", "ViennaRNA"),
                     ("Minerva finetuned, in genome", "ViennaRNA"),
                     ("Minerva finetuned, tRNA alone", "Minerva base, tRNA alone"),
                     ("Minerva finetuned, in genome", "Minerva base, in genome")):
            d = bs[a][2] - bs[b][2]
            pt = pooled(counts[a][ki])[2] - pooled(counts[b][ki])[2]
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            fa = [prf(*counts[a][k])[2] for k in ki]
            fb = [prf(*counts[b][k])[2] for k in ki]
            w = sum(x > y for x, y in zip(fa, fb)); l = sum(x < y for x, y in zip(fa, fb))
            flag = "" if lo > 0 or hi < 0 else "  (crosses zero)"
            print(f"{a + ' - ' + b:52s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {w}-{l}{flag}")
            out["deltas"][f"{subset_name} | {a} - {b}"] = {
                "point": pt, "ci": [lo, hi], "better": w, "worse": l}

    for k, g in enumerate(genes):
        out["per_gene"][g] = {n: dict(zip(("tp", "n_pred", "n_true"),
                                          map(int, counts[n][k]))) for n in names}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
