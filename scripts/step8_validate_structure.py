"""Grade the predictors against base pairs that evolution proves are real.

Step 7 derived, for each human mitochondrial tRNA, the set of position pairs
that covary across thousands of animal species and stay Watson-Crick
compatible throughout. Those are true base pairs established without any
model, any folding algorithm, or any geometric assumption of mine.

Covariation cannot see a pair that never varies, so this measures recall on
the pairs it *can* see, not the whole structure. That is the right direction
of error: every pair in the reference is a real one, so a predictor that
misses them is genuinely wrong.

    python scripts/step8_validate_structure.py --adapter outputs/adapters/lora-r8/adapter
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import RNA
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from step2_benchmark import Folder  # noqa: E402
from mitominerva.cloverleaf import revcomp  # noqa: E402
from mitominerva.loading import load_model  # noqa: E402
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito  # noqa: E402

CONFIDENT = 0.5


def vienna_pairs(seq):
    structure, _ = RNA.fold(seq.replace("T", "U"))
    stack, pairs = [], set()
    for k, ch in enumerate(structure, start=1):
        if ch == "(":
            stack.append(k)
        elif ch == ")":
            pairs.add((stack.pop(), k))
    return pairs


def recovered(matrix, pairs, threshold=CONFIDENT):
    """How many reference pairs the map calls, and their mean probability."""
    hit, probs = 0, []
    for p in pairs:
        v = float(matrix[p["i"] - 1, p["j"] - 1])
        probs.append(v)
        hit += v >= threshold
    return hit, (float(np.mean(probs)) if probs else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covariation", default="data/processed/trna_covariation.json")
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--adapter", help="finetuned adapter; omit for the base model")
    ap.add_argument("--genome-map", help="whole-window map from the same weights")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--out", default="outputs/step8_covariation_validation.json")
    args = ap.parse_args()

    cov = json.load(open(args.covariation))["genes"]
    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    feats = {f["name"]: f for f in m.other_features if f["type"] == "tRNA"}

    tok, model = load_model(args.model, args.device, args.dtype, args.adapter)
    fold = Folder(model, tok, args.device)
    gmap = np.load(args.genome_map).astype(np.float32) if args.genome_map else None

    print(f"{'gene':7s} {'pairs':>5s} {'species':>8s} | {'isolated':>10s} "
          f"{'in genome':>10s} {'Vienna':>9s} | mean prob")
    print("-" * 74)

    rows, tot = [], {"pairs": 0, "iso": 0, "gen": 0, "vie": 0}
    for gene, d in sorted(cov.items()):
        pairs = d["pairs"]
        if len(pairs) < args.min_pairs:
            continue
        f = feats[gene]
        minus = f["strand"] == -1
        raw = rcrs[f["start"] - 1:f["end"]]
        seq = (revcomp(raw) if minus else raw).upper()
        if seq != d["human_seq"]:
            print(f"{gene}: sequence mismatch, skipping")
            continue

        iso = fold(seq)
        n_iso, p_iso = recovered(iso, pairs)

        n_gen = None
        if gmap is not None:
            span = m.token_span(f["start"], f["end"])
            sub = gmap[span[0]:span[1], span[0]:span[1]]
            if minus:
                sub = sub[::-1, ::-1]
            if sub.shape[0] == len(seq):
                n_gen, _ = recovered(sub, pairs)

        vp = vienna_pairs(seq)
        n_vie = sum((p["i"], p["j"]) in vp or (p["j"], p["i"]) in vp for p in pairs)

        tot["pairs"] += len(pairs); tot["iso"] += n_iso; tot["vie"] += n_vie
        if n_gen is not None:
            tot["gen"] += n_gen

        rows.append({"gene": gene, "n_pairs": len(pairs), "n_species": d["n_species"],
                     "isolated": n_iso, "genome": n_gen, "vienna": n_vie,
                     "mean_prob": p_iso})
        g = f"{n_gen}/{len(pairs)}" if n_gen is not None else "-"
        print(f"{gene:7s} {len(pairs):5d} {d['n_species']:8,} | "
              f"{n_iso:>6d}/{len(pairs):<3d} {g:>10s} {n_vie:>5d}/{len(pairs):<3d} | "
              f"{p_iso:.3f}")

    n = tot["pairs"]
    print("-" * 74)
    print(f"{'TOTAL':7s} {n:5d} {'':8s} | {tot['iso']:>6d}/{n:<3d} "
          f"{tot['gen']:>6d}/{n:<3d} {tot['vie']:>5d}/{n:<3d}")
    print(f"\nrecall of evolution-proven base pairs:")
    print(f"  Minerva, tRNA alone   {100*tot['iso']/n:5.1f}%")
    if gmap is not None:
        print(f"  Minerva, in genome    {100*tot['gen']/n:5.1f}%")
    print(f"  ViennaRNA             {100*tot['vie']/n:5.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"adapter": args.adapter, "genome_map": args.genome_map,
                   "total": tot, "genes": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
