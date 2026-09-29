"""Is Minerva's mutation score just conservation in disguise?

Step 5 finetuned on 15,589 animal mitochondrial genomes and the disease AUC
jumped from 0.601 to 0.744. Many of those genomes are mammals whose tRNAs
closely resemble ours, so the model may have learned nothing more than which
bases are conserved across animals -- the signal existing clinical predictors
already use.

This scores the same ClinVar variants three ways on the same footing:

  conservation   built from the same corpus by step6_build_conservation
  Minerva        the finetuned masked-language-model surprise
  ViennaRNA      the thermodynamic baseline

and then asks the question that matters: does Minerva add anything on top of
conservation, or is it redundant with it?

    python scripts/step6_evaluate.py --scores outputs/step3_mutations_ft.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from step4_evaluate import (  # noqa: E402
    BENIGN, COMPLEMENT, PATHOGENIC, auc, auc_ci, paired_delta, vienna_scores,
)
from mitominerva.cloverleaf import revcomp  # noqa: E402
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito  # noqa: E402

PSEUDO = 0.5  # Laplace-style prior, so an allele never seen is not -inf


def spearman(a, b):
    """Rank correlation, ties averaged. Avoids a scipy dependency."""
    def rank(v):
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
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def conservation_scores(cons, gene, pos, wt, alt):
    """Allele counts at one human position, turned into three scores.

    Returns None when the column was never covered, so the variant can be
    dropped rather than silently scored as average.
    """
    g = cons["genes"].get(gene)
    if not g:
        return None
    col = g["positions"].get(str(pos))
    if not col:
        return None
    total = sum(col.values())
    if total < 50:  # too thin to mean anything
        return None
    f_alt = (col.get(alt, 0) + PSEUDO) / (total + 4 * PSEUDO)
    f_wt = (col.get(wt, 0) + PSEUDO) / (total + 4 * PSEUDO)
    return {
        "cons_n": total,
        "cons_alt_rarity": -math.log(f_alt),
        "cons_wt_frequency": f_wt,
        "cons_log_odds": math.log(f_wt / f_alt),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/step3_mutations_ft.json")
    ap.add_argument("--baseline-scores", default="outputs/step3_mutations.json",
                    help="the pre-finetuning scores, for the before/after column")
    ap.add_argument("--labels", default="data/processed/clinvar_trna.json")
    ap.add_argument("--conservation", default="data/processed/trna_conservation.json")
    ap.add_argument("--out", default="outputs/step6_conservation_eval.json")
    args = ap.parse_args()

    scores = json.load(open(args.scores))
    cons = json.load(open(args.conservation))
    labels = json.load(open(args.labels))
    base = {}
    if os.path.exists(args.baseline_scores):
        base = {(r["rcrs_pos"], r["wt"], r["mut"]): r["llr"]
                for r in json.load(open(args.baseline_scores))}

    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    trna_feat = {f["name"]: f for f in m.other_features if f["type"] == "tRNA"}
    by_key = {(r["rcrs_pos"], r["wt"], r["mut"]): r for r in scores}

    rows, no_cons, missed = [], 0, 0
    vienna_cache = {}
    for lab in labels:
        cls = lab["clinvar"]
        if cls in PATHOGENIC:
            y = 1
        elif cls in BENIGN:
            y = 0
        else:
            continue

        f = trna_feat[lab["trna"]]
        minus = f["strand"] == -1
        ref = COMPLEMENT[lab["ref"]] if minus else lab["ref"]
        alt = COMPLEMENT[lab["alt"]] if minus else lab["alt"]
        rec = by_key.get((lab["pos"], ref, alt))
        if rec is None:
            missed += 1
            continue

        c = conservation_scores(cons, lab["trna"], rec["transcript_pos"], ref, alt)
        if c is None:
            no_cons += 1
            continue

        raw = rcrs[f["start"] - 1:f["end"]]
        wt_seq = (revcomp(raw) if minus else raw).upper()
        i = rec["transcript_pos"] - 1
        ck = (lab["trna"], i, alt)
        if ck not in vienna_cache:
            vienna_cache[ck] = vienna_scores(wt_seq, i, alt)

        rows.append({
            **{k: lab[k] for k in ("trna", "pos", "ref", "alt", "clinvar")},
            "label": y, "llr": rec["llr"], "pairs_lost": rec["pairs_lost"],
            "llr_base": base.get((lab["pos"], ref, alt)),
            **c, **vienna_cache[ck],
        })

    n_pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} variants scored ({n_pos} pathogenic, {len(rows) - n_pos} benign)")
    print(f"  dropped: {missed} unmatched, {no_cons} without conservation coverage\n")

    y = [r["label"] for r in rows]

    def z(key, sign=1.0):
        v = np.array([r[key] for r in rows], float) * sign
        return (v - v.mean()) / (v.std() or 1.0)

    combos = {
        "Conservation: alt allele rarity": z("cons_alt_rarity"),
        "Conservation: position conserved": z("cons_wt_frequency"),
        "Conservation: log-odds wt/alt": z("cons_log_odds"),
        "Minerva finetuned: surprise": z("llr", -1.0),
        "Minerva finetuned: folding lost": z("pairs_lost"),
        "ViennaRNA: destabilisation": z("vienna_ddg"),
        "Minerva + conservation": z("llr", -1.0) + z("cons_log_odds"),
        "Conservation + ViennaRNA": z("cons_log_odds") + z("vienna_ddg"),
    }
    have_base = all(r["llr_base"] is not None for r in rows)
    if have_base:
        combos["Minerva base (pre-finetune)"] = z("llr_base", -1.0)

    print(f"{'score':34s} {'AUC':>6s}  {'95% CI':>16s}")
    print("-" * 60)
    results = {}
    for name, v in combos.items():
        a = auc(v, y)
        lo, hi = auc_ci(v, y)
        results[name] = {"auc": a, "ci": [lo, hi]}
        print(f"{name:34s} {a:>6.3f}  [{lo:+.3f}, {hi:+.3f}]")

    best_cons = max([k for k in combos if k.startswith("Conservation:")],
                    key=lambda k: results[k]["auc"])
    minerva = "Minerva finetuned: surprise"

    print("\n--- is Minerva more than conservation? ---")
    rho = spearman([r["llr"] for r in rows], [-r["cons_log_odds"] for r in rows])
    print(f"rank correlation, Minerva surprise vs conservation: {rho:+.3f}")

    d1 = paired_delta(combos[minerva], combos[best_cons], y)
    print(f"Minerva minus best conservation: "
          f"{d1[0]:+.3f}  95% CI [{d1[1]:+.3f}, {d1[2]:+.3f}]")

    d2 = paired_delta(combos["Minerva + conservation"], combos[best_cons], y)
    print(f"adding Minerva to conservation:  "
          f"{d2[0]:+.3f}  95% CI [{d2[1]:+.3f}, {d2[2]:+.3f}]")

    print("\nlabel mix:", dict(Counter(r["clinvar"] for r in rows)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "n": len(rows), "n_pathogenic": n_pos,
            "conservation_source": {k: cons[k] for k in
                                    ("genomes", "tRNAs_tallied", "min_identity")},
            "results": results,
            "spearman_minerva_vs_conservation": rho,
            "minerva_minus_conservation": {"point": d1[0], "ci": [d1[1], d1[2]]},
            "minerva_added_to_conservation": {"point": d2[0], "ci": [d2[1], d2[2]]},
            "variants": rows,
        }, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
