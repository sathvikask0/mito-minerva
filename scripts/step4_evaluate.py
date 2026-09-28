"""Step 4 -- do the scores separate real disease mutations from harmless ones?

Joins the step-3 scores to ClinVar's labels and asks a single question of each
score: given one pathogenic and one benign variant at random, how often does
the score rank the pathogenic one as worse?  That is the AUC.  0.5 is a coin
flip; 1.0 is perfect.

ViennaRNA is scored the same way on the same variants, so the comparison is
like for like.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import RNA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.cloverleaf import revcomp
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito

PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


def auc(scores, labels):
    """Probability a random positive outranks a random negative (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(scores, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pi, ni = np.flatnonzero(y), np.flatnonzero(~y)
    boots = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        boots.append(auc(s[idx], y[idx]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def vienna_scores(wt_seq, i, mut_base):
    """Destabilisation and structure change from ViennaRNA's MFE fold."""
    mut_seq = wt_seq[:i] + mut_base + wt_seq[i + 1:]
    s_wt, e_wt = RNA.fold(wt_seq.replace("T", "U"))
    s_mu, e_mu = RNA.fold(mut_seq.replace("T", "U"))
    return {"vienna_ddg": float(e_mu - e_wt),
            "vienna_bpdist": float(RNA.bp_distance(s_wt, s_mu))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/step3_mutations.json")
    ap.add_argument("--labels", default="data/processed/clinvar_trna.json")
    ap.add_argument("--context-llr", default="outputs/step3b_context_llr.json")
    ap.add_argument("--high-confidence", action="store_true",
                    help="keep only Pathogenic and Benign, dropping the 'Likely' calls")
    ap.add_argument("--out", default="outputs/step4_evaluation.json")
    args = ap.parse_args()

    scores = json.load(open(args.scores))
    labels = json.load(open(args.labels))
    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    trna_feat = {f["name"]: f for f in m.other_features if f["type"] == "tRNA"}

    by_key = {(r["rcrs_pos"], r["wt"], r["mut"]): r for r in scores}
    ctx = {}
    if os.path.exists(args.context_llr):
        ctx = {(r["rcrs_pos"], r["wt"], r["mut"]): r["llr_ctx"]
               for r in json.load(open(args.context_llr))}

    rows, missed = [], 0
    vienna_cache = {}
    for lab in labels:
        cls = lab["clinvar"]
        if args.high_confidence and cls not in ("Pathogenic", "Benign"):
            continue
        if cls in PATHOGENIC:
            y = 1
        elif cls in BENIGN:
            y = 0
        else:
            continue  # uncertain, conflicting, not provided

        f = trna_feat[lab["trna"]]
        minus = f["strand"] == -1
        # Step 3 scored transcript alleles; ClinVar reports heavy-strand alleles.
        ref = COMPLEMENT[lab["ref"]] if minus else lab["ref"]
        alt = COMPLEMENT[lab["alt"]] if minus else lab["alt"]
        rec = by_key.get((lab["pos"], ref, alt))
        if rec is None:
            missed += 1
            continue

        raw = rcrs[f["start"] - 1:f["end"]]
        wt_seq = (revcomp(raw) if minus else raw).upper()
        i = rec["transcript_pos"] - 1
        assert wt_seq[i] == ref, f"{lab} expected {ref} got {wt_seq[i]}"
        ck = (lab["trna"], i, alt)
        if ck not in vienna_cache:
            vienna_cache[ck] = vienna_scores(wt_seq, i, alt)

        rows.append({**{k: lab[k] for k in ("trna", "pos", "ref", "alt", "clinvar")},
                     "label": y,
                     "pairs_lost": rec["pairs_lost"], "llr": rec["llr"],
                     "stem_lost": rec["stem_lost"], "delta_l1": rec["delta_l1"],
                     "llr_ctx": ctx.get((lab["pos"], ref, alt)),
                     **vienna_cache[ck]})

    n_pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} variants matched  ({n_pos} pathogenic, {len(rows) - n_pos} benign)"
          f"{f'  [{missed} unmatched]' if missed else ''}\n")

    y = [r["label"] for r in rows]

    def z(key, sign=1.0):
        v = np.array([r[key] for r in rows], float) * sign
        return (v - v.mean()) / (v.std() or 1.0)

    have_ctx = all(r.get("llr_ctx") is not None for r in rows)
    combos = {
        "Minerva: folding lost": z("pairs_lost"),
        "Minerva: stem pairs lost": z("stem_lost"),
        "Minerva: surprise (-LLR)": z("llr", -1.0),
        "Minerva: both combined": z("pairs_lost") + z("llr", -1.0),
        **({"Minerva: surprise in context": z("llr_ctx", -1.0),
            "Minerva: context + folding": z("llr_ctx", -1.0) + z("pairs_lost"),
            "Minerva: all three": z("llr_ctx", -1.0) + z("llr", -1.0) + z("pairs_lost"),
            } if have_ctx else {}),
        "ViennaRNA: destabilisation": z("vienna_ddg"),
        "ViennaRNA: structure change": z("vienna_bpdist"),
        "ViennaRNA: both combined": z("vienna_ddg") + z("vienna_bpdist"),
    }

    print(f"{'score':30s} {'AUC':>6s}  {'95% CI':>14s}")
    print("-" * 54)
    results = {}
    for name, v in combos.items():
        a = auc(v, y)
        lo, hi = auc_ci(v, y)
        results[name] = {"auc": a, "ci": [lo, hi]}
        print(f"{name:30s} {a:>6.3f}  [{lo:.3f}, {hi:.3f}]")

    best_m = max([k for k in combos if k.startswith("Minerva")],
                 key=lambda k: results[k]["auc"])
    best_v = max([k for k in combos if k.startswith("Vienna")],
                 key=lambda k: results[k]["auc"])
    delta = paired_delta(combos[best_m], combos[best_v], y)
    print(f"\nbest Minerva ({best_m.split(': ')[1]}) minus "
          f"best ViennaRNA ({best_v.split(': ')[1]}): "
          f"{delta[0]:+.3f}  95% CI [{delta[1]:+.3f}, {delta[2]:+.3f}]")

    print("\nlabel mix:", dict(Counter(r["clinvar"] for r in rows)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n": len(rows), "n_pathogenic": n_pos, "results": results,
                   "best_minerva": best_m, "best_vienna": best_v,
                   "delta_auc": {"point": delta[0], "ci": [delta[1], delta[2]]},
                   "variants": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


def paired_delta(a, b, y, n=2000, seed=0):
    """Bootstrap the AUC difference on the same resampled variants."""
    rng = np.random.default_rng(seed)
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(y, bool)
    pi, ni = np.flatnonzero(y), np.flatnonzero(~y)
    d = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
        d.append(auc(a[idx], y[idx]) - auc(b[idx], y[idx]))
    return float(auc(a, y) - auc(b, y)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


if __name__ == "__main__":
    raise SystemExit(main())
