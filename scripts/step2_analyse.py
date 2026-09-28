"""Step 2b -- the go / no-go: does Minerva fold the mitochondrial tRNAs?

Scores the cached base_pairing map against the cloverleaf geometry for all 22
tRNAs, against length-matched decoys drawn from the same genome, and against
ViennaRNA's thermodynamic fold as a baseline.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minerva.rna_structure import call_structure
from mitominerva.cloverleaf import (
    MITO_ANTICODONS,
    acceptor_stem_pairs,
    anticodon_stem_pairs,
    find_anticodon,
    revcomp,
    score_stem,
)
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito


def transcript_view(matrix, seq, minus):
    """Put a slice into transcript orientation.

    The model always reads the heavy strand, so a light-strand tRNA was shown
    the reverse complement of its own transcript.  Flipping both axes of the
    map and reverse-complementing the sequence recovers transcript coordinates.
    """
    if not minus:
        return matrix, seq.upper()
    return matrix[::-1, ::-1], revcomp(seq).upper()


def score_trna(matrix, seq, name):
    """Acceptor- and anticodon-stem recovery for one tRNA, in transcript order."""
    L = len(seq)
    acc = score_stem(matrix, acceptor_stem_pairs(L), name="acceptor")
    ac_start = find_anticodon(seq, MITO_ANTICODONS[name]) if name in MITO_ANTICODONS else None
    if ac_start is None:
        anti = None
    else:
        anti = score_stem(matrix, anticodon_stem_pairs(ac_start, L), name="anticodon")
    return acc, anti, ac_start


def vienna_stems(seq, L, ac_start):
    """Same two stems, but scored on ViennaRNA's minimum-free-energy fold."""
    try:
        import RNA
    except ImportError:
        return None
    structure, _ = RNA.fold(seq.replace("T", "U"))
    partners = np.full(L, -1, dtype=int)
    stack = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            j = stack.pop()
            partners[i], partners[j] = j, i
    acc = sum(1 for i, j in acceptor_stem_pairs(L) if partners[i] == j)
    anti = (
        sum(1 for i, j in anticodon_stem_pairs(ac_start, L) if partners[i] == j)
        if ac_start is not None else None
    )
    return structure, acc, anti


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="outputs/full_window")
    ap.add_argument("--n-decoys", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bp = np.load(os.path.join(args.maps, "base_pairing.npy")).astype(np.float32)
    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    assert bp.shape[0] == m.n_tokens, f"{bp.shape} vs {m.n_tokens} tokens"

    trnas = sorted(
        [f for f in m.other_features if f["type"] == "tRNA"], key=lambda f: f["start"]
    )
    print(f"base_pairing map {bp.shape}, {len(trnas)} tRNAs\n")

    header = (f"{'tRNA':7s} {'strand':6s} {'len':>4s}  {'acceptor 7bp':>13s} "
              f"{'anticodon 5bp':>14s}  {'Vienna acc':>10s}  dot-bracket")
    print(header)
    print("-" * len(header))

    rows, occupied = [], []
    for f in trnas:
        s0, s1 = m.token_span(f["start"], f["end"])
        occupied.append((s0, s1))
        sub = bp[s0:s1, s0:s1]
        seq = rcrs[f["start"] - 1:f["end"]]
        minus = f["strand"] == -1
        mat, tseq = transcript_view(sub, seq, minus)

        acc, anti, ac_start = score_trna(mat, tseq, f["name"])
        vienna = vienna_stems(tseq, len(tseq), ac_start)
        st = call_structure(mat, tseq, name=f["name"])

        rows.append({
            "name": f["name"], "product": f["product"], "strand": "-" if minus else "+",
            "start": f["start"], "end": f["end"], "length": len(tseq),
            "acceptor_recovered": acc.n_recovered, "acceptor_mean": acc.mean_score,
            "acceptor_shift": acc.shift,
            "anticodon_recovered": anti.n_recovered if anti else None,
            "anticodon_expected": len(anti.scores) if anti else None,
            "anticodon_mean": anti.mean_score if anti else None,
            "anticodon_found": ac_start,
            "vienna_acceptor": vienna[1] if vienna else None,
            "vienna_anticodon": vienna[2] if vienna else None,
            "vienna_structure": vienna[0] if vienna else None,
            "dot_bracket": st.dot_bracket,
            "n_pairs": len(st.pairs),
            "canonical_fraction": st.canonical_fraction(),
        })
        ant = f"{anti.n_recovered}/{len(anti.scores)} ({anti.mean_score:.2f})" if anti else "no anticodon"
        print(f"{f['name']:7s} {'-' if minus else '+':6s} {len(tseq):>4d}  "
              f"{acc.n_recovered}/7 ({acc.mean_score:.2f}){'':>3s} {ant:>14s}  "
              f"{(str(vienna[1]) + '/7') if vienna else 'n/a':>10s}  {st.dot_bracket}")

    # Decoys: same length, same genome, same lower-case regions, no tRNA overlap.
    rng = np.random.default_rng(args.seed)
    lower = [i for i, ch in enumerate(m.token_string) if ch.islower()]
    tok_of = {i: m.char_to_token[i] for i in lower}
    lower_tokens = sorted({tok_of[i] for i in lower})
    taken = np.zeros(m.n_tokens, bool)
    for s0, s1 in occupied:
        taken[s0:s1] = True

    decoy_acc, lengths = [], [r["length"] for r in rows]
    tries = 0
    while len(decoy_acc) < args.n_decoys and tries < args.n_decoys * 200:
        tries += 1
        L = int(rng.choice(lengths))
        s0 = int(rng.choice(lower_tokens))
        s1 = s0 + L
        if s1 > m.n_tokens or taken[s0:s1].any():
            continue
        if not all(t in set(lower_tokens) for t in (s0, s1 - 1)):
            continue
        sub = bp[s0:s1, s0:s1]
        if sub.shape[0] != L:
            continue
        decoy_acc.append(score_stem(sub, acceptor_stem_pairs(L), name="acceptor"))

    print()
    summarise(rows, decoy_acc)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/step2_trna_scores.json", "w") as fh:
        json.dump({"trnas": rows,
                   "decoys": {"n": len(decoy_acc),
                              "acceptor_recovered": [d.n_recovered for d in decoy_acc],
                              "acceptor_mean": [d.mean_score for d in decoy_acc]}},
                  fh, indent=2)
    print("\nwrote outputs/step2_trna_scores.json")
    return 0


def summarise(rows, decoys):
    acc = np.array([r["acceptor_recovered"] for r in rows])
    vie = np.array([r["vienna_acceptor"] for r in rows])
    ant = np.array([r["anticodon_recovered"] for r in rows if r["anticodon_recovered"] is not None])
    ante = np.array([r["anticodon_expected"] for r in rows if r["anticodon_expected"] is not None])
    dec = np.array([d.n_recovered for d in decoys]) if decoys else np.array([0])

    print("=" * 72)
    print(f"acceptor stem   Minerva {acc.sum()}/{7 * len(rows)} bp "
          f"({acc.sum() / (7 * len(rows)):.0%})   "
          f"tRNAs with >=5/7: {(acc >= 5).sum()}/{len(rows)}")
    print(f"                Vienna  {vie.sum()}/{7 * len(rows)} bp "
          f"({vie.sum() / (7 * len(rows)):.0%})   "
          f"tRNAs with >=5/7: {(vie >= 5).sum()}/{len(rows)}")
    print(f"anticodon stem  Minerva {ant.sum()}/{ante.sum()} bp "
          f"({ant.sum() / max(ante.sum(), 1):.0%})")
    print(f"decoys (n={len(decoys)})  {dec.mean():.2f}/7 bp mean, "
          f">=5/7 in {(dec >= 5).mean():.1%} of them")
    plus = np.array([r["acceptor_recovered"] for r in rows if r["strand"] == "+"])
    minus = np.array([r["acceptor_recovered"] for r in rows if r["strand"] == "-"])
    print(f"by strand       heavy(+) {plus.mean():.1f}/7 (n={len(plus)})   "
          f"light(-) {minus.mean():.1f}/7 (n={len(minus)})")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
