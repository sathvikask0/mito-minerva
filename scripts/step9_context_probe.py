"""Where does genomic context help, and where does it drown the tRNA out?

TRNV is the one tRNA the finetuned model still gets wrong inside the genome:
10/13 evolution-proven pairs alone, 0/13 in context. TRNV is also the only
tRNA wedged between the two rRNA genes, which are themselves large folded
structures. Before spending GPU time scanning thousands of species, it is
worth knowing whether that is a TRNV quirk or a general weakness near rRNA -
because if it is general, every species' TRNV score would be noise.

Two conditions at each flank size:

  real       the actual neighbouring genome sequence
  shuffled   the same flanks, dinucleotide-shuffled

Dinucleotide shuffling preserves length and base composition but destroys
structure. If real flanks break the prediction and shuffled ones of identical
composition do not, the damage comes from what the neighbours *are*, not from
mere sequence length.

    python scripts/step9_context_probe.py --adapter outputs/adapters/lora-r8/adapter
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from step2_benchmark import Folder  # noqa: E402
from step8_validate_structure import recovered  # noqa: E402
from mitominerva.cloverleaf import revcomp  # noqa: E402
from mitominerva.loading import load_model  # noqa: E402
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito  # noqa: E402

FLANKS = [0, 25, 50, 100, 200, 400, 800]
# rCRS coordinates of the two rRNA genes, for the adjacency column.
RRNA = [(648, 1601), (1671, 3229)]


def dinuc_shuffle(seq, rng):
    """Altschul-Erikson: preserve dinucleotide frequencies, destroy structure."""
    if len(seq) < 4:
        return seq
    for _ in range(20):
        edges = {}
        for a, b in zip(seq, seq[1:]):
            edges.setdefault(a, []).append(b)
        for v in edges.values():
            rng.shuffle(v)
        out, cur = [seq[0]], seq[0]
        ok = True
        used = {k: 0 for k in edges}
        for _ in range(len(seq) - 1):
            if used[cur] >= len(edges[cur]):
                ok = False
                break
            nxt = edges[cur][used[cur]]
            used[cur] += 1
            out.append(nxt)
            cur = nxt
        if ok and len(out) == len(seq):
            return "".join(out)
    return "".join(rng.sample(list(seq), len(seq)))


def gap_to_rrna(start, end):
    """Bases between this gene and the nearest rRNA; 0 if it touches one."""
    return min(max(0, rs - end, start - re_) for rs, re_ in RRNA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covariation", default="data/processed/trna_covariation.json")
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--adapter")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/step9_context_probe.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cov = json.load(open(args.covariation))["genes"]
    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    feats = {f["name"]: f for f in m.other_features if f["type"] == "tRNA"}

    tok, model = load_model(args.model, args.device, args.dtype, args.adapter)
    fold = Folder(model, tok, args.device)

    genes = sorted(g for g in cov if len(cov[g]["pairs"]) >= 5)
    head = "  ".join(f"{f:>4d}" for f in FLANKS)
    print(f"{'gene':7s} {'gap':>5s} {'pairs':>5s} | real flanks: {head}")
    print(f"{'':7s} {'':5s} {'':5s} | shuffled:    {head}")
    print("-" * (34 + 6 * len(FLANKS)))

    rows = []
    for gene in genes:
        pairs = cov[gene]["pairs"]
        f = feats[gene]
        minus = f["strand"] == -1
        L = f["end"] - f["start"] + 1
        gap = gap_to_rrna(f["start"], f["end"])

        real, shuf = [], []
        for flank in FLANKS:
            lo = max(1, f["start"] - flank)
            hi = min(len(rcrs), f["end"] + flank)
            left, core, right = (rcrs[lo - 1:f["start"] - 1].upper(),
                                 rcrs[f["start"] - 1:f["end"]].upper(),
                                 rcrs[f["end"]:hi].upper())

            for tag, (l, r) in (("real", (left, right)),
                                ("shuf", (dinuc_shuffle(left, rng),
                                          dinuc_shuffle(right, rng)))):
                win = l + core + r
                off = len(l)
                if minus:
                    win = revcomp(win)
                    off = len(r)  # revcomp puts the right flank first
                mat = fold(win)
                # The window was already reverse-complemented above, so the
                # map is in transcript coordinates. Flipping again here would
                # double-correct -- that bug made flank=0 disagree with the
                # isolated figure in step 8.
                sub = mat[off:off + L, off:off + L]
                n, _ = recovered(sub, pairs)
                (real if tag == "real" else shuf).append(n)

        rows.append({"gene": gene, "n_pairs": len(pairs), "gap_to_rrna": gap,
                     "flanks": FLANKS, "real": real, "shuffled": shuf})
        r = "  ".join(f"{v:>4d}" for v in real)
        s = "  ".join(f"{v:>4d}" for v in shuf)
        print(f"{gene:7s} {gap:>5d} {len(pairs):>5d} | real         {r}")
        print(f"{'':7s} {'':5s} {'':5s} | shuffled     {s}")

    tot_r = [sum(r["real"][i] for r in rows) for i in range(len(FLANKS))]
    tot_s = [sum(r["shuffled"][i] for r in rows) for i in range(len(FLANKS))]
    n = sum(r["n_pairs"] for r in rows)
    print("-" * (34 + 6 * len(FLANKS)))
    print(f"{'TOTAL':7s} {'':5s} {n:>5d} | real         "
          + "  ".join(f"{v:>4d}" for v in tot_r))
    print(f"{'':7s} {'':5s} {'':5s} | shuffled     "
          + "  ".join(f"{v:>4d}" for v in tot_s))
    print("\nrecall %:")
    print("  flank      " + "  ".join(f"{f:>5d}" for f in FLANKS))
    print("  real       " + "  ".join(f"{100*v/n:>5.1f}" for v in tot_r))
    print("  shuffled   " + "  ".join(f"{100*v/n:>5.1f}" for v in tot_s))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"adapter": args.adapter, "flanks": FLANKS,
                   "total_pairs": n, "total_real": tot_r,
                   "total_shuffled": tot_s, "genes": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
