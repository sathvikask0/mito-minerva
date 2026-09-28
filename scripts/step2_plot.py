"""Step 2c -- draw the predicted base-pairing map for every mitochondrial tRNA.

A cloverleaf has a recognisable signature in a contact map: the acceptor stem
is an anti-diagonal in the top-right corner (first bases against last bases),
and the D-, anticodon- and T-arms are three short anti-diagonals stepping down
the main diagonal.  The expected acceptor and anticodon pairs are overlaid so a
hit or a miss is visible at a glance.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.cloverleaf import (
    MITO_ANTICODONS,
    acceptor_stem_pairs,
    anticodon_stem_pairs,
    find_anticodon,
    revcomp,
)
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="outputs/full_window")
    ap.add_argument("--mode", default="isolated", choices=["isolated", "genome"])
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"outputs/step2_trna_maps_{args.mode}.png"

    bp = np.load(os.path.join(args.maps, "base_pairing.npy")).astype(np.float32)
    fold = None
    if args.mode == "isolated":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from step2_benchmark import Folder
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.float16
        ).to(args.device).eval()
        fold = Folder(model, tok, args.device)
    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    trnas = sorted([f for f in m.other_features if f["type"] == "tRNA"],
                   key=lambda f: f["start"])

    fig, axes = plt.subplots(4, 6, figsize=(19, 13))
    for ax in axes.ravel():
        ax.axis("off")

    for ax, f in zip(axes.ravel(), trnas):
        seq = rcrs[f["start"] - 1:f["end"]]
        minus = f["strand"] == -1
        if fold is not None:
            seq = (revcomp(seq) if minus else seq).upper()
            sub = fold(seq)
        else:
            s0, s1 = m.token_span(f["start"], f["end"])
            sub = bp[s0:s1, s0:s1]
            if minus:
                sub, seq = sub[::-1, ::-1], revcomp(seq)
            seq = seq.upper()
        L = len(seq)

        ax.axis("on")
        ax.imshow(sub, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")

        acc = acceptor_stem_pairs(L)
        ax.scatter([j for _, j in acc], [i for i, _ in acc],
                   s=22, facecolors="none", edgecolors="#1f77b4", linewidths=1.1)
        ac_start = find_anticodon(seq, MITO_ANTICODONS[f["name"]])
        if ac_start is not None:
            anti = anticodon_stem_pairs(ac_start, L)
            ax.scatter([j for _, j in anti], [i for i, _ in anti],
                       s=22, facecolors="none", edgecolors="#2ca02c", linewidths=1.1)

        strand = "light" if minus else "heavy"
        ax.set_title(f"{f['name']}  {f['product'].replace('tRNA-', '')}  "
                     f"{L} nt  {strand}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    mode = ("each gene folded on its own"
            if args.mode == "isolated" else "sliced from the whole-genome window")
    fig.suptitle(
        f"Minerva base-pairing predictions for the 22 human mitochondrial tRNAs "
        f"({mode})\n"
        "blue = expected acceptor stem, green = expected anticodon stem, "
        "read in transcript orientation",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
