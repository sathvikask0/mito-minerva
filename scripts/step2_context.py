"""Step 2d -- does Minerva need the tRNA isolated, or does genomic context help?

The whole-genome run recovers only a few cloverleafs.  Three explanations are
worth separating: the model may need a short input, the acceptor stem may be
there but pairing to somewhere outside the gene, or the signal may not be
there at all.  This runs each tRNA at three context widths and reports where
the 5' end's strongest partner actually is.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.cloverleaf import (
    MITO_ANTICODONS, acceptor_stem_pairs, anticodon_stem_pairs,
    find_anticodon, revcomp, score_stem,
)
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--flanks", type=int, nargs="+", default=[0, 50, 200, 1000])
    args = ap.parse_args()

    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    trnas = sorted([f for f in m.other_features if f["type"] == "tRNA"],
                   key=lambda f: f["start"])

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype)
    ).to(args.device).eval()

    full = np.load("outputs/full_window/base_pairing.npy").astype(np.float32)

    print(f"{'tRNA':7s} {'str':4s}" + "".join(f"{('f=' + str(f)):>10s}" for f in args.flanks)
          + f"{'genome':>10s}   5' partner in the genome map")
    rows = []
    for f in trnas:
        seq = rcrs[f["start"] - 1:f["end"]]
        minus = f["strand"] == -1
        L = len(seq)
        scores = {}
        for flank in args.flanks:
            lo = max(1, f["start"] - flank)
            hi = min(len(rcrs), f["end"] + flank)
            sub_seq = rcrs[lo - 1:hi].lower()
            text = f"<+>{sub_seq}"
            ids = tok(text, return_tensors="pt")["input_ids"]
            with torch.inference_mode():
                out = model.predict_contacts(input_ids=ids.to(args.device),
                                             head_names=["base_pairing"])
            mat = (out if not isinstance(out, dict) else out["predictions"]["base_pairing"])
            mat = mat.float().cpu().numpy()
            if mat.ndim == 3:
                mat = mat[0]
            # Drop the strand-marker row/column, then crop to the gene.
            off = (f["start"] - lo) + 1
            g = mat[off:off + L, off:off + L]
            gseq = seq
            if minus:
                g, gseq = g[::-1, ::-1], revcomp(seq)
            scores[flank] = score_stem(g, acceptor_stem_pairs(L), name="acceptor")

        # Where does the genome-context map send the tRNA's first base?
        s0, s1 = m.token_span(f["start"], f["end"])
        row = full[s0 if not minus else s1 - 1]
        best = int(np.argmax(row))
        inside = s0 <= best < s1
        rel = (best - s0) if not minus else (s1 - 1 - best)

        line = f"{f['name']:7s} {'-' if minus else '+':4s}"
        line += "".join(f"{scores[fl].n_recovered:>4d}/7{'':4s}" for fl in args.flanks)
        gsub = full[s0:s1, s0:s1]
        if minus:
            gsub = gsub[::-1, ::-1]
        gen = score_stem(gsub, acceptor_stem_pairs(L), name="acceptor")
        line += f"{gen.n_recovered:>6d}/7    "
        line += (f"pos {rel} (in gene, expect {L - 2}), p={row[best]:.2f}"
                 if inside else f"OUTSIDE the gene, p={row[best]:.2f}")
        print(line)
        rows.append({"name": f["name"], "strand": "-" if minus else "+", "length": L,
                     "by_flank": {str(k): v.n_recovered for k, v in scores.items()},
                     "genome": gen.n_recovered,
                     "five_prime_partner_rel": rel if inside else None,
                     "five_prime_partner_p": float(row[best])})

    print()
    for fl in args.flanks:
        v = np.array([r["by_flank"][str(fl)] for r in rows])
        print(f"flank {fl:>4d} nt : {v.sum():>3d}/154 bp ({v.sum()/154:.0%})  "
              f">=5/7 in {(v >= 5).sum()}/22")
    v = np.array([r["genome"] for r in rows])
    print(f"whole genome : {v.sum():>3d}/154 bp ({v.sum()/154:.0%})  >=5/7 in {(v >= 5).sum()}/22")

    with open("outputs/step2_context.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
