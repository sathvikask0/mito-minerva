"""Step 3b -- recompute the likelihood score with genomic context.

Step 2 found that flanking sequence destroys the *folding* prediction, so step 3
folded every tRNA in isolation.  But masked-token prediction is the task the
model was actually trained on, and it was trained on whole genomes, so the
likelihood score may want the context that the folding score does not.  This
recomputes log P(base) at every tRNA position with real flanking sequence.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.cloverleaf import revcomp
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito
from mitominerva.loading import load_model

BASES = "ACGT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--adapter", help="directory holding a finetuned LoRA adapter")
    ap.add_argument("--flank", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="outputs/step3b_context_llr.json")
    args = ap.parse_args()

    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    trnas = sorted([f for f in m.other_features if f["type"] == "tRNA"],
                   key=lambda f: f["start"])

    tok, model = load_model(args.model, args.device, args.dtype, args.adapter)
    vocab = tok.get_vocab()
    mask_id = vocab["<mask>"]
    base_ids = [vocab[b.lower()] for b in BASES]

    records = []
    t0 = time.time()
    for f in trnas:
        lo = max(1, f["start"] - args.flank)
        hi = min(len(rcrs), f["end"] + args.flank)
        window = rcrs[lo - 1:hi].lower()
        ids = tok(f"<+>{window}", return_tensors="pt")["input_ids"][0]
        minus = f["strand"] == -1
        L = f["end"] - f["start"] + 1

        # Token index of rCRS position p is (p - lo) + 1, the +1 for the marker.
        positions = [(p, (p - lo) + 1) for p in range(f["start"], f["end"] + 1)]
        logps = {}
        for k in range(0, len(positions), args.batch):
            chunk = positions[k:k + args.batch]
            batch = ids.repeat(len(chunk), 1)
            for r, (_, ti) in enumerate(chunk):
                batch[r, ti] = mask_id
            with torch.inference_mode():
                out = model(input_ids=batch.to(args.device))
            lg = out.logits.float()
            for r, (p, ti) in enumerate(chunk):
                lp = torch.log_softmax(lg[r, ti], dim=-1)
                logps[p] = [lp[b].item() for b in base_ids]

        for p in range(f["start"], f["end"] + 1):
            heavy = rcrs[p - 1].upper()
            if heavy not in BASES:
                continue
            wt_lp = logps[p][BASES.index(heavy)]
            for alt in BASES:
                if alt == heavy:
                    continue
                # Reported in transcript alleles, to match step 3.
                t_ref, t_alt = (revcomp(heavy), revcomp(alt)) if minus else (heavy, alt)
                records.append({
                    "trna": f["name"], "rcrs_pos": p,
                    "wt": t_ref, "mut": t_alt,
                    "llr_ctx": round(logps[p][BASES.index(alt)] - wt_lp, 4),
                })
        print(f"{f['name']:7s} window {hi - lo + 1:5d} nt, {L:3d} positions  "
              f"{time.time() - t0:6.1f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(records, fh)
    print(f"\n{len(records)} entries in {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
