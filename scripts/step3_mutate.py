"""Step 3 -- change every base of every mitochondrial tRNA, one at a time.

Each point mutation gets two independent scores:

* **structure disruption** -- refold the mutant and measure how much of the
  wild-type base-pairing survives.  This is what ViennaRNA can also do.
* **masked log-likelihood ratio** -- mask the position, ask the model what
  belongs there, and take ``log P(mutant) - log P(wild type)``.  A strongly
  negative value means the model is confident the wild-type base belongs.
  ViennaRNA has no equivalent; this is the reason to use a language model.

Step 2 showed the model only folds these tRNAs when the gene is given on its
own, so every sequence here is folded in isolation.
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

from mitominerva.cloverleaf import (
    MITO_ANTICODONS, acceptor_stem_pairs, anticodon_stem_pairs,
    find_anticodon, revcomp, score_stem,
)
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito

BASES = "ACGT"
CONFIDENT = 0.5


def to_rcrs(name_start, name_end, minus, i):
    """Transcript index -> 1-based rCRS position."""
    return (name_end - i) if minus else (name_start + i)


def fold_batch(model, tok, seqs, device, batch_size=128):
    """base_pairing maps for a list of equal-length sequences."""
    maps = []
    for k in range(0, len(seqs), batch_size):
        chunk = seqs[k:k + batch_size]
        ids = tok([f"<+>{s.lower()}" for s in chunk],
                  return_tensors="pt", padding=True)["input_ids"].to(device)
        with torch.inference_mode():
            out = model.predict_contacts(input_ids=ids, head_names=["base_pairing"])
        mat = out if not isinstance(out, dict) else out["predictions"]["base_pairing"]
        mat = mat.float().cpu().numpy()
        if mat.ndim == 2:
            mat = mat[None]
        maps.append(mat[:, 1:, 1:])  # drop the strand-marker row/column
    return np.concatenate(maps, axis=0)


def masked_llr(model, tok, seq, device, batch_size=64):
    """log P(base) at every position, from a separate masked pass per position."""
    vocab = tok.get_vocab()
    mask_id = vocab["<mask>"]
    L = len(seq)
    base_ids = {b: vocab[b.lower()] for b in BASES}
    ids = tok(f"<+>{seq.lower()}", return_tensors="pt")["input_ids"][0]
    logps = np.zeros((L, len(BASES)), dtype=np.float32)
    for k in range(0, L, batch_size):
        idx = list(range(k, min(k + batch_size, L)))
        batch = ids.repeat(len(idx), 1)
        for r, i in enumerate(idx):
            batch[r, i + 1] = mask_id  # +1 for the <+> marker
        with torch.inference_mode():
            out = model(input_ids=batch.to(device))
        lg = out.logits.float()
        for r, i in enumerate(idx):
            lp = torch.log_softmax(lg[r, i + 1], dim=-1)
            for c, b in enumerate(BASES):
                logps[i, c] = lp[base_ids[b]].item()
    return logps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--only", nargs="*", help="restrict to these tRNA names")
    ap.add_argument("--out", default="outputs/step3_mutations.json")
    args = ap.parse_args()

    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    trnas = sorted([f for f in m.other_features if f["type"] == "tRNA"],
                   key=lambda f: f["start"])
    if args.only:
        trnas = [f for f in trnas if f["name"] in args.only]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype)
    ).to(args.device).eval()

    records = []
    t_all = time.time()
    for f in trnas:
        t0 = time.time()
        raw = rcrs[f["start"] - 1:f["end"]]
        minus = f["strand"] == -1
        seq = (revcomp(raw) if minus else raw).upper()
        L = len(seq)

        wt = fold_batch(model, tok, [seq], args.device)[0]
        wt_pairs = [(i, j) for i in range(L) for j in range(i + 1, L)
                    if wt[i, j] >= CONFIDENT]
        ac_start = find_anticodon(seq, MITO_ANTICODONS[f["name"]])
        stems = acceptor_stem_pairs(L) + (
            anticodon_stem_pairs(ac_start, L) if ac_start is not None else [])
        wt_stem = score_stem(wt, stems, name="stems").n_recovered

        logps = masked_llr(model, tok, seq, args.device)

        muts, mut_seqs = [], []
        for i in range(L):
            for b in BASES:
                if b == seq[i]:
                    continue
                muts.append((i, b))
                mut_seqs.append(seq[:i] + b + seq[i + 1:])
        maps = fold_batch(model, tok, mut_seqs, args.device)

        for (i, b), mm in zip(muts, maps):
            lost = (sum(1 for p, q in wt_pairs if mm[p, q] < CONFIDENT) / len(wt_pairs)
                    if wt_pairs else 0.0)
            stem_after = score_stem(mm, stems, name="stems").n_recovered
            records.append({
                "trna": f["name"], "product": f["product"],
                "strand": "-" if minus else "+",
                "rcrs_pos": to_rcrs(f["start"], f["end"], minus, i),
                "transcript_pos": i + 1,
                "wt": seq[i], "mut": b,
                "pairs_lost": round(float(lost), 4),
                "delta_l1": round(float(np.abs(mm - wt).mean()), 6),
                "stem_wt": wt_stem, "stem_mut": int(stem_after),
                "stem_lost": int(wt_stem - stem_after),
                "llr": round(float(logps[i, BASES.index(b)] - logps[i, BASES.index(seq[i])]), 4),
            })
        print(f"{f['name']:7s} L={L:3d} {len(muts):4d} mutants  "
              f"wt_pairs={len(wt_pairs):3d} wt_stems={wt_stem:2d}/{len(stems)}  "
              f"{time.time() - t0:5.1f}s")

    print(f"\n{len(records)} mutations in {time.time() - t_all:.0f}s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(records, fh)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
