"""Step 2a -- run Minerva over the whole mitochondrial coding window.

Takes about three minutes on the CPU.  Caches the base_pairing map so the
analysis in step2_analyse.py can be re-run without paying for inference again.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.mito import fetch_rcrs, tokenize_mito
from mitominerva.sanity import check_contact_map
from mitominerva.loading import load_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="cpu", help="cpu fits the full window; mps caps near 7168 tokens")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--adapter", help="directory holding a finetuned LoRA adapter")
    ap.add_argument("--heads", nargs="+", default=["base_pairing", "protein", "repeat"])
    ap.add_argument("--out", default="outputs/full_window")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m = tokenize_mito(fetch_rcrs(), drop_dloop=True)
    print(f"window rCRS {m.window[0]}..{m.window[1]}  {m.n_tokens:,} tokens")

    tok, model = load_model(args.model, args.device, args.dtype, args.adapter)
    ids = tok(m.token_string, return_tensors="pt")["input_ids"]
    assert ids.shape[1] == m.n_tokens

    t0 = time.time()
    with torch.inference_mode():
        out = model.predict_contacts(
            input_ids=ids.to(args.device), head_names=args.heads, return_dict=True
        )
    print(f"inference {time.time() - t0:.1f}s on {args.device}/{args.dtype}")

    # With return_dict=True the maps live under 'predictions', keyed by head name.
    preds = out["predictions"] if isinstance(out, dict) and "predictions" in out else out
    for head in args.heads:
        a = preds[head]
        if a.ndim == 3:
            a = a[0]
        a = a.float().cpu().numpy()
        if head == "base_pairing":
            check_contact_map(a, name=head)
        path = os.path.join(args.out, f"{head}.npy")
        np.save(path, a.astype(np.float16))
        print(f"  {head:13s} {a.shape} min={a.min():.4f} max={a.max():.4f} "
              f">0.5: {int((a > 0.5).sum()):,}  -> {path}")

    np.save(os.path.join(args.out, "token_ids.npy"), ids[0].numpy())
    with open(os.path.join(args.out, "token_string.txt"), "w") as fh:
        fh.write(m.token_string)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
