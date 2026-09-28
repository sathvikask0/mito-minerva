"""How long a Minerva input can this machine actually handle?

Runs predict_contacts at increasing sequence lengths and reports wall time and
peak memory, so we know where the M3 Pro gives out and Modal has to take over.
"""

import argparse
import gc
import os
import resource
import sys
import time

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--heads", nargs="+", default=["base_pairing"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype
    ).to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}  {n_params/1e6:.0f}M params  {args.dtype}  device={args.device}")
    print(f"load {time.time() - t0:.1f}s   heads={args.heads}\n")

    text = open("data/processed/mito_coding_window.txt").read()
    ids_all = tok(text, return_tensors="pt")["input_ids"][0]

    print(f"{'tokens':>8} {'seconds':>9} {'peak RSS':>10}  result")
    for n in args.lengths:
        if n > len(ids_all):
            print(f"{n:>8}  (longer than the {len(ids_all)}-token input, skipped)")
            continue
        ids = ids_all[:n].unsqueeze(0)
        out = None
        gc.collect()
        t0 = time.time()
        try:
            with torch.inference_mode():
                out = model.predict_contacts(input_ids=ids, head_names=args.heads)
            if args.device == "mps":
                torch.mps.synchronize()
            m = out[args.heads[0]] if isinstance(out, dict) else out
            desc = f"{tuple(m.shape)} min={m.min():.3f} max={m.max():.3f}"
        except Exception as exc:  # noqa: BLE001 - we want the reason, not a traceback
            desc = f"FAILED: {type(exc).__name__}: {str(exc)[:80]}"
        print(f"{n:>8} {time.time() - t0:>9.1f} {peak_rss_gb():>9.1f}G  {desc}")
        del out
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
