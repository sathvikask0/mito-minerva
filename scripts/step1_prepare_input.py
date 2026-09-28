"""Step 1 -- build and sanity-check the Minerva input for the human mtDNA.

Fetches the rCRS (NC_012920.1), converts it to Minerva's mixed-token format with
the vertebrate mitochondrial genetic code, verifies the reconstruction against
the real tokenizer, and reports where each tRNA lands in token space.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer

from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito

MODEL_CONTEXT = {"gbrixi/minerva-mlm": 4096, "gbrixi/minerva-mlm-8k": 8192}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", default="data/raw/NC_012920.1.gb")
    ap.add_argument("--tokenizer", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    gb = fetch_rcrs(args.gb)
    record = load_record(gb)
    print(f"{record.id}  {len(record.seq):,} bp  ({record.description})")
    n_ambiguous = sum(1 for b in str(record.seq).upper() if b not in "ACGT")
    if n_ambiguous:
        pos = [i + 1 for i, b in enumerate(str(record.seq).upper()) if b not in "ACGT"]
        print(f"  ambiguous bases: {n_ambiguous} at {pos} -> tokenized as <unk>")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    results = {}
    for label, drop in (("full circle", False), ("D-loop excised", True)):
        m = tokenize_mito(gb, drop_dloop=drop)

        # The tokenizer is character-level over {aa, acgt} plus the two strand
        # markers, so our reconstructed length must match it exactly.
        ids = tok(m.token_string, return_tensors="pt")["input_ids"][0]
        assert len(ids) == m.n_tokens, f"{len(ids)} != {m.n_tokens}"
        n_unk = int((ids == tok.get_vocab()["<unk>"]).sum())

        print(f"\n--- {label}: rCRS {m.window[0]}..{m.window[1]} "
              f"({m.window[1] - m.window[0] + 1:,} bp)")
        print(f"    tokens {m.n_tokens:,}   <unk> {n_unk}   "
              f"chars {len(m.token_string):,}")
        for model, ctx in MODEL_CONTEXT.items():
            verdict = "fits" if m.n_tokens <= ctx else f"OVER by {m.n_tokens - ctx:,}"
            print(f"      {model:26s} ctx {ctx:>5,}  {verdict}")

        trnas = [f for f in m.other_features if f["type"] == "tRNA"]
        rrnas = [f for f in m.other_features if f["type"] == "rRNA"]
        print(f"    features in window: {len(m.cds)} CDS, {len(trnas)} tRNA, {len(rrnas)} rRNA")

        if drop:
            results = summarise(m, trnas, rrnas)

    os.makedirs(args.out, exist_ok=True)
    m = tokenize_mito(gb, drop_dloop=True)
    with open(os.path.join(args.out, "mito_coding_window.txt"), "w") as fh:
        fh.write(m.token_string)
    with open(os.path.join(args.out, "mito_coding_window.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}/mito_coding_window.txt and .json")
    return 0


def summarise(m, trnas, rrnas) -> dict:
    """Print and return the token span of every tRNA and rRNA."""
    print("\n    tRNA / rRNA token spans (all must be lower-case DNA):")
    out = {"window": list(m.window), "n_tokens": m.n_tokens, "features": []}
    for f in sorted(trnas + rrnas, key=lambda x: x["start"]):
        span = m.token_span(f["start"], f["end"])
        chars = m.token_string[span[0]:span[1]] if span else ""
        # Token index == character index only before any strand marker, so read
        # the actual characters through the char map instead.
        idx = [i for i, t in enumerate(m.char_to_token) if span[0] <= t < span[1]]
        chars = "".join(m.token_string[i] for i in idx)
        clean = chars.islower() and set(chars) <= set("acgtn")
        strand = "+" if f["strand"] == 1 else "-"
        print(f"      {f['type']:5s} {f['name']:12s} {f['start']:>6}-{f['end']:<6} {strand} "
              f"tokens {span[0]:>5}-{span[1]:<5} len {f['end'] - f['start'] + 1:>5} "
              f"{'ok' if clean else 'CONTAMINATED: ' + chars[:40]}")
        out["features"].append({
            "type": f["type"], "name": f["name"], "product": f["product"],
            "start": f["start"], "end": f["end"], "strand": f["strand"],
            "token_start": span[0], "token_end": span[1], "seq": chars,
        })
    out["cds"] = [
        {"name": c["gene_name"], "start": c["start"], "end": c["end"],
         "strand": "+" if c["orientation"] else "-", "n_aa": len(c["seq"])}
        for c in m.cds
    ]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
