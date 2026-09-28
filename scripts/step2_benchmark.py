"""Step 2e -- the definitive go / no-go.

Evaluates three predictors on the 22 mitochondrial tRNAs and on length-matched
decoys drawn from the same genome:

* Minerva on the isolated gene,
* Minerva on the whole-genome window (the cached map),
* ViennaRNA's minimum-free-energy fold.

A predictor is only useful here if it separates tRNAs from decoys, so every
number is reported for both.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import RNA
from minerva.rna_structure import call_structure
from mitominerva.cloverleaf import (
    MITO_ANTICODONS, acceptor_stem_pairs, anticodon_stem_pairs,
    find_anticodon, revcomp, score_stem,
)
from mitominerva.mito import fetch_rcrs, load_record, tokenize_mito


def vienna_partners(seq):
    structure, _ = RNA.fold(seq.upper().replace("T", "U"))
    partners = np.full(len(seq), -1, dtype=int)
    stack = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            j = stack.pop()
            partners[i], partners[j] = j, i
    return structure, partners


def vienna_matrix(seq):
    """A 0/1 'contact map' from the MFE fold, so it scores through score_stem."""
    _, partners = vienna_partners(seq)
    mat = np.zeros((len(seq), len(seq)), dtype=np.float32)
    for i, j in enumerate(partners):
        if j >= 0:
            mat[i, j] = 1.0
    return mat


class Folder:
    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device

    def __call__(self, seq):
        ids = self.tok(f"<+>{seq.lower()}", return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            out = self.model.predict_contacts(
                input_ids=ids.to(self.device), head_names=["base_pairing"]
            )
        mat = out if not isinstance(out, dict) else out["predictions"]["base_pairing"]
        mat = mat.float().cpu().numpy()
        if mat.ndim == 3:
            mat = mat[0]
        return mat[1:, 1:]  # drop the strand-marker row/column


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gbrixi/minerva-mlm-8k")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n-decoys", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gb = fetch_rcrs()
    rcrs = str(load_record(gb).seq)
    m = tokenize_mito(gb, drop_dloop=True)
    trnas = sorted([f for f in m.other_features if f["type"] == "tRNA"],
                   key=lambda f: f["start"])
    genome_map = np.load("outputs/full_window/base_pairing.npy").astype(np.float32)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=getattr(torch, args.dtype)
    ).to(args.device).eval()
    fold = Folder(model, tok, args.device)

    header = (f"{'tRNA':7s} {'s':2s} {'len':>4s} | {'isolated':>18s} | "
              f"{'genome':>8s} | {'Vienna':>18s} | dot-bracket (isolated)")
    print(header)
    print("-" * 118)

    rows = []
    for f in trnas:
        seq = rcrs[f["start"] - 1:f["end"]]
        minus = f["strand"] == -1
        tseq = (revcomp(seq) if minus else seq).upper()
        L = len(tseq)
        acc_pairs = acceptor_stem_pairs(L)
        ac_start = find_anticodon(tseq, MITO_ANTICODONS[f["name"]])
        ant_pairs = anticodon_stem_pairs(ac_start, L) if ac_start is not None else []

        iso = fold(tseq)
        s0, s1 = m.token_span(f["start"], f["end"])
        gen = genome_map[s0:s1, s0:s1]
        if minus:
            gen = gen[::-1, ::-1]
        vie = vienna_matrix(tseq)
        vstruct, _ = vienna_partners(tseq)

        a_iso = score_stem(iso, acc_pairs, name="acc")
        a_gen = score_stem(gen, acc_pairs, name="acc")
        a_vie = score_stem(vie, acc_pairs, name="acc")
        n_iso = score_stem(iso, ant_pairs, name="ant") if ant_pairs else None
        n_vie = score_stem(vie, ant_pairs, name="ant") if ant_pairs else None

        st = call_structure(iso, tseq, name=f["name"])
        rows.append({
            "name": f["name"], "strand": "-" if minus else "+", "length": L,
            "iso_acc": a_iso.n_recovered, "gen_acc": a_gen.n_recovered,
            "vie_acc": a_vie.n_recovered,
            "iso_ant": n_iso.n_recovered if n_iso else None,
            "vie_ant": n_vie.n_recovered if n_vie else None,
            "n_ant": len(ant_pairs),
            "iso_dot": st.dot_bracket, "vienna_dot": vstruct,
            "canonical_fraction": st.canonical_fraction(),
        })
        print(f"{f['name']:7s} {'-' if minus else '+':2s} {L:>4d} | "
              f"acc {a_iso.n_recovered}/7  ant {n_iso.n_recovered if n_iso else 0}/{len(ant_pairs)} | "
              f"  acc {a_gen.n_recovered}/7 | "
              f"acc {a_vie.n_recovered}/7  ant {n_vie.n_recovered if n_vie else 0}/{len(ant_pairs)} | "
              f"{st.dot_bracket}")

    decoys = run_decoys(args, rcrs, m, trnas, fold, genome_map)
    summarise(rows, decoys)
    with open("outputs/step2_benchmark.json", "w") as fh:
        json.dump({"trnas": rows, "decoys": decoys}, fh, indent=2)
    print("\nwrote outputs/step2_benchmark.json")
    return 0


def run_decoys(args, rcrs, m, trnas, fold, genome_map):
    """Length-matched windows from the same genome, outside every tRNA."""
    rng = np.random.default_rng(args.seed)
    blocked = np.zeros(len(rcrs) + 2, bool)
    for f in m.other_features:
        if f["type"] == "tRNA":
            blocked[f["start"] - 10:f["end"] + 10] = True
    # Only lower-case (non-protein) territory, so decoys are comparable input.
    lower_pos = {g for (g, _), ch in zip(m.char_to_genome, m.token_string)
                 if g != -1 and ch.islower()}
    lengths = [f["end"] - f["start"] + 1 for f in trnas]

    out = []
    tries = 0
    while len(out) < args.n_decoys and tries < args.n_decoys * 300:
        tries += 1
        L = int(rng.choice(lengths))
        start = int(rng.integers(m.window[0], m.window[1] - L))
        rng_pos = range(start, start + L)
        if blocked[start:start + L].any() or not all(p in lower_pos for p in rng_pos):
            continue
        seq = rcrs[start - 1:start + L - 1].upper()
        if "N" in seq:
            continue
        iso = fold(seq)
        s0, s1 = m.token_span(start, start + L - 1)
        gen = genome_map[s0:s1, s0:s1]
        if gen.shape[0] != L:
            continue
        acc = acceptor_stem_pairs(L)
        out.append({
            "start": start, "length": L,
            "iso_acc": score_stem(iso, acc, name="acc").n_recovered,
            "gen_acc": score_stem(gen, acc, name="acc").n_recovered,
            "vie_acc": score_stem(vienna_matrix(seq), acc, name="acc").n_recovered,
        })
    return out


def summarise(rows, decoys):
    n = len(rows)
    tot = 7 * n
    print("\n" + "=" * 78)
    print(f"{'predictor':22s} {'tRNA acceptor':>16s} {'>=5/7':>8s} "
          f"{'decoy >=5/7':>13s} {'decoy mean':>12s}")
    for key, label in (("iso_acc", "Minerva, isolated"),
                       ("gen_acc", "Minerva, whole genome"),
                       ("vie_acc", "ViennaRNA MFE")):
        v = np.array([r[key] for r in rows])
        d = np.array([x[key] for x in decoys]) if decoys else np.array([0])
        print(f"{label:22s} {v.sum():>6d}/{tot} ({v.sum()/tot:>3.0%}) "
              f"{(v >= 5).sum():>4d}/{n} {(d >= 5).mean():>12.1%} {d.mean():>11.2f}/7")
    ia = np.array([r["iso_ant"] for r in rows if r["iso_ant"] is not None])
    va = np.array([r["vie_ant"] for r in rows if r["vie_ant"] is not None])
    na = np.array([r["n_ant"] for r in rows if r["iso_ant"] is not None])
    print(f"\nanticodon stem       Minerva isolated {ia.sum()}/{na.sum()} ({ia.sum()/na.sum():.0%})"
          f"   ViennaRNA {va.sum()}/{na.sum()} ({va.sum()/na.sum():.0%})")
    cf = np.array([r["canonical_fraction"] for r in rows])
    print(f"canonical pairs in Minerva's isolated calls: {cf.mean():.0%} mean")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
