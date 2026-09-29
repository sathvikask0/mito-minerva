"""Derive true base pairs from evolution, then grade the model against them.

Everything so far scored the model against a *geometric assumption*: that the
acceptor stem is base i paired with base L-2-i, and the anticodon stem sits a
fixed offset from the anticodon. That is textbook tRNA architecture, but it is
an assumption, and both Minerva and ViennaRNA were being graded against it.

Comparative sequence analysis does not assume anything. If two positions
really pair, a mutation in one is compensated by a mutation in the other,
because an unpaired stem costs the organism. Those compensatory changes are
physical evidence of a base pair, and they are the standard by which RNA
secondary structures were established long before any predictor existed.

This aligns every animal mitochondrial tRNA to its human counterpart, measures
mutual information between every pair of columns, corrects it for background
(APC), and keeps pairs that both covary strongly and are compatible with
Watson-Crick pairing. That set is the reference the model is then scored on.

    python scripts/step7_covariation.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from step6_build_conservation import (  # noqa: E402
    REQUIRED_LINEAGE, HELD_OUT, BASES, amino_acid, build_aligner, human_trnas,
)

GAP = "-"
WC = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G"), ("G", "T"), ("T", "G")}


def collect_alignments(gz, human, by_aa, min_identity=0.55, limit=None):
    """One gapped string per species per gene, in human coordinates."""
    aligner = build_aligner()
    aln: dict[str, list[str]] = defaultdict(list)
    genomes = 0

    with gzip.open(gz, "rt") as fh:
        for record in SeqIO.parse(fh, "genbank"):
            if REQUIRED_LINEAGE not in record.annotations.get("taxonomy", []):
                continue
            if record.name.startswith(HELD_OUT) or record.id.startswith(HELD_OUT):
                continue
            genomes += 1
            if limit and genomes > limit:
                break
            if genomes % 2000 == 0:
                print(f"  {genomes:,} genomes", flush=True)

            for feat in record.features:
                if feat.type != "tRNA":
                    continue
                aa = amino_acid((feat.qualifiers.get("product") or [""])[0])
                if not aa or aa not in by_aa:
                    continue
                try:
                    seq = str(feat.extract(record.seq)).upper()
                except Exception:
                    continue
                if not 50 <= len(seq) <= 110 or set(seq) - set(BASES):
                    continue

                best_gene, best, best_score = None, None, float("-inf")
                for gene in by_aa[aa]:
                    a = aligner.align(human[gene]["seq"], seq)[0]
                    if a.score > best_score:
                        best_gene, best, best_score = gene, a, a.score

                ref = human[best_gene]["seq"]
                row = [GAP] * len(ref)
                matches = 0
                for (hs, he), (qs, _) in zip(*best.aligned):
                    for k in range(he - hs):
                        row[hs + k] = seq[qs + k]
                        matches += ref[hs + k] == seq[qs + k]
                if matches / len(ref) >= min_identity:
                    aln[best_gene].append("".join(row))
    return aln, genomes


def mutual_information(cols_i, cols_j):
    """MI in bits between two alignment columns, gaps excluded pairwise."""
    pairs = [(a, b) for a, b in zip(cols_i, cols_j) if a != GAP and b != GAP]
    n = len(pairs)
    if n < 200:
        return 0.0, 0
    joint = Counter(pairs)
    pi = Counter(a for a, _ in pairs)
    pj = Counter(b for _, b in pairs)
    mi = 0.0
    for (a, b), c in joint.items():
        pij = c / n
        mi += pij * math.log2(pij / ((pi[a] / n) * (pj[b] / n)))
    return mi, n


def wc_fraction(cols_i, cols_j):
    """Share of species whose two bases could actually pair."""
    pairs = [(a, b) for a, b in zip(cols_i, cols_j) if a != GAP and b != GAP]
    if not pairs:
        return 0.0
    return sum((a, b) in WC for a, b in pairs) / len(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default="data/raw/mitochondrion.1.genomic.gbff.gz")
    ap.add_argument("--out", default="data/processed/trna_covariation.json")
    ap.add_argument("--limit", type=int, help="stop after N genomes (for a quick test)")
    ap.add_argument("--min-species", type=int, default=200)
    ap.add_argument("--min-wc", type=float, default=0.9,
                    help="a real pair is Watson-Crick or GU in nearly every species")
    ap.add_argument("--min-sep", type=int, default=3, help="ignore near-diagonal pairs")
    args = ap.parse_args()

    human = human_trnas()
    by_aa = defaultdict(list)
    for name, info in human.items():
        aa = amino_acid(info["product"])
        if aa:
            by_aa[aa].append(name)

    print("aligning animal tRNAs to their human counterparts...")
    aln, genomes = collect_alignments(args.gz, human, by_aa, limit=args.limit)
    print(f"{genomes:,} genomes; alignments per gene: "
          f"{min(len(v) for v in aln.values()):,}-{max(len(v) for v in aln.values()):,}")

    out = {}
    for gene, rows in sorted(aln.items()):
        if len(rows) < args.min_species:
            continue
        L = len(human[gene]["seq"])
        cols = [[r[i] for r in rows] for i in range(L)]

        mi = np.zeros((L, L))
        wc = np.zeros((L, L))
        for i in range(L):
            for j in range(i + args.min_sep + 1, L):
                m, n = mutual_information(cols[i], cols[j])
                mi[i, j] = mi[j, i] = m
                if m > 0:
                    w = wc_fraction(cols[i], cols[j])
                    wc[i, j] = wc[j, i] = w

        # Average product correction: strips the background signal that comes
        # from a column simply being variable, leaving genuine covariation.
        mean_i = mi.mean(axis=1, keepdims=True)
        apc = (mean_i @ mean_i.T) / mi.mean() if mi.mean() > 0 else 0
        mic = mi - apc
        np.fill_diagonal(mic, 0)

        # A pair is called when it covaries in the top tier AND the two bases
        # can physically pair in nearly every species.
        cand = [(mic[i, j], i + 1, j + 1, wc[i, j])
                for i in range(L) for j in range(i + args.min_sep + 1, L)
                if wc[i, j] >= args.min_wc]
        cand.sort(reverse=True)

        # Greedy: each position pairs at most once, as in a real structure.
        used, pairs = set(), []
        for score, i, j, w in cand:
            if score <= 0 or i in used or j in used:
                continue
            used |= {i, j}
            pairs.append({"i": i, "j": j, "mi_apc": round(float(score), 4),
                          "wc_fraction": round(float(w), 3)})

        out[gene] = {
            "human_seq": human[gene]["seq"],
            "length": L,
            "n_species": len(rows),
            "pairs": pairs,
        }
        print(f"  {gene:6s} L={L:3d}  {len(rows):6,} species  "
              f"{len(pairs):2d} covarying pairs")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"genomes": genomes, "min_wc": args.min_wc,
                   "min_species": args.min_species, "genes": out}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
