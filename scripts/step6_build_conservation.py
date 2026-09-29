"""Per-position conservation of the 22 human mitochondrial tRNAs across animals.

Step 5 finetuned Minerva on 15,589 animal mitochondrial genomes and its
mutation scores improved sharply. The obvious objection is that the model has
simply learned which bases are conserved across those animals -- which is what
existing clinical predictors already use. This builds the conservation signal
explicitly, from the same corpus, so the two can be compared head to head.

For every animal genome the annotated tRNAs are extracted in transcript
orientation, matched to their human counterpart by amino acid (and by
alignment for the two Leu and two Ser genes), aligned, and tallied into a
per-human-position allele count.

    python scripts/step6_build_conservation.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

from Bio import SeqIO
from Bio.Align import PairwiseAligner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.cloverleaf import MITO_ANTICODONS
from mitominerva.mito import fetch_rcrs, load_record

REQUIRED_LINEAGE = "Metazoa"
HELD_OUT = "NC_012920"
BASES = "ACGT"


def human_trnas():
    """The 22 human tRNAs as transcripts, keyed by gene name."""
    path = fetch_rcrs()
    record = load_record(path)
    out = {}
    for feat in record.features:
        if feat.type != "tRNA":
            continue
        name = (feat.qualifiers.get("gene") or feat.qualifiers.get("product"))[0]
        name = name.replace("MT-", "").upper()
        seq = str(feat.extract(record.seq)).upper()
        product = feat.qualifiers.get("product", [name])[0]
        out[name] = {"seq": seq, "product": product}
    return out


def amino_acid(product: str) -> str | None:
    """'tRNA-Leu (CUN)' -> 'LEU'. Returns None for anything unparseable."""
    p = product.strip().lower()
    if not p.startswith("trna-"):
        return None
    aa = p[5:].split("(")[0].split("-")[0].strip()
    return aa.upper()[:3] or None


def build_aligner() -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score = 2
    a.mismatch_score = -1
    a.open_gap_score = -5
    a.extend_gap_score = -1
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default="data/raw/mitochondrion.1.genomic.gbff.gz")
    ap.add_argument("--out", default="data/processed/trna_conservation.json")
    ap.add_argument("--min-identity", type=float, default=0.55,
                    help="reject an alignment below this, so a mis-annotated "
                         "or wildly divergent tRNA cannot pollute the counts")
    args = ap.parse_args()

    human = human_trnas()
    print(f"{len(human)} human tRNAs")

    # amino acid -> candidate human gene names (Leu and Ser have two each)
    by_aa: dict[str, list[str]] = defaultdict(list)
    for name, info in human.items():
        aa = amino_acid(info["product"])
        if aa:
            by_aa[aa].append(name)
    dupes = {k: v for k, v in by_aa.items() if len(v) > 1}
    print(f"amino acids with two genes: {dupes}")

    aligner = build_aligner()
    # counts[gene][human_position_1based][allele] = n
    counts: dict[str, dict[int, Counter]] = {
        g: defaultdict(Counter) for g in human
    }
    n_species = Counter()
    genomes = kept = 0

    with gzip.open(args.gz, "rt") as fh:
        for record in SeqIO.parse(fh, "genbank"):
            lineage = record.annotations.get("taxonomy", [])
            if REQUIRED_LINEAGE not in lineage:
                continue
            if record.name.startswith(HELD_OUT) or record.id.startswith(HELD_OUT):
                continue
            genomes += 1
            if genomes % 1000 == 0:
                print(f"  {genomes:,} genomes, {kept:,} tRNAs tallied", flush=True)

            for feat in record.features:
                if feat.type != "tRNA":
                    continue
                product = (feat.qualifiers.get("product") or [""])[0]
                aa = amino_acid(product)
                if not aa or aa not in by_aa:
                    continue
                try:
                    seq = str(feat.extract(record.seq)).upper()
                except Exception:
                    continue
                if not 50 <= len(seq) <= 110 or set(seq) - set(BASES):
                    continue

                # Pick the better-matching human gene when the amino acid has
                # two (Leu, Ser); otherwise there is only one candidate.
                best_gene, best_aln, best_score = None, None, float("-inf")
                for gene in by_aa[aa]:
                    aln = aligner.align(human[gene]["seq"], seq)[0]
                    if aln.score > best_score:
                        best_gene, best_aln, best_score = gene, aln, aln.score

                ref = human[best_gene]["seq"]
                matches = sum(
                    1
                    for (hs, he), (qs, qe) in zip(*best_aln.aligned)
                    for k in range(he - hs)
                    if ref[hs + k] == seq[qs + k]
                )
                if matches / len(ref) < args.min_identity:
                    continue

                for (hs, he), (qs, qe) in zip(*best_aln.aligned):
                    for k in range(he - hs):
                        counts[best_gene][hs + k + 1][seq[qs + k]] += 1
                kept += 1
                n_species[best_gene] += 1

    out = {
        "source": os.path.basename(args.gz),
        "held_out": HELD_OUT,
        "genomes": genomes,
        "tRNAs_tallied": kept,
        "min_identity": args.min_identity,
        "genes": {
            gene: {
                "human_seq": human[gene]["seq"],
                "n_species": n_species[gene],
                "positions": {
                    str(pos): dict(alleles)
                    for pos, alleles in sorted(counts[gene].items())
                },
            }
            for gene in human
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)

    print(f"\n{genomes:,} animal genomes, {kept:,} tRNAs aligned")
    thin = [g for g in human if n_species[g] < 500]
    print(f"species per gene: min {min(n_species.values()):,} "
          f"max {max(n_species.values()):,}")
    if thin:
        print(f"thin coverage (<500 species): {thin}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
