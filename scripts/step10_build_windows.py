"""Extract every tRNA plus 200nt of its own genomic context, per species.

Restricted to species that also have a maximum lifespan in AnAge, since those
are the only ones the longevity analysis can use. Mitochondrial genomes are
circular, so flanks wrap around the origin rather than being truncated.

    python scripts/step10_build_windows.py --anage <dir>/anage_data.txt
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter

from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from step6_build_conservation import REQUIRED_LINEAGE, HELD_OUT, BASES  # noqa: E402

FLANK = 200


def load_anage(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ml = r["Maximum longevity (yrs)"].strip()
            if not ml:
                continue
            mass = r["Body mass (g)"].strip() or r["Adult weight (g)"].strip()
            out[f"{r['Genus'].strip()} {r['Species'].strip()}"] = {
                "lifespan_years": float(ml),
                "body_mass_g": float(mass) if mass else None,
                "class": r["Class"].strip(),
                "order": r["Order"].strip(),
                "family": r["Family"].strip(),
                "data_quality": r["Data quality"].strip(),
            }
    return out


def circular_slice(seq, start, end):
    """0-based [start, end) with wraparound, as a circular genome requires."""
    n = len(seq)
    if start < 0:
        return seq[start % n:] + seq[:end]
    if end > n:
        return seq[start:] + seq[:end - n]
    return seq[start:end]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default="data/raw/mitochondrion.1.genomic.gbff.gz")
    ap.add_argument("--anage", required=True)
    ap.add_argument("--out", default="data/processed/trna_windows.jsonl")
    ap.add_argument("--flank", type=int, default=FLANK)
    ap.add_argument("--min-trnas", type=int, default=15,
                    help="skip poorly annotated genomes")
    args = ap.parse_args()

    anage = load_anage(args.anage)
    print(f"AnAge: {len(anage):,} species with a lifespan")

    kept, seen, classes = 0, set(), Counter()
    n_trnas = []
    with gzip.open(args.gz, "rt") as fh, open(args.out, "w") as out:
        for record in SeqIO.parse(fh, "genbank"):
            if REQUIRED_LINEAGE not in record.annotations.get("taxonomy", []):
                continue
            if record.name.startswith(HELD_OUT) or record.id.startswith(HELD_OUT):
                continue
            org = record.annotations.get("organism", "").strip()
            meta = anage.get(org)
            if not meta or org in seen:
                continue

            try:
                genome = str(record.seq).upper()
            except Exception:
                # a few entries are CONTIG references carrying no sequence
                continue
            if set(genome) - set(BASES):
                # a handful of genomes carry ambiguity codes; flanks would be
                # unreadable to a model with no N token
                continue

            trnas = []
            for feat in record.features:
                if feat.type != "tRNA":
                    continue
                product = (feat.qualifiers.get("product") or [""])[0]
                try:
                    core = str(feat.extract(record.seq)).upper()
                except Exception:
                    continue
                if not 50 <= len(core) <= 110 or set(core) - set(BASES):
                    continue
                s = int(feat.location.start)
                e = int(feat.location.end)
                left = circular_slice(genome, s - args.flank, s)
                right = circular_slice(genome, e, e + args.flank)
                if feat.location.strand == -1:
                    comp = str.maketrans("ACGT", "TGCA")
                    left, right = (right.translate(comp)[::-1],
                                   left.translate(comp)[::-1])
                if len(left) != args.flank or len(right) != args.flank:
                    continue
                trnas.append({"product": product, "seq": core,
                              "left": left, "right": right})

            if len(trnas) < args.min_trnas:
                continue
            seen.add(org)
            n_trnas.append(len(trnas))
            classes[meta["class"]] += 1
            out.write(json.dumps({
                "accession": record.id, "organism": org,
                "lineage": record.annotations.get("taxonomy", []),
                "gc": round(100 * (genome.count("G") + genome.count("C")) / len(genome), 2),
                "bp": len(genome), **meta, "trnas": trnas,
            }) + "\n")
            kept += 1
            if kept % 250 == 0:
                print(f"  {kept:,} species written", flush=True)

    n_trnas.sort()
    print(f"\n{kept:,} species with lifespan and >= {args.min_trnas} tRNAs")
    print(f"tRNAs per species: median {n_trnas[len(n_trnas) // 2]}, "
          f"total {sum(n_trnas):,}")
    print("by class:", dict(classes.most_common(8)))
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
