"""Step 5a -- build a training corpus of animal mitochondrial genomes.

Minerva has never seen mitochondrial DNA.  RefSeq publishes every complete
mitochondrial genome it holds in one GenBank file, so the corpus is that file
filtered to Metazoa and converted to the same mixed-token format as the human
input, under the vertebrate/invertebrate mitochondrial genetic code.

Human (NC_012920) is held out, so nothing we evaluate on is trained on.
"""

import argparse
import gzip
import json
import os
import sys
import urllib.request

from Bio import SeqIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minerva.data import extract_and_tokenize_gb
from mitominerva.mito import RCRS_ACCESSION

URL = ("https://ftp.ncbi.nlm.nih.gov/refseq/release/mitochondrion/"
       "mitochondrion.1.genomic.gbff.gz")

#: Plant and fungal mitochondrial genomes are an order of magnitude larger and
#: use different genetic codes, so the corpus is animals only.
REQUIRED_LINEAGE = "Metazoa"
#: 2 = vertebrate mito, 5 = invertebrate mito, 9/13/14/21 = other animal codes.
ANIMAL_TABLES = {2, 5, 9, 13, 14, 21, 24, 33}
MAX_LEN = 30_000  # a normal animal mitochondrial genome is 15-20 kb


def download(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, path)
    print(f"{path}  {os.path.getsize(path) / 1e6:.0f} MB")
    return path


def translation_table(record):
    for feat in record.features:
        if feat.type == "CDS" and "transl_table" in feat.qualifiers:
            return int(feat.qualifiers["transl_table"][0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default="data/raw/mitochondrion.1.genomic.gbff.gz")
    ap.add_argument("--out", default="data/processed/metazoa_mito.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="stop after N genomes (0 = all)")
    args = ap.parse_args()

    download(args.gz)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    kept = skipped = held_out = 0
    total_tokens = 0
    tmp = args.out + ".one.gb"
    with gzip.open(args.gz, "rt") as fh, open(args.out, "w") as out:
        for record in SeqIO.parse(fh, "genbank"):
            lineage = record.annotations.get("taxonomy", [])
            if REQUIRED_LINEAGE not in lineage:
                skipped += 1
                continue
            if len(record.seq) > MAX_LEN or len(record.seq) < 5_000:
                skipped += 1
                continue
            if record.id.split(".")[0] == RCRS_ACCESSION.split(".")[0]:
                held_out += 1
                continue
            table = translation_table(record) or 2
            if table not in ANIMAL_TABLES:
                skipped += 1
                continue

            SeqIO.write(record, tmp, "genbank")
            try:
                recs = extract_and_tokenize_gb(
                    tmp, use_existing_translations=True, overlap_mode="expand",
                    translation_table=table,
                )
            except Exception:
                skipped += 1
                continue
            seq = recs[0]["sequence"]
            if len(seq) < 3_000:
                skipped += 1
                continue
            out.write(json.dumps({
                "accession": record.id,
                "organism": record.annotations.get("organism", ""),
                "lineage": lineage[:8],
                "table": table,
                "bp": len(record.seq),
                "text": seq,
            }) + "\n")
            kept += 1
            total_tokens += len(seq)
            if kept % 500 == 0:
                print(f"  {kept:6d} genomes, {total_tokens/1e6:.1f}M characters")
            if args.limit and kept >= args.limit:
                break
    if os.path.exists(tmp):
        os.remove(tmp)

    print(f"\nkept {kept:,} animal mitochondrial genomes "
          f"({total_tokens/1e6:.1f}M characters, ~{total_tokens/1e6:.1f}M tokens)")
    print(f"skipped {skipped:,}; held out human rCRS ({held_out})")
    print(f"wrote {args.out}  {os.path.getsize(args.out)/1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
