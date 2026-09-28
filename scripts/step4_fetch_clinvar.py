"""Step 4a -- pull clinically labelled mitochondrial tRNA variants from ClinVar.

MITOMAP sits behind a bot wall, so the labels come from ClinVar instead, which
NCBI serves through the same E-utilities API used to fetch the reference.  Only
single-base substitutions inside a tRNA gene are kept, because those are the
only ones step 3 scored.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.mito import fetch_rcrs, tokenize_mito

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

#: NC_012920 gene names -> the HGNC symbols ClinVar indexes them under.
GENE_SYMBOL = {
    "TRNF": "MT-TF", "TRNV": "MT-TV", "TRNL1": "MT-TL1", "TRNI": "MT-TI",
    "TRNQ": "MT-TQ", "TRNM": "MT-TM", "TRNW": "MT-TW", "TRNA": "MT-TA",
    "TRNN": "MT-TN", "TRNC": "MT-TC", "TRNY": "MT-TY", "TRNS1": "MT-TS1",
    "TRND": "MT-TD", "TRNK": "MT-TK", "TRNG": "MT-TG", "TRNR": "MT-TR",
    "TRNH": "MT-TH", "TRNS2": "MT-TS2", "TRNL2": "MT-TL2", "TRNE": "MT-TE",
    "TRNT": "MT-TT", "TRNP": "MT-TP",
}

TITLE_RE = re.compile(r"m\.(\d+)([ACGT])>([ACGT])\s*$")


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as fh:
                return json.load(fh)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def search(symbol):
    q = urllib.parse.urlencode({"db": "clinvar", "term": f"{symbol}[gene]",
                                "retmax": 1000, "retmode": "json"})
    return get(f"{EUTILS}/esearch.fcgi?{q}")["esearchresult"].get("idlist", [])


def summarise(ids):
    out = []
    for k in range(0, len(ids), 200):
        q = urllib.parse.urlencode({"db": "clinvar", "id": ",".join(ids[k:k + 200]),
                                    "retmode": "json"})
        res = get(f"{EUTILS}/esummary.fcgi?{q}").get("result", {})
        for uid in res.get("uids", []):
            out.append(res[uid])
        time.sleep(0.4)
    return out


def classification(rec):
    for key in ("germline_classification", "clinical_significance",
                "rcv_accession", "classifications"):
        blob = rec.get(key)
        if isinstance(blob, dict):
            desc = blob.get("description")
            if desc:
                return desc
        if isinstance(blob, dict) and "germline_classification" in blob:
            desc = blob["germline_classification"].get("description")
            if desc:
                return desc
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/clinvar_trna.json")
    args = ap.parse_args()

    m = tokenize_mito(fetch_rcrs(), drop_dloop=True)
    spans = {f["name"]: (f["start"], f["end"])
             for f in m.other_features if f["type"] == "tRNA"}

    seen, rows = set(), []
    for name, symbol in GENE_SYMBOL.items():
        ids = search(symbol)
        recs = summarise(ids) if ids else []
        lo, hi = spans[name]
        kept = 0
        for r in recs:
            title = r.get("title", "")
            mt = TITLE_RE.search(title)
            if not mt:
                continue
            pos, ref, alt = int(mt.group(1)), mt.group(2), mt.group(3)
            if not (lo <= pos <= hi):
                continue
            sig = classification(r)
            if not sig:
                continue
            key = (pos, ref, alt)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"trna": name, "symbol": symbol, "pos": pos,
                         "ref": ref, "alt": alt, "clinvar": sig,
                         "title": title,
                         "review": r.get("germline_classification", {}).get("review_status")})
            kept += 1
        print(f"{symbol:8s} {len(ids):4d} records -> {kept:3d} tRNA substitutions")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)

    from collections import Counter
    print(f"\n{len(rows)} labelled substitutions")
    for sig, n in Counter(r["clinvar"] for r in rows).most_common():
        print(f"  {n:4d}  {sig}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
