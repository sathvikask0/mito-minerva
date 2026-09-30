"""Base pairs of human mitochondrial tRNAs, read off experimental 3D structures.

The covariation reference in step 7 was derived from the same 15,589 genomes
the model was finetuned on, so a model that learned covariation could score
well against it for that reason alone. Experimentally solved structures owe
nothing to those genomes: they are atoms placed into X-ray or cryo-EM density.

For each human mt-tRNA with a structure in the PDB:

  1. find chains by sequence search (>=90% identity), keep Homo sapiens only
  2. take up to --per-trna best-resolution chains
  3. read atoms from the RCSB ModelServer, one chain at a time
  4. call canonical pairs (Watson-Crick and G-U wobble) from hydrogen-bond
     donor-acceptor distances
  5. align the chain to the human transcript to put pairs in its coordinates
  6. keep the maximum nested (pseudoknot-free) subset, preferring stacked
     pairs -- that is secondary structure, which is what every predictor
     here predicts; tertiary contacts such as G19:C56 cross the stems and drop
  7. a pair enters the reference if it appears in at least half the chains

Unlike covariation this gives the whole secondary structure, so precision as
well as recall can be measured.

    python scripts/step13_pdb_structures.py
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter

from Bio.Align import PairwiseAligner
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from step6_build_conservation import human_trnas  # noqa: E402

HUMAN = 9606
CACHE = "data/raw/pdb"
HB = 3.6  # angstrom; tolerant of 2-3.5 A cryo-EM models

# donor/acceptor atom pairs for each canonical pair, (atom on first, atom on second)
PAIR_ATOMS = {
    ("G", "C"): [("N1", "N3"), ("N2", "O2"), ("O6", "N4")],
    ("A", "U"): [("N6", "O4"), ("N1", "N3")],
    ("G", "U"): [("O6", "N3"), ("N1", "O2")],
}
NEED = {("G", "C"): 2, ("A", "U"): 2, ("G", "U"): 2}


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    return json.loads(raw) if raw else {}


def search_chains(seq):
    q = {"query": {"type": "terminal", "service": "sequence",
                   "parameters": {"evalue_cutoff": 10, "identity_cutoff": 0.9,
                                  "sequence_type": "rna", "value": seq}},
         "return_type": "polymer_instance",
         "request_options": {"paginate": {"start": 0, "rows": 1000},
                             "results_content_type": ["experimental"]}}
    return [x["identifier"] for x in post("https://search.rcsb.org/rcsbsearch/v2/query",
                                          q).get("result_set", [])]


def chain_metadata(ids):
    q = """query($ids:[String!]!){ polymer_entity_instances(instance_ids:$ids){ rcsb_id
      rcsb_polymer_entity_instance_container_identifiers{ entry_id asym_id auth_asym_id }
      polymer_entity{ rcsb_entity_source_organism{ ncbi_taxonomy_id scientific_name }
        entity_poly{ pdbx_seq_one_letter_code_can }
        entry{ rcsb_entry_info{ resolution_combined experimental_method } } } } }"""
    out = {}
    for k in range(0, len(ids), 50):
        d = post("https://data.rcsb.org/graphql",
                 {"query": q, "variables": {"ids": ids[k:k + 50]}})
        for x in d["data"]["polymer_entity_instances"]:
            pe = x["polymer_entity"]
            c = x["rcsb_polymer_entity_instance_container_identifiers"]
            org = (pe.get("rcsb_entity_source_organism") or [{}])[0]
            info = pe["entry"]["rcsb_entry_info"]
            out[x["rcsb_id"]] = {
                "entry": c["entry_id"], "asym": c["asym_id"], "auth": c["auth_asym_id"],
                "taxid": org.get("ncbi_taxonomy_id"), "organism": org.get("scientific_name"),
                "seq": pe["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", ""),
                "resolution": (info.get("resolution_combined") or [None])[0],
                "method": info.get("experimental_method"),
            }
    return out


def chain_atoms(entry, asym):
    """{label_seq_id: {atom_name: (x, y, z)}} for the first model of one chain."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{entry}_{asym}.cif"
    if not os.path.exists(path):
        url = (f"https://models.rcsb.org/v1/{entry.lower()}/atoms?"
               f"label_asym_id={asym}&encoding=cif")
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(url, timeout=120).read()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
        with open(path, "wb") as fh:
            fh.write(data)
    d = MMCIF2Dict(path)
    seq_ids = d["_atom_site.label_seq_id"]
    names = d["_atom_site.label_atom_id"]
    xs, ys, zs = (d[f"_atom_site.Cartn_{a}"] for a in "xyz")
    models = d.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(names))
    first = models[0]
    res = {}
    for sid, nm, x, y, z, mdl in zip(seq_ids, names, xs, ys, zs, models):
        if mdl != first or sid in (".", "?"):
            continue
        res.setdefault(int(sid), {})[nm.strip('"')] = (float(x), float(y), float(z))
    return res


def dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def canonical_pairs(atoms, can_seq, min_sep=4):
    """Pairs (i, j, mean H-bond distance) in label_seq_id numbering."""
    ids = sorted(atoms)
    hits = []
    for a in range(len(ids)):
        i = ids[a]
        bi = can_seq[i - 1] if i - 1 < len(can_seq) else "N"
        for b in range(a + 1, len(ids)):
            j = ids[b]
            if j - i < min_sep:
                continue
            bj = can_seq[j - 1] if j - 1 < len(can_seq) else "N"
            key, first, second = None, None, None
            if (bi, bj) in PAIR_ATOMS:
                key, first, second = (bi, bj), atoms[i], atoms[j]
            elif (bj, bi) in PAIR_ATOMS:
                key, first, second = (bj, bi), atoms[j], atoms[i]
            if key is None:
                continue
            ds = [dist(first[p], second[q]) for p, q in PAIR_ATOMS[key]
                  if p in first and q in second]
            close = [x for x in ds if x <= HB]
            if len(close) >= NEED[key]:
                hits.append((i, j, sum(close) / len(close)))
    # one partner per residue: keep the tightest
    hits.sort(key=lambda h: h[2])
    used, out = set(), []
    for i, j, d in hits:
        if i in used or j in used:
            continue
        used |= {i, j}
        out.append((i, j, d))
    return out


def max_nested(pairs):
    """Maximum-weight pseudoknot-free subset; stacked pairs weigh more."""
    ps = {(i, j) for i, j, _ in pairs}
    w = {(i, j): 2.0 if ((i + 1, j - 1) in ps or (i - 1, j + 1) in ps) else 1.0
         for i, j in ps}
    if not ps:
        return []
    pos = sorted({x for p in ps for x in p})
    idx = {p: k for k, p in enumerate(pos)}
    n = len(pos)
    partner = {}
    for i, j in ps:
        partner[idx[i]] = idx[j]
        partner[idx[j]] = idx[i]
    from functools import lru_cache
    sys.setrecursionlimit(10000)

    @lru_cache(None)
    def best(a, b):
        if a >= b:
            return 0.0, ()
        # a unpaired
        s, sel = best(a + 1, b)
        c = partner.get(a)
        if c is not None and a < c <= b:
            s1, sel1 = best(a + 1, c - 1)
            s2, sel2 = best(c + 1, b)
            tot = s1 + s2 + w[(pos[a], pos[c])]
            if tot > s:
                s, sel = tot, sel1 + sel2 + ((pos[a], pos[c]),)
        return s, sel

    return sorted(best(0, n - 1)[1])


def build_aligner():
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score, a.mismatch_score = 2, -1
    a.open_gap_score, a.extend_gap_score = -5, -1
    a.target_end_gap_score = a.query_end_gap_score = 0  # leaders, trailers, CCA
    return a


def dot_bracket(L, pairs):
    s = ["."] * L
    for i, j in pairs:
        s[i - 1], s[j - 1] = "(", ")"
    return "".join(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-trna", type=int, default=5)
    ap.add_argument("--min-identity", type=float, default=0.95)
    ap.add_argument("--out", default="data/processed/pdb_trna_pairs.json")
    args = ap.parse_args()

    human = human_trnas()
    aligner = build_aligner()
    result = {}

    for gene in sorted(human):
        tx = human[gene]["seq"]
        ids = search_chains(tx.replace("T", "U"))
        if not ids:
            continue
        meta = chain_metadata(ids)
        cands = [(k, m) for k, m in meta.items()
                 if m["taxid"] == HUMAN and m["resolution"]]
        cands.sort(key=lambda km: km[1]["resolution"])
        if not cands:
            print(f"{gene:6s} {len(ids)} chains, none human -- skipped")
            continue

        per_chain = []
        for key, m in cands:
            if len(per_chain) >= args.per_trna:
                break
            can = m["seq"].upper().replace("T", "U")
            aln = aligner.align(tx, can.replace("U", "T"))[0]
            to_tx, matches = {}, 0
            for (ts, te), (qs, qe) in zip(*aln.aligned):
                for k in range(te - ts):
                    to_tx[int(qs + k + 1)] = int(ts + k + 1)  # label_seq_id -> transcript pos
                    matches += tx[ts + k] == can[qs + k].replace("U", "T")
            ident = matches / len(tx)
            if ident < args.min_identity:
                continue
            try:
                atoms = chain_atoms(m["entry"], m["asym"])
            except Exception as e:
                print(f"  {key}: download failed ({e})")
                continue
            modelled = sum(1 for s in atoms if s in to_tx)
            raw = canonical_pairs(atoms, can)
            mapped = [(to_tx[i], to_tx[j], d) for i, j, d in raw
                      if i in to_tx and j in to_tx]
            nested = max_nested(mapped)
            per_chain.append({"chain": key, "resolution": m["resolution"],
                              "method": m["method"], "identity": round(ident, 3),
                              "modelled": modelled, "canonical": len(mapped),
                              "nested": [list(p) for p in nested]})

        if not per_chain:
            print(f"{gene:6s} no human chain passed identity {args.min_identity}")
            continue
        votes = Counter(tuple(p) for c in per_chain for p in c["nested"])
        need = math.ceil(len(per_chain) / 2)
        consensus = sorted(p for p, v in votes.items() if v >= need)
        consensus = max_nested([(i, j, 0.0) for i, j in consensus])
        L = len(tx)
        result[gene] = {"human_seq": tx, "length": L, "chains": per_chain,
                        "consensus": [list(p) for p in consensus]}
        chains = ", ".join(f"{c['chain']}({c['resolution']}A)" for c in per_chain)
        print(f"{gene:6s} {len(consensus):2d} pairs from {len(per_chain)} chain(s): {chains}")
        print(f"       {tx}")
        print(f"       {dot_bracket(L, consensus)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"hbond_cutoff_A": HB, "min_identity": args.min_identity,
                   "genes": result}, fh, indent=2)
    print(f"\n{len(result)} tRNAs with experimental secondary structure; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
