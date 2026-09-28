"""Check that the mixed-token string really is the rCRS, base for base."""

import os
import sys

import pytest
from Bio.Seq import Seq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitominerva.mito import MITO_TRANSLATION_TABLE, fetch_rcrs, load_record, tokenize_mito

GB = fetch_rcrs()
RCRS = str(load_record(GB).seq)  # 0-based; rCRS position p is RCRS[p - 1]


@pytest.fixture(scope="module", params=[False, True], ids=["full", "nodloop"])
def tk(request):
    return tokenize_mito(GB, drop_dloop=request.param)


def test_lowercase_chars_match_the_reference(tk):
    """Every lower-case character is the rCRS base at the position it claims."""
    n = 0
    for ch, (gs, ge) in zip(tk.token_string, tk.char_to_genome):
        if not ch.islower():
            continue
        assert ge - gs == 1, f"{ch} at {gs} spans {ge - gs} bases"
        assert ch == RCRS[gs - 1].lower(), f"{ch!r} != {RCRS[gs - 1]!r} at rCRS {gs}"
        n += 1
    assert n > 4000


def test_amino_acids_match_their_codons(tk):
    """Every upper-case character translates from the codon it is mapped to."""
    table = MITO_TRANSLATION_TABLE
    checked = 0
    for ch, (gs, ge) in zip(tk.token_string, tk.char_to_genome):
        if gs == -1 or ch.islower():
            continue
        codon = Seq(RCRS[gs - 1:ge - 1])
        if len(codon) < 3:  # incomplete terminal codon, completed by polyA
            continue
        fwd = str(codon.translate(table=table))
        rev = str(codon.reverse_complement().translate(table=table))
        assert ch in (fwd, rev, "M"), f"{ch!r} not in {{{fwd},{rev}}} at rCRS {gs}-{ge}"
        checked += 1
    assert checked > 3500


def test_every_genome_base_in_window_is_covered_once(tk):
    """Each base is emitted once, except where annotated genes really overlap.

    Human mtDNA packs ATP8/ATP6 and ND4L/ND4 in two reading frames, and
    ``overlap_mode="expand"`` deliberately emits both proteins, so those bases
    appear twice.  Stop codons carry no amino acid, so they are the only bases
    allowed to go missing.
    """
    lo, hi = tk.window
    covered = {}
    for (gs, ge) in tk.char_to_genome:
        if gs == -1:
            continue
        for p in range(gs, ge):
            covered[p] = covered.get(p, 0) + 1

    cds = [c for c in tk.cds]
    overlaps = {
        p
        for a in cds
        for b in cds
        if a is not b
        for p in range(max(a["start"], b["start"]), min(a["end"], b["end"]) + 1)
    }
    doubled = {p for p, c in covered.items() if c > 1}
    assert doubled <= overlaps, f"doubled outside a gene overlap: {sorted(doubled - overlaps)[:10]}"

    stop_codons = {
        p
        for c in cds
        for p in (range(c["end"] - 2, c["end"] + 1) if c["orientation"]
                  else range(c["start"], c["start"] + 3))
    }
    missing = {p for p in range(lo, hi + 1) if p not in covered}
    assert missing <= stop_codons, f"dropped a non-stop base: {sorted(missing - stop_codons)[:10]}"
    assert len(missing) == 21


def test_all_22_trnas_are_present_and_are_plain_dna(tk):
    trnas = [f for f in tk.other_features if f["type"] == "tRNA"]
    assert len(trnas) == 22
    for f in trnas:
        span = tk.token_span(f["start"], f["end"])
        assert span is not None, f"{f['name']} not in window"
        chars = "".join(
            tk.token_string[i]
            for i, t in enumerate(tk.char_to_token)
            if span[0] <= t < span[1]
        )
        assert chars == RCRS[f["start"] - 1:f["end"]].lower(), f["name"]


def test_token_count(tk):
    assert tk.n_tokens == (9040 if tk.window[0] == 1 else 7918)
