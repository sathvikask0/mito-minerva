"""Turn the human mitochondrial reference (rCRS) into Minerva's mixed-token input.

Minerva expects a genome as a single string in which protein-coding genes appear
as upper-case amino acids, everything else appears as lower-case DNA, and each
segment is prefixed by a strand marker (``<+>`` / ``<->``).  ``minerva.data.
extract_and_tokenize_gb`` produces that string from a GenBank file, but it does
not tell you which genome base each character came from.  This module wraps it
and rebuilds that mapping, so a tRNA annotation can be turned into a token range
in the model's input.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from minerva.data import extract_and_tokenize_gb

RCRS_ACCESSION = "NC_012920.1"
RCRS_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={acc}&rettype=gbwithparts&retmode=text"
)

# The rCRS control region, as annotated in NC_012920: join(16024..16569, 1..576).
# It carries no tRNA, rRNA or CDS, so dropping it linearises the circle at the
# one place where nothing of interest is cut.  1-based inclusive.
DLOOP_END_5P = 576      # last base of the 1..576 half
DLOOP_START_3P = 16024  # first base of the 16024..16569 half

# Vertebrate mitochondrial genetic code.
MITO_TRANSLATION_TABLE = 2

STRAND_MARKERS = ("<+>", "<->")


@dataclass
class MitoTokenization:
    """A mixed-token Minerva input plus the provenance of every character."""

    token_string: str
    #: ``char_to_genome[i]`` is the half-open genome interval covered by
    #: ``token_string[i]``, in 1-based rCRS coordinates, or ``(-1, -1)`` for the
    #: characters of a strand marker.
    char_to_genome: list[tuple[int, int]]
    #: ``char_to_token[i]`` is the index of the token that ``token_string[i]``
    #: belongs to.  Strand markers are three characters but one token.
    char_to_token: list[int]
    n_tokens: int
    cds: list[dict]
    other_features: list[dict] = field(default_factory=list)
    #: rCRS base at which this window starts (1-based), for reporting.
    window: tuple[int, int] = (1, 16569)
    name: str = RCRS_ACCESSION

    def token_span(self, start: int, end: int) -> tuple[int, int] | None:
        """Token indices covering 1-based inclusive genome range ``start..end``.

        Returns a half-open ``(first, last + 1)`` pair, or ``None`` when the
        range falls outside this window.
        """
        hits = [
            self.char_to_token[i]
            for i, (gs, ge) in enumerate(self.char_to_genome)
            if gs != -1 and gs <= end and ge > start
        ]
        if not hits:
            return None
        return min(hits), max(hits) + 1


def fetch_rcrs(path: str = "data/raw/NC_012920.1.gb") -> str:
    """Download the rCRS GenBank record from NCBI unless it is already present."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(RCRS_URL.format(acc=RCRS_ACCESSION), path)
    return path


def load_record(path: str) -> SeqRecord:
    return next(SeqIO.parse(path, "genbank"))


def excise_dloop(record: SeqRecord) -> tuple[SeqRecord, int]:
    """Cut the circular genome open at the control region.

    Returns the sub-record spanning ``577..16023`` and the offset (576) that
    must be added to its 0-based coordinates to get 1-based rCRS positions.
    """
    sub = record[DLOOP_END_5P:DLOOP_START_3P - 1]
    sub.name = record.name
    sub.id = record.id
    sub.annotations = dict(record.annotations)
    sub.annotations["molecule_type"] = "DNA"
    return sub, DLOOP_END_5P


def _cds_char_spans(feature: dict, offset: int) -> list[tuple[int, int]]:
    """Genome interval covered by each amino-acid character of a CDS token.

    ``feature`` is one entry of ``extract_and_tokenize_gb``'s ``features`` list:
    0-based half-open ``start``/``end`` and ``orientation`` (True = forward).
    Amino acid *i* maps to codon *i* read 5'->3' along the coding strand.  The
    result is in 1-based rCRS coordinates.
    """
    n_aa = len(feature["seq"])
    start, end = feature["start"], feature["end"]
    spans = []
    for i in range(n_aa):
        if feature["orientation"]:
            gs, ge = start + 3 * i, start + 3 * i + 3
        else:
            gs, ge = end - 3 * (i + 1), end - 3 * i
        # An incomplete terminal codon (common in human mtDNA, where the stop is
        # completed by polyadenylation) can run past the annotated end.
        gs, ge = max(gs, start), min(ge, end)
        spans.append((gs + 1 + offset, ge + 1 + offset))
    return spans


def tokenize_mito(
    gb_path: str = "data/raw/NC_012920.1.gb",
    *,
    drop_dloop: bool = True,
    use_existing_translations: bool = True,
) -> MitoTokenization:
    """Build the Minerva mixed-token string for the human mitochondrial genome."""
    record = load_record(gb_path)
    offset = 0
    window = (1, len(record.seq))
    if drop_dloop:
        record, offset = excise_dloop(record)
        window = (offset + 1, offset + len(record.seq))

    tmp = f"{gb_path}.window.gb" if drop_dloop else gb_path
    if drop_dloop:
        SeqIO.write(record, tmp, "genbank")

    recs = extract_and_tokenize_gb(
        tmp,
        use_existing_translations=use_existing_translations,
        overlap_mode="expand",
        other_feature_types_to_track=["tRNA", "rRNA", "D-loop", "misc_feature"],
        translation_table=MITO_TRANSLATION_TABLE,
    )
    if drop_dloop:
        os.remove(tmp)
    rec = recs[0]

    # Walk the emitted segments in order, pairing each with the CDS or
    # intergenic region it came from.  CDS segments are upper case (protein),
    # intergenic segments are lower case (DNA) -- the same test Minerva uses.
    char_to_genome: list[tuple[int, int]] = []
    char_to_token: list[int] = []
    cds_iter = iter(rec["features"])
    ig_iter = iter(rec["intergenic_regions"])
    token_idx = 0

    for segment in rec["tokens"]:
        marker, content = segment[:3], segment[3:]
        assert marker in STRAND_MARKERS, f"unexpected segment prefix {marker!r}"
        char_to_genome.extend([(-1, -1)] * 3)
        char_to_token.extend([token_idx] * 3)
        token_idx += 1

        if content.islower():
            region = next(ig_iter)
            assert region["end"] - region["start"] == len(content)
            for j in range(len(content)):
                pos = region["start"] + j + 1 + offset
                char_to_genome.append((pos, pos + 1))
                char_to_token.append(token_idx)
                token_idx += 1
        else:
            feature = next(cds_iter)
            assert len(feature["seq"]) == len(content)
            for span in _cds_char_spans(feature, offset):
                char_to_genome.append(span)
                char_to_token.append(token_idx)
                token_idx += 1

    token_string = rec["sequence"]
    assert len(char_to_genome) == len(token_string)

    cds = [{**f, "start": f["start"] + 1 + offset, "end": f["end"] + offset} for f in rec["features"]]
    others = [
        {**f, "start": f["start"] + 1 + offset, "end": f["end"] + offset}
        for f in rec["other_features"]
    ]

    return MitoTokenization(
        token_string=token_string,
        char_to_genome=char_to_genome,
        char_to_token=char_to_token,
        n_tokens=token_idx,
        cds=cds,
        other_features=others,
        window=window,
    )
