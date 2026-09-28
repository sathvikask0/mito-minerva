"""Score a predicted base-pairing map against the canonical tRNA cloverleaf.

There is no curated per-tRNA reference structure in this repo, so the ground
truth used here is the cloverleaf's geometry, which is fixed by standard tRNA
numbering and can be anchored on two things we know independently of any
prediction:

* the **acceptor stem**, 7 bp joining the first 7 bases of the mature tRNA to
  the last 7 before the discriminator base.  Mitochondrial tRNA genes do not
  encode the 3' CCA, so the gene ends at the discriminator (position 73) and
  base ``i`` is expected to pair with base ``L - 2 - i``.
* the **anticodon stem**, 5 bp flanking the 7-base anticodon loop.  The
  anticodon triplet itself is known from the amino acid the tRNA carries, so it
  can be located by sequence rather than by counting.

Both are read in *transcript* coordinates.  Eight of the 22 human mitochondrial
tRNAs are encoded on the light strand, so their transcript is the reverse
complement of what the model was shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Anticodon of each human mitochondrial tRNA, 5'->3' in the transcript.
#: Keyed by the gene name used in NC_012920.
MITO_ANTICODONS = {
    "TRNF": "GAA", "TRNV": "TAC", "TRNL1": "TAA", "TRNI": "GAT",
    "TRNQ": "TTG", "TRNM": "CAT", "TRNW": "TCA", "TRNA": "TGC",
    "TRNN": "GTT", "TRNC": "GCA", "TRNY": "GTA", "TRNS1": "TGA",
    "TRND": "GTC", "TRNK": "TTT", "TRNG": "TCC", "TRNR": "TCG",
    "TRNH": "GTG", "TRNS2": "GCT", "TRNL2": "TAG", "TRNE": "TTC",
    "TRNT": "TGT", "TRNP": "TGG",
}

ACCEPTOR_STEM_BP = 7
ANTICODON_STEM_BP = 5
#: The anticodon occupies positions 34-36; the stem resumes at 31 and 39, so
#: the base 3 before the anticodon pairs with the base 5 after it.
_AC_STEM_5P_GAP = 3
_AC_STEM_3P_GAP = 5

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@dataclass
class StemScore:
    """How much of one expected stem the model actually predicted."""

    name: str
    expected: list[tuple[int, int]]
    #: Predicted probability at each expected pair, best over the register
    #: tolerance.
    scores: list[float]
    #: Register shift that scored best, in bases.
    shift: int

    @property
    def n_recovered(self) -> int:
        return sum(1 for s in self.scores if s >= 0.5)

    @property
    def fraction(self) -> float:
        return self.n_recovered / len(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def acceptor_stem_pairs(length: int) -> list[tuple[int, int]]:
    """Expected acceptor-stem pairs, 0-based, for a gene of *length* bases."""
    return [(i, length - 2 - i) for i in range(ACCEPTOR_STEM_BP)]


def find_anticodon(seq: str, anticodon: str) -> int | None:
    """0-based index of the anticodon triplet.

    Position 33, immediately 5' of the anticodon, is an invariant uridine in
    essentially every tRNA, and it holds for all 22 human mitochondrial tRNAs.
    That is a far better anchor than counting bases, because the mitochondrial
    tRNAs are truncated by varying amounts -- tRNA-Ser(AGY) has no D-arm at all,
    which pulls its anticodon eight bases earlier than length alone predicts.
    So candidates preceded by U are preferred, and position only breaks ties.
    """
    seq = seq.upper().replace("U", "T")
    hits = [i for i in range(1, len(seq) - 2) if seq[i:i + 3] == anticodon.upper()]
    if not hits:
        return None
    expected = round(33 / 73 * len(seq))
    return min(hits, key=lambda i: (seq[i - 1] != "T", abs(i - expected)))


def anticodon_stem_pairs(ac_start: int, length: int) -> list[tuple[int, int]]:
    """Expected anticodon-stem pairs, 0-based, given the anticodon's index."""
    pairs = []
    for k in range(ANTICODON_STEM_BP):
        i = ac_start - _AC_STEM_5P_GAP - k
        j = ac_start + _AC_STEM_3P_GAP + k
        if 0 <= i and j < length:
            pairs.append((i, j))
    return pairs


def score_stem(
    matrix: np.ndarray,
    expected: list[tuple[int, int]],
    *,
    name: str,
    max_shift: int = 1,
) -> StemScore:
    """Best score for *expected* over a small register shift.

    Annotated gene boundaries can be off by a base, which would slide the whole
    acceptor stem, so the stem is scored at each shift and the best is kept.
    """
    n = matrix.shape[0]
    best, best_shift = None, 0
    for shift in range(-max_shift, max_shift + 1):
        scores = []
        for i, j in expected:
            jj = j + shift
            if 0 <= i < n and 0 <= jj < n:
                scores.append(float(max(matrix[i, jj], matrix[jj, i])))
            else:
                scores.append(0.0)
        if best is None or np.mean(scores) > np.mean(best):
            best, best_shift = scores, shift
    return StemScore(name=name, expected=expected, scores=best, shift=best_shift)
