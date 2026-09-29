Every number in step 5 had a hidden weakness: we graded the model against a
definition of "correct" that **we wrote ourselves**.

A tRNA is drawn as a cloverleaf, and the textbook says which letters pair with
which. We encoded that rule and marked the model right or wrong against it.
But ViennaRNA was marked against the same rule. If the rule is wrong for some
tRNAs, both tools were graded against a fiction.

Human mitochondrial tRNAs are exactly where that matters. They are notorious
oddballs — several are missing entire arms of the cloverleaf.

## Letting evolution define the right answer

There is a way to find real base pairs without assuming anything.

If two positions in a tRNA genuinely touch each other, they are stuck
together. When one mutates, the molecule breaks — unless the partner mutates
to match. So over millions of years you see the two positions **changing
together**. A position that pairs with nothing changes freely.

That pattern is physical evidence of contact, and it is how RNA structures
were worked out decades before any software existed.

We had the raw material already: 16,080 animal mitochondrial genomes. For each
of the 22 human tRNAs we lined up its counterpart from **8,397 to 15,552
species**, then looked for pairs of positions that change in lockstep and can
always physically pair.

## Two checks that this reference is trustworthy

**It found a known deformity by itself.** One human tRNA, the one for serine
called TRNS2, is famous for being broken — it is missing a whole arm of the
cloverleaf. We did not tell the method this. It returned **7 pairs for TRNS2
and 10 to 26 for every other tRNA**. It found the missing arm on its own.

**It mostly agrees with the textbook rule** — confirming 106 of our 154
assumed pairs. The disagreements are expected rather than worrying: if a pair
never changes in any species, there is nothing to observe, so this method is
blind to the most conserved pairs. It undercounts; it does not misreport.

## The result

Scored on 409 base pairs that evolution proves are real:

| | before finetuning | **after** | ViennaRNA |
| --- | --- | --- | --- |
| tRNA on its own | 39.6% | **63.8%** | 40.8% |
| **inside the genome** | 27.9% | **62.8%** | — |

Before, Minerva was no better than the standard physics tool, and fell apart
once the tRNA sat in its real genomic surroundings. After, it finds **1.56
times more real base pairs than ViennaRNA**, and it barely loses anything when
you put the tRNA back where it lives.

This is the claim that survived. It was measured against structure we did not
define.

## The one failure, and why it is interesting

TRNV gets 10 of 13 pairs right on its own and **0 of 13** inside the genome.
We checked whether this was a bug in how we cut the genome up. It is not — the
coordinates are exactly right.

TRNV is the only tRNA sandwiched between the two large rRNA genes, which are
themselves enormous folded structures. The most likely explanation is that
their base pairing dominates that stretch and drowns out the small tRNA. A
real limitation, and a specific one worth knowing about.
