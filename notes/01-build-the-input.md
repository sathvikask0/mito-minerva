Minerva does not read raw DNA. It wants a specific format: genes written out as
protein, everything else as lowercase DNA, with markers saying which direction
each piece runs.

We took the standard human mitochondrial reference sequence
(rCRS, NC_012920.1) and converted it.

**The problem:** the model can only read 8,192 units at a time. The full
circular genome came to **9,040** — about 10% too long.

**The fix:** mitochondrial DNA is a loop, so it has no natural start or end. We
cut the loop at the **control region**, a stretch that contains no genes at all.
That dropped it to **7,918** units, which fits, and we lost nothing we care
about — all 22 tRNAs are still there.

We also built a lookup table mapping every character back to its exact position
in the reference, so later steps can point at a specific letter. Tests check
this against the real sequence letter by letter.

**Two oddities worth knowing:** position 3,107 of the reference is a
placeholder letter that does not really exist (it is kept so the numbering
matches the original 1981 sequence). And two pairs of genes physically overlap,
sharing the same letters in different reading frames.
