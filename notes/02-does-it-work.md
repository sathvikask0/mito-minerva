This was the go/no-go. If the model could not find the clover shape, the whole
project was dead and we would have had to retrain it from scratch.

**How we checked.** We did not use a hand-curated answer key. Instead we used
two features every clover must have, which we know from the sequence alone:

- the **acceptor stem** — the two ends of the strand stick to each other,
  7 pairs deep
- the **anticodon stem** — 5 pairs around the business end that reads the
  genetic code

We also ran **200 decoys**: random chunks of the same genome, same length, that
are *not* tRNAs. If the model calls those clovers too, it is just guessing.

**Results:**

| what we fed it | acceptor stem found | tRNAs mostly right |
| --- | --- | --- |
| one tRNA on its own | **72%** | 17 of 22 |
| the whole genome at once | 24% | 5 of 22 |
| ViennaRNA (the standard old tool) | 83% | 19 of 22 |

Decoys scored essentially zero. So when Minerva calls a clover, it means it —
it is not just sticking the ends of any short sequence together.

**The twist: context breaks it.** Give the model the tRNA plus just 50 letters
of its real neighbours and performance collapses from 69% to 30%.

That is the opposite of what we expected. We chose mitochondrial DNA *because*
the whole thing fits in one window — and one big window turns out to be the
model's worst setting.

**Verdict: go, with a change of plan.** Feed tRNAs one at a time. On the
anticodon stem — the part that matters most for disease — Minerva actually
beats the old tool, 55% to 35%. And the tRNA behind the most famous
mitochondrial disease mutation comes out perfect in every setting.

Side benefit: single tRNAs are tiny, so everything runs in milliseconds. **No
GPU needed.**
