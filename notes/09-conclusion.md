We set out to predict which mutations break human mitochondrial tRNAs, using an
AI trained on bacterial DNA. Here is what held up and what didn't.

## What we can claim

**Fine-tuning taught the model to read mitochondrial tRNA shape, better than
the standard physics tool.**

Scored on 409 base pairs that evolution shows are real, across all 22 human
mitochondrial tRNAs:

| | recall | 95% range |
| --- | --- | --- |
| **Minerva, fine-tuned** | **63.8%** | 59.3 – 67.5 |
| Minerva, fine-tuned, inside the genome | 62.8% | 56.9 – 67.4 |
| ViennaRNA (standard tool) | 40.8% | 33.0 – 48.5 |
| Minerva, before fine-tuning | 39.6% | 32.4 – 46.6 |
| Minerva, before fine-tuning, inside the genome | 27.9% | 17.8 – 38.8 |

Fine-tuned Minerva beats ViennaRNA by **23 points (12.6 to 32.6)** and wins on
16 of the 22 tRNAs. The ranges come from resampling whole tRNAs, not single
pairs, so they aren't flattered by counting related pairs as independent.

Before fine-tuning, putting a tRNA back inside its genome wrecked the
prediction. After, it makes no difference (+1.0 points, range −4.8 to +8.0).

It cost about **$24** of rented GPU in total.

## How it works, in one paragraph

The model never outputs a shape directly. Inside it, "attention" lets each
letter look at every other letter. Some of those attention patterns learn to
point from a letter to the partner it pairs with. A tiny add-on of 41 numbers,
trained by the model's authors on bacterial RNA, reads those patterns and turns
them into a pairing map. We **never retrained that add-on** and never showed
it a single mitochondrial pair. Fine-tuning only changed the attention, and
the add-on read the improvement out on its own.

## What we cannot claim

**Disease prediction.** The model's scores separate disease mutations from
harmless ones (0.744, where 0.5 is a coin flip), but simply counting how
conserved each letter is across animals gets 0.706. The gap is too small to
trust with only 52 known disease mutations.

**Longevity.** Across 1,321 animals living 2 to 211 years, long-lived species
do not have sturdier tRNAs once you compare each animal with its close
relatives.

## The honest caveat

Our "real" base pairs come from spotting letters that change together across
thousands of species, and those are the same 15,589 genomes the model trained
on. So the model may have learned exactly the pattern we then graded it on.

That doesn't make the result wrong. Learning which letters change together
from raw DNA, with no alignment and no labels, is the same thing that makes
protein AI models work. But it means the grading is independent of *our
assumptions*, not of the *training data*. A fully independent test would use
lab-solved 3D structures of human mitochondrial tRNAs. That's the next step if
anyone picks this up.

## What we'd tell someone else

- Check a good-looking number against the most boring explanation you can
  build. Our disease result looked strong until we did.
- Don't grade a model against a definition you wrote yourself.
- Compare animals with their relatives, or you'll mostly measure "mammal vs
  fish".
- Measure costs; don't guess them. Our first two estimates were off by 4× and
  5×.
