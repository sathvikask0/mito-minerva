The final test. Do our scores actually pick out real disease mutations?

**The data.** MITOMAP, the usual catalogue, blocks automated access. We used
**ClinVar** instead — NCBI's public variant database, same source we got the
reference sequence from. Filtering to single-letter changes inside the 22 tRNAs
gave **363 labelled variants: 52 disease-causing, 311 harmless.**

**The measure.** Pick one disease mutation and one harmless one at random. How
often does the score rank the disease one as worse? That is the AUC.
0.5 is a coin flip. 1.0 is perfect. A tool people actually use scores about 0.9.

## The answer: no

| score | AUC | |
| --- | --- | --- |
| Minerva — shape damage | 0.53 | coin flip |
| Minerva — surprise | 0.58 | barely better |
| Minerva — surprise, with context | 0.49 | coin flip |
| **Minerva — best combination** | **0.60** | weak |
| ViennaRNA — structure change | 0.52 | coin flip |
| **ViennaRNA — destabilisation** | **0.66** | weak, but better |

**Minerva lost to the 30-year-old physics tool, 0.60 to 0.66.** The gap is not
statistically firm (95% CI −0.16 to +0.04), so the fair reading is "no better",
not "clearly worse".

Either way, neither is good enough to be useful. Published tools for this exact
job score around 0.9.

We re-ran it keeping only the most confident labels — same answer.

## Why it did not work

The pitch was that Minerva reads folding from a single sequence, with no need
to compare across species. The tools that score 0.9 do exactly that comparison:
they check whether a letter stayed the same across millions of years of
evolution. That turns out to carry most of the signal, and Minerva has no
access to it.

Adding context made the surprise score *worse* — down to a coin flip. That
matches what we found in step 2: this model gets confused by real genomic
neighbours. It is trained on bacteria, and mitochondrial DNA is packed far more
tightly than any bacterial genome.

## What is honestly true

- **Step 2 was a real positive.** Minerva does fold mitochondrial tRNAs it has
  never seen, better than physics at the anticodon arm. That is a genuine
  finding about a bacterial model transferring to human DNA.
- **Step 4 is a real negative.** That folding ability does not translate into
  predicting which mutations cause disease.

Both are worth knowing. The second one is why we are stopping here rather than
building on it.

## What we would try next

1. **Fine-tune on animal mitochondrial genomes.** This was always the fallback.
   The model has never seen mitochondrial DNA; a few thousand animal
   mitochondrial genomes are freely available.
2. **Get the MITOMAP set.** 52 disease variants is thin. MITOMAP has several
   times more, which would make the test sharper.
3. **Stop.** The honest option. Conservation-based tools already do this well.
