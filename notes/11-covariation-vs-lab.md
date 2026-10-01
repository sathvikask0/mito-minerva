Step 9 answered one circularity worry: the model was graded against lab
structures it could not have seen. A second one remained. The model was
fine-tuned on the same genomes that produced our covariation reference in
Step 7. Maybe it had simply learned to reproduce that covariation signal,
and any "win" was really a win for covariation.

That is easy to test, and it needs no GPU: **grade the covariation reference
itself against the lab structures**, with the same scoring as Step 9.

## Result

Same 8 tRNAs, same 132 lab-measured base pairs:

| | finds real pairs (recall) | its pairs are real (precision) | overall (F1) |
| --- | --- | --- | --- |
| **Minerva, fine-tuned** | **94%** | 67% | **0.78** |
| Minerva, fine-tuned, inside the genome | 82% | 67% | 0.74 |
| ViennaRNA | 68% | 57% | 0.62 |
| Minerva, before fine-tuning | 61% | 64% | 0.63 |
| **Covariation reference (Step 7)** | 56% | 58% | **0.57** |

- **The fine-tuned model beats the covariation reference on all 8 tRNAs:**
  +0.21 F1 (95% range +0.14 to +0.29).
- Inside the genome it still wins, +0.16 (range +0.02 to +0.29), on 6 of 8.
- ViennaRNA and the original model are statistically tied with it.

So the model is not just copying the covariation reference. It agrees with
the lab structures more than the reference does.

## What the reference gets wrong

Looking pair by pair, the reference misses much of the **acceptor stem** in 6
of the 8 tRNAs (17 of its 58 misses). These positions are likely too
conserved to covary: if a base pair almost never changes, there are no
compensating changes to detect. Many of its extra pairs sit far from any real
helix and look like tertiary contacts or noise.

## What this changes

- **Step 7's headline undersells the model.** That reference is only about
  58% precise against the lab structures, so some of the model's "misses" in
  Step 7 were the reference's mistakes, not the model's.
- **The honest claim is narrow.** Our covariation pipeline is simple: each
  species is aligned to human one at a time, related species are not
  down-weighted, and pairs are picked greedily. A structure-aware covariation
  tool (Infernal with R-scape, for example) would do better. So this shows the
  model beats *this* covariation reference, not covariation in general.

Script: `scripts/step15_covariation_vs_pdb.py`, output
`outputs/step15_covariation_vs_pdb.json`. It reads committed files only.
