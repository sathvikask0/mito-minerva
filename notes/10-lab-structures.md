The conclusion had one weakness we flagged ourselves: our "real" base pairs
came from the same 15,589 genomes the model trained on. The model might have
learned exactly the pattern we graded it on.

So we graded it against something it could not have seen: **the actual 3D
shapes of human mitochondrial tRNAs, measured in the lab.**

## Where the shapes come from

Scientists freeze molecules and image them with X-rays or electron
microscopes, then build atom-by-atom models. These models are public in the
Protein Data Bank. Human mitochondrial tRNAs show up in them because they get
photographed while bound to the mitochondrial ribosome or to the enzymes that
process them.

We searched every one of the 22 human tRNAs. **8 have usable human
structures**: H, I, M, Q, R, S2, V and Y, from 36 models at 1.9–4.3 Å. (A
ninth, L2, exists only from pig, so we left it out.)

For each model we read the base pairs directly off the atoms: two letters
count as paired when their hydrogen-bonding atoms are within 3.6 Å, the
distance at which they physically hold on to each other. Where several models
exist, a pair counts only if at least half of them agree.

Seven of the eight came out as clean cloverleaves. The eighth, TRNS2, is the
tRNA famous for missing an arm, and its structures are partial because they
were captured mid-processing. We report results with and without it.

## Result

132 base pairs across 8 tRNAs, none of which the model or its training data
had any hand in:

| | finds real pairs (recall) | its pairs are real (precision) | overall (F1) |
| --- | --- | --- | --- |
| **Minerva, fine-tuned** | **94%** | 67% | **0.78** |
| Minerva, fine-tuned, inside the genome | 82% | 67% | 0.74 |
| ViennaRNA | 68% | 57% | 0.62 |
| Minerva, before fine-tuning | 61% | 64% | 0.63 |
| Minerva, before fine-tuning, inside the genome | 33% | 58% | 0.42 |

What holds up:

- **Fine-tuned Minerva finds far more of the real pairs than ViennaRNA:**
  +25.8 points (95% range +6.1 to +45.5). That repeats the +23 we measured
  against evolution, this time on truly independent data.
- **Fine-tuning clearly helped.** Against the lab structures, fine-tuned beats
  the original model by 0.15 F1 (range +0.07 to +0.25), on 6 of 8 tRNAs.
- **Minerva is consistent; ViennaRNA is hit or miss.** Fine-tuned Minerva
  scores 0.74–0.86 F1 on seven of the eight tRNAs. ViennaRNA scores
  0.83–0.95 on three of them, but 0.29–0.54 on four.

What doesn't:

- **Precision is only about the same as ViennaRNA's** (+9.7 points, range
  crossing zero). About a third of the fine-tuned model's pairs aren't in the
  lab structure. We looked at what they are: **69% sit directly on the end of a
  real helix.** The model tends to extend a stem by one pair too many. It
  isn't inventing structure that isn't there.
- **So the overall F1 lead over ViennaRNA is borderline:** +0.16, with a range
  of −0.003 to +0.32 on all eight tRNAs, and +0.004 to +0.34 without TRNS2. It
  is most likely real, but eight tRNAs can't prove it.
- **TRNV still fails inside the genome** (0 of 15 pairs), the same
  long-window weakness found earlier. On its own it gets all 15.

## What this changes

The circularity worry is answered: the model's advantage in **finding** real
base pairs is not an artifact of grading it on its own training data. The
honest refinement is that it wins by catching pairs ViennaRNA misses, not by
making fewer mistakes.

Remaining limits: only 8 tRNAs, and cryo-EM models are sometimes built with
help from templates, so a modeller's expectations can leak into a structure.
But none of that connects to our genomes or our model.
