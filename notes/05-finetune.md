Step 4 said no. But it also said *why*: the model has never seen mitochondrial
DNA. It was trained on bacteria, and mitochondrial genomes are packed far more
tightly — genes butt up against each other, and the tRNAs sit in the cracks
between them.

So the obvious repair is to show it some. There are thousands of animal
mitochondrial genomes free to download. We fine-tune on those, hold out human,
then re-run steps 2 through 4 and see if anything moved.

**Human is held out** so we never train on what we test on.

## Laptop or rented GPU?

We measured it rather than guessed.

Training is far heavier than the prediction work so far, because the computer
has to remember every intermediate step in order to work backwards and adjust
the model. On your laptop:

| sequence length | speed |
| --- | --- |
| 512 units | 0.62 s per step |
| 1,024 units | 1.47 s per step |
| 2,048 units | **out of memory** |

The corpus is roughly 120 million units of text. One pass over it, at the
biggest chunk size that fits, would take about **48 hours** — and it would be
processing one sequence at a time, which is the slowest possible way to do it.

A rented A100 GPU does the same pass in **about an hour**, with four times the
chunk size and eight sequences at once. Three passes is a few hours and
roughly **$10**.

**So: rented GPU for the training.** Everything else stays on the laptop —
building the data, and all the scoring afterwards, which we already know runs
in minutes.

## What success looks like

We are not looking for a small nudge. Step 2 found the model is fragile: real
genomic neighbours break it. If fine-tuning fixes that fragility, we should see
folding hold up in context instead of collapsing. If it only moves the disease
scores from 0.60 to 0.63, that is noise, not a result.

*In progress: building the corpus.*
