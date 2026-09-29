Step 4 said no. But it also said *why*: the model has never seen mitochondrial
DNA. It was trained on bacteria, and mitochondrial genomes are packed far more
tightly — genes butt up against each other, and the tRNAs sit in the cracks
between them.

So the obvious repair is to show it some. There are thousands of animal
mitochondrial genomes free to download. We fine-tune on those, hold out human,
then re-run steps 2 through 4 and see if anything moved.

**Human is held out** so we never train on what we test on.

## The corpus

Downloaded from NCBI's RefSeq collection, filtered to animals only:

| | |
| --- | --- |
| genomes | **15,589** |
| total text | 137.8 million units |
| typical size | 16,469 units each |
| human | removed |

It spans the animal kingdom — about 7,800 vertebrates, 5,700 crustaceans and
insects, 858 molluscs, and a long tail of worms, spiders and sea urchins. Wide
variety is the point: we want the model to learn what mitochondrial DNA looks
like in general, not to memorise one animal.

## Laptop or rented GPU?

We measured it rather than guessed.

Training is far heavier than the prediction work so far, because the computer
has to remember every intermediate step in order to work backwards and adjust
the model. On your laptop:

| chunk size | speed |
| --- | --- |
| 512 units | 0.62 s per step |
| 1,024 units | 1.47 s per step |
| 2,048 units | **out of memory** |

One pass over 137.8 million units, at the biggest chunk that fits, would take
about **48 hours** — processing one sequence at a time, the slowest possible
way.

So we rented an A100 GPU. Measured there: **5.5 seconds per step, 2,626 steps,
just under 4 hours** for one pass, at four times the chunk size and eight
sequences at once. That is about **$10**.

An earlier estimate here said one hour per pass and $10 for three passes. That
was a guess and it was wrong by roughly three times. The numbers above are
measured.

**So: one pass on a rented GPU.** Everything else stays on the laptop —
building the data, and all the scoring afterwards, which we already know runs
in minutes. If the model is still visibly improving when the pass ends, we can
pay for another.

## A bug worth recording

The first launch reported a training error of 2.954 while the held-out error
was 1.476 — exactly double. Not a coincidence, and not the model misbehaving.

To save memory, the trainer processes two batches before each adjustment and
averages them. A recent change to the training library skips that averaging
when the model's code is written a certain way — which Minerva's is — because
it assumes the model does the averaging itself. Minerva doesn't. So every
adjustment was twice as large as intended, quietly doubling the learning rate.

Caught it ten minutes into a four-hour run, cost about fifty cents. After the
fix the training error reads 1.468, right where the held-out error sits.

The general lesson: when a number is off by exactly 2, suspect arithmetic
before biology.

## What success looks like

We are not looking for a small nudge. Step 2 found the model is fragile: real
genomic neighbours break it. If fine-tuning fixes that fragility, we should see
folding hold up in context instead of collapsing. If it only moves the disease
scores from 0.60 to 0.63, that is noise, not a result.

The number to beat is **1.4764** — how well the untrained model predicts
held-out mitochondrial DNA. For reference, pure guessing among the four DNA
letters scores 1.386, so the bacterial model starts off barely better than
chance on this material. That gap is the whole reason for this step.

*Done: 2,626 steps, 3h51m on an A100, held-out loss 1.4764 -> 0.7435, about $10.
The validation curve fell monotonically and train and validation stayed on top
of each other throughout, so nothing was memorised. See step 6 for what the
finetuned model can actually do.*
