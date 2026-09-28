Now the actual experiment. For all 22 tRNAs, change every letter to each of the
three alternatives and measure the damage. That is roughly 4,600 mutations.

Each one gets **two independent scores**:

1. **Shape damage** — refold the mutant and count how much of the original
   folding is lost. The old tool can do this too.
2. **Surprise** — hide the letter, ask the model what belongs there, and see
   how strongly it prefers the original. The old tool *cannot* do this at all.
   This is the real reason to use an AI model.

These two can disagree, and that is the interesting part: a mutation might
leave the shape intact but still look deeply wrong to the model, or vice versa.

## Done — 4,524 mutations in about two minutes

On a laptop, no GPU.

**The two scores barely agree with each other** (correlation 0.19). That is the
encouraging part: they are not measuring the same thing twice. Shape damage and
surprise are genuinely separate signals, so combining them should beat either
one alone.

**First look at the famous one.** m.3243A>G — the mutation behind MELAS, a
serious mitochondrial disease — lands in the **96th percentile for surprise**
across all 4,524 mutations. The model strongly expects the original letter
there.

On shape damage alone it is only middling, 56th percentile. So the shape score
by itself would have missed it, and the surprise score caught it.

One data point is not evidence. Step 4 is the real test.
