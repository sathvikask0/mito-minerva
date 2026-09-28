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
