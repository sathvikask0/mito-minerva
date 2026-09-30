The question: do animals that live a long time have mitochondrial tRNAs whose
shape survives random mutations better?

The idea behind it: mitochondrial DNA mutates fast and is barely repaired. A
mouse has to keep its mitochondria working for three years; a bowhead whale for
two centuries. So long-lived animals might be pushed by evolution toward tRNAs
that still fold properly after a letter changes.

## What we did

For **1,321 animals** with a known maximum lifespan, we took every tRNA, changed
every letter to each of the other three, and asked the model how much of the
folded shape broke. The average damage per mutation is that species'
**fragility**.

1,044 of them also had a known body mass, which we need as a control.

## The trap

The raw numbers show a weak link — but in the **wrong direction**: longer-lived
species looked slightly *more* fragile (r = +0.08).

And that link turns out to be fake. Mammals, birds, fish and reptiles differ
from each other in lifespan *and* in their tRNAs, for reasons that have nothing
to do with each other. Comparing a whale to a goldfish mostly measures "mammal
vs fish".

The fix is to compare each animal only with its close relatives — whales with
whales, rockfish with rockfish — and ask whether the longer-lived relative has
the sturdier tRNAs.

## Result

| compared only within | species | link | chance it's luck |
| --- | --- | --- | --- |
| the same family | 911 | +0.05 | 15% |
| the same order | 1,026 | −0.01 | 76% |

**No link.** Among close relatives, living longer has nothing to do with how
fragile the tRNAs are. A second way of measuring fragility gave the same answer.

One group looked promising: reptiles, at −0.48, in the predicted direction. It
disappears when turtles are compared only with turtles and lizards with lizards
(−0.06). It was just "turtles live long and have different tRNAs", not a real
effect.

## What this means

The hypothesis is wrong, or at least undetectable this way. Long-lived animals
are not buying longevity through mutation-proof tRNA shapes — they presumably
handle mitochondrial damage some other way.

Total cost for this step: about **$8** of GPU time, including two runs lost to
cancellations before the job was made to survive them.
