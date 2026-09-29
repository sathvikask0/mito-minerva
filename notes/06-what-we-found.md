Step 5 worked, but not in the way we expected. The result is about
**structure**, not disease.

## The real finding

Minerva can now read a mitochondrial genome and correctly work out how each
tRNA folds — while it is still sitting inside the genome, surrounded by its
real neighbours.

| | before | after | ViennaRNA |
| --- | --- | --- | --- |
| acceptor stem, tRNA alone | 72% | **93%** | 83% |
| acceptor stem, **inside the genome** | **24%** | **89%** | — |
| anticodon stem | 55% | **95%** | 35% |
| scrambled decoys wrongly accepted | 0% | 0% | 1.5% |

The 24% was the thing that nearly killed this project. Give the model a tRNA
on its own and it folded it; put that same tRNA back where it actually lives
and the answer fell apart. After one pass over 15,589 animal mitochondrial
genomes, that collapse is gone.

It also beats the standard physics-based tool on both measurements, and it
still refuses to see a cloverleaf in scrambled sequence — so it has learned
what a real tRNA looks like, not "say cloverleaf to everything".

## The disease result did not hold up

Fine-tuning lifted disease prediction from 0.578 to 0.744 (where 0.5 is a coin
flip). That looks good until you ask the obvious question: the model just read
15,589 animal genomes, many of them mammals with tRNAs much like ours. Has it
learned anything, or has it just memorised **which letters rarely change
across animals**?

So we measured that directly. From the same genomes we counted, for every
position in every human tRNA, how often each letter appears across 259,964
tRNAs from 16,080 species. Then we scored the same disease variants with
nothing but that count.

| score | AUC |
| --- | --- |
| Minerva, fine-tuned | 0.744 |
| **Plain conservation counting** | **0.706** |
| ViennaRNA | 0.664 |
| Minerva, before fine-tuning | 0.578 |

Minerva is ahead — by +0.038, with a confidence interval of −0.052 to +0.124.
That interval crosses zero, so **we cannot claim Minerva beats simple
counting**. Adding Minerva on top of conservation gains +0.026, which also
crosses zero.

The two scores only agree at a rank correlation of 0.42, so they are genuinely
not the same signal. But with 52 disease variants to test on, we do not have
the evidence to say Minerva is better. Reporting 0.744 without this check
would have been the easiest kind of self-deception.

## What this means

We set out to predict which mutations cause disease. We did not succeed at
that. What we built instead is a tool that reads mitochondrial tRNA structure
accurately in its native genomic context, which the previous approach could
not do, and which conservation counting cannot do at all — conservation tells
you *whether* a position matters, never *what shape* the molecule takes.

So the useful thing we have is a structure reader. The next question is what
that is good for.
