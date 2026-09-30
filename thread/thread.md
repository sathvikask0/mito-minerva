# Twitter/X thread: mito-minerva

Images are in this `thread/` folder. Each tweet says which one to attach.
Character counts are checked (links count as 23 on X).

---

## 1/11
📎 attach: `1_hero_arcs.png`

```
I took an AI trained only on bacterial DNA and taught it to read human mitochondrial DNA.

It now works out how mitochondrial tRNAs fold far better than the standard tool, checked against lab-measured 3D structures.

Then I tried to break every claim. 🧵
```

## 2/11

```
Why this is interesting:

Mitochondria were once free-living bacteria. Their tRNAs are tiny molecules that must fold into a precise shape, and mutations that break that shape cause disease.

So: can a model trained on bacteria read them?
```

## 3/11
📎 attach: `4_training.png`

```
Out of the box: barely. On mitochondrial DNA it scored about as well as random guessing.

So I fine-tuned it (Minerva, an open 650M-param genome model) on 15,589 animal mitochondrial genomes, human held out.

1 GPU, ~4 hours, ~$10. Error on unseen genomes halved.
```

## 4/11
📎 attach: `3_context.png`

```
The big fix:

Before, it could fold a tRNA on its own, but put that tRNA back inside its real genome and it fell apart (40% → 28% of real pairs found).

After fine-tuning: 64% alone, 63% inside the genome. The surrounding DNA no longer confuses it.
```

## 5/11

```
How do you grade this without fooling yourself?

Not against my own idea of the shape. If two letters truly pair, evolution changes them together. Across 8,000–15,000 species per tRNA, those lockstep changes reveal the real pairs.

It even found a famously odd tRNA by itself.
```

## 6/11

```
But there was a catch.

Those species are the same genomes the model trained on. It could have learned the exact pattern I was grading it on.

So I found lab-measured 3D structures of human mitochondrial tRNAs (X-ray and cryo-EM) and graded against those instead.
```

## 7/11
📎 attach: `2_lab_structures.png`

```
Against 8 lab-solved tRNAs (132 base pairs):

Real pairs it finds: 94% vs the standard tool's 68%. A clear win.
Its pairs that are real: 67% vs 57%. Not a clear win.

It catches far more real pairs, but isn't more precise. It tends to stretch stems one pair too far.
```

## 8/11
📎 attach: `5_disease.png`

```
The result I killed: disease prediction.

Scoring known disease vs harmless mutations, it hit 0.74 (0.5 = coin flip). Looked great.

Then I built the boring baseline: just count how conserved each letter is across animals. 0.71.

Gap not significant. No claim.
```

## 9/11
📎 attach: `6_longevity.png`

```
Then a longevity idea: long-lived animals run their mitochondria for decades on DNA that's barely repaired. Maybe they evolved sturdier tRNAs?

I scanned 1,321 species living 2 to 211 years.

Compared with close relatives: no link at all (p = 0.15).
```

## 10/11

```
What I'd tell anyone doing ML for biology:

• Test a good number against the most boring explanation
• Don't grade a model against a definition you wrote
• Compare animals with their relatives, or you just measure "mammal vs fish"
• Measure costs, don't guess
```

## 11/11

```
Everything is open: code, every experiment as a commit, and a plain-language write-up. Total compute: $24.

Code: https://github.com/sathvikask0/mito-minerva
Write-up: https://sathvikask0.github.io/mito-minerva/
Model: https://huggingface.co/sathvikask/mito-minerva
```
