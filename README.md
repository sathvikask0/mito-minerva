# mito-minerva

Probing human mitochondrial DNA with [Minerva](https://github.com/garykbrixi/minerva),
a 650M-parameter genome language model trained on bacterial DNA.

Mitochondria descend from bacteria, so a bacterial genome model may transfer to
the mitochondrial genome. If it does, its single-sequence coevolution
predictions give a way to score how badly a given mutation disrupts a
mitochondrial tRNA — without alignments, and without a structure.

Site with the plain-language write-up: https://sathvikask0.github.io/mito-minerva/
Finetuned model (LoRA adapter): https://huggingface.co/sathvikask/mito-minerva

```python
from peft import PeftModel
# base = AutoModelForMaskedLM.from_pretrained("gbrixi/minerva-mlm-8k", trust_remote_code=True,
#                                             revision="df01967534e5414af838665f715fa2033f4c9012")
model = PeftModel.from_pretrained(base, "sathvikask/mito-minerva").merge_and_unload()
```

## Result

**One positive, two negatives.**

After LoRA finetuning on 15,589 animal mitochondrial genomes (human held out),
Minerva recovers base pairs of the 22 human mitochondrial tRNAs far better
than ViennaRNA, and genomic context no longer degrades it. Scored against 409
base pairs derived from covariation across 8,397–15,552 species per gene, 95%
intervals from a gene-level bootstrap:

| predictor | recall | 95% CI |
| --- | --- | --- |
| Minerva finetuned, tRNA alone | **63.8%** | 59.3–67.5 |
| Minerva finetuned, in genome | **62.8%** | 56.9–67.4 |
| ViennaRNA MFE | 40.8% | 33.0–48.5 |
| Minerva base, tRNA alone | 39.6% | 32.4–46.6 |
| Minerva base, in genome | 27.9% | 17.8–38.8 |

Finetuned minus ViennaRNA: **+23.0 points [+12.6, +32.6]**, better on 16 of 22
tRNAs. The `base_pairing` head (a 41-parameter logistic regression over the
last two layers' attention) was never retrained; only the attention changed.

The negatives:

- **Pathogenicity.** Finetuned masked-marginal LLR reaches AUC 0.744 on ClinVar,
  but a plain conservation count from the same corpus reaches 0.706; the
  difference is +0.038 [−0.052, +0.124] at 52 pathogenic variants.
- **Longevity.** Across 1,321 species, per-genome tRNA structural fragility does
  not track maximum lifespan once phylogeny is controlled (within family
  r=+0.05, p=0.15; within order r=−0.01, p=0.76).

**Independent check against experimental structures.** The covariation
reference shares its source genomes with the finetuning corpus, so the model
could have learned the statistics it is graded on. To rule that out, pairs
were read off experimental 3D models in the PDB (`scripts/step13_pdb_structures.py`):
8 human mt-tRNAs (H, I, M, Q, R, S2, V, Y) from 36 X-ray/cryo-EM chains at
1.9–4.3 Å, canonical pairs called from hydrogen-bond donor–acceptor distances
(≤3.6 Å), reduced to the maximum nested structure, majority vote across chains.
132 pairs; scored with `scripts/step14_grade_vs_pdb.py`, gene-level bootstrap:

| predictor | precision | recall | F1 |
| --- | --- | --- | --- |
| Minerva finetuned, tRNA alone | 67.0% | **93.9%** | **0.782** |
| Minerva finetuned, in genome | 66.7% | 81.8% | 0.735 |
| ViennaRNA MFE | 57.3% | 68.2% | 0.623 |
| Minerva base, tRNA alone | 64.3% | 61.4% | 0.628 |
| Minerva base, in genome | 58.1% | 32.6% | 0.417 |

Finetuned − ViennaRNA: recall **+25.8 [+6.1, +45.5]**, precision +9.7
[−4.3, +23.5], F1 +0.159 [−0.003, +0.319] (without the partially modelled
TRNS2: +0.171 [+0.004, +0.342]). Finetuned − base, F1: +0.154 [+0.068, +0.246].
The recall advantage replicates on independent ground truth; precision does
not improve, and 69% of the finetuned model's false positives stack directly
on a true helix, i.e. stems over-extended by a pair. Limits: n=8 tRNAs, and
cryo-EM tRNA models can carry template-derived assumptions.

The model code is fetched with `trust_remote_code`; results are pinned to
Hub revision `df01967` (`src/mitominerva/loading.py`).

Total compute: $24.35 of Modal credit (one A100 finetune, L4 inference).

## Plan

1. **Build the input.** Convert the rCRS (NC_012920.1) to Minerva's mixed-token
   format under the vertebrate mitochondrial code, and check it fits the context. ✅
2. **Go / no-go.** Does the `base_pairing` head recover the cloverleaf folds of
   the 22 mitochondrial tRNAs? ✅ yes, but only on near-isolated genes — see below.
3. **Mutate.** ✅ 4,524 point mutations scored two ways.
4. **Validate.** ❌ Against ClinVar, base Minerva does not beat ViennaRNA and
   neither is accurate enough to use. See below.
5. **Finetune** on animal mitochondrial genomes. ✅ Held-out loss 1.476 → 0.744;
   genomic-context collapse fixed.
6. **Is the disease score just conservation?** ❌ Can't distinguish it.
7. **Grade against covariation, not geometry.** ✅ 63.8% vs ViennaRNA 40.8%.
8. **Context probe.** rRNA adjacency is not what breaks TRNV; the full 8k
   window is. Isolated folding is within 0.5 points of the best flank.
9. **Longevity scan** over 1,321 species. ❌ No association after phylogenetic
   control.
10. **Experimental structures.** ✅ Recall advantage over ViennaRNA replicates on
    8 PDB-derived tRNA structures (+25.8 points); F1 advantage borderline.

## Setup

```bash
uv venv --python 3.12
uv pip install -e .
```

## Step 1 — the input

```bash
.venv/bin/python scripts/step1_prepare_input.py
```

Minerva's input is one string: proteins as upper-case amino acids, everything
else as lower-case DNA, each segment prefixed with `<+>` or `<->`. Since tRNAs
and rRNAs are not CDS features, they stay as raw DNA — exactly the form the
`base_pairing` head reads.

The whole 16,569 bp circle comes to **9,040 tokens**, which overruns the 8,192
context of `minerva-mlm-8k` by 848. The fix is to cut the circle at the control
region: the D-loop (`join(16024..16569, 1..576)`) holds no tRNA, rRNA or CDS, so
dropping it is the one cut that severs nothing of interest.

| window | bases | tokens | fits 8k? |
| --- | --- | --- | --- |
| full circle 1..16569 | 16,569 | 9,040 | no, over by 848 |
| **coding window 577..16023** | 15,447 | **7,918** | **yes** |

The coding window still carries all 13 CDS, both rRNAs and **all 22 tRNAs**,
from tRNA-Phe (577) to tRNA-Pro (16023).

Two details worth knowing:

- rCRS position 3107 is an `N` — a historical placeholder kept so that numbering
  matches the original Cambridge sequence. Minerva's vocabulary has no `n`, so it
  becomes a single `<unk>`. It sits inside the 16S rRNA, far from any tRNA, and
  we keep it so our coordinates stay rCRS coordinates.
- ATP8/ATP6 and ND4L/ND4 overlap in two reading frames. With
  `overlap_mode="expand"` both proteins are emitted, so 47 bases appear twice in
  the token string. This is intended; `tests/test_mito.py` pins it down.

`mitominerva.mito` rebuilds the map from every character of the token string
back to its rCRS position, which is what lets step 3 point at a specific base.
The tests check that mapping against the reference base by base.

```bash
.venv/bin/python -m pytest tests/ -q
```

## Running it: this Mac vs. a GPU

`predict_contacts` materialises the full `[1, 20, L, L]` attention map for each
of the two layers the head reads, so memory grows with L². Measured on an M3 Pro
with 18 GB unified memory, `minerva-mlm-8k`:

| tokens | MPS float32 | MPS float16 | CPU float32 |
| --- | --- | --- | --- |
| 4,096 | 2.9 s | 2.3 s | 15.2 s |
| 6,144 | 15.5 s | 3.6 s | |
| 7,168 | out of memory | 5.3 s | |
| 7,552 | out of memory | out of memory | |
| **7,918 (full window)** | out of memory | out of memory | **187 s** |

float16 tracks float32 closely on MPS (correlation 0.99998, max difference 0.03
at L=4096); bfloat16 does not (max difference 0.26) and should not be used. The
CPU path agrees with MPS to correlation 0.999985, with 1018 of 1020 confident
pairs identical, so it is a trustworthy fallback and not just a slower one.

**The whole-genome run does fit on this machine, on the CPU, in about three
minutes.** GPU memory tops out near 7,168 tokens, but the CPU is not bound by
it. So step 2 needs no GPU: run the full 7,918-token window on the CPU once.
A GPU is still worth having for step 3, where roughly 4,600 forward passes over
mutated tRNAs would take hours locally.

### MPS fails silently

When a Metal command buffer runs out of memory, PyTorch does not raise. It
prints `kIOGPUCommandBufferCallbackErrorOutOfMemory` to stderr and returns a
tensor that was never written — in practice a map pinned flat at the 0.0018
sigmoid floor, which is easy to mistake for a real, confidently-negative
prediction. `mitominerva.sanity.check_contact_map` rejects those; call it on
every prediction made on this machine.

## Step 2 — does Minerva fold the mitochondrial tRNAs?

```bash
.venv/bin/python scripts/step2_predict.py     # whole window, ~2 min on CPU
.venv/bin/python scripts/step2_benchmark.py   # the go / no-go table
.venv/bin/python scripts/step2_context.py     # why context matters
.venv/bin/python scripts/step2_plot.py        # outputs/step2_trna_maps_isolated.png
```

### Ground truth

No curated per-tRNA reference structure is used. The target is the cloverleaf's
geometry, anchored on two things known independently of any prediction:

- the **acceptor stem**, 7 bp joining the first 7 bases to the last 7. Because
  mitochondrial tRNA genes do not encode the 3' CCA, base `i` should pair with
  base `L - 2 - i`.
- the **anticodon stem**, 5 bp flanking the anticodon loop. The anticodon is
  known from the amino acid, so it is located by sequence. 21 of the 22 sit
  immediately after the invariant U33, which is what the search keys on.

Everything is read in transcript orientation: 8 of the 22 tRNAs are on the
light strand, so the model was shown the reverse complement of their transcript.

### Result

| predictor | tRNA acceptor stem | tRNAs ≥5/7 | decoys ≥5/7 |
| --- | --- | --- | --- |
| Minerva, gene isolated | 111/154 (72%) | 17/22 | 0.5% |
| Minerva, whole-genome window | 37/154 (24%) | 5/22 | 0.0% |
| ViennaRNA MFE | 128/154 (83%) | 19/22 | 1.5% |

Anticodon stem: **Minerva 55%, ViennaRNA 35%**. 95% of the pairs Minerva calls
are Watson-Crick or G-U wobble.

Decoys are 200 length-matched windows drawn from the same genome, outside every
tRNA, from non-protein territory. They score essentially zero, so Minerva is not
simply folding the ends of any short sequence together — when it calls a
cloverleaf, it means it.

**Verdict: the model transfers, but the whole-genome premise does not.** Minerva
folds isolated mitochondrial tRNAs about as well as thermodynamics does, and
noticeably better at the anticodon arm, which is the part step 4 cares about
most. What fails is reading those folds out of a whole-genome pass.

### Context destroys the signal

The same tRNAs, folded with increasing amounts of their real flanking sequence:

| flanking sequence | acceptor stem | tRNAs ≥5/7 |
| --- | --- | --- |
| none (gene only) | 106/154 (69%) | 16/22 |
| ±50 nt | 46/154 (30%) | 4/22 |
| ±200 nt | 42/154 (27%) | 4/22 |
| ±1000 nt | 33/154 (21%) | 4/22 |
| whole 7,918-token window | 37/154 (24%) | 5/22 |

Fifty bases of real genomic flank is enough to lose most of it. Five tRNAs
(Leu(UUR), Gln, Asn, Ser(UCN), Thr) hold their fold at every context width and
give a textbook cloverleaf in the whole-genome map; the rest collapse.

This is worth stating plainly because it inverts the original motivation. The
appeal of mtDNA was that the whole genome fits one 8k window — but the
whole-window pass is the regime where the model does *worst*. Steps 3 and 4 have
to fold tRNAs on their own, which is cheap: 70-token inputs run in milliseconds,
so no GPU is needed for the mutation scan either.

## Steps 3 and 4 — mutate, then validate

```bash
.venv/bin/python scripts/step3_mutate.py         # 4,524 mutants, ~2 min
.venv/bin/python scripts/step3b_context_llr.py   # likelihood with flanks
.venv/bin/python scripts/step4_fetch_clinvar.py  # labels from NCBI
.venv/bin/python scripts/step4_evaluate.py
```

Every base of every tRNA was changed to each alternative and scored two ways:
the fraction of wild-type base pairs lost on refolding, and the masked
log-likelihood ratio against the wild-type base. The second has no ViennaRNA
equivalent and was the reason to reach for a language model.

MITOMAP is behind a bot wall, so labels come from ClinVar via the same
E-utilities API used for the reference: **363 substitutions inside the 22
tRNAs, 52 pathogenic and 311 benign.**

### Result: no separation worth having

AUC, with bootstrap 95% CIs over 2,000 resamples:

| score | AUC | 95% CI |
| --- | --- | --- |
| Minerva, pairs lost | 0.526 | 0.441–0.609 |
| Minerva, stem pairs lost | 0.574 | 0.505–0.643 |
| Minerva, masked LLR | 0.578 | 0.489–0.674 |
| Minerva, LLR in ±300 nt context | 0.485 | 0.391–0.584 |
| **Minerva, best combination** | **0.601** | 0.524–0.674 |
| ViennaRNA, base-pair distance | 0.524 | 0.446–0.605 |
| **ViennaRNA, ΔΔG** | **0.664** | 0.581–0.741 |

Best Minerva minus best ViennaRNA: **−0.063**, CI −0.164 to +0.043. So Minerva
is not ahead; whether it is genuinely behind is not resolved at this sample
size. Restricting to `Pathogenic` and `Benign` only (dropping every "Likely")
gives the same ordering with wider intervals.

Published predictors for this task report AUCs near 0.9, so neither number here
is usable.

### Why

The pitch was alignment-free scoring from a single sequence. The tools that
reach 0.9 are conservation-based — they ask whether a base held still across
evolutionary time — and that appears to carry most of the signal. Minerva has
no access to it.

Adding genomic context made the likelihood score *worse*, down to chance. That
is the same fragility step 2 found in the folding head, now in the language
modelling head, and it is the clearest sign that mitochondrial DNA is out of
distribution for a bacterially-trained model.

### What is actually true here

Step 2 is a real positive: a model trained only on bacteria folds human
mitochondrial tRNAs it has never seen, and beats thermodynamics at the
anticodon arm. Step 4 is a real negative: that ability does not transfer to
ranking mutations by pathogenicity. Both results are reproducible from this
repo.

So we took the fallback from the original plan and finetuned on animal
mitochondrial genomes.

## Steps 5–9 — finetune, then try hard to disprove it

**Finetuning** (`scripts/modal_train.py`, A100-80GB, 3h51m, ~$10). LoRA r=8 on
`wqkv, wo, w1, w2, w3`; 15,589 RefSeq Metazoa genomes, human held out; 300
genomes held out for validation *before* tiling into 4,096-token blocks, so no
genome leaks across the split. One epoch, 2,626 steps. Validation loss fell
monotonically 1.476 → 0.741 with train and validation overlapping throughout.
Two gotchas: Minerva refuses padded batches without flash-attn, so every block
is exactly `block_size` with the last one back-shifted; and transformers skips
its gradient-accumulation normalisation because Minerva's `forward` takes
`**kwargs`, which silently doubled loss and gradient until
`trainer.model_accepts_loss_kwargs = False`.

**Conservation check** (`scripts/step6_*`). Per-position allele counts over
259,964 animal tRNAs aligned to human. Conservation alone scores 0.706 against
the finetuned model's 0.744; not separable at n=52.

**Covariation reference** (`scripts/step7_covariation.py`,
`scripts/step8_validate_structure.py`). APC-corrected mutual information
between alignment columns, keeping pairs that are Watson-Crick/GU in ≥90% of
species, greedily one partner per position. It recovers the known D-arm loss of
mt-tRNA-Ser(AGY) blind (TRNS2: 7 pairs; every other tRNA 10–26). Intervals in
`scripts/step12_confidence.py`.

**Context probe** (`scripts/step9_context_probe.py`). Recall against flank
size, with dinucleotide-shuffled flanks as control: 63.8% (0 nt), 64.3%
(200 nt), 53.8% (800 nt); real flanks beat shuffled at every large size. TRNV
scores 10/13 even with 800 nt of real rRNA each side, yet 0/13 in the full
7,918-token window — so the failure is the long window, not rRNA.

**Longevity** (`scripts/step10_build_windows.py`, `scripts/modal_fragility.py`,
`scripts/step11_longevity.py`). 1,334 species with both an AnAge maximum
lifespan and a well-annotated genome; every base of every tRNA mutated to all
three alternatives, fragility = mean fraction of predicted pairs lost. Controls
for log body mass and GC (GC alone correlates r=−0.29 with fragility) and for
phylogeny by centring within family and order, with permutation inside those
groups. The raw association is weak and in the wrong direction (r=+0.08); it
vanishes within groups. A reptile signal (r=−0.48) is turtles vs squamates and
disappears within order.

## Layout

```
src/mitominerva/mito.py     rCRS -> Minerva tokens, with rCRS coordinates preserved
src/mitominerva/sanity.py   reject silently-corrupted MPS predictions
src/mitominerva/cloverleaf.py  score a map against the cloverleaf geometry
scripts/step1_prepare_input.py
scripts/step2_predict.py    whole-window inference, cached to outputs/
scripts/step2_benchmark.py  Minerva vs ViennaRNA vs decoys -- the go / no-go
scripts/step2_context.py    how recovery decays with flanking sequence
scripts/step2_plot.py       contact maps for all 22 tRNAs
scripts/step3_mutate.py     every point mutation, scored two ways
scripts/step3b_context_llr.py  the likelihood score with flanking sequence
scripts/step4_fetch_clinvar.py labelled variants from NCBI
scripts/step4_evaluate.py   AUC against ClinVar, with ViennaRNA as baseline
scripts/step5_build_corpus.py  15,589 animal mitochondrial genomes, human held out
scripts/modal_train.py      LoRA finetune on Modal (A100)
scripts/step6_*.py          conservation baseline for the disease score
scripts/step7_covariation.py   base pairs from covariation across species
scripts/step8_validate_structure.py  recall of covariation pairs
scripts/step9_context_probe.py  recall vs flank size, real vs shuffled
scripts/step10_build_windows.py  per-species tRNAs for the longevity scan
scripts/modal_fragility.py  per-species mutational fragility on Modal (L4)
scripts/step11_longevity.py fragility vs lifespan with phylogenetic control
scripts/step12_confidence.py   gene-level bootstrap on the headline numbers
scripts/step13_pdb_structures.py  base pairs from experimental PDB structures
scripts/step14_grade_vs_pdb.py    precision/recall/F1 against those structures
src/mitominerva/loading.py  load Minerva with a LoRA adapter merged in
scripts/build_site.py       rebuilds docs/ from notes/ and outputs/
scripts/bench_device.py     how long an input this machine can handle
notes/                      the plain-language write-up behind the site
docs/                       the published site (GitHub Pages)
tests/test_mito.py          the token string really is the rCRS, base for base
data/processed/             the mixed-token string and the tRNA token spans
```
