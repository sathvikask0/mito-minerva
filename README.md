# mito-minerva

Probing human mitochondrial DNA with [Minerva](https://github.com/garykbrixi/minerva),
a 650M-parameter genome language model trained on bacterial DNA.

Mitochondria descend from bacteria, so a bacterial genome model may transfer to
the mitochondrial genome. If it does, its single-sequence coevolution
predictions give a way to score how badly a given mutation disrupts a
mitochondrial tRNA — without alignments, and without a structure.

## Plan

1. **Build the input.** Convert the rCRS (NC_012920.1) to Minerva's mixed-token
   format under the vertebrate mitochondrial code, and check it fits the context. ✅
2. **Go / no-go.** Does the `base_pairing` head recover the cloverleaf folds of
   the 22 mitochondrial tRNAs? If not, finetune on animal mitochondrial genomes.
3. **Mutate.** Change each tRNA base in turn; score the disruption to the
   predicted fold and to the model's likelihood.
4. **Validate.** Compare those scores against MITOMAP pathogenic variants
   (e.g. m.3243A>G in tRNA-Leu(UUR)) versus benign population variants.

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

## Layout

```
src/mitominerva/mito.py     rCRS -> Minerva tokens, with rCRS coordinates preserved
src/mitominerva/sanity.py   reject silently-corrupted MPS predictions
scripts/step1_prepare_input.py
scripts/bench_device.py     how long an input this machine can handle
tests/test_mito.py          the token string really is the rCRS, base for base
data/processed/             the mixed-token string and the tRNA token spans
```
