"""How fragile is each species' mitochondrial tRNA set?

The longevity question: a long-lived animal has to keep its mitochondria
working for decades on DNA that mutates fast and is barely repaired. So it
may be under pressure to carry tRNAs whose *shape* survives a random letter
change, not merely tRNAs that work when perfect.

For each species this mutates every base of every annotated tRNA and asks how
often the predicted structure falls apart. Each tRNA is folded inside +/-200nt
of its own genomic context, which step 9 measured as the best-scoring setting
(64.3%, above both isolated and whole-genome) and which is far cheaper than a
whole-genome pass.

    modal run scripts/modal_fragility.py --limit 10      # benchmark
    modal run --detach scripts/modal_fragility.py        # the real scan
"""
from __future__ import annotations

import modal

APP_NAME = "mito-fragility"
MODEL_ID = "gbrixi/minerva-mlm-8k"
FLANK = 200
CONFIDENT = 0.5

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0", "transformers==5.17.0", "peft==0.21.0",
        "biopython", "numpy", "minerva-dna==0.1.0", "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)

app = modal.App(APP_NAME, image=image)
corpus_vol = modal.Volume.from_name("mito-corpus")
runs_vol = modal.Volume.from_name("mito-runs")
hf_vol = modal.Volume.from_name("mito-hf-cache", create_if_missing=True)
VOLUMES = {"/corpus": corpus_vol, "/runs": runs_vol, "/hf": hf_vol}

COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def revcomp(s):
    return "".join(COMPLEMENT.get(c, "N") for c in reversed(s.upper()))


@app.function(gpu="L4", volumes=VOLUMES, timeout=6 * 60 * 60)
def scan(limit: int | None = None, batch_size: int = 128, flank: int = 0,
         adapter_dir: str = "/runs/lora-r8/adapter", run_name: str = "fragility-full"):
    import json, os, time
    import numpy as np
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from peft import PeftModel

    print(torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model = model.to(torch.bfloat16).cuda().eval()

    def fold_many(seqs, core_lo, core_len):
        """Contact maps for a batch of equal-length windows, cropped to core."""
        out = []
        for s in range(0, len(seqs), batch_size):
            chunk = seqs[s:s + batch_size]
            ids = tok([f"<+>{x.lower()}" for x in chunk],
                      return_tensors="pt")["input_ids"].cuda()
            with torch.inference_mode():
                r = model.predict_contacts(input_ids=ids, head_names=["base_pairing"])
            mat = r["predictions"]["base_pairing"] if isinstance(r, dict) else r
            mat = mat.float().cpu().numpy()
            if mat.ndim == 2:  # the batch axis is squeezed for a single sequence
                mat = mat[None]
            lo = core_lo + 1  # +1 for the <+> marker
            out.append(mat[:, lo:lo + core_len, lo:lo + core_len])
        return np.concatenate(out, axis=0)

    # Results are appended one species at a time and the volume is committed
    # every few species. The first full run saved only at the end, so when it
    # was cancelled 520 species in, all of that work was lost.
    os.makedirs(f"/runs/{run_name}", exist_ok=True)
    out_path = f"/runs/{run_name}/fragility.jsonl"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as prev:
            for l in prev:
                try:
                    done.add(json.loads(l)["accession"])
                except Exception:
                    pass  # a line cut off mid-write by a cancellation
    print(f"resuming: {len(done)} species already done", flush=True)
    out_fh = open(out_path, "a")

    records, t0, new_n = [], time.time(), 0
    with open("/corpus/trna_windows.jsonl") as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                break
            sp = json.loads(line)
            if sp["accession"] in done:
                continue
            per_trna, t_sp = [], time.time()
            for t in sp["trnas"]:
                core = t["seq"]
                # The windows are stored with 200nt flanks; trim to the size
                # being used. Step 9 measured 63.8% recall with no flank and
                # 64.3% with 200 -- half a point, for 6.7x the tokens.
                left = t["left"][len(t["left"]) - flank:] if flank else ""
                right = t["right"][:flank] if flank else ""
                L = len(core)
                win = left + core + right
                wt = fold_many([win], len(left), L)[0]
                wt_pairs = [(i, j) for i in range(L) for j in range(i + 1, L)
                            if wt[i, j] >= CONFIDENT]
                if not wt_pairs:
                    continue
                muts = [(i, b) for i in range(L) for b in "ACGT" if b != core[i]]
                wins = [left + core[:i] + b + core[i + 1:] + right for i, b in muts]
                maps = fold_many(wins, len(left), L)
                lost = [
                    sum(1 for p, q in wt_pairs if mm[p, q] < CONFIDENT) / len(wt_pairs)
                    for mm in maps
                ]
                per_trna.append({
                    "product": t["product"], "length": L,
                    "wt_pairs": len(wt_pairs),
                    "mean_fraction_lost": float(np.mean(lost)),
                    "fraction_breaking": float(np.mean([x >= 0.5 for x in lost])),
                })
            if per_trna:
                rec = {
                    "accession": sp["accession"], "organism": sp["organism"],
                    "lineage": sp["lineage"], "n_trnas": len(per_trna),
                    "fragility": float(np.mean([t["mean_fraction_lost"] for t in per_trna])),
                    "breaking": float(np.mean([t["fraction_breaking"] for t in per_trna])),
                    "gc": sp.get("gc"), "trnas": per_trna,
                }
                records.append(rec)
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                new_n += 1
                if new_n % 20 == 0:
                    runs_vol.commit()
            if new_n and new_n % 10 == 0:
                el = time.time() - t0
                print(f"  {len(done) + new_n} species ({new_n} this run), {el:.0f}s, "
                      f"{el / new_n:.2f}s/species", flush=True)
            elif limit and limit <= 10:
                print(f"  {sp['organism']}: {time.time() - t_sp:.2f}s, "
                      f"{len(per_trna)} tRNAs", flush=True)

    out_fh.close()
    runs_vol.commit()
    elapsed = time.time() - t0
    per = elapsed / max(1, len(records))

    print(f"\n{len(records)} species in {elapsed:.0f}s = {per:.2f}s/species")
    print(f"projected 1,334 species: {1334 * per / 3600:.2f} h "
          f"= ${1334 * per / 3600 * 0.80:.2f} on an L4")
    return {"species": len(records), "seconds": elapsed, "per_species": per}


@app.local_entrypoint()
def main(limit: int = 0, flank: int = 0, batch_size: int = 128, wait: bool = False):
    """Spawn the scan so it outlives this client.

    .remote() blocks on the call, and when the local client died -- laptop
    asleep, shell closed -- Modal cancelled the in-flight input even under
    --detach. That killed two full runs, at 520 and 95 species. A spawned
    call is not tied to its caller.
    """
    kw = dict(limit=limit or None, flank=flank, batch_size=batch_size)
    if wait:
        r = scan.remote(**kw)
        print(f"\n{r['species']} species, {r['per_species']:.2f} s/species")
    else:
        call = scan.spawn(**kw)
        print(f"spawned {call.object_id}; safe to disconnect")
