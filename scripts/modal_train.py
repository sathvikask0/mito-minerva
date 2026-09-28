"""LoRA-finetune Minerva on animal mitochondrial genomes, on a rented GPU.

Minerva was pretrained on bacterial DNA. Steps 2-4 showed it can fold an
isolated mitochondrial tRNA but loses the signal once the tRNA sits in genomic
context, and that its mutation scores do not beat ViennaRNA. The hypothesis
here is that the model has simply never seen a mitochondrial genome. This job
shows it 15,589 of them.

Usage (from the repo root, with the venv active):

    modal run scripts/modal_train.py --smoke      # ~5 min, proves the pipeline
    modal run --detach scripts/modal_train.py     # the real run

Outputs land on the ``mito-runs`` volume; pull them with

    modal volume get mito-runs /<run-name> outputs/finetune
"""
from __future__ import annotations

import modal

APP_NAME = "mito-minerva-train"
MODEL_ID = "gbrixi/minerva-mlm-8k"

# Pinned to the versions the laptop half of this project is developed against,
# so a result from the GPU is reproducible locally.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "datasets==4.8.5",
        "accelerate",
        "biopython",
        "minerva-dna==0.1.0",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)

app = modal.App(APP_NAME, image=image)

corpus_vol = modal.Volume.from_name("mito-corpus")
runs_vol = modal.Volume.from_name("mito-runs")
hf_vol = modal.Volume.from_name("mito-hf-cache", create_if_missing=True)

VOLUMES = {"/corpus": corpus_vol, "/runs": runs_vol, "/hf": hf_vol}


def build_blocks(path, tokenizer, block_size, val_genomes, limit=None):
    """Read the corpus, hold out whole genomes, tile the rest into blocks.

    Splitting by genome *before* tiling matters: blocks from one mitochondrion
    are near-identical to each other, so letting them straddle the train/val
    boundary would make validation loss meaningless.
    """
    import json

    from datasets import Dataset, DatasetDict

    records = []
    with open(path) as fh:
        for line in fh:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    print(f"read {len(records):,} genomes from {path}", flush=True)

    # Deterministic holdout so reruns compare against the same genomes.
    import random

    rng = random.Random(42)
    rng.shuffle(records)
    val, train = records[:val_genomes], records[val_genomes:]
    print(f"train {len(train):,} genomes / validation {len(val):,} genomes", flush=True)

    def tile(recs):
        """Cut each genome into blocks of exactly ``block_size`` tokens.

        Uniform length is not cosmetic: without flash-attn installed, Minerva
        refuses a batch containing padding (modeling_minerva.py:487). Rather
        than drop each genome's short tail, the final block is back-shifted to
        end at the sequence end, so every token is seen and only a little of
        the tail is seen twice.
        """
        out, skipped = [], 0
        for rec in recs:
            # Strand markers are 3 chars but 1 token; collapse them so the
            # slicing below counts tokens, then restore.
            packed = rec["text"].replace("<+>", "\x01").replace("<->", "\x02")
            n = len(packed)
            if n < block_size:
                skipped += 1
                continue
            starts = list(range(0, n - block_size + 1, block_size))
            if starts[-1] + block_size < n:
                starts.append(n - block_size)
            for i in starts:
                out.append(
                    packed[i : i + block_size]
                    .replace("\x01", "<+>")
                    .replace("\x02", "<->")
                )
        if skipped:
            print(f"skipped {skipped} genomes shorter than {block_size} tokens", flush=True)
        return out

    splits = DatasetDict(
        {
            "train": Dataset.from_dict({"text": tile(train)}),
            "validation": Dataset.from_dict({"text": tile(val)}),
        }
    )
    print(
        f"blocks of {block_size}: train {len(splits['train']):,} / "
        f"validation {len(splits['validation']):,}",
        flush=True,
    )

    def encode(batch):
        # No special_tokens_mask: the collator only needs input_ids, and a
        # stray column would ride through the collator into model(**inputs).
        return tokenizer(batch["text"], truncation=True, max_length=block_size)

    return splits.map(
        encode,
        batched=True,
        num_proc=8,
        remove_columns=["text"],
        desc="tokenizing",
    )


@app.function(
    gpu="A100-80GB",
    volumes=VOLUMES,
    timeout=8 * 60 * 60,
)
def train(
    smoke: bool = False,
    block_size: int = 4096,
    batch_size: int = 8,
    grad_accum: int = 2,
    epochs: float = 3.0,
    lr: float = 1e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    run_name: str = "lora-r8",
):
    import json
    import os
    import time

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments
    from minerva.finetuning import MinervaTrainer, apply_lora
    from minerva.masking import DataCollatorForMinervaMLM

    print(torch.cuda.get_device_name(0), flush=True)

    if smoke:
        run_name, epochs = "smoke", 1.0

    out_dir = f"/runs/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16
    )

    ds = build_blocks(
        "/corpus/metazoa_mito.jsonl",
        tok,
        block_size,
        val_genomes=20 if smoke else 300,
        limit=200 if smoke else None,
    )

    model = apply_lora(model, r=lora_r, alpha=lora_alpha)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # The collator writes pad_token_id (not -100) into the unmasked label
    # positions, so the loss has to ignore that id or it trains the model to
    # emit padding everywhere.
    collator = DataCollatorForMinervaMLM(tokenizer=tok, mask_prob=0.30)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        max_steps=30 if smoke else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10 if smoke else 200,
        bf16=True,
        logging_steps=10 if smoke else 50,
        eval_strategy="steps",
        eval_steps=10 if smoke else 500,
        save_strategy="steps",
        save_steps=10 if smoke else 500,
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="none",
        seed=42,
    )

    trainer = MinervaTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        ignore_token_id=tok.pad_token_id,
    )

    t0 = time.time()
    baseline = trainer.evaluate()
    print(f"before training: {baseline}", flush=True)

    result = trainer.train()
    final = trainer.evaluate()
    elapsed = time.time() - t0

    trainer.save_model(f"{out_dir}/adapter")
    tok.save_pretrained(f"{out_dir}/adapter")

    summary = {
        "run_name": run_name,
        "smoke": smoke,
        "model": MODEL_ID,
        "block_size": block_size,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "epochs": epochs,
        "lr": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_blocks": len(ds["train"]),
        "val_blocks": len(ds["validation"]),
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "eval_loss_before": baseline["eval_loss"],
        "eval_loss_after": final["eval_loss"],
        "gpu": torch.cuda.get_device_name(0),
        "wall_seconds": round(elapsed, 1),
    }
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    runs_vol.commit()

    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    smoke: bool = False,
    epochs: float = 1.0,
    block_size: int = 4096,
    batch_size: int = 8,
):
    summary = train.remote(
        smoke=smoke, epochs=epochs, block_size=block_size, batch_size=batch_size
    )
    before, after = summary["eval_loss_before"], summary["eval_loss_after"]
    print(
        f"\nheld-out loss {before:.4f} -> {after:.4f} "
        f"({after - before:+.4f}) in {summary['wall_seconds'] / 60:.1f} min"
    )
