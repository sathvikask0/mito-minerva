"""Load Minerva, optionally with a finetuned LoRA adapter merged in.

Steps 2-4 all reach past the language-model head into ``predict_contacts`` and
the interaction heads. A PEFT wrapper hides those methods, so the adapter is
merged back into the base weights rather than left wrapped -- the returned
object is an ordinary ``MinervaForMaskedLM`` that every existing script can
use unchanged.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODEL = "gbrixi/minerva-mlm-8k"


def load_model(
    model_id: str = DEFAULT_MODEL,
    device: str = "cpu",
    dtype: str = "float32",
    adapter: str | None = None,
):
    """Return ``(tokenizer, model)`` ready for inference.

    ``adapter`` is a directory holding a PEFT adapter (what step 5 writes).
    When given, the base checkpoint is loaded in float32 before merging so the
    merge arithmetic does not happen in half precision, then cast to ``dtype``.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    target = getattr(torch, dtype)
    load_dtype = torch.float32 if adapter else target

    model = AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=load_dtype
    )

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        model = model.to(target)

    return tok, model.to(device).eval()
