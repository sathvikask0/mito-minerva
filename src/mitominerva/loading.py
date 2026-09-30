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
# The checkpoint loads with trust_remote_code, so an upstream commit can change
# the code that runs. This is the revision every result here was produced with;
# a later upstream commit (ba9bba9) left all eight files we download unchanged.
DEFAULT_REVISION = "df01967534e5414af838665f715fa2033f4c9012"


def load_model(
    model_id: str = DEFAULT_MODEL,
    device: str = "cpu",
    dtype: str = "float32",
    adapter: str | None = None,
    revision: str = DEFAULT_REVISION,
):
    """Return ``(tokenizer, model)`` ready for inference.

    ``adapter`` is a directory holding a PEFT adapter (what step 5 writes).
    When given, the base checkpoint is loaded in float32 before merging so the
    merge arithmetic does not happen in half precision, then cast to ``dtype``.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    target = getattr(torch, dtype)
    load_dtype = torch.float32 if adapter else target

    model = AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=load_dtype, revision=revision
    )

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        model = model.to(target)

    return tok, model.to(device).eval()
