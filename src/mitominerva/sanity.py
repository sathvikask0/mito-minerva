"""Guard against silently corrupted contact maps.

On Apple Silicon, a Metal command buffer that runs out of memory does not raise
in PyTorch -- it logs ``kIOGPUCommandBufferCallbackErrorOutOfMemory`` to stderr
and hands back a tensor full of whatever was in the buffer.  Observed failure
modes are an all-constant map (every entry at the 0.0018 sigmoid floor) and a
map where a huge fraction of entries sit above 0.5.  Both look like ordinary
arrays, so every prediction has to be checked before it is believed.
"""

from __future__ import annotations

import numpy as np

#: A healthy base_pairing map bottoms out here; an exactly-zero minimum means
#: the buffer was never written.
SIGMOID_FLOOR = 1e-4
#: Real base pairing is sparse: roughly one partner per position, so far under
#: 1% of an L x L map should be confident.
MAX_CONFIDENT_FRACTION = 0.01


class CorruptPredictionError(RuntimeError):
    pass


def check_contact_map(matrix, *, name: str = "contact map") -> np.ndarray:
    """Raise unless *matrix* looks like a real prediction."""
    a = np.asarray(matrix, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    frac = float((a > 0.5).mean())

    if not np.isfinite(a).all():
        raise CorruptPredictionError(f"{name}: contains NaN or inf")
    if lo < SIGMOID_FLOOR:
        raise CorruptPredictionError(f"{name}: minimum {lo:.2e} is below the sigmoid floor")
    if hi - lo < 1e-6:
        raise CorruptPredictionError(
            f"{name}: constant at {lo:.5f} -- the GPU buffer was never written "
            f"(out of memory at length {a.shape[-1]})"
        )
    if hi < 0.5:
        raise CorruptPredictionError(
            f"{name}: no entry above 0.5 (max {hi:.4f}) -- prediction is empty"
        )
    if frac > MAX_CONFIDENT_FRACTION:
        raise CorruptPredictionError(
            f"{name}: {frac:.1%} of entries above 0.5, far more than base pairing allows"
        )
    return a
