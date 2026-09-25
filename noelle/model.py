"""Model selection and execution, independent of the input pipeline.

Architectures belong here. The pipeline owns text preparation and result
formatting; callers never need to instantiate the underlying model class.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import jax
import jax.numpy as jnp

from noelle.exp.dec.smollm import SmollmSystemOne


class BackboneFamily(Enum):
    EB = "eb"  # encoder backbone
    DB = "db"  # decoder backbone


class Arch(Enum):
    SmollmDB135m = 0

    @property
    def family(self) -> BackboneFamily:
        if self is Arch.SmollmDB135m:
            return BackboneFamily.DB
        raise ValueError(f"unsupported architecture: {self}")


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "recovered/rlcd-final.safetensors"


@jax.jit
def _smollm_logits(model: SmollmSystemOne, ids, ids_mask, ctx_ids, ctx_mask):
    return model(ids, ids_mask=ids_mask, ctx_ids=ctx_ids, ctx_mask=ctx_mask)[0, :, 0]


class Runtime:
    """Load an architecture and execute it on already prepared JAX arrays."""

    def __init__(self, arch: Arch = Arch.SmollmDB135m, *, checkpoint: str | Path | None = None):
        if not isinstance(arch, Arch):
            raise TypeError(f"arch must be an Arch value, got {arch!r}")
        self.arch = arch
        path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        if arch is Arch.SmollmDB135m:
            self._model = SmollmSystemOne.init(path)
        else:
            raise ValueError(f"unsupported architecture: {arch}")

    def logits(self, ids, ids_mask, ctx_ids, ctx_mask, *, choice_count: int) -> jax.Array:
        """Return one logit for each real choice, preserving input order."""
        if self.arch is Arch.SmollmDB135m:
            values = _smollm_logits(self._model, ids, ids_mask, ctx_ids, ctx_mask)
            return values[:choice_count].astype(jnp.float32)
        raise ValueError(f"unsupported architecture: {self.arch}")
