import jax
import jax.numpy as jnp


def build_rope(seq_len: int, head_dim: int, theta: float = 100000.0) -> tuple[jax.Array, jax.Array]:
    """Return (cos, sin) each shaped (seq_len, head_dim//2), float32."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    dim = jnp.arange(head_dim // 2, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (2.0 * dim / head_dim))
    freqs = jnp.outer(pos, inv_freq)  # (L, D/2)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate-half RoPE. x: (..., seq_len, n_heads, head_dim), cos/sin: (seq_len, head_dim//2)."""
    # move seq dim for broadcasting: cos/sin -> (seq_len, 1, D/2)
    seq_len = x.shape[-3]
    cos = cos[:seq_len, None, :]
    sin = sin[:seq_len, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    # upcast for stability, cast back
    dtype = x.dtype
    x1 = x1.astype(jnp.float32)
    x2 = x2.astype(jnp.float32)
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    _ = rotated  # keep rotate_half semantics explicit
    return out.astype(dtype)
