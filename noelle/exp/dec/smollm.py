import pathlib
import typing as tp

import jax
import jax.numpy as jnp
from safetensors.flax import load_file
from taktiny import nn
from taktiny.utils.typing import QuantConfig

from noelle.exp.utils import apply_rope, build_rope


class SmollmConfig:
  dim: int = 576
  inter_dim: int = 1536
  max_pos_emb: int = 8192
  head_dim: int = 64
  n_heads: int = 9
  n_layers: int = 30
  n_kv_heads: int = 3
  epsilon: float = 1e-05
  rope_theta: float = 100000
  dtype: str = "bfloat16"
  wte_size: int = 49152

@nn.module
class MLP:
    def __init__(self, config: SmollmConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.w1_proj = nn.Linear(
            config.dim,
            2 * config.inter_dim,
            rngs=rngs,
            bias=False,
            quant=quant,
            dtype=config.dtype
        )
        self.w2_proj = nn.Linear(
            config.inter_dim,
            config.dim,
            rngs=rngs,
            bias=False,
            quant=quant,
            dtype=config.dtype,
        )

    def __call__(self, x: jax.Array):
        z = self.w1_proj(x)
        z1, z2 = jnp.split(z, 2, axis=-1)
        return self.w2_proj(jax.nn.silu(z1) * z2)

@nn.module
class Attention:
    def __init__(self, config: SmollmConfig, *, rngs: nn.Rngs, quant: QuantConfig = None, sep_q: bool=False):
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.repeat = config.n_heads // config.n_kv_heads
        self.scale = config.head_dim ** -0.5
        if sep_q:
            self.q_proj = nn.Linear(
                config.dim,
                (config.n_heads, config.head_dim),
                bias=False,
                rngs=rngs,
                dtype=config.dtype,
                quant=quant
            )
            self.kv_proj = nn.Linear(
                config.dim,
                (2 * config.n_kv_heads, config.head_dim),
                bias=False,
                rngs=rngs,
                dtype=config.dtype,
                quant=quant
            )
            self.qkv_proj = None
        else:
            self.qkv_proj = nn.Linear(
                config.dim,
                (config.n_heads + 2 * config.n_kv_heads, config.head_dim),
                bias=False,
                rngs=rngs,
                dtype=config.dtype,
                quant=quant
            )
            self.q_proj = None
            self.kv_proj = None

        self.o_proj = nn.Linear(
            (config.n_heads, config.head_dim),
            config.dim,
            bias=False,
            rngs=rngs,
            dtype=config.dtype,
            quant=quant
        )

    def __call__(self, x: jax.Array, *, ctx: jax.Array | None = None, ctx_mask: jax.Array | None = None, mask: jax.Array | None = None, is_causal: bool = True, pos_emb: tuple[jax.Array, jax.Array], ctx_pos_emb: tuple[jax.Array, jax.Array] | None = None):
        cos_q, sin_q = pos_emb
        if ctx is None:
            assert self.qkv_proj is not None
            # self-attn, causal: x [..., T, C]
            h = self.qkv_proj(x)  # [..., T, 15, 64]
            q, k, v = jnp.split(h, [self.n_heads, self.n_heads + self.n_kv_heads], axis=-2)
            q = apply_rope(q, cos_q, sin_q)
            k = apply_rope(k, cos_q, sin_q)
            attn_mask = None
            if mask is not None:
                mask = mask.astype(bool)
                # Fully padded choices still need one safe key for a finite softmax.
                safe_mask = mask | (~jnp.any(mask, axis=-1, keepdims=True) & (jnp.arange(mask.shape[-1]) == 0))
                attn_mask = safe_mask[:, None, None, :]
            a = jax.nn.dot_product_attention(q, k, v, mask=attn_mask, is_causal=is_causal)
            return self.o_proj(a)

        # cross-attn: x [B, Ts, C], ctx [B, Tc, C] or [B, k, Tc, C]
        assert self.q_proj is not None and self.kv_proj
        q = self.q_proj(x)  # [B, Ts, 9, 64]
        kv = self.kv_proj(ctx)  # [..., Tc, 6, 64]
        k, v = jnp.split(kv, 2, axis=-2)

        b, nq, h = q.shape
        _, n, s, nk, _ = k.shape
        r = nq // nk

        # Cross attention uses choice-local positions, independent of prompt padding.
        q = apply_rope(q[:, None], cos_q[:1], sin_q[:1])[:, 0]
        if ctx_pos_emb is not None:
            cos_c, sin_c = ctx_pos_emb
            k = apply_rope(k, cos_c, sin_c)

        q = q.reshape(b, nk, r, h)
        dk = float(h)
        s = jnp.einsum('bnrh,bksnh->bknrs', q, k) * (dk ** -0.5)
        if ctx_mask is not None:
            # ctx_mask: [b k t]
            # s       : [b k n r t]
            ctx_mask = ctx_mask.astype(bool)
            safe_mask = ctx_mask | (~jnp.any(ctx_mask, axis=-1, keepdims=True) & (jnp.arange(ctx_mask.shape[-1]) == 0))
            s = jnp.where(safe_mask[:, :, None, None, :], s, -jnp.inf)
        a = jnp.einsum('bknrs,bksnh->bknrh', jax.nn.softmax(s, axis=-1), v).reshape(b, n, nq, h)
        return self.o_proj(a)  # [B, k, Ts, C]

@nn.module
class Decoder:
    def __init__(self, config: SmollmConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.norm1 = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.norm2 = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.attn = Attention(config, rngs=rngs, quant=quant)
        self.mlp = MLP(config, rngs=rngs, quant=quant)

    def __call__(self, x: jax.Array, *, mask: jax.Array | None = None, pos_emb: tuple[jax.Array, jax.Array]):
        h = x + self.attn(self.norm1(x), mask=mask, pos_emb=pos_emb)
        return h + self.mlp(self.norm2(h))

@nn.module
class ChoiceEncoder:
    """Contextualize tokens within each choice before cross attention."""
    def __init__(self, config: SmollmConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.norm1 = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.norm2 = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.attn = Attention(config, rngs=rngs, quant=quant)
        self.mlp = MLP(config, rngs=rngs, quant=quant)

    def __call__(self, x: jax.Array, *, mask: jax.Array | None, pos_emb: tuple[jax.Array, jax.Array]):
        # x: [B * K, Tc, C]; attention is bidirectional within each choice.
        h = x + self.attn(self.norm1(x), mask=mask, is_causal=False, pos_emb=pos_emb)
        h = h + self.mlp(self.norm2(h))
        return h if mask is None else jnp.where(mask[..., None], h, 0)

@nn.module
class CtxDecoder:
    """Cross decoder for x_attn: deep state queries vs stacked choices."""
    def __init__(self, config: SmollmConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.norm1 = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.norm2 = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.norm3 = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.attn = Attention(config, rngs=rngs, quant=quant, sep_q=True)
        self.mlp = MLP(config, rngs=rngs, quant=quant)

    def __call__(self, x: jax.Array, *, ctx: jax.Array, ctx_mask: jax.Array | None, pos_emb: tuple[jax.Array, jax.Array], ctx_pos_emb: tuple[jax.Array, jax.Array]):
        # x: [B, C], ctx: [B, k, Tc, C]
        x = self.attn(self.norm1(x), ctx=self.norm2(ctx), ctx_mask=ctx_mask, pos_emb=pos_emb, ctx_pos_emb=ctx_pos_emb)
        return self.mlp(self.norm3(x))


def _encode_choices(module, x, mask, cos, sin):
    return module(x, mask=mask, pos_emb=(cos, sin))


def _cross_attend(module, query, ctx, ctx_mask, cos_q, sin_q, cos_c, sin_c):
    return module(query, ctx=ctx, ctx_mask=ctx_mask, pos_emb=(cos_q, sin_q), ctx_pos_emb=(cos_c, sin_c))


def _decode_layer(layer, x, mask, cos, sin):
    return layer(x, mask=mask, pos_emb=(cos, sin)), None

@nn.module
class Model:
    def __init__(self, config: SmollmConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.config = config
        self.wte = nn.Embedding(config.wte_size, config.dim, dtype=config.dtype, rngs=rngs)
        self.layers = nn.SeqStack([
            tp.cast(nn.Module, Decoder)(config, rngs=rngs, quant=quant) 
            for _ in range(config.n_layers)
        ])
        self.choice_encoder = ChoiceEncoder(config, rngs=rngs, quant=quant)
        self.ctx_layer = CtxDecoder(config, rngs=rngs, quant=quant)

    def __call__(self, ids: jax.Array, *, ctx_ids: jax.Array, ctx_mask: jax.Array | None = None, ids_mask: jax.Array | None = None, freeze_backbone: bool = False, remat_decision: bool = False, remat_backbone: bool = False):
        cfg = self.config
        x = self.wte(ids)  # [B, Ts, C]
        ts = x.shape[-2]
        cos_q, sin_q = build_rope(ts, cfg.head_dim, cfg.rope_theta)
        decode_layer = jax.checkpoint(_decode_layer) if remat_backbone else _decode_layer
        x, _ = self.layers(decode_layer, x, ids_mask, cos_q, sin_q)
        if freeze_backbone:
            x = jax.lax.stop_gradient(x)

        ctx = self.wte(ctx_ids)  # [B, k, Tc, C]
        if freeze_backbone:
            ctx = jax.lax.stop_gradient(ctx)
        tc = ctx.shape[-2]
        cos_c, sin_c = build_rope(tc, cfg.head_dim, cfg.rope_theta)
        batch, choices = ctx.shape[:2]
        flat_mask = None if ctx_mask is None else ctx_mask.reshape(batch * choices, tc)
        encode_choices = jax.checkpoint(_encode_choices) if remat_decision else _encode_choices
        cross_attend = jax.checkpoint(_cross_attend) if remat_decision else _cross_attend
        ctx = encode_choices(
            self.choice_encoder, ctx.reshape(batch * choices, tc, cfg.dim),
            flat_mask, cos_c, sin_c,
        ).reshape(batch, choices, tc, cfg.dim)
        return cross_attend(self.ctx_layer, x[:, -1], ctx, ctx_mask, cos_q, sin_q, cos_c, sin_c)

@nn.module
class SmollmSystemOne:
    def __init__(self, config: SmollmConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.model = Model(config, rngs=rngs, quant=quant)
        self.norm = nn.RMSNorm(config.dim, config.epsilon, dtype='float32')
        self.d_head = nn.Linear(
            config.dim,
            1,
            bias=True,
            dtype=config.dtype,
            rngs=rngs,
            quant=quant
        )

    def __call__(self, ids: jax.Array, *, ctx_ids: jax.Array, ctx_mask: jax.Array | None = None, ids_mask: jax.Array | None = None, freeze_backbone: bool = False, remat_decision: bool = False, remat_backbone: bool = False):
        x = self.model(ids, ctx_ids=ctx_ids, ctx_mask=ctx_mask, ids_mask=ids_mask, freeze_backbone=freeze_backbone, remat_decision=remat_decision, remat_backbone=remat_backbone)
        # x: [B, k, C]
        x = self.norm(x)
        logits = self.d_head(x)  # [B, k, 1]
        if ctx_mask is not None:
            logits = jnp.where(jnp.any(ctx_mask, axis=-1, keepdims=True), logits, -jnp.inf)
        return logits
    
    @classmethod
    def init(cls, path: pathlib.Path | str, quant: QuantConfig = None) -> 'SmollmSystemOne':
        config = SmollmConfig()
        rngs = nn.Rngs(0)
        state = load_file(path)
        model = tp.cast(nn.Module, cls)(config, rngs=rngs, quant=quant)
        expected = model.flat_state_dict()
        missing = sorted(expected.keys() - state.keys())
        unexpected = sorted(state.keys() - expected.keys())
        if missing or unexpected:
            raise ValueError(f"checkpoint keys do not match model: missing={missing}, unexpected={unexpected}")
        mismatched = [
            key for key, value in expected.items()
            if state[key].shape != value.shape
        ]
        if mismatched:
            raise ValueError(f"checkpoint tensor shapes do not match model: {mismatched}")
        model.load_flat_state_dict(state)
        return model
