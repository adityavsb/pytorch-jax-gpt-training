"""Pure JAX GPT-2-style transformer using lax.scan over stacked block weights."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from bench.spec import LN_EPS, ModelSpec, init_numpy_params


def layer_norm(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    """LayerNorm computed in FP32 to match PyTorch autocast behavior."""
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
    normed = (x32 - mean) * jax.lax.rsqrt(var + LN_EPS)
    return normed * w.astype(jnp.float32) + b.astype(jnp.float32)


def block_forward(
    x: jax.Array,
    p: dict[str, jax.Array],
    *,
    n_head: int,
    compute_dtype: jnp.dtype,
) -> jax.Array:
    """Single pre-LN transformer block with explicit causal attention and FP32 softmax."""
    bsz, seq_len, d_model = x.shape
    head_dim = d_model // n_head
    scale = 1.0 / math.sqrt(head_dim)

    h = layer_norm(x, p["ln1_w"], p["ln1_b"]).astype(compute_dtype)
    qkv = h @ p["qkv_w"].astype(compute_dtype) + p["qkv_b"].astype(compute_dtype)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(bsz, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) * scale
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]
    scores = jnp.where(causal_mask, scores, -1e30)
    attn = jax.nn.softmax(scores, axis=-1).astype(compute_dtype)
    y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    y = y.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
    attn_out = y @ p["proj_w"].astype(compute_dtype) + p["proj_b"].astype(compute_dtype)
    x = x.astype(jnp.float32) + attn_out.astype(jnp.float32)

    h2 = layer_norm(x, p["ln2_w"], p["ln2_b"]).astype(compute_dtype)
    mlp_h = jax.nn.gelu(
        h2 @ p["fc1_w"].astype(compute_dtype) + p["fc1_b"].astype(compute_dtype),
        approximate=True,
    )
    mlp_out = mlp_h @ p["fc2_w"].astype(compute_dtype) + p["fc2_b"].astype(compute_dtype)
    return x + mlp_out.astype(jnp.float32)


def model_forward(
    params: dict[str, object],
    idx: jax.Array,
    *,
    spec: ModelSpec,
    compute_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Full GPT forward pass returning FP32 logits (B, S, V)."""
    _, seq_len = idx.shape
    wte = params["wte"]
    wpe = params["wpe"]
    assert isinstance(wte, jax.Array) and isinstance(wpe, jax.Array)
    pos = jnp.arange(seq_len, dtype=jnp.int32)
    x = (wte[idx] + wpe[pos]).astype(jnp.float32)

    blocks = params["blocks"]

    def scan_body(carry: jax.Array, layer_p: dict[str, jax.Array]) -> tuple[jax.Array, None]:
        return block_forward(carry, layer_p, n_head=spec.n_head, compute_dtype=compute_dtype), None

    x, _ = jax.lax.scan(scan_body, x, blocks)
    x = layer_norm(x, params["ln_f_w"], params["ln_f_b"])  # type: ignore[arg-type]
    logits = x.astype(compute_dtype) @ wte.astype(compute_dtype).T
    return logits.astype(jnp.float32)


def _block_from_flat(flat: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {
        "ln1_w": flat[f"{prefix}ln1.w"],
        "ln1_b": flat[f"{prefix}ln1.b"],
        "qkv_w": flat[f"{prefix}attn.qkv.w"],
        "qkv_b": flat[f"{prefix}attn.qkv.b"],
        "proj_w": flat[f"{prefix}attn.proj.w"],
        "proj_b": flat[f"{prefix}attn.proj.b"],
        "ln2_w": flat[f"{prefix}ln2.w"],
        "ln2_b": flat[f"{prefix}ln2.b"],
        "fc1_w": flat[f"{prefix}mlp.fc1.w"],
        "fc1_b": flat[f"{prefix}mlp.fc1.b"],
        "fc2_w": flat[f"{prefix}mlp.fc2.w"],
        "fc2_b": flat[f"{prefix}mlp.fc2.b"],
    }


def block_pytree_from_numpy(
    flat: dict[str, np.ndarray],
    prefix: str = "",
) -> dict[str, jax.Array]:
    """Convert canonical single-block NumPy dict into a JAX pytree."""
    return {k: jnp.asarray(v) for k, v in _block_from_flat(flat, prefix).items()}


def pytree_from_numpy(flat: dict[str, np.ndarray], spec: ModelSpec) -> dict[str, object]:
    """Convert canonical flat NumPy dict into a stacked JAX parameter pytree."""
    per_layer = [_block_from_flat(flat, f"blocks.{i}.") for i in range(spec.n_layer)]
    stacked_blocks = {
        key: jnp.asarray(np.stack([layer[key] for layer in per_layer], axis=0))
        for key in per_layer[0]
    }
    return {
        "wte": jnp.asarray(flat["wte"]),
        "wpe": jnp.asarray(flat["wpe"]),
        "blocks": stacked_blocks,
        "ln_f_w": jnp.asarray(flat["ln_f.w"]),
        "ln_f_b": jnp.asarray(flat["ln_f.b"]),
    }


def pytree_to_numpy(params: dict[str, object], spec: ModelSpec) -> dict[str, np.ndarray]:
    """Convert stacked JAX parameter pytree back to canonical flat NumPy dict."""
    out: dict[str, np.ndarray] = {
        "wte": np.asarray(params["wte"]),
        "wpe": np.asarray(params["wpe"]),
    }
    blocks = params["blocks"]
    assert isinstance(blocks, dict)
    key_map = {
        "ln1_w": "ln1.w",
        "ln1_b": "ln1.b",
        "qkv_w": "attn.qkv.w",
        "qkv_b": "attn.qkv.b",
        "proj_w": "attn.proj.w",
        "proj_b": "attn.proj.b",
        "ln2_w": "ln2.w",
        "ln2_b": "ln2.b",
        "fc1_w": "mlp.fc1.w",
        "fc1_b": "mlp.fc1.b",
        "fc2_w": "mlp.fc2.w",
        "fc2_b": "mlp.fc2.b",
    }
    blocks_np = {k: np.asarray(v) for k, v in blocks.items()}
    for i in range(spec.n_layer):
        for jkey, ckey in key_map.items():
            out[f"blocks.{i}.{ckey}"] = blocks_np[jkey][i].copy()
    out["ln_f.w"] = np.asarray(params["ln_f_w"]).copy()
    out["ln_f.b"] = np.asarray(params["ln_f_b"]).copy()
    return out


def init_jax_params(spec: ModelSpec, seed: int = 0) -> dict[str, object]:
    """Initialize GPT parameters directly on the current JAX default device."""
    return pytree_from_numpy(init_numpy_params(spec, seed=seed), spec)


def count_params(params: dict[str, object]) -> int:
    """Total scalar parameter count across a JAX pytree."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
