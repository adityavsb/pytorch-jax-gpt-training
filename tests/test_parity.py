"""CPU parity tests: block forward, full forward, loss, gradients, and 1 AdamW step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.nn.functional as F

from bench.model_jax import (
    block_forward,
    block_pytree_from_numpy,
    model_forward,
    pytree_from_numpy,
    pytree_to_numpy,
)
from bench.model_torch import GPTModel, TransformerBlock
from bench.spec import (
    ADAMW_BETA1,
    ADAMW_BETA2,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    get_spec,
    init_block_numpy_params,
    init_numpy_params,
)


def test_single_block_forward_parity() -> None:
    spec = get_spec("tiny")
    rng = np.random.default_rng(123)
    w_np = init_block_numpy_params(spec, rng)
    x_np = rng.normal(0.0, 1.0, size=(2, spec.seq_len, spec.d_model)).astype(np.float32)

    t_block = TransformerBlock(spec)
    t_block.load_numpy(w_np)
    with torch.no_grad():
        y_torch = t_block(torch.from_numpy(x_np)).numpy()

    j_block = block_pytree_from_numpy(w_np)
    y_jax = np.asarray(
        block_forward(
            jnp.asarray(x_np),
            j_block,
            n_head=spec.n_head,
            compute_dtype=jnp.float32,
        )
    )

    max_diff = float(np.max(np.abs(y_torch - y_jax)))
    assert max_diff < 1e-5, f"Block forward mismatch: {max_diff}"


def test_full_model_forward_loss_grad_and_adamw_parity() -> None:
    spec = get_spec("tiny")
    w_np = init_numpy_params(spec, seed=42)
    rng = np.random.default_rng(7)
    tokens_np = rng.integers(0, spec.vocab, size=(4, spec.seq_len + 1), dtype=np.int64)

    # PyTorch path
    t_model = GPTModel(spec)
    t_model.load_numpy(w_np)
    t_opt = torch.optim.AdamW(
        t_model.parameters(),
        lr=ADAMW_LR,
        betas=(ADAMW_BETA1, ADAMW_BETA2),
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
    )
    t_tokens = torch.from_numpy(tokens_np)
    t_logits = t_model(t_tokens[:, :-1])
    t_loss = F.cross_entropy(t_logits.flatten(0, 1), t_tokens[:, 1:].flatten())
    t_opt.zero_grad(set_to_none=True)
    t_loss.backward()

    # Extract torch gradients in canonical layout before optimizer step
    t_grads_np = {
        "wte": t_model.wte.weight.grad.detach().numpy().copy(),
        "wpe": t_model.wpe.weight.grad.detach().numpy().copy(),
        "ln_f.w": t_model.ln_f.weight.grad.detach().numpy().copy(),
        "ln_f.b": t_model.ln_f.bias.grad.detach().numpy().copy(),
    }
    for i, b in enumerate(t_model.blocks):
        t_grads_np[f"blocks.{i}.ln1.w"] = b.ln1.weight.grad.detach().numpy().copy()
        t_grads_np[f"blocks.{i}.ln1.b"] = b.ln1.bias.grad.detach().numpy().copy()
        t_grads_np[f"blocks.{i}.attn.qkv.w"] = b.attn.qkv.weight.grad.detach().numpy().T.copy()
        t_grads_np[f"blocks.{i}.attn.qkv.b"] = b.attn.qkv.bias.grad.detach().numpy().copy()
        t_grads_np[f"blocks.{i}.attn.proj.w"] = b.attn.proj.weight.grad.detach().numpy().T.copy()
        t_grads_np[f"blocks.{i}.attn.proj.b"] = b.attn.proj.bias.grad.detach().numpy().copy()
        t_grads_np[f"blocks.{i}.ln2.w"] = b.ln2.weight.grad.detach().numpy().copy()
        t_grads_np[f"blocks.{i}.ln2.b"] = b.ln2.bias.grad.detach().numpy().copy()
        t_grads_np[f"blocks.{i}.mlp.fc1.w"] = b.mlp.fc1.weight.grad.detach().numpy().T.copy()
        t_grads_np[f"blocks.{i}.mlp.fc1.b"] = b.mlp.fc1.bias.grad.detach().numpy().copy()
        t_grads_np[f"blocks.{i}.mlp.fc2.w"] = b.mlp.fc2.weight.grad.detach().numpy().T.copy()
        t_grads_np[f"blocks.{i}.mlp.fc2.b"] = b.mlp.fc2.bias.grad.detach().numpy().copy()

    t_opt.step()
    t_updated = t_model.to_numpy()

    # JAX path
    j_params = pytree_from_numpy(w_np, spec)
    j_tokens = jnp.asarray(tokens_np)

    def loss_fn(p: dict[str, object]) -> tuple[jax.Array, jax.Array]:
        logits = model_forward(p, j_tokens[:, :-1], spec=spec, compute_dtype=jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, j_tokens[:, 1:]).mean()
        return loss, logits

    (j_loss, j_logits), j_grads = jax.value_and_grad(loss_fn, has_aux=True)(j_params)
    j_grads_np = pytree_to_numpy(j_grads, spec)

    j_opt = optax.adamw(
        learning_rate=ADAMW_LR,
        b1=ADAMW_BETA1,
        b2=ADAMW_BETA2,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
    )
    opt_state = j_opt.init(j_params)
    updates, _ = j_opt.update(j_grads, opt_state, j_params)
    j_new_params = optax.apply_updates(j_params, updates)
    j_updated = pytree_to_numpy(j_new_params, spec)

    # Compare logits, loss, gradients, and post-step weights
    logit_diff = float(np.max(np.abs(t_logits.detach().numpy() - np.asarray(j_logits))))
    assert logit_diff < 1e-5, f"Logit mismatch: {logit_diff}"

    loss_diff = abs(float(t_loss.detach().numpy()) - float(j_loss))
    assert loss_diff < 1e-5, f"Loss mismatch: {loss_diff}"

    for key in t_grads_np:
        tg = t_grads_np[key]
        jg = j_grads_np[key]
        rel = float(np.linalg.norm(tg - jg) / max(np.linalg.norm(tg), 1e-8))
        assert rel < 1e-4, f"Grad mismatch on {key}: rel={rel}"

    for key in t_updated:
        max_w_diff = float(np.max(np.abs(t_updated[key] - j_updated[key])))
        assert max_w_diff < 1e-5, f"Updated weight mismatch on {key}: {max_w_diff}"
