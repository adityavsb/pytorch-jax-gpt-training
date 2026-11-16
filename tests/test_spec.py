"""Tests for ModelSpec parameter and FLOP formulas against both frameworks."""

from __future__ import annotations

import jax
import torch

from bench.model_jax import init_jax_params
from bench.model_torch import GPTModel
from bench.spec import PRESETS, ModelSpec, get_spec


def test_medium_exact_param_count() -> None:
    spec = get_spec("medium")
    assert spec.n_params() == 303_884_288
    assert spec.grad_bytes_fp32 == 4 * 303_884_288


def test_presets_match_torch_and_jax_models() -> None:
    for name, spec in PRESETS.items():
        if name == "medium":
            # Construct without allocating ~1.2 GB in CPU CI.
            with torch.device("meta"):
                m_torch = GPTModel(spec)
            torch_count = sum(p.numel() for p in m_torch.parameters())
            jax_shapes = jax.eval_shape(lambda s=spec: init_jax_params(s, seed=0))
            jax_count = sum(int(x.size) for x in jax.tree_util.tree_leaves(jax_shapes))
        else:
            m_torch = GPTModel(spec)
            torch_count = sum(p.numel() for p in m_torch.parameters())
            jax_params = init_jax_params(spec, seed=0)
            jax_count = sum(int(x.size) for x in jax.tree_util.tree_leaves(jax_params))

        assert torch_count == spec.n_params(), f"{name} torch mismatch"
        assert jax_count == spec.n_params(), f"{name} jax mismatch"


def test_train_flops_monotonic() -> None:
    spec = ModelSpec()
    f1 = spec.train_flops_per_step(4)
    f2 = spec.train_flops_per_step(8)
    assert f2 == 2.0 * f1
    assert f1 > 0.0
