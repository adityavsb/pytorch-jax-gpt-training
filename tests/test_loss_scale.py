"""Tests for JAX DynamicLossScale matching PyTorch GradScaler behavior."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch

from bench.spec import LOSS_SCALE_GROWTH_INTERVAL, LOSS_SCALE_INIT
from bench.step_jax import DynamicLossScale


def test_dynamic_loss_scale_backoff_and_growth() -> None:
    ls = DynamicLossScale.create(init_scale=LOSS_SCALE_INIT)
    assert float(ls.scale) == LOSS_SCALE_INIT

    # Overflow backs off by 0.5 and resets tracker.
    bad_grads = {"w": jnp.asarray([1.0, jnp.inf], dtype=jnp.float32)}
    _, is_finite = ls.unscale(bad_grads)
    assert not bool(is_finite)
    ls = ls.update(is_finite)
    assert float(ls.scale) == LOSS_SCALE_INIT * 0.5
    assert int(ls.growth_tracker) == 0

    # Growth occurs after LOSS_SCALE_GROWTH_INTERVAL consecutive finite steps.
    cur_scale = float(ls.scale)
    for _ in range(LOSS_SCALE_GROWTH_INTERVAL - 1):
        ls = ls.update(jnp.asarray(True))
        assert float(ls.scale) == cur_scale
    ls = ls.update(jnp.asarray(True))
    assert float(ls.scale) == cur_scale * 2.0
    assert int(ls.growth_tracker) == 0


def test_dynamic_loss_scale_matches_torch_grad_scaler_schedule() -> None:
    ls = DynamicLossScale.create(init_scale=1024.0)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)

    # Simulate an overflow step in JAX and verify initial scale + backoff match.
    ls_after_inf = ls.update(jnp.asarray(False))
    assert float(ls_after_inf.scale) == scaler.get_scale() * scaler.get_backoff_factor()
    assert scaler.get_growth_interval() == LOSS_SCALE_GROWTH_INTERVAL

    # Verify finite unscale values match division by scale.
    raw = np.array([2048.0, -4096.0], dtype=np.float32)
    unscaled, is_finite = ls.unscale({"x": jnp.asarray(raw)})
    assert bool(is_finite)
    np.testing.assert_allclose(np.asarray(unscaled["x"]), raw / 1024.0)
