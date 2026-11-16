"""CPU test verifying 2-device JAX batch sharding equals 1-device execution."""

from __future__ import annotations

import os
import subprocess
import sys


def test_jax_2device_sharding_matches_1device() -> None:
    script = """
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import jax
import numpy as np
from bench.spec import get_spec, init_numpy_params
from bench.model_jax import pytree_from_numpy, pytree_to_numpy
from bench.step_jax import build_jax_step, inspect_hlo_allreduce

assert len(jax.devices()) >= 2, f"Expected >=2 host devices, got {jax.devices()}"
spec = get_spec("tiny")
w_np = init_numpy_params(spec, seed=11)
rng = np.random.default_rng(5)
tokens_np = rng.integers(0, spec.vocab, size=(4, spec.seq_len + 1), dtype=np.int32)

h1 = build_jax_step(
    spec,
    sample_batch_size=4,
    n_devices=1,
    precision="fp32",
    donate_state=False,
    initial_params=pytree_from_numpy(w_np, spec),
)
b1 = jax.device_put(tokens_np, h1.batch_sharding)
s1, l1 = h1.step_fn(h1.state, b1)
p1 = pytree_to_numpy(s1.params, spec)

h2 = build_jax_step(
    spec,
    sample_batch_size=4,
    n_devices=2,
    precision="fp32",
    donate_state=False,
    initial_params=pytree_from_numpy(w_np, spec),
)
b2 = jax.device_put(tokens_np, h2.batch_sharding)
s2, l2 = h2.step_fn(h2.state, b2)
p2 = pytree_to_numpy(s2.params, spec)

assert abs(float(l1) - float(l2)) < 1e-5, f"Loss diff: {abs(float(l1) - float(l2))}"
for k in p1:
    diff = float(np.max(np.abs(p1[k] - p2[k])))
    assert diff < 1e-5, f"Param {k} diff: {diff}"

hlo_stats = inspect_hlo_allreduce(h2.hlo_text)
assert hlo_stats["all_reduce_count"] >= 1, f"No all-reduce found in 2-device HLO: {hlo_stats}"
"""
    env = os.environ.copy()
    res = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
