"""JAX training step builder: jit, dynamic FP16 loss scaling, and GSPMD batch sharding."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

# Preallocate up to 92% of GPU memory so the 10-14 GB T4 window does not OOM under
# JAX's default 75% (~11.3 GB) cap. Must be set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402

from bench.model_jax import init_jax_params, model_forward  # noqa: E402
from bench.spec import (  # noqa: E402
    ADAMW_BETA1,
    ADAMW_BETA2,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    LOSS_SCALE_BACKOFF_FACTOR,
    LOSS_SCALE_GROWTH_FACTOR,
    LOSS_SCALE_GROWTH_INTERVAL,
    LOSS_SCALE_INIT,
    ModelSpec,
)


class DynamicLossScale(NamedTuple):
    """Dynamic FP16 loss scaler mirroring PyTorch GradScaler defaults."""

    scale: jax.Array
    growth_tracker: jax.Array

    @classmethod
    def create(cls, init_scale: float = LOSS_SCALE_INIT) -> DynamicLossScale:
        return cls(
            scale=jnp.asarray(init_scale, dtype=jnp.float32),
            growth_tracker=jnp.asarray(0, dtype=jnp.int32),
        )

    def unscale(self, grads: Any) -> tuple[Any, jax.Array]:
        inv_scale = jnp.asarray(1.0, dtype=jnp.float32) / self.scale
        unscaled = jax.tree_util.tree_map(lambda g: g * inv_scale, grads)
        leaves = jax.tree_util.tree_leaves(unscaled)
        finite_flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
        is_finite = jnp.all(jnp.stack(finite_flags))
        return unscaled, is_finite

    def update(self, is_finite: jax.Array) -> DynamicLossScale:
        def on_finite() -> DynamicLossScale:
            next_tracker = self.growth_tracker + 1
            grow = next_tracker >= LOSS_SCALE_GROWTH_INTERVAL
            next_scale = jnp.where(grow, self.scale * LOSS_SCALE_GROWTH_FACTOR, self.scale)
            reset_tracker = jnp.where(grow, jnp.asarray(0, dtype=jnp.int32), next_tracker)
            return DynamicLossScale(scale=next_scale, growth_tracker=reset_tracker)

        def on_overflow() -> DynamicLossScale:
            next_scale = jnp.maximum(
                jnp.asarray(1.0, dtype=jnp.float32),
                self.scale * LOSS_SCALE_BACKOFF_FACTOR,
            )
            return DynamicLossScale(
                scale=next_scale,
                growth_tracker=jnp.asarray(0, dtype=jnp.int32),
            )

        return jax.lax.cond(is_finite, on_finite, on_overflow)


class TrainState(NamedTuple):
    """Full replicated JAX training state."""

    params: dict[str, object]
    opt_state: optax.OptState
    loss_scale: DynamicLossScale


@dataclass
class JaxStepHandle:
    """Container for compiled JAX train_step, initial state, shardings, and HLO stats."""

    state: TrainState
    mesh: Mesh
    batch_sharding: NamedSharding
    repl_sharding: NamedSharding
    step_fn: Callable[[TrainState, jax.Array], tuple[TrainState, jax.Array]]
    compile_s: float
    hlo_text: str


def build_mesh(n_devices: int) -> Mesh:
    devices = jax.devices()[:n_devices]
    if len(devices) < n_devices:
        raise RuntimeError(f"Requested {n_devices} JAX devices, found {len(devices)}")
    return Mesh(np.array(devices), ("data",))


def make_synthetic_pool(
    spec: ModelSpec,
    batch_size: int,
    *,
    batch_sharding: NamedSharding,
    num_batches: int = 8,
    seed: int = 0,
) -> list[jax.Array]:
    """Pre-generate synthetic token batches and place them on device(s) before timing."""
    rng = np.random.default_rng(seed)
    pool: list[jax.Array] = []
    for _ in range(num_batches):
        arr = rng.integers(0, spec.vocab, size=(batch_size, spec.seq_len + 1), dtype=np.int32)
        pool.append(jax.device_put(arr, batch_sharding))
    jax.block_until_ready(pool)
    return pool


def block_until_ready(obj: Any) -> None:
    jax.block_until_ready(obj)


def peak_memory_gb(devices: list[Any] | None = None) -> float:
    devs = devices if devices is not None else jax.devices()
    peaks: list[int] = []
    for d in devs:
        stats = d.memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            peaks.append(int(stats["peak_bytes_in_use"]))
    return (max(peaks) / (1024.0**3)) if peaks else 0.0


def inspect_hlo_allreduce(hlo_text: str) -> dict[str, int]:
    """Count all-reduce instructions and sum their element bytes from compiled HLO text."""
    dtype_bytes = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "s32": 4, "s64": 8}
    pattern = re.compile(r"=\s*([a-z0-9]+)\[([0-9,]*)\][^{]*\ball-reduce\b")
    count = 0
    total_bytes = 0
    for line in hlo_text.splitlines():
        if "all-reduce(" not in line and "all-reduce-start(" not in line:
            continue
        count += 1
        match = pattern.search(line)
        if match:
            dt, dims_str = match.group(1), match.group(2)
            numel = 1
            if dims_str:
                for dim in dims_str.split(","):
                    numel *= int(dim)
            total_bytes += numel * dtype_bytes.get(dt, 4)
    return {"all_reduce_count": count, "all_reduce_bytes": total_bytes}


def make_optimizer() -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=ADAMW_LR,
        b1=ADAMW_BETA1,
        b2=ADAMW_BETA2,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
    )


def build_jax_step(
    spec: ModelSpec,
    *,
    sample_batch_size: int,
    n_devices: int = 1,
    precision: str = "fp16",
    seed: int = 0,
    donate_state: bool = True,
    initial_params: dict[str, object] | None = None,
) -> JaxStepHandle:
    """Create mesh, replicated state, and AOT-compiled train_step."""
    import time

    if precision not in ("fp32", "fp16"):
        raise ValueError(f"Unknown precision {precision!r}; expected 'fp32' or 'fp16'")

    compute_dtype = jnp.float16 if precision == "fp16" else jnp.float32
    use_loss_scale = precision == "fp16"

    mesh = build_mesh(n_devices)
    batch_sharding = NamedSharding(mesh, PartitionSpec("data"))
    repl_sharding = NamedSharding(mesh, PartitionSpec())

    params = initial_params if initial_params is not None else init_jax_params(spec, seed=seed)
    tx = make_optimizer()
    opt_state = tx.init(params)
    loss_scale = DynamicLossScale.create(LOSS_SCALE_INIT if use_loss_scale else 1.0)
    state = TrainState(params=params, opt_state=opt_state, loss_scale=loss_scale)
    state = jax.device_put(state, repl_sharding)

    def loss_fn(p: dict[str, object], tokens: jax.Array) -> jax.Array:
        logits = model_forward(p, tokens[:, :-1], spec=spec, compute_dtype=compute_dtype)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:]).mean()

    def _train_step(cur: TrainState, tokens: jax.Array) -> tuple[TrainState, jax.Array]:
        def scaled_loss(p: dict[str, object]) -> jax.Array:
            loss = loss_fn(p, tokens)
            return loss * cur.loss_scale.scale if use_loss_scale else loss

        loss_scaled, grads = jax.value_and_grad(scaled_loss)(cur.params)
        if use_loss_scale:
            grads, is_finite = cur.loss_scale.unscale(grads)
            loss_out = loss_scaled / cur.loss_scale.scale

            def apply_fn(s: TrainState, g: Any) -> TrainState:
                updates, new_opt = tx.update(g, s.opt_state, s.params)
                new_params = optax.apply_updates(s.params, updates)
                return TrainState(
                    params=new_params,
                    opt_state=new_opt,
                    loss_scale=s.loss_scale.update(jnp.asarray(True)),
                )

            def skip_fn(s: TrainState, _g: Any) -> TrainState:
                return TrainState(
                    params=s.params,
                    opt_state=s.opt_state,
                    loss_scale=s.loss_scale.update(jnp.asarray(False)),
                )

            new_state = jax.lax.cond(is_finite, apply_fn, skip_fn, cur, grads)
            return new_state, loss_out

        updates, new_opt = tx.update(grads, cur.opt_state, cur.params)
        new_params = optax.apply_updates(cur.params, updates)
        return TrainState(
            params=new_params, opt_state=new_opt, loss_scale=cur.loss_scale
        ), loss_scaled

    jit_kwargs: dict[str, Any] = {
        "in_shardings": (repl_sharding, batch_sharding),
        "out_shardings": (repl_sharding, repl_sharding),
    }
    if donate_state:
        jit_kwargs["donate_argnums"] = (0,)

    jitted_step = partial(jax.jit, **jit_kwargs)(_train_step)

    dummy_tokens = jax.device_put(
        np.zeros((sample_batch_size, spec.seq_len + 1), dtype=np.int32),
        batch_sharding,
    )
    t0 = time.perf_counter()
    compiled = jitted_step.lower(state, dummy_tokens).compile()
    compile_s = time.perf_counter() - t0
    hlo_text = compiled.as_text()

    return JaxStepHandle(
        state=state,
        mesh=mesh,
        batch_sharding=batch_sharding,
        repl_sharding=repl_sharding,
        step_fn=compiled,
        compile_s=compile_s,
        hlo_text=hlo_text,
    )
