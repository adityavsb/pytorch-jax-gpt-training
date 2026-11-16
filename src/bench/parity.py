"""Correctness check: FP32 forward pass on single block and full model from shared NumPy weights."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import numpy as np  # noqa: E402

from bench.gates import check_parity_gate  # noqa: E402
from bench.results import (  # noqa: E402
    collect_provenance,
    ensure_results_dir,
    require_stage_passed,
    write_json_atomic,
)
from bench.spec import get_spec, init_block_numpy_params, init_numpy_params  # noqa: E402


def _shared_inputs(spec_name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = get_spec(spec_name)
    rng_x = np.random.default_rng(100)
    x_block = rng_x.normal(0.0, 1.0, size=(2, spec.seq_len, spec.d_model)).astype(np.float32)
    rng_tok = np.random.default_rng(101)
    tokens = rng_tok.integers(0, spec.vocab, size=(2, spec.seq_len), dtype=np.int64)
    return x_block, tokens


def _run_torch(preset: str, device_kind: str, parity_dir: Path) -> None:
    import torch

    from bench.model_torch import GPTModel, TransformerBlock

    spec = get_spec(preset)
    device = torch.device("cuda:0" if device_kind == "cuda" else "cpu")
    x_block, tokens = _shared_inputs(preset)

    block_w = init_block_numpy_params(spec, np.random.default_rng(77))
    block = TransformerBlock(spec).to(device)
    block.load_numpy(block_w)
    block.eval()
    with torch.no_grad():
        y_block = block(torch.from_numpy(x_block).to(device)).cpu().numpy()
    del block

    full_w = init_numpy_params(spec, seed=0)
    model = GPTModel(spec).to(device)
    model.load_numpy(full_w)
    model.eval()
    with torch.no_grad():
        y_full = model(torch.from_numpy(tokens).to(device)).cpu().numpy()
    del model

    np.save(parity_dir / "torch_block.npy", y_block)
    np.save(parity_dir / "torch_full.npy", y_full)


def _run_jax(preset: str, parity_dir: Path) -> None:
    import jax
    import jax.numpy as jnp

    from bench.model_jax import (
        block_forward,
        block_pytree_from_numpy,
        model_forward,
        pytree_from_numpy,
    )

    spec = get_spec(preset)
    x_block, tokens = _shared_inputs(preset)

    block_w = init_block_numpy_params(spec, np.random.default_rng(77))
    j_block = block_pytree_from_numpy(block_w)
    y_block = np.asarray(
        jax.jit(lambda x, p: block_forward(x, p, n_head=spec.n_head, compute_dtype=jnp.float32))(
            jnp.asarray(x_block), j_block
        )
    )
    del j_block

    full_w = init_numpy_params(spec, seed=0)
    j_params = pytree_from_numpy(full_w, spec)
    y_full = np.asarray(
        jax.jit(lambda p, t: model_forward(p, t, spec=spec, compute_dtype=jnp.float32))(
            j_params, jnp.asarray(tokens, dtype=jnp.int32)
        )
    )
    del j_params

    np.save(parity_dir / "jax_block.npy", y_block)
    np.save(parity_dir / "jax_full.npy", y_full)


def _compare(out_dir: Path, parity_dir: Path, preset: str, *, force_gates: bool) -> int:
    t_block = np.load(parity_dir / "torch_block.npy")
    j_block = np.load(parity_dir / "jax_block.npy")
    t_full = np.load(parity_dir / "torch_full.npy")
    j_full = np.load(parity_dir / "jax_full.npy")

    block_diff = float(np.max(np.abs(t_block - j_block)))
    full_diff = float(np.max(np.abs(t_full - j_full)))
    passed, reasons = check_parity_gate(block_diff, full_diff)

    payload = {
        "stage": "parity",
        "preset": preset,
        "block_max_abs_diff": block_diff,
        "full_max_abs_diff": full_diff,
        "passed": passed or force_gates,
        "forced": bool(force_gates and not passed),
        "reasons": reasons,
        "provenance": collect_provenance(),
    }
    write_json_atomic(out_dir / "parity.json", payload)
    print(
        f"[parity] preset={preset} block_max_diff={block_diff:.3e} "
        f"full_max_diff={full_diff:.3e} passed={payload['passed']}"
    )
    if not payload["passed"]:
        print(f"Parity gate failed: {reasons}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FP32 forward parity check between PyTorch and JAX"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--framework", choices=["torch", "jax"])
    group.add_argument("--compare", action="store_true")
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="medium")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--force-gates", action="store_true")
    args = parser.parse_args(argv)

    out_dir = ensure_results_dir(args.out)
    require_stage_passed(out_dir, "allreduce.json", force=args.force_gates)

    parity_dir = out_dir / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)

    if args.framework == "torch":
        _run_torch(args.preset, args.device, parity_dir)
        return 0
    if args.framework == "jax":
        _run_jax(args.preset, parity_dir)
        return 0
    return _compare(out_dir, parity_dir, args.preset, force_gates=args.force_gates)


if __name__ == "__main__":
    raise SystemExit(main())
