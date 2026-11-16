"""Quick fail-fast smoke check on GPT-2 small (or tiny on CPU) before long runs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

from bench.results import collect_provenance, ensure_results_dir, write_json_atomic  # noqa: E402
from bench.spec import get_spec  # noqa: E402
from bench.timing import measure_loop  # noqa: E402


def _run_torch(preset: str, device_kind: str, steps: int) -> dict[str, object]:
    import torch

    from bench.step_torch import build_torch_step, make_synthetic_pool, peak_memory_gb, sync_device

    spec = get_spec(preset)
    device = torch.device("cuda:0" if device_kind == "cuda" else "cpu")
    precision = "fp16" if device_kind == "cuda" else "fp32"
    handle = build_torch_step(spec, device=device, mode="eager", precision=precision, seed=0)
    pool = make_synthetic_pool(spec, 2, device=device, num_batches=4, seed=1)
    stats = measure_loop(
        handle.step_fn,
        lambda: sync_device(device),
        pool,
        warmup_steps=1,
        timed_steps=steps,
        loss_extract_fn=lambda t: float(t.detach().item()),
    )
    stats["peak_mem_gb"] = peak_memory_gb(device)
    stats["provenance"] = collect_provenance("torch")
    return stats


def _run_jax(preset: str, device_kind: str, steps: int) -> dict[str, object]:
    import jax

    from bench.step_jax import build_jax_step, make_synthetic_pool, peak_memory_gb

    spec = get_spec(preset)
    precision = "fp16" if device_kind == "cuda" else "fp32"
    handle = build_jax_step(spec, sample_batch_size=2, n_devices=1, precision=precision, seed=0)
    pool = make_synthetic_pool(
        spec, 2, batch_sharding=handle.batch_sharding, num_batches=4, seed=1
    )
    state_box = [handle.state]

    def step_wrapper(batch: jax.Array) -> jax.Array:
        new_state, loss = handle.step_fn(state_box[0], batch)
        state_box[0] = new_state
        return loss

    stats = measure_loop(
        step_wrapper,
        lambda: jax.block_until_ready(state_box[0]),
        pool,
        warmup_steps=1,
        timed_steps=steps,
        loss_extract_fn=float,
    )
    stats["peak_mem_gb"] = peak_memory_gb(list(handle.mesh.devices.flat))
    stats["compile_s"] = handle.compile_s
    stats["provenance"] = collect_provenance("jax")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-fast smoke test on small/tiny preset")
    parser.add_argument("--framework", choices=["torch", "jax"], required=True)
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_dir = ensure_results_dir(args.out)
    res = (
        _run_torch(args.preset, args.device, args.steps)
        if args.framework == "torch"
        else _run_jax(args.preset, args.device, args.steps)
    )
    ok = bool(res["finite"]) and math.isfinite(float(res["loss_last"]))  # type: ignore[arg-type]
    res["stage"] = f"smoke_{args.framework}"
    res["preset"] = args.preset
    res["passed"] = ok
    write_json_atomic(Path(out_dir) / f"smoke_{args.framework}.json", res)
    print(f"[smoke:{args.framework}] preset={args.preset} loss_last={res['loss_last']:.4f} ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
