"""Experiment 1: Measure 2-GPU allreduce bandwidth and latency in PyTorch NCCL and JAX psum."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

from bench.gates import check_allreduce_gate  # noqa: E402
from bench.results import (  # noqa: E402
    collect_provenance,
    ensure_results_dir,
    read_json,
    require_stage_passed,
    write_json_atomic,
)
from bench.spec import get_spec  # noqa: E402
from bench.timing import estimate_sync_time_s, fit_allreduce_linear  # noqa: E402

GPU_SIZES_BYTES = [2**k * 1024 * 1024 for k in range(0, 9)]  # 1 MB .. 256 MB
CPU_SIZES_BYTES = [2**k * 64 * 1024 for k in range(0, 6)]  # 64 KB .. 2 MB


def _torch_allreduce_worker(
    rank: int,
    world_size: int,
    device_kind: str,
    sizes_bytes: list[int],
    warmup: int,
    iters: int,
    out_file: str,
    init_method: str | None = None,
) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

    backend = "nccl" if device_kind == "cuda" else "gloo"
    if init_method is not None:
        dist.init_process_group(
            backend=backend, init_method=init_method, rank=rank, world_size=world_size
        )
    else:
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if device_kind == "cuda":
            torch.cuda.set_device(local_rank)

    try:
        dev = (
            torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
            if device_kind == "cuda"
            else torch.device("cpu")
        )
        dtypes = [("fp32", torch.float32)]
        if device_kind == "cuda":
            dtypes.append(("fp16", torch.float16))

        results: dict[str, Any] = {}
        for label, dt in dtypes:
            elem_size = torch.tensor([], dtype=dt).element_size()
            points: list[dict[str, Any]] = []
            for nbytes in sizes_bytes:
                numel = max(1, nbytes // elem_size)
                buf = torch.ones(numel, dtype=dt, device=dev)
                for _ in range(warmup):
                    dist.all_reduce(buf)
                if device_kind == "cuda":
                    torch.cuda.synchronize(dev)

                samples: list[float] = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    dist.all_reduce(buf)
                    if device_kind == "cuda":
                        torch.cuda.synchronize(dev)
                    samples.append(time.perf_counter() - t0)
                med = float(np.median(samples))
                points.append({"bytes": nbytes, "median_s": med, "min_s": float(min(samples))})

            fit = fit_allreduce_linear(
                [p["bytes"] for p in points], [p["median_s"] for p in points]
            )
            results[label] = {"points": points, "fit": fit}

        if rank == 0:
            write_json_atomic(
                out_file,
                {
                    "backend": f"torch_{backend}",
                    "results": results,
                    "provenance": collect_provenance("torch"),
                },
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_torch_allreduce(
    out_dir: Path,
    device_kind: str,
    sizes_bytes: list[int],
    warmup: int,
    iters: int,
) -> None:
    import tempfile

    import torch.multiprocessing as mp

    out_file = str(out_dir / "allreduce_torch.json")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _torch_allreduce_worker(
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            device_kind,
            sizes_bytes,
            warmup,
            iters,
            out_file,
        )
        return

    with tempfile.TemporaryDirectory() as tmp:
        init_file = f"file://{Path(tmp) / 'init'}"
        mp.spawn(
            _torch_allreduce_worker,
            args=(2, device_kind, sizes_bytes, warmup, iters, out_file, init_file),
            nprocs=2,
            join=True,
        )


def _run_jax_allreduce(
    out_dir: Path,
    device_kind: str,
    sizes_bytes: list[int],
    warmup: int,
    iters: int,
    *,
    force_gates: bool,
) -> int:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devs = jax.devices()[:2]
    if len(devs) < 2:
        raise SystemExit(f"JAX allreduce requires 2 devices, found {len(devs)}")

    mesh = Mesh(np.array(devs), ("i",))
    sharding = NamedSharding(mesh, PartitionSpec("i"))
    pmap_sum = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i", devices=devs)
    dtypes = [("fp32", jnp.float32)]
    if device_kind == "cuda":
        dtypes.append(("fp16", jnp.float16))

    jax_results: dict[str, Any] = {}
    for label, dt in dtypes:
        elem_size = 4 if dt == jnp.float32 else 2
        points: list[dict[str, Any]] = []
        for nbytes in sizes_bytes:
            numel = max(1, nbytes // elem_size)
            host_arr = np.ones((2, numel), dtype=np.float32 if dt == jnp.float32 else np.float16)
            sharded = jax.device_put(host_arr, sharding)
            for _ in range(warmup):
                sharded = pmap_sum(sharded)
            jax.block_until_ready(sharded)

            samples: list[float] = []
            for _ in range(iters):
                t0 = time.perf_counter()
                sharded = pmap_sum(sharded)
                jax.block_until_ready(sharded)
                samples.append(time.perf_counter() - t0)
            med = float(np.median(samples))
            points.append({"bytes": nbytes, "median_s": med, "min_s": float(min(samples))})

        fit = fit_allreduce_linear([p["bytes"] for p in points], [p["median_s"] for p in points])
        jax_results[label] = {"points": points, "fit": fit}

    torch_path = out_dir / "allreduce_torch.json"
    if not torch_path.exists():
        raise SystemExit(f"Missing {torch_path}; run --backend torch first")
    torch_data = read_json(torch_path)

    grad_bytes_fp32 = get_spec("medium").grad_bytes_fp32
    torch_fp32_fit = torch_data["results"]["fp32"]["fit"]
    jax_fp32_fit = jax_results["fp32"]["fit"]

    sync_est_torch_s = estimate_sync_time_s(torch_fp32_fit, grad_bytes_fp32)
    sync_est_jax_s = estimate_sync_time_s(jax_fp32_fit, grad_bytes_fp32)

    if device_kind == "cpu":
        passed = bool(force_gates)
        reasons = [] if passed else ["CPU allreduce requires --force-gates"]
    else:
        passed, reasons = check_allreduce_gate(torch_fp32_fit["r2"])

    payload = {
        "stage": "allreduce",
        "device_mode": device_kind,
        "grad_bytes_fp32": grad_bytes_fp32,
        "torch": torch_data["results"],
        "jax": jax_results,
        "sync_estimate_s": sync_est_torch_s,
        "sync_estimate_jax_s": sync_est_jax_s,
        "passed": passed or force_gates,
        "forced": bool(force_gates and not passed),
        "reasons": reasons,
        "provenance_torch": torch_data.get("provenance"),
        "provenance_jax": collect_provenance("jax"),
    }
    write_json_atomic(out_dir / "allreduce.json", payload)
    print(
        f"[allreduce] torch_fp32 beta={torch_fp32_fit['beta_gbs']:.2f} GB/s "
        f"R2={torch_fp32_fit['r2']:.4f} | jax_fp32 beta={jax_fp32_fit['beta_gbs']:.2f} GB/s | "
        f"est_sync={sync_est_torch_s * 1000:.1f} ms | passed={payload['passed']}"
    )
    if not payload["passed"]:
        print(f"Allreduce gate failed: {reasons}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    # Pre-parse device/backend so XLA_FLAGS can be set prior to importing jax on CPU.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=["torch", "jax"], required=True)
    pre.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    known, _ = pre.parse_known_args(argv)
    if known.backend == "jax" and known.device == "cpu":
        os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

    parser = argparse.ArgumentParser(description="Experiment 1: 2-GPU allreduce sweep")
    parser.add_argument("--backend", choices=["torch", "jax"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--force-gates", action="store_true")
    args = parser.parse_args(argv)

    out_dir = ensure_results_dir(args.out)
    require_stage_passed(out_dir, "topo.json", force=args.force_gates)
    sizes = GPU_SIZES_BYTES if args.device == "cuda" else CPU_SIZES_BYTES

    if args.backend == "torch":
        _run_torch_allreduce(out_dir, args.device, sizes, args.warmup, args.iters)
        return 0
    return _run_jax_allreduce(
        out_dir, args.device, sizes, args.warmup, args.iters, force_gates=args.force_gates
    )


if __name__ == "__main__":
    raise SystemExit(main())
