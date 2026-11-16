"""Experiment 3: 1-GPU benchmark across PyTorch eager, torch.compile, and JAX jit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

from bench.results import (  # noqa: E402
    collect_provenance,
    ensure_results_dir,
    read_json,
    require_stage_passed,
    write_json_atomic,
)
from bench.spec import get_spec  # noqa: E402
from bench.timing import aggregate_process_runs, measure_loop  # noqa: E402


def _worker_torch(
    preset: str,
    device_kind: str,
    mode: str,
    batch_size: int,
    warmup: int,
    steps: int,
    worker_out: str,
    trace_out: str | None,
) -> int:
    import torch

    from bench.step_torch import (
        build_torch_step,
        make_synthetic_pool,
        peak_memory_gb,
        reset_peak_memory,
        sync_device,
    )

    spec = get_spec(preset)
    device = torch.device("cuda:0" if device_kind == "cuda" else "cpu")
    precision = "fp16" if device_kind == "cuda" else "fp32"

    try:
        handle = build_torch_step(
            spec,
            device=device,
            mode=mode,
            precision=precision,
            seed=0,
            check_graph_breaks=(mode == "compile"),
        )
        pool = make_synthetic_pool(spec, batch_size, device=device, num_batches=8, seed=1)
        reset_peak_memory(device)
        res = measure_loop(
            handle.step_fn,
            lambda: sync_device(device),
            pool,
            warmup_steps=warmup,
            timed_steps=steps,
            loss_extract_fn=lambda t: float(t.detach().item()),
        )
        res["peak_mem_gb"] = peak_memory_gb(device)
        res["graph_breaks"] = handle.graph_breaks

        if trace_out is not None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                _ = handle.step_fn(pool[0])
                sync_device(device)
            Path(trace_out).parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(trace_out)

        write_json_atomic(worker_out, res)
        return 0
    except Exception as exc:
        write_json_atomic(worker_out, {"error": repr(exc), "oom": "OutOfMemoryError" in repr(exc)})
        return 1


def _worker_jax(
    preset: str,
    device_kind: str,
    batch_size: int,
    warmup: int,
    steps: int,
    worker_out: str,
    trace_dir: str | None,
) -> int:
    import jax

    from bench.step_jax import build_jax_step, make_synthetic_pool, peak_memory_gb

    spec = get_spec(preset)
    precision = "fp16" if device_kind == "cuda" else "fp32"
    try:
        handle = build_jax_step(
            spec,
            sample_batch_size=batch_size,
            n_devices=1,
            precision=precision,
            seed=0,
        )
        pool = make_synthetic_pool(
            spec,
            batch_size,
            batch_sharding=handle.batch_sharding,
            num_batches=8,
            seed=1,
        )
        state_box = [handle.state]

        def step_wrapper(batch: jax.Array) -> jax.Array:
            new_state, loss = handle.step_fn(state_box[0], batch)
            state_box[0] = new_state
            return loss

        res = measure_loop(
            step_wrapper,
            lambda: jax.block_until_ready(state_box[0]),
            pool,
            warmup_steps=warmup,
            timed_steps=steps,
            loss_extract_fn=float,
        )
        res["compile_s"] = handle.compile_s
        res["peak_mem_gb"] = peak_memory_gb(list(handle.mesh.devices.flat))

        if trace_dir is not None:
            try:
                Path(trace_dir).mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(trace_dir)
                _ = step_wrapper(pool[0])
                jax.block_until_ready(state_box[0])
                jax.profiler.stop_trace()
            except Exception:
                pass

        write_json_atomic(worker_out, res)
        return 0
    except Exception as exc:
        is_oom = "RESOURCE_EXHAUSTED" in repr(exc) or "out of memory" in repr(exc).lower()
        write_json_atomic(worker_out, {"error": repr(exc), "oom": is_oom})
        return 1


def _spawn_worker(cmd: list[str], worker_out: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if worker_out.exists():
        data = json.loads(worker_out.read_text(encoding="utf-8"))
        if proc.returncode != 0:
            data.setdefault("stderr", proc.stderr[-1000:])
        return data
    return {"error": proc.stderr[-1000:] or f"Worker exited {proc.returncode}", "oom": False}


def _run_torch_parent(
    out_dir: Path,
    preset: str,
    device_kind: str,
    batch_star: int,
    repeats: int,
    warmup: int,
    steps: int,
) -> int:
    spec = get_spec(preset)
    flops = spec.train_flops_per_step(batch_star)
    rows: list[dict[str, Any]] = []
    precision = "fp16" if device_kind == "cuda" else "fp32"

    for mode in ("eager", "compile"):
        runs: list[dict[str, Any]] = []
        failed_err: str | None = None
        for r in range(repeats):
            with tempfile.TemporaryDirectory() as tmp:
                w_out = Path(tmp) / "worker.json"
                trace_arg = (
                    ["--trace-out", str(out_dir / "traces" / f"torch_{mode}.json")]
                    if r == 0
                    else []
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "bench.single",
                    "--worker",
                    "--framework",
                    "torch",
                    "--mode",
                    mode,
                    "--preset",
                    preset,
                    "--device",
                    device_kind,
                    "--batch",
                    str(batch_star),
                    "--warmup",
                    str(warmup),
                    "--steps",
                    str(steps),
                    "--worker-out",
                    str(w_out),
                    *trace_arg,
                ]
                res = _spawn_worker(cmd, w_out)
                if "error" in res:
                    failed_err = str(res["error"])
                    break
                runs.append(res)

        if failed_err is not None:
            rows.append(
                {
                    "framework": "torch",
                    "mode": mode,
                    "precision": precision,
                    "batch_per_gpu": batch_star,
                    "finite": False,
                    "error": failed_err,
                }
            )
            continue

        row = aggregate_process_runs(
            runs,
            global_batch=batch_star,
            seq_len=spec.seq_len,
            flops_per_step=flops,
        )
        row["framework"] = "torch"
        row["mode"] = mode
        row["precision"] = precision
        row["batch_per_gpu"] = batch_star
        if mode == "compile":
            row["graph_breaks"] = runs[0].get("graph_breaks")
        rows.append(row)
        print(
            f"[single:torch:{mode}] step={row['step_s']['median']*1000:.1f} ms "
            f"tok/s={row['tokens_per_s']['median']:.0f} mem={row['peak_mem_gb']:.2f} GB"
        )

    write_json_atomic(
        out_dir / "single_torch.json",
        {
            "stage": "single_torch",
            "preset": preset,
            "batch_per_gpu": batch_star,
            "rows": rows,
            "provenance": collect_provenance("torch"),
        },
    )
    eager_ok = any(r.get("mode") == "eager" and r.get("finite") for r in rows)
    return 0 if eager_ok else 1


def _run_jax_parent(
    out_dir: Path,
    preset: str,
    device_kind: str,
    batch_star: int,
    repeats: int,
    warmup: int,
    steps: int,
    *,
    force_gates: bool,
) -> int:
    spec = get_spec(preset)
    flops = spec.train_flops_per_step(batch_star)
    precision = "fp16" if device_kind == "cuda" else "fp32"

    runs: list[dict[str, Any]] = []
    failed_res: dict[str, Any] | None = None
    for r in range(repeats):
        with tempfile.TemporaryDirectory() as tmp:
            w_out = Path(tmp) / "worker.json"
            trace_arg = (
                ["--trace-out", str(out_dir / "traces" / "jax_single")] if r == 0 else []
            )
            cmd = [
                sys.executable,
                "-m",
                "bench.single",
                "--worker",
                "--framework",
                "jax",
                "--mode",
                "jit",
                "--preset",
                preset,
                "--device",
                device_kind,
                "--batch",
                str(batch_star),
                "--warmup",
                str(warmup),
                "--steps",
                str(steps),
                "--worker-out",
                str(w_out),
                *trace_arg,
            ]
            res = _spawn_worker(cmd, w_out)
            if "error" in res:
                failed_res = res
                break
            runs.append(res)

    torch_path = out_dir / "single_torch.json"
    if not torch_path.exists():
        raise SystemExit(f"Missing {torch_path}; run --framework torch first")
    torch_data = read_json(torch_path)
    rows = list(torch_data["rows"])

    if failed_res is not None:
        jax_row: dict[str, Any] = {
            "framework": "jax",
            "mode": "jit",
            "precision": precision,
            "batch_per_gpu": batch_star,
            "finite": False,
            "oom": bool(failed_res.get("oom", False)),
            "error": failed_res.get("error"),
        }
    else:
        jax_row = aggregate_process_runs(
            runs,
            global_batch=batch_star,
            seq_len=spec.seq_len,
            flops_per_step=flops,
        )
        jax_row["framework"] = "jax"
        jax_row["mode"] = "jit"
        jax_row["precision"] = precision
        jax_row["batch_per_gpu"] = batch_star
        print(
            f"[single:jax:jit] step={jax_row['step_s']['median']*1000:.1f} ms "
            f"tok/s={jax_row['tokens_per_s']['median']:.0f} mem={jax_row['peak_mem_gb']:.2f} GB"
        )

    rows.append(jax_row)

    valid_torch = [r for r in rows if r["framework"] == "torch" and r.get("finite")]
    chosen_torch_mode = (
        min(valid_torch, key=lambda r: r["step_s"]["median"])["mode"] if valid_torch else "eager"
    )
    all_ok = bool(valid_torch) and bool(jax_row.get("finite"))

    payload = {
        "stage": "single",
        "preset": preset,
        "spec": spec.to_dict(),
        "batch_per_gpu": batch_star,
        "rows": rows,
        "chosen_torch_mode": chosen_torch_mode,
        "passed": all_ok or force_gates,
        "forced": bool(force_gates and not all_ok),
        "provenance_torch": torch_data.get("provenance"),
        "provenance_jax": collect_provenance("jax"),
    }
    write_json_atomic(out_dir / "single.json", payload)
    print(f"[single] chosen_torch_mode={chosen_torch_mode} passed={payload['passed']}")
    return 0 if payload["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 3: 1-GPU benchmark")
    parser.add_argument("--framework", choices=["torch", "jax"], required=True)
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="medium")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", default=".")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--force-gates", action="store_true")

    # Worker-only flags
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=["eager", "compile", "jit"], default="eager")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--worker-out")
    parser.add_argument("--trace-out")
    args = parser.parse_args(argv)

    if args.worker:
        assert args.worker_out is not None
        if args.framework == "torch":
            return _worker_torch(
                args.preset,
                args.device,
                args.mode,
                args.batch,
                args.warmup,
                args.steps,
                args.worker_out,
                args.trace_out,
            )
        return _worker_jax(
            args.preset,
            args.device,
            args.batch,
            args.warmup,
            args.steps,
            args.worker_out,
            args.trace_out,
        )

    out_dir = ensure_results_dir(args.out)
    gate_data = require_stage_passed(out_dir, "gate.json", force=args.force_gates)
    batch_star = int(gate_data.get("batch_star", 4))

    if args.framework == "torch":
        return _run_torch_parent(
            out_dir, args.preset, args.device, batch_star, args.repeats, args.warmup, args.steps
        )
    return _run_jax_parent(
        out_dir,
        args.preset,
        args.device,
        batch_star,
        args.repeats,
        args.warmup,
        args.steps,
        force_gates=args.force_gates,
    )


if __name__ == "__main__":
    raise SystemExit(main())
