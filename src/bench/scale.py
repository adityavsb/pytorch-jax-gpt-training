"""Experiments 4 & 5: 2-GPU weak scaling (DDP / mesh) and strong scaling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

from bench.gates import check_strong_scaling_flag, check_weak_scaling_gate  # noqa: E402
from bench.results import (  # noqa: E402
    collect_provenance,
    ensure_results_dir,
    read_json,
    require_stage_passed,
    write_json_atomic,
)
from bench.spec import get_spec  # noqa: E402
from bench.timing import aggregate_process_runs, measure_loop  # noqa: E402


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _torch_worker_dist(
    rank: int,
    world_size: int,
    preset: str,
    device_kind: str,
    mode: str,
    local_batch: int,
    warmup: int,
    steps: int,
    worker_out: str,
    trace_out: str | None,
    init_method: str | None = None,
) -> None:
    import torch
    import torch.distributed as dist

    from bench.step_torch import (
        build_torch_step,
        cleanup_distributed,
        make_synthetic_pool,
        peak_memory_gb,
        reset_peak_memory,
        sync_device,
    )

    backend = "nccl" if device_kind == "cuda" else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if init_method is not None:
        dist.init_process_group(
            backend=backend, init_method=init_method, rank=rank, world_size=world_size
        )
    else:
        dist.init_process_group(backend=backend)
        if device_kind == "cuda":
            torch.cuda.set_device(local_rank)

    try:
        device = (
            torch.device(f"cuda:{local_rank}") if device_kind == "cuda" else torch.device("cpu")
        )
        precision = "fp16" if device_kind == "cuda" else "fp32"
        spec = get_spec(preset)

        handle = build_torch_step(
            spec,
            device=device,
            mode=mode,
            precision=precision,
            distributed=True,
            local_rank=local_rank,
            seed=0,
        )
        pool = make_synthetic_pool(
            spec,
            local_batch,
            device=device,
            num_batches=8,
            seed=100 + rank,
        )
        reset_peak_memory(device)
        dist.barrier()
        res = measure_loop(
            handle.step_fn,
            lambda: sync_device(device),
            pool,
            warmup_steps=warmup,
            timed_steps=steps,
            loss_extract_fn=lambda t: float(t.detach().item()),
        )
        res["peak_mem_gb"] = peak_memory_gb(device)

        if trace_out is not None and rank == 0:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                _ = handle.step_fn(pool[0])
                sync_device(device)
            Path(trace_out).parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(trace_out)

        if rank == 0:
            write_json_atomic(worker_out, res)
    except Exception as exc:
        if rank == 0:
            write_json_atomic(
                worker_out,
                {"error": repr(exc), "oom": "OutOfMemoryError" in repr(exc)},
            )
        raise
    finally:
        cleanup_distributed()


def _worker_jax_2gpu(
    preset: str,
    device_kind: str,
    global_batch: int,
    warmup: int,
    steps: int,
    worker_out: str,
    trace_dir: str | None,
) -> int:
    import jax

    from bench.step_jax import (
        build_jax_step,
        inspect_hlo_allreduce,
        make_synthetic_pool,
        peak_memory_gb,
    )

    spec = get_spec(preset)
    precision = "fp16" if device_kind == "cuda" else "fp32"
    try:
        handle = build_jax_step(
            spec,
            sample_batch_size=global_batch,
            n_devices=2,
            precision=precision,
            seed=0,
        )
        pool = make_synthetic_pool(
            spec,
            global_batch,
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
        res["hlo_allreduce"] = inspect_hlo_allreduce(handle.hlo_text)

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


def _run_torch_repeat(
    preset: str,
    device_kind: str,
    mode: str,
    local_batch: int,
    warmup: int,
    steps: int,
    trace_out: str | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        w_out = Path(tmp) / "worker.json"
        trace_arg = ["--trace-out", trace_out] if trace_out else []
        if device_kind == "cuda":
            torchrun_path = Path(sys.executable).parent / "torchrun"
            torchrun_bin = (
                str(torchrun_path) if torchrun_path.exists() else shutil.which("torchrun")
            )
            if not torchrun_bin:
                raise SystemExit("torchrun executable not found")
            cmd = [
                torchrun_bin,
                "--nproc_per_node=2",
                f"--master_port={_find_free_port()}",
                "-m",
                "bench.scale",
                "--worker",
                "--framework",
                "torch",
                "--mode",
                mode,
                "--preset",
                preset,
                "--device",
                device_kind,
                "--local-batch",
                str(local_batch),
                "--warmup",
                str(warmup),
                "--steps",
                str(steps),
                "--worker-out",
                str(w_out),
                *trace_arg,
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "bench.scale",
                "--worker",
                "--framework",
                "torch",
                "--mode",
                mode,
                "--preset",
                preset,
                "--device",
                "cpu",
                "--local-batch",
                str(local_batch),
                "--warmup",
                str(warmup),
                "--steps",
                str(steps),
                "--worker-out",
                str(w_out),
                *trace_arg,
            ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if w_out.exists():
            return dict(json.loads(w_out.read_text(encoding="utf-8")))
        return {"error": proc.stderr[-1000:] or f"Worker failed ({proc.returncode})"}


def _run_jax_repeat(
    preset: str,
    device_kind: str,
    global_batch: int,
    warmup: int,
    steps: int,
    trace_dir: str | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        w_out = Path(tmp) / "worker.json"
        trace_arg = ["--trace-out", trace_dir] if trace_dir else []
        cmd = [
            sys.executable,
            "-m",
            "bench.scale",
            "--worker",
            "--framework",
            "jax",
            "--preset",
            preset,
            "--device",
            device_kind,
            "--local-batch",
            str(global_batch // 2),
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--worker-out",
            str(w_out),
            *trace_arg,
        ]
        env = os.environ.copy()
        if device_kind == "cpu":
            env.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if w_out.exists():
            return dict(json.loads(w_out.read_text(encoding="utf-8")))
        return {"error": proc.stderr[-1000:] or f"JAX worker failed ({proc.returncode})"}


def _lookup_1gpu_baseline(single_data: dict[str, Any], framework: str, mode: str) -> dict[str, Any]:
    for row in single_data["rows"]:
        if row["framework"] == framework and row["mode"] == mode:
            return row
    raise SystemExit(f"Missing 1-GPU baseline row for {framework}/{mode} in single.json")


def _update_scale_json(
    out_dir: Path,
    regime: str,
    framework: str,
    entry: dict[str, Any],
    *,
    force_gates: bool,
) -> dict[str, Any]:
    scale_path = out_dir / "scale.json"
    scale_data = read_json(scale_path) if scale_path.exists() else {"stage": "scale"}
    section = scale_data.get(regime, {"rows": {}})
    section["rows"][framework] = entry

    if regime == "weak":
        all_rows = section["rows"]
        passed = all(r.get("passed", False) for r in all_rows.values()) and len(all_rows) > 0
        section["passed"] = passed or force_gates
        section["forced"] = bool(force_gates and not passed)
    elif regime == "strong" and "weak" in scale_data:
        weak_rows = scale_data["weak"].get("rows", {})
        if framework in weak_rows and entry.get("finite") and weak_rows[framework].get("finite"):
            entry["strong_vs_weak"] = check_strong_scaling_flag(
                weak_efficiency=float(weak_rows[framework]["scaling_efficiency"]),
                strong_efficiency=float(entry["scaling_efficiency"]),
            )
        section["passed"] = all(r.get("finite", False) for r in section["rows"].values())

    scale_data[regime] = section
    scale_data["passed"] = bool(scale_data.get("weak", {}).get("passed", False))
    write_json_atomic(scale_path, scale_data)
    return scale_data


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--framework", choices=["torch", "jax"], required=True)
    pre.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    known, _ = pre.parse_known_args(argv)
    if known.framework == "jax" and known.device == "cpu":
        os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

    parser = argparse.ArgumentParser(description="Experiments 4 & 5: 2-GPU weak/strong scaling")
    regime_group = parser.add_mutually_exclusive_group()
    regime_group.add_argument("--weak", action="store_true")
    regime_group.add_argument("--strong", action="store_true")
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
    parser.add_argument("--mode", default="eager")
    parser.add_argument("--local-batch", type=int, default=2)
    parser.add_argument("--worker-out")
    parser.add_argument("--trace-out")
    args = parser.parse_args(argv)

    if args.worker:
        assert args.worker_out is not None
        if args.framework == "torch":
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                _torch_worker_dist(
                    int(os.environ["RANK"]),
                    int(os.environ["WORLD_SIZE"]),
                    args.preset,
                    args.device,
                    args.mode,
                    args.local_batch,
                    args.warmup,
                    args.steps,
                    args.worker_out,
                    args.trace_out,
                )
            else:
                import torch.multiprocessing as mp

                with tempfile.TemporaryDirectory() as tmp:
                    init_file = f"file://{Path(tmp) / 'init'}"
                    mp.spawn(
                        _torch_worker_dist,
                        args=(
                            2,
                            args.preset,
                            args.device,
                            args.mode,
                            args.local_batch,
                            args.warmup,
                            args.steps,
                            args.worker_out,
                            args.trace_out,
                            init_file,
                        ),
                        nprocs=2,
                        join=True,
                    )
            return 0
        return _worker_jax_2gpu(
            args.preset,
            args.device,
            args.local_batch * 2,
            args.warmup,
            args.steps,
            args.worker_out,
            args.trace_out,
        )

    if not args.weak and not args.strong:
        raise SystemExit("Specify either --weak or --strong")
    regime = "weak" if args.weak else "strong"

    out_dir = ensure_results_dir(args.out)
    single_data = require_stage_passed(out_dir, "single.json", force=args.force_gates)
    allreduce_data = require_stage_passed(out_dir, "allreduce.json", force=args.force_gates)
    gate_data = require_stage_passed(out_dir, "gate.json", force=args.force_gates)

    if regime == "strong":
        scale_prev = read_json(out_dir / "scale.json") if (out_dir / "scale.json").exists() else {}
        if not scale_prev.get("weak", {}).get("passed", False) and not args.force_gates:
            raise SystemExit("Gate blocked: weak scaling must pass before running strong scaling.")

    batch_star = int(gate_data.get("batch_star", 4))
    if regime == "weak":
        local_batch = batch_star
        global_batch = 2 * batch_star
    else:
        if batch_star < 2 or batch_star % 2 != 0:
            raise SystemExit(f"Strong scaling requires even batch_star >= 2, got {batch_star}")
        local_batch = batch_star // 2
        global_batch = batch_star

    chosen_mode = str(single_data.get("chosen_torch_mode", "eager"))
    mode = chosen_mode if args.framework == "torch" else "jit"
    baseline = _lookup_1gpu_baseline(single_data, args.framework, mode)
    if not baseline.get("finite", False):
        raise SystemExit(f"1-GPU baseline for {args.framework}/{mode} is invalid/failed.")

    spec = get_spec(args.preset)
    flops = spec.train_flops_per_step(global_batch)
    sync_est_s = float(allreduce_data.get("sync_estimate_s", 0.0))

    runs: list[dict[str, Any]] = []
    err_info: dict[str, Any] | None = None
    for _ in range(args.repeats):
        res = (
            _run_torch_repeat(
                args.preset, args.device, mode, local_batch, args.warmup, args.steps, None
            )
            if args.framework == "torch"
            else _run_jax_repeat(
                args.preset, args.device, global_batch, args.warmup, args.steps, None
            )
        )
        if "error" in res:
            err_info = res
            break
        runs.append(res)

    if err_info is not None:
        entry: dict[str, Any] = {
            "framework": args.framework,
            "mode": mode,
            "regime": regime,
            "local_batch": local_batch,
            "global_batch": global_batch,
            "finite": False,
            "passed": bool(args.force_gates),
            "error": err_info.get("error"),
        }
        _update_scale_json(out_dir, regime, args.framework, entry, force_gates=args.force_gates)
        return 0 if args.force_gates else 1

    entry = aggregate_process_runs(
        runs,
        global_batch=global_batch,
        seq_len=spec.seq_len,
        flops_per_step=flops,
    )
    entry["framework"] = args.framework
    entry["mode"] = mode
    entry["regime"] = regime
    entry["local_batch"] = local_batch
    entry["global_batch"] = global_batch
    entry["provenance"] = collect_provenance(args.framework)
    if args.framework == "jax":
        entry["hlo_allreduce"] = runs[0].get("hlo_allreduce")

    t1_s = float(baseline["step_s"]["median"])
    t2_s = float(entry["step_s"]["median"])
    tok1 = float(baseline["tokens_per_s"]["median"])
    tok2 = float(entry["tokens_per_s"]["median"])
    entry["scaling_efficiency"] = tok2 / (2.0 * tok1)
    entry["delta_step_s"] = t2_s - t1_s

    if regime == "weak":
        gate_eval = check_weak_scaling_gate(t1_s=t1_s, t2_s=t2_s, sync_estimate_s=sync_est_s)
        entry["sync_estimate_s"] = sync_est_s
        entry["passed"] = bool(gate_eval["passed"]) or bool(args.force_gates)
        entry["reasons"] = gate_eval["reasons"]
        if not gate_eval["passed"] and not args.force_gates:
            # Capture a diagnostic trace on failure as specified by the design.
            trace_target = (
                str(out_dir / "traces" / "torch_ddp_fail.json")
                if args.framework == "torch"
                else str(out_dir / "traces" / "jax_2gpu_fail")
            )
            if args.framework == "torch":
                _run_torch_repeat(args.preset, args.device, mode, local_batch, 2, 2, trace_target)
            else:
                _run_jax_repeat(args.preset, args.device, global_batch, 2, 2, trace_target)
    else:
        entry["passed"] = bool(entry["finite"])

    _update_scale_json(out_dir, regime, args.framework, entry, force_gates=args.force_gates)
    print(
        f"[scale:{regime}:{args.framework}] global_batch={global_batch} "
        f"step={t2_s * 1000:.1f} ms tok/s={tok2:.0f} eff={entry['scaling_efficiency'] * 100:.1f}% "
        f"delta={entry['delta_step_s'] * 1000:+.1f} ms"
    )
    return 0 if entry["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
