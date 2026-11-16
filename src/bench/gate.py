"""Experiment 2: Workload gate on 1 T4 GPU (PyTorch batch search, FP32 vs FP16, memory, TFLOP/s)."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import torch

from bench.gates import (
    GATE_TARGET_MEM_HIGH_GB,
    GATE_TARGET_MEM_LOW_GB,
    check_workload_gate,
)
from bench.results import (
    collect_provenance,
    ensure_results_dir,
    require_stage_passed,
    write_json_atomic,
)
from bench.spec import T4_FP16_PEAK_TFLOPS, get_spec
from bench.step_torch import (
    build_torch_step,
    make_synthetic_pool,
    peak_memory_gb,
    reset_peak_memory,
    sync_device,
)
from bench.timing import measure_loop, summarize_steps


def _probe_batch_cuda(preset: str, batch_size: int, precision: str) -> tuple[bool, float]:
    spec = get_spec(preset)
    device = torch.device("cuda:0")
    handle = None
    pool = None
    try:
        handle = build_torch_step(spec, device=device, mode="eager", precision=precision, seed=0)
        pool = make_synthetic_pool(spec, batch_size, device=device, num_batches=2, seed=1)
        reset_peak_memory(device)
        for i in range(3):
            loss = handle.step_fn(pool[i % len(pool)])
            sync_device(device)
            _ = float(loss.detach().item())
        mem_gb = peak_memory_gb(device)
        return True, mem_gb
    except torch.cuda.OutOfMemoryError:
        return False, float("inf")
    finally:
        del handle, pool
        gc.collect()
        torch.cuda.empty_cache()


def _find_batches_cuda(preset: str) -> tuple[int, float, int, list[dict[str, Any]]]:
    search_log: list[dict[str, Any]] = []
    b_star: int | None = None
    b_star_mem = 0.0

    for b in range(2, 66, 2):
        ok, mem = _probe_batch_cuda(preset, b, "fp16")
        search_log.append({"batch": b, "precision": "fp16", "ok": ok, "peak_mem_gb": mem})
        print(f"[gate] probe fp16 batch={b} ok={ok} peak_mem={mem:.2f} GB")
        if not ok or mem > GATE_TARGET_MEM_HIGH_GB:
            break
        b_star = b
        b_star_mem = mem
        if mem >= GATE_TARGET_MEM_LOW_GB:
            # Stay inside [10, 14] GB; keep checking if next even batch still fits <= 14 GB.
            continue

    if b_star is None:
        raise SystemExit("Even batch=2 exceeded 14 GB or OOM'd on FP16; workload does not fit.")

    candidates = list(range(b_star, 0, -2))
    if 1 not in candidates:
        candidates.append(1)

    b_cmp: int | None = None
    for b in candidates:
        ok, mem = _probe_batch_cuda(preset, b, "fp32")
        search_log.append({"batch": b, "precision": "fp32", "ok": ok, "peak_mem_gb": mem})
        print(f"[gate] probe fp32 batch={b} ok={ok} peak_mem={mem:.2f} GB")
        if ok:
            b_cmp = b
            break

    if b_cmp is None:
        raise SystemExit("No batch size >= 1 fit in FP32 for the precision comparison gate.")
    return b_star, b_star_mem, b_cmp, search_log


def _time_torch_config(
    preset: str,
    device: torch.device,
    batch_size: int,
    precision: str,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    spec = get_spec(preset)
    handle = build_torch_step(spec, device=device, mode="eager", precision=precision, seed=0)
    pool = make_synthetic_pool(spec, batch_size, device=device, num_batches=8, seed=1)
    reset_peak_memory(device)
    try:
        res = measure_loop(
            handle.step_fn,
            lambda: sync_device(device),
            pool,
            warmup_steps=warmup,
            timed_steps=steps,
            loss_extract_fn=lambda t: float(t.detach().item()),
        )
        res["peak_mem_gb"] = peak_memory_gb(device)
        res["summary"] = summarize_steps(res["step_times_s"])
        med_s = res["summary"]["median"]
        flops = spec.train_flops_per_step(batch_size)
        res["tokens_per_s"] = (batch_size * spec.seq_len) / med_s
        res["achieved_tflops"] = (flops / med_s) / 1e12
        return res
    finally:
        del handle, pool
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 2: 1-GPU workload & FP16 gate")
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="medium")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--force-gates", action="store_true")
    args = parser.parse_args(argv)

    out_dir = ensure_results_dir(args.out)
    require_stage_passed(out_dir, "parity.json", force=args.force_gates)

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    spec = get_spec(args.preset)

    if args.device == "cuda":
        b_star, _, b_cmp, search_log = _find_batches_cuda(args.preset)
    else:
        b_star, b_cmp, search_log = 4, 4, []

    fp32_cmp = _time_torch_config(args.preset, device, b_cmp, "fp32", args.warmup, args.steps)
    fp16_cmp = _time_torch_config(args.preset, device, b_cmp, "fp16", args.warmup, args.steps)
    fp16_star = (
        fp16_cmp
        if b_star == b_cmp
        else _time_torch_config(args.preset, device, b_star, "fp16", args.warmup, args.steps)
    )

    gate_eval = check_workload_gate(
        fp32_steps=fp32_cmp["step_times_s"],
        fp16_cmp_steps=fp16_cmp["step_times_s"],
        fp16_star_steps=fp16_star["step_times_s"],
        peak_mem_star_gb=float(fp16_star["peak_mem_gb"]),
    )

    passed = bool(gate_eval["passed"])
    payload: dict[str, Any] = {
        "stage": "gate",
        "preset": args.preset,
        "spec": spec.to_dict(),
        "batch_star": b_star,
        "batch_cmp": b_cmp,
        "search_log": search_log,
        "fp32_cmp": fp32_cmp,
        "fp16_cmp": fp16_cmp,
        "fp16_star": fp16_star,
        "t4_fp16_peak_tflops": T4_FP16_PEAK_TFLOPS,
        "pct_of_fp16_peak": (fp16_star["achieved_tflops"] / T4_FP16_PEAK_TFLOPS) * 100.0,
        "speedup_fp16_vs_fp32": gate_eval["speedup_fp16_vs_fp32"],
        "non_overlapping": gate_eval["non_overlapping"],
        "passed": passed or args.force_gates,
        "forced": bool(args.force_gates and not passed),
        "reasons": gate_eval["reasons"],
        "provenance": collect_provenance("torch"),
    }
    write_json_atomic(Path(out_dir) / "gate.json", payload)
    print(
        f"[gate] b*={b_star} b_cmp={b_cmp} speedup={gate_eval['speedup_fp16_vs_fp32']:.2f}x "
        f"mem={fp16_star['peak_mem_gb']:.2f} GB tflops={fp16_star['achieved_tflops']:.1f} "
        f"passed={payload['passed']}"
    )
    if not payload["passed"]:
        print(f"Workload gate failed: {gate_eval['reasons']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
