"""Timing loops, statistical summaries, and allreduce linear bandwidth fitting."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def summarize_steps(
    step_times_s: list[float],
    *,
    process_medians: list[float] | None = None,
) -> dict[str, Any]:
    """Compute robust summary statistics over step durations in seconds."""
    if not step_times_s:
        raise ValueError("step_times_s must be non-empty")
    arr = np.asarray(step_times_s, dtype=np.float64)
    out: dict[str, Any] = {
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "count": int(arr.size),
    }
    if process_medians is not None:
        out["process_medians"] = [float(x) for x in process_medians]
    return out


def measure_loop(
    step_fn: Callable[[Any], Any],
    sync_fn: Callable[[], None],
    pool: list[Any],
    *,
    warmup_steps: int = 5,
    timed_steps: int = 30,
    loss_extract_fn: Callable[[Any], float] | None = None,
) -> dict[str, Any]:
    """Run warmup_steps (recording first_step_s) followed by timed_steps with device syncs."""
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    if timed_steps < 1:
        raise ValueError("timed_steps must be >= 1")

    def _to_loss(out: Any) -> float:
        if loss_extract_fn is not None:
            return float(loss_extract_fn(out))
        return float(out)

    # Step 0: includes JIT / Inductor warmup if not precompiled.
    sync_fn()
    t0 = time.perf_counter()
    out0 = step_fn(pool[0])
    sync_fn()
    first_step_s = time.perf_counter() - t0
    loss_first = _to_loss(out0)

    for i in range(1, warmup_steps):
        _ = step_fn(pool[i % len(pool)])
    sync_fn()

    step_times_s: list[float] = []
    last_out: Any = out0
    for i in range(timed_steps):
        batch = pool[(warmup_steps + i) % len(pool)]
        t_start = time.perf_counter()
        last_out = step_fn(batch)
        sync_fn()
        step_times_s.append(time.perf_counter() - t_start)

    loss_last = _to_loss(last_out)
    finite = math.isfinite(loss_first) and math.isfinite(loss_last)
    return {
        "first_step_s": float(first_step_s),
        "step_times_s": step_times_s,
        "loss_first": loss_first,
        "loss_last": loss_last,
        "finite": finite,
    }


def aggregate_process_runs(
    runs: list[dict[str, Any]],
    *,
    global_batch: int,
    seq_len: int,
    flops_per_step: float,
) -> dict[str, Any]:
    """Aggregate multiple isolated process runs into one result row."""
    all_steps: list[float] = []
    proc_medians: list[float] = []
    first_steps: list[float] = []
    peaks_gb: list[float] = []
    losses_first: list[float] = []
    losses_last: list[float] = []
    compile_times: list[float] = []
    all_finite = True

    for run in runs:
        steps = [float(x) for x in run["step_times_s"]]
        all_steps.extend(steps)
        proc_medians.append(float(np.median(steps)))
        first_steps.append(float(run["first_step_s"]))
        peaks_gb.append(float(run.get("peak_mem_gb", 0.0)))
        losses_first.append(float(run["loss_first"]))
        losses_last.append(float(run["loss_last"]))
        all_finite = all_finite and bool(run.get("finite", True))
        if "compile_s" in run and run["compile_s"] is not None:
            compile_times.append(float(run["compile_s"]))

    summary = summarize_steps(all_steps, process_medians=proc_medians)
    tokens_per_step = global_batch * seq_len
    med_s = summary["median"]
    min_s = summary["min"]
    max_s = summary["max"]

    result: dict[str, Any] = {
        "step_s": summary,
        "tokens_per_s": {
            "median": tokens_per_step / med_s,
            "min": tokens_per_step / max_s,
            "max": tokens_per_step / min_s,
        },
        "achieved_tflops": (flops_per_step / med_s) / 1e12,
        "peak_mem_gb": max(peaks_gb) if peaks_gb else 0.0,
        "first_step_s": float(np.median(first_steps)),
        "loss_first": float(np.median(losses_first)),
        "loss_last": float(np.median(losses_last)),
        "finite": all_finite,
        "n_processes": len(runs),
    }
    if compile_times:
        result["compile_s"] = float(np.median(compile_times))
    return result


def fit_allreduce_linear(sizes_bytes: list[int], times_s: list[float]) -> dict[str, float]:
    """Fit t = alpha + n / beta to allreduce measurements and compute R^2."""
    if len(sizes_bytes) != len(times_s) or len(sizes_bytes) < 2:
        raise ValueError("Need at least 2 matching (size, time) points")
    x = np.asarray(sizes_bytes, dtype=np.float64)
    y = np.asarray(times_s, dtype=np.float64)

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    beta_gbs = (1.0 / slope) / 1e9 if slope > 0 else float("inf")

    return {
        "alpha_us": float(intercept * 1e6),
        "beta_gbs": float(beta_gbs),
        "r2": float(r2),
        "intercept_s": float(intercept),
        "slope_s_per_byte": float(slope),
    }


def estimate_sync_time_s(fit: dict[str, float], payload_bytes: int) -> float:
    """Estimate allreduce wall time (seconds) from a fitted linear model."""
    return float(fit["intercept_s"] + payload_bytes * fit["slope_s_per_byte"])
