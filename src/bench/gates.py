"""Pure pass/fail gate functions and threshold constants."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.spec import T4_EXPECTED_CC, T4_EXPECTED_GPU_NAME

ALLREDUCE_MIN_R2: float = 0.98
PARITY_MAX_ABS_DIFF: float = 1e-3

GATE_MIN_FP16_SPEEDUP: float = 1.5
GATE_MIN_MEM_GB: float = 8.0
GATE_TARGET_MEM_LOW_GB: float = 8.5
GATE_TARGET_MEM_HIGH_GB: float = 14.0
GATE_MIN_STEP_S: float = 0.05

WEAK_SCALE_SYNC_TOLERANCE: float = 1.25


def check_topo_gate(
    gpus: list[dict[str, str]],
    *,
    link_type: str,
    torch_gpus: int,
    jax_gpus: int,
) -> tuple[bool, list[str]]:
    """Verify exactly 2 Tesla T4s (CC 7.5), visible to both stacks, without unexpected NVLink."""
    reasons: list[str] = []
    if len(gpus) != 2:
        reasons.append(f"Expected 2 GPUs, found {len(gpus)}")
    for i, gpu in enumerate(gpus):
        name = gpu.get("name", "")
        cc = gpu.get("compute_cap", "")
        if T4_EXPECTED_GPU_NAME not in name:
            reasons.append(f"GPU {i} name {name!r} does not match {T4_EXPECTED_GPU_NAME!r}")
        if cc and cc != T4_EXPECTED_CC:
            reasons.append(f"GPU {i} compute cap {cc!r} != {T4_EXPECTED_CC!r}")
    if torch_gpus != 2:
        reasons.append(f"PyTorch sees {torch_gpus} GPUs (expected 2)")
    if jax_gpus != 2:
        reasons.append(f"JAX sees {jax_gpus} GPUs (expected 2)")
    if link_type.upper().startswith("NV"):
        reasons.append(f"Unexpected NVLink interconnect detected ({link_type}); update expectation")
    return (len(reasons) == 0), reasons


def check_allreduce_gate(r2: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if r2 < ALLREDUCE_MIN_R2:
        reasons.append(f"Allreduce linear fit R^2={r2:.4f} < {ALLREDUCE_MIN_R2}")
    return (len(reasons) == 0), reasons


def check_parity_gate(block_max_diff: float, full_max_diff: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if block_max_diff >= PARITY_MAX_ABS_DIFF:
        reasons.append(
            f"Block parity max abs diff {block_max_diff:.3e} >= {PARITY_MAX_ABS_DIFF:.1e}"
        )
    if full_max_diff >= PARITY_MAX_ABS_DIFF:
        reasons.append(
            f"Full model parity max abs diff {full_max_diff:.3e} >= {PARITY_MAX_ABS_DIFF:.1e}"
        )
    return (len(reasons) == 0), reasons


def check_workload_gate(
    *,
    fp32_steps: list[float],
    fp16_cmp_steps: list[float],
    fp16_star_steps: list[float],
    peak_mem_star_gb: float,
) -> dict[str, Any]:
    """Check FP16 speedup over FP32 at b_cmp, plus memory and step duration at b*."""
    fp32_med = float(np.median(fp32_steps))
    fp16_cmp_med = float(np.median(fp16_cmp_steps))
    fp16_star_med = float(np.median(fp16_star_steps))

    speedup = fp32_med / fp16_cmp_med if fp16_cmp_med > 0 else 0.0
    non_overlapping = float(np.max(fp16_cmp_steps)) < float(np.min(fp32_steps))

    reasons: list[str] = []
    if speedup < GATE_MIN_FP16_SPEEDUP:
        reasons.append(
            f"FP16 speedup {speedup:.2f}x < {GATE_MIN_FP16_SPEEDUP:.2f}x at common batch"
        )
    if not non_overlapping:
        reasons.append("FP16 max step >= FP32 min step at common batch (distributions overlap)")
    if peak_mem_star_gb < GATE_MIN_MEM_GB:
        reasons.append(f"Peak memory {peak_mem_star_gb:.2f} GB < {GATE_MIN_MEM_GB:.1f} GB")
    if fp16_star_med < GATE_MIN_STEP_S:
        reasons.append(
            f"FP16 median step {fp16_star_med * 1000:.1f} ms < {GATE_MIN_STEP_S * 1000:.0f} ms"
        )

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "speedup_fp16_vs_fp32": speedup,
        "non_overlapping": non_overlapping,
    }


def check_weak_scaling_gate(
    *,
    t1_s: float,
    t2_s: float,
    sync_estimate_s: float,
) -> dict[str, Any]:
    """Check whether 2-GPU step overhead matches the measured allreduce estimate."""
    delta_s = t2_s - t1_s
    limit_s = WEAK_SCALE_SYNC_TOLERANCE * sync_estimate_s
    passed = delta_s <= limit_s
    reasons = []
    if not passed:
        reasons.append(
            f"Weak-scaling step delta {delta_s * 1000:.1f} ms exceeds "
            f"{WEAK_SCALE_SYNC_TOLERANCE:.2f}x sync estimate ({limit_s * 1000:.1f} ms)"
        )
    return {
        "passed": passed,
        "reasons": reasons,
        "delta_s": delta_s,
        "limit_s": limit_s,
    }


def check_strong_scaling_flag(
    *, weak_efficiency: float, strong_efficiency: float
) -> dict[str, Any]:
    """Flag if strong scaling efficiency unexpectedly meets or exceeds weak scaling."""
    anomaly = strong_efficiency >= weak_efficiency
    return {
        "strong_ge_weak": anomaly,
        "note": (
            "Strong scaling efficiency >= weak scaling; sync cost is not the sole bottleneck."
            if anomaly
            else "Strong scaling efficiency < weak scaling as expected."
        ),
    }
