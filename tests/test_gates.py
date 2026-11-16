"""Tests for pure stage-gate logic."""

from __future__ import annotations

from bench.gates import (
    check_allreduce_gate,
    check_parity_gate,
    check_strong_scaling_flag,
    check_topo_gate,
    check_weak_scaling_gate,
    check_workload_gate,
)


def test_topo_gate() -> None:
    gpus = [
        {"name": "Tesla T4", "compute_cap": "7.5"},
        {"name": "Tesla T4", "compute_cap": "7.5"},
    ]
    ok, reasons = check_topo_gate(gpus, link_type="PHB", torch_gpus=2, jax_gpus=2)
    assert ok and not reasons

    bad_nv, nv_reasons = check_topo_gate(gpus, link_type="NV4", torch_gpus=2, jax_gpus=2)
    assert not bad_nv and nv_reasons

    bad_count, _ = check_topo_gate(gpus[:1], link_type="PHB", torch_gpus=1, jax_gpus=1)
    assert not bad_count


def test_allreduce_and_parity_gates() -> None:
    assert check_allreduce_gate(0.995)[0]
    assert not check_allreduce_gate(0.90)[0]

    assert check_parity_gate(1e-5, 5e-5)[0]
    assert not check_parity_gate(1e-5, 2e-3)[0]


def test_workload_gate() -> None:
    res = check_workload_gate(
        fp32_steps=[0.70, 0.72, 0.71],
        fp16_cmp_steps=[0.32, 0.33, 0.34],
        fp16_star_steps=[0.34, 0.35, 0.36],
        peak_mem_star_gb=12.1,
    )
    assert res["passed"]
    assert res["speedup_fp16_vs_fp32"] > 2.0

    low_mem = check_workload_gate(
        fp32_steps=[0.70, 0.71],
        fp16_cmp_steps=[0.30, 0.31],
        fp16_star_steps=[0.30, 0.31],
        peak_mem_star_gb=6.5,
    )
    assert not low_mem["passed"]

    overlap = check_workload_gate(
        fp32_steps=[0.35, 0.70],
        fp16_cmp_steps=[0.30, 0.36],
        fp16_star_steps=[0.30, 0.36],
        peak_mem_star_gb=12.0,
    )
    assert not overlap["passed"]


def test_scaling_gates() -> None:
    ok = check_weak_scaling_gate(t1_s=0.33, t2_s=0.60, sync_estimate_s=0.30)
    assert ok["passed"]

    too_slow = check_weak_scaling_gate(t1_s=0.33, t2_s=0.80, sync_estimate_s=0.30)
    assert not too_slow["passed"]

    flag = check_strong_scaling_flag(weak_efficiency=0.72, strong_efficiency=0.75)
    assert flag["strong_ge_weak"]
