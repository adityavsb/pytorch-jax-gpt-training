"""Tests for timing summaries and allreduce linear bandwidth fitting."""

from __future__ import annotations

import pytest

from bench.timing import estimate_sync_time_s, fit_allreduce_linear, summarize_steps


def test_summarize_steps() -> None:
    s = summarize_steps([0.1, 0.2, 0.3, 0.4, 0.5], process_medians=[0.2, 0.3, 0.4])
    assert s["median"] == pytest.approx(0.3)
    assert s["min"] == pytest.approx(0.1)
    assert s["max"] == pytest.approx(0.5)
    assert s["count"] == 5
    assert s["process_medians"] == [0.2, 0.3, 0.4]


def test_fit_allreduce_linear_recovers_known_params() -> None:
    alpha_s = 50e-6  # 50 us
    beta_gbs = 4.0  # 4 GB/s -> 4e9 bytes/s
    sizes = [2**k * 1024 * 1024 for k in range(0, 9)]  # 1 MB .. 256 MB
    times = [alpha_s + n / (beta_gbs * 1e9) for n in sizes]

    fit = fit_allreduce_linear(sizes, times)
    assert fit["alpha_us"] == pytest.approx(50.0, rel=1e-4)
    assert fit["beta_gbs"] == pytest.approx(4.0, rel=1e-4)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)

    est = estimate_sync_time_s(fit, 1_200_000_000)
    assert est == pytest.approx(0.30005, rel=1e-4)
