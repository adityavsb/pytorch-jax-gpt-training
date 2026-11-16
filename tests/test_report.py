"""Golden-fixture tests for README report rendering and marker preservation."""

from __future__ import annotations

from pathlib import Path

from bench.report import (
    END_MARKER,
    START_MARKER,
    plot_allreduce,
    render_report_block,
    update_readme,
)
from bench.results import write_json_atomic


def _write_golden_results(d: Path) -> None:
    write_json_atomic(
        d / "topo.json",
        {
            "gpus": [
                {"name": "Tesla T4", "compute_cap": "7.5", "memory_mib": "15360"},
                {"name": "Tesla T4", "compute_cap": "7.5", "memory_mib": "15360"},
            ],
            "link_type": "PHB",
            "driver": "550.90.07",
            "provenance_torch": {
                "python": "3.11.10",
                "git_sha": "abc1234",
                "torch": {"torch": "2.5.1", "cuda": "12.4", "cudnn": 90100, "nccl": "2.21.5"},
            },
            "provenance_jax": {"jax": {"jax": "0.4.35", "jaxlib": "0.4.35"}},
        },
    )
    pts = [
        {"bytes": 1_000_000, "median_s": 0.0003},
        {"bytes": 100_000_000, "median_s": 0.0253},
    ]
    fit = {
        "alpha_us": 50.0,
        "beta_gbs": 4.0,
        "r2": 0.9999,
        "intercept_s": 5e-5,
        "slope_s_per_byte": 2.5e-10,
    }
    write_json_atomic(
        d / "allreduce.json",
        {
            "grad_bytes_fp32": 1_215_537_152,
            "sync_estimate_s": 0.304,
            "torch": {"fp32": {"points": pts, "fit": fit}},
            "jax": {"fp32": {"points": pts, "fit": fit}},
        },
    )
    write_json_atomic(
        d / "parity.json",
        {"block_max_abs_diff": 3.1e-6, "full_max_abs_diff": 4.8e-5, "passed": True},
    )
    write_json_atomic(
        d / "gate.json",
        {
            "batch_star": 8,
            "batch_cmp": 4,
            "fp32_cmp": {"summary": {"median": 0.710}},
            "fp16_cmp": {"summary": {"median": 0.330}},
            "fp16_star": {
                "summary": {"median": 0.350},
                "peak_mem_gb": 12.1,
                "achieved_tflops": 24.4,
            },
            "speedup_fp16_vs_fp32": 2.15,
            "passed": True,
        },
    )
    write_json_atomic(
        d / "single.json",
        {
            "chosen_torch_mode": "eager",
            "provenance_torch": {
                "python": "3.11.10",
                "git_sha": "abc1234",
                "torch": {"torch": "2.5.1", "cuda": "12.4", "cudnn": 90100, "nccl": "2.21.5"},
            },
            "rows": [
                {
                    "framework": "torch",
                    "mode": "eager",
                    "finite": True,
                    "step_s": {"median": 0.331},
                    "tokens_per_s": {"median": 12375, "min": 12000, "max": 12500},
                },
                {
                    "framework": "torch",
                    "mode": "compile",
                    "finite": True,
                    "step_s": {"median": 0.360},
                    "tokens_per_s": {"median": 11377, "min": 11000, "max": 11600},
                },
                {
                    "framework": "jax",
                    "mode": "jit",
                    "finite": True,
                    "step_s": {"median": 0.335},
                    "tokens_per_s": {"median": 12226, "min": 12100, "max": 12400},
                },
            ],
        },
    )
    write_json_atomic(
        d / "scale.json",
        {
            "weak": {
                "rows": {
                    "torch": {
                        "finite": True,
                        "tokens_per_s": {"median": 13500, "min": 13200, "max": 13800},
                        "scaling_efficiency": 0.545,
                        "delta_step_s": 0.275,
                        "sync_estimate_s": 0.304,
                    },
                    "jax": {
                        "finite": True,
                        "tokens_per_s": {"median": 13200, "min": 13000, "max": 13400},
                        "scaling_efficiency": 0.540,
                        "delta_step_s": 0.285,
                        "sync_estimate_s": 0.304,
                    },
                }
            },
            "strong": {
                "rows": {
                    "torch": {
                        "finite": True,
                        "tokens_per_s": {"median": 9200, "min": 9000, "max": 9400},
                        "scaling_efficiency": 0.372,
                        "strong_vs_weak": {"strong_ge_weak": False},
                    },
                    "jax": {
                        "finite": True,
                        "tokens_per_s": {"median": 9000, "min": 8900, "max": 9100},
                        "scaling_efficiency": 0.368,
                        "strong_vs_weak": {"strong_ge_weak": False},
                    },
                }
            },
        },
    )


def test_render_and_update_readme_preserves_surrounding_text(tmp_path: Path) -> None:
    _write_golden_results(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Header\n\n{START_MARKER}\nold content\n{END_MARKER}\n\n## Footer\nKeep me.\n",
        encoding="utf-8",
    )

    block = render_report_block(tmp_path)
    assert "2× Tesla T4" in block
    assert "PyTorch (`eager`)" in block
    assert "JAX (`jit`)" in block
    assert "12,375" in block

    update_readme(readme, block)
    updated = readme.read_text(encoding="utf-8")
    assert updated.startswith("# Header\n\n")
    assert updated.endswith("\n\n## Footer\nKeep me.\n")
    assert "old content" not in updated

    png = plot_allreduce(tmp_path)
    assert png.exists() and png.stat().st_size > 0
