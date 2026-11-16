"""Render README summary block and allreduce plot from stage JSON outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from bench.results import read_json

START_MARKER = "<!-- results:start -->"
END_MARKER = "<!-- results:end -->"


def _fmt_toks(row: dict[str, Any] | None) -> str:
    if not row or not row.get("finite"):
        err = (row or {}).get("error")
        return "OOM" if (row or {}).get("oom") else ("failed" if err else "—")
    t = row["tokens_per_s"]
    return f"{t['median']:,.0f} [{t['min']:,.0f}–{t['max']:,.0f}]"


def _fmt_weak_cell(row: dict[str, Any] | None) -> str:
    if not row or not row.get("finite"):
        return _fmt_toks(row)
    eff = row["scaling_efficiency"] * 100.0
    delta_ms = row["delta_step_s"] * 1000.0
    est_ms = row.get("sync_estimate_s", 0.0) * 1000.0
    return f"{_fmt_toks(row)} ({eff:.1f}%, Δ {delta_ms:+.0f} ms vs {est_ms:.0f} ms)"


def _fmt_strong_cell(row: dict[str, Any] | None) -> str:
    if not row or not row.get("finite"):
        return _fmt_toks(row)
    eff = row["scaling_efficiency"] * 100.0
    flag = " ⚠" if row.get("strong_vs_weak", {}).get("strong_ge_weak") else ""
    return f"{_fmt_toks(row)} ({eff:.1f}%{flag})"


def render_report_block(results_dir: Path) -> str:
    """Build the markdown block to place between results markers."""
    topo = read_json(results_dir / "topo.json")
    allreduce = read_json(results_dir / "allreduce.json")
    parity = read_json(results_dir / "parity.json")
    gate = read_json(results_dir / "gate.json")
    single = read_json(results_dir / "single.json")
    scale = read_json(results_dir / "scale.json")

    gpus = topo.get("gpus") or []
    gpu_name = gpus[0]["name"] if gpus else "CPU"
    cc = gpus[0].get("compute_cap", "?") if gpus else "cpu"
    mem_mib = gpus[0].get("memory_mib", "?") if gpus else "0"
    link = topo.get("link_type", "UNKNOWN")
    driver = topo.get("driver", "?")
    t_prov = (topo.get("provenance_torch") or {}).get("torch", {})
    j_prov = (topo.get("provenance_jax") or {}).get("jax", {})

    machine_line = (
        f"**Machine:** {len(gpus) or topo.get('jax_gpus', 1)}× {gpu_name} · CC {cc} · "
        f"{mem_mib} MiB each · link: `{link}` · driver `{driver}` · "
        f"CUDA `{t_prov.get('cuda', 'n/a')}`"
    )

    b_star = gate["batch_star"]
    b_cmp = gate["batch_cmp"]
    t_fp16 = gate["fp16_cmp"]["summary"]["median"]
    t_fp32 = gate["fp32_cmp"]["summary"]["median"]
    speedup = gate["speedup_fp16_vs_fp32"]
    mem_star = gate["fp16_star"]["peak_mem_gb"]
    tflops = gate["fp16_star"]["achieved_tflops"]
    gate_line = (
        f"**Gate (batch\\*={b_star}, cmp={b_cmp}):** "
        f"FP16 `{t_fp16:.3f} s` vs FP32 `{t_fp32:.3f} s` "
        f"(`{speedup:.2f}×`), `{mem_star:.2f} GB` peak, `{tflops:.1f}` / `65.0` TFLOP/s"
    )

    chosen_mode = single.get("chosen_torch_mode", "eager")
    rows_single = single.get("rows", [])
    torch_1g = next(
        (r for r in rows_single if r["framework"] == "torch" and r["mode"] == chosen_mode), None
    )
    torch_other = next(
        (r for r in rows_single if r["framework"] == "torch" and r["mode"] != chosen_mode), None
    )
    jax_1g = next((r for r in rows_single if r["framework"] == "jax"), None)

    weak_rows = scale.get("weak", {}).get("rows", {})
    strong_rows = scale.get("strong", {}).get("rows", {})

    table_lines = [
        "| Stack | 1 GPU (tok/s) | 2 GPU Weak (tok/s, eff, Δ vs sync) "
        "| 2 GPU Strong (tok/s, eff) |",
        "|---|---|---|---|",
        (
            f"| PyTorch (`{chosen_mode}`) | {_fmt_toks(torch_1g)} | "
            f"{_fmt_weak_cell(weak_rows.get('torch'))} | "
            f"{_fmt_strong_cell(strong_rows.get('torch'))} |"
        ),
        (
            f"| JAX (`jit`) | {_fmt_toks(jax_1g)} | "
            f"{_fmt_weak_cell(weak_rows.get('jax'))} | "
            f"{_fmt_strong_cell(strong_rows.get('jax'))} |"
        ),
    ]
    if torch_other is not None:
        table_lines.append(
            f"| PyTorch (`{torch_other['mode']}`, unused for 2 GPU) | "
            f"{_fmt_toks(torch_other)} | — | — |"
        )

    b_diff = parity["block_max_abs_diff"]
    f_diff = parity["full_max_abs_diff"]
    p_status = "✓" if parity.get("passed") else "✗"
    parity_line = (
        f"**Correctness (FP32 forward):** block `{b_diff:.2e}`, "
        f"full model `{f_diff:.2e}` (`< 1e-3`) {p_status}"
    )

    t_fit = allreduce["torch"]["fp32"]["fit"]
    j_fit = allreduce["jax"]["fp32"]["fit"]
    est_ms = allreduce["sync_estimate_s"] * 1000.0
    allreduce_line = (
        f"**Allreduce (FP32):** NCCL `α = {t_fit['alpha_us']:.1f} µs`, "
        f"`β = {t_fit['beta_gbs']:.2f} GB/s` (`R² = {t_fit['r2']:.4f}`, "
        f"1.22 GB sync est `{est_ms:.0f} ms`); "
        f"JAX `psum` `β = {j_fit['beta_gbs']:.2f} GB/s`"
    )

    prov_root = single.get("provenance_torch") or topo.get("provenance_torch") or {}
    versions_line = (
        f"**Versions:** Python `{prov_root.get('python', '?')}` · "
        f"PyTorch `{t_prov.get('torch', '?')}` · "
        f"JAX `{j_prov.get('jax', '?')}` / jaxlib `{j_prov.get('jaxlib', '?')}` · "
        f"cuDNN `{t_prov.get('cudnn', '?')}` · NCCL `{t_prov.get('nccl', '?')}` · "
        f"git `{prov_root.get('git_sha', '?')}`"
    )

    return "\n\n".join(
        [
            machine_line,
            gate_line,
            "\n".join(table_lines),
            parity_line,
            allreduce_line,
            versions_line,
        ]
    )


def update_readme(readme_path: Path, block: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"Markers {START_MARKER} .. {END_MARKER} not found in {readme_path}")
    replacement = f"{START_MARKER}\n{block}\n{END_MARKER}"
    readme_path.write_text(pattern.sub(replacement, text, count=1), encoding="utf-8")


def plot_allreduce(results_dir: Path) -> Path:
    import os

    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    data = read_json(results_dir / "allreduce.json")
    grad_mb = data["grad_bytes_fp32"] / 1e6

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    styles = [
        ("torch", "fp32", "PyTorch NCCL FP32", "o-", "#1f77b4"),
        ("torch", "fp16", "PyTorch NCCL FP16", "s--", "#1f77b4"),
        ("jax", "fp32", "JAX psum FP32", "^-", "#d95f02"),
        ("jax", "fp16", "JAX psum FP16", "v--", "#d95f02"),
    ]
    for fw, dt, label, fmt, color in styles:
        entry = data.get(fw, {}).get(dt)
        if not entry:
            continue
        pts = entry["points"]
        fit = entry["fit"]
        xs = np.array([p["bytes"] / 1e6 for p in pts])
        ys = np.array([p["median_s"] * 1000.0 for p in pts])
        ax.plot(xs, ys, fmt, color=color, label=f"{label} ({fit['beta_gbs']:.2f} GB/s)")

    ax.set_xlabel("Payload size (MB)")
    ax.set_ylabel("Allreduce time (ms)")
    ax.set_title(f"2-GPU Allreduce Scaling (1.22 GB FP32 model grad = {grad_mb:.0f} MB)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_png = results_dir / "allreduce.png"
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate README table and allreduce plot from JSON"
    )
    parser.add_argument("--results", required=True, help="Results directory containing stage JSONs")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    block = render_report_block(results_dir)
    update_readme(Path(args.readme), block)
    if not args.no_plot:
        plot_allreduce(results_dir)
    print(f"[report] Updated {args.readme} from {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
