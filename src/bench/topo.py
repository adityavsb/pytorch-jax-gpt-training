"""Stage 0: Verify hardware topology (2x NVIDIA T4, CC 7.5), link type, and framework visibility."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

from bench.gates import check_topo_gate  # noqa: E402
from bench.results import (  # noqa: E402
    collect_provenance,
    ensure_results_dir,
    read_json,
    write_json_atomic,
)


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _read_ram_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / (1024.0**2)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024.0**3)
    except Exception:
        pass
    return 0.0


def parse_nvidia_smi_query(raw: str) -> tuple[list[dict[str, str]], str]:
    gpus: list[dict[str, str]] = []
    driver = ""
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "compute_cap": parts[1],
                "memory_mib": parts[2],
            }
        )
        driver = parts[3]
    return gpus, driver


def parse_topo_link(topo_text: str) -> str:
    """Extract GPU0 <-> GPU1 interconnect code (e.g. PHB, PIX, SYS, NV4) from nvidia-smi topo -m."""
    for line in topo_text.splitlines():
        stripped = line.strip()
        if re.match(r"^GPU0\b", stripped):
            tokens = stripped.split()
            if len(tokens) >= 3:
                return tokens[2]
    return "UNKNOWN"


def _probe_torch(device_kind: str) -> int:
    import torch

    if device_kind == "cpu":
        a = torch.randn(64, 64)
        _ = (a @ a).sum().item()
        return 0
    count = torch.cuda.device_count()
    for idx in range(count):
        dev = torch.device(f"cuda:{idx}")
        a = torch.randn(128, 128, device=dev, dtype=torch.float16)
        _ = (a @ a).sum().item()
        torch.cuda.synchronize(dev)
    return count


def _probe_jax(device_kind: str) -> int:
    import jax
    import jax.numpy as jnp

    devs = jax.devices()
    if device_kind == "cpu":
        x = jnp.ones((64, 64), dtype=jnp.float32)
        _ = float((x @ x).sum().block_until_ready())
        return len(devs)
    for d in devs:
        x = jax.device_put(jnp.ones((128, 128), dtype=jnp.float16), d)
        _ = float((x @ x).sum().block_until_ready())
    return len(devs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect hardware topology and framework visibility")
    parser.add_argument("--framework", choices=["torch", "jax"], required=True)
    parser.add_argument("--out", required=True, help="Results directory path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--force-gates", action="store_true")
    args = parser.parse_args(argv)

    out_dir = ensure_results_dir(args.out)

    if args.framework == "torch":
        torch_gpus = _probe_torch(args.device)
        prov = collect_provenance("torch")
        if args.device == "cuda":
            smi_text = _run_cmd(["nvidia-smi"])
            topo_text = _run_cmd(["nvidia-smi", "topo", "-m"])
            query_text = _run_cmd(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            )
            gpus, driver = parse_nvidia_smi_query(query_text)
            link_type = parse_topo_link(topo_text)
        else:
            smi_text = "CPU mode (nvidia-smi skipped)"
            topo_text = "CPU mode"
            gpus, driver = [], "cpu"
            link_type = "CPU"

        cpu_count = os.cpu_count() or 0
        ram_gb = _read_ram_gb()

        txt = (
            f"=== nvidia-smi ===\n{smi_text}\n\n"
            f"=== nvidia-smi topo -m ===\n{topo_text}\n\n"
            f"=== host ===\ncpus={cpu_count}, ram_gb={ram_gb:.1f}, link={link_type}\n"
            f"torch={prov.get('torch')}\n"
        )
        (out_dir / "topo.txt").write_text(txt, encoding="utf-8")

        payload: dict[str, Any] = {
            "stage": "topo",
            "device_mode": args.device,
            "gpus": gpus,
            "driver": driver,
            "link_type": link_type,
            "cpu_count": cpu_count,
            "ram_gb": round(ram_gb, 2),
            "torch_gpus": torch_gpus,
            "provenance_torch": prov,
        }
        write_json_atomic(out_dir / "topo_torch.json", payload)
        print(f"[topo:torch] gpus={len(gpus)} link={link_type} torch_gpus={torch_gpus}")
        return 0

    # JAX pass: finalize topo.json
    jax_gpus = _probe_jax(args.device)
    prov_jax = collect_provenance("jax")
    torch_path = out_dir / "topo_torch.json"
    if not torch_path.exists():
        raise SystemExit(f"Run --framework torch before --framework jax (missing {torch_path})")

    data = read_json(torch_path)
    data["jax_gpus"] = jax_gpus
    data["provenance_jax"] = prov_jax

    topo_txt_path = out_dir / "topo.txt"
    if topo_txt_path.exists():
        existing = topo_txt_path.read_text(encoding="utf-8")
        topo_txt_path.write_text(existing + f"jax={prov_jax.get('jax')}\n", encoding="utf-8")

    if args.device == "cpu":
        passed = bool(args.force_gates)
        reasons = [] if passed else ["CPU mode requires --force-gates to pass hardware gate"]
    else:
        passed, reasons = check_topo_gate(
            data["gpus"],
            link_type=data["link_type"],
            torch_gpus=int(data["torch_gpus"]),
            jax_gpus=jax_gpus,
        )

    data["passed"] = passed or bool(args.force_gates)
    data["forced"] = bool(args.force_gates and not passed)
    data["reasons"] = reasons
    write_json_atomic(out_dir / "topo.json", data)
    print(f"[topo:jax] jax_gpus={jax_gpus} passed={data['passed']} reasons={reasons}")
    if not data["passed"]:
        print(f"Topology gate failed: {reasons}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
