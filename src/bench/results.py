"""Atomic JSON I/O, stage-gate enforcement, and environment provenance capture."""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ENV_KEYS_OF_INTEREST = (
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def ensure_results_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return sha or "nogit", dirty
    except Exception:
        return "nogit", False


def collect_provenance(
    framework: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture git revision, Python/OS info, relevant env vars, and framework versions."""
    sha, dirty = _git_info()
    prov: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "env": {k: os.environ[k] for k in ENV_KEYS_OF_INTEREST if k in os.environ},
    }

    if framework == "torch":
        import torch

        torch_info: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            try:
                torch_info["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
            except Exception:
                torch_info["nccl"] = None
            torch_info["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        prov["torch"] = torch_info

    if framework == "jax":
        import jax
        import jaxlib

        prov["jax"] = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
        }

    if extra:
        prov.update(extra)
    return prov


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically via temp file + fsync + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return dict(json.load(handle))


def require_stage_passed(
    out_dir: str | Path,
    filename: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Load a prior stage JSON and abort unless it passed (or force=True)."""
    path = Path(out_dir) / filename
    if not path.exists():
        if force:
            print(f"[WARN] Missing prior stage {path}; continuing due to --force", file=sys.stderr)
            return {}
        raise SystemExit(f"Gate blocked: required prior stage output not found: {path}")

    data = read_json(path)
    if not data.get("passed", False):
        if force:
            print(f"[WARN] Stage {path} did not pass; continuing due to --force", file=sys.stderr)
            return data
        reasons = data.get("reasons", [])
        raise SystemExit(f"Gate blocked: {path} failed ({reasons}). Re-run or pass --force.")
    return data
