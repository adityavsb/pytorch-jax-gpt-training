#!/usr/bin/env bash
set -euo pipefail

# Create isolated environments (.venv-torch and .venv-jax) so PyTorch and JAX
# keep their own CUDA/cuDNN wheels without dependency conflicts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROBE=0
if [[ "${1:-}" == "--probe" ]]; then
  PROBE=1
fi

install_venv() {
  local venv_dir="$1"
  local req_file="$2"

  if command -v uv >/dev/null 2>&1; then
    uv venv "$venv_dir" --python python3
    if [[ "$PROBE" -eq 1 ]]; then
      uv pip install -p "$venv_dir" --upgrade torch numpy matplotlib || true
    else
      uv pip install -p "$venv_dir" -r "$req_file"
    fi
    uv pip install -p "$venv_dir" --no-deps -e .
  else
    python3 -m venv "$venv_dir"
    "$venv_dir/bin/python" -m pip install --upgrade pip
    "$venv_dir/bin/python" -m pip install -r "$req_file"
    "$venv_dir/bin/python" -m pip install --no-deps -e .
  fi
}

echo "[setup] Creating .venv-torch..."
if [[ "$PROBE" -eq 1 ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv-torch --python python3
    uv pip install -p .venv-torch torch numpy matplotlib
    uv pip install -p .venv-torch --no-deps -e .
  else
    python3 -m venv .venv-torch
    .venv-torch/bin/python -m pip install --upgrade pip
    .venv-torch/bin/python -m pip install torch numpy matplotlib
    .venv-torch/bin/python -m pip install --no-deps -e .
  fi
else
  install_venv .venv-torch requirements/torch.txt
fi

echo "[setup] Creating .venv-jax..."
if [[ "$PROBE" -eq 1 ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv-jax --python python3
    uv pip install -p .venv-jax "jax[cuda12]" optax numpy matplotlib
    uv pip install -p .venv-jax --no-deps -e .
  else
    python3 -m venv .venv-jax
    .venv-jax/bin/python -m pip install --upgrade pip
    .venv-jax/bin/python -m pip install "jax[cuda12]" optax numpy matplotlib
    .venv-jax/bin/python -m pip install --no-deps -e .
  fi
else
  install_venv .venv-jax requirements/jax.txt
fi

echo "[setup] Resolved versions:"
.venv-torch/bin/python -c "import torch, numpy, matplotlib; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'numpy:', numpy.__version__, 'matplotlib:', matplotlib.__version__)"
.venv-jax/bin/python -c "import jax, jaxlib, optax, numpy; print('jax:', jax.__version__, 'jaxlib:', jaxlib.__version__, 'optax:', optax.__version__, 'numpy:', numpy.__version__)"
