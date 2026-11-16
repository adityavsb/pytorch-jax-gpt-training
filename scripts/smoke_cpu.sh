#!/usr/bin/env bash
set -euo pipefail

# End-to-end CPU smoke test of the entire stage pipeline on the tiny preset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON_BIN:-python3}"
TMP_OUT="$(mktemp -d)"
TMP_README="${TMP_OUT}/README.md"
cp README.md "$TMP_README"
trap 'rm -rf "$TMP_OUT"' EXIT

"$PY" -m bench.topo --framework torch --device cpu --force-gates --out "$TMP_OUT"
"$PY" -m bench.topo --framework jax   --device cpu --force-gates --out "$TMP_OUT"

"$PY" -m bench.smoke --framework torch --preset tiny --device cpu --steps 2 --out "$TMP_OUT"
"$PY" -m bench.smoke --framework jax   --preset tiny --device cpu --steps 2 --out "$TMP_OUT"

"$PY" -m bench.allreduce --backend torch --device cpu --warmup 1 --iters 3 --force-gates --out "$TMP_OUT"
"$PY" -m bench.allreduce --backend jax   --device cpu --warmup 1 --iters 3 --force-gates --out "$TMP_OUT"

"$PY" -m bench.parity --framework torch --preset tiny --device cpu --force-gates --out "$TMP_OUT"
"$PY" -m bench.parity --framework jax   --preset tiny --device cpu --force-gates --out "$TMP_OUT"
"$PY" -m bench.parity --compare         --preset tiny --device cpu --force-gates --out "$TMP_OUT"

"$PY" -m bench.gate --preset tiny --device cpu --warmup 1 --steps 3 --force-gates --out "$TMP_OUT"

"$PY" -m bench.single --framework torch --preset tiny --device cpu --repeats 1 --warmup 1 --steps 3 --force-gates --out "$TMP_OUT"
"$PY" -m bench.single --framework jax   --preset tiny --device cpu --repeats 1 --warmup 1 --steps 3 --force-gates --out "$TMP_OUT"

"$PY" -m bench.scale --weak   --framework torch --preset tiny --device cpu --repeats 1 --warmup 1 --steps 3 --force-gates --out "$TMP_OUT"
"$PY" -m bench.scale --weak   --framework jax   --preset tiny --device cpu --repeats 1 --warmup 1 --steps 3 --force-gates --out "$TMP_OUT"

"$PY" -m bench.scale --strong --framework torch --preset tiny --device cpu --repeats 1 --warmup 1 --steps 3 --force-gates --out "$TMP_OUT"
"$PY" -m bench.scale --strong --framework jax   --preset tiny --device cpu --repeats 1 --warmup 1 --steps 3 --force-gates --out "$TMP_OUT"

"$PY" -m bench.report --results "$TMP_OUT" --readme "$TMP_README"

echo "[smoke_cpu] Full pipeline passed on CPU (tiny preset)."
