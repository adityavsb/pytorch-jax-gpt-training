#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLBACKEND=Agg

DATE_TAG="${RUN_DATE:-$(date -u +%Y-%m-%d)}"
OUT="${RESULTS_DIR:-results/${DATE_TAG}}"
PRESET="${PRESET:-medium}"
REPEATS="${REPEATS:-3}"

mkdir -p "$OUT/logs"

T=".venv-torch/bin/python"
TR=".venv-torch/bin/torchrun"
J=".venv-jax/bin/python"

run_logged() {
  local name="$1"
  shift
  echo "=== [stage: ${name}] ==="
  "$@" 2>&1 | tee "$OUT/logs/${name}.log"
}

run_logged topo_torch "$T" -m bench.topo --framework torch --out "$OUT"
run_logged topo_jax   "$J" -m bench.topo --framework jax   --out "$OUT"

run_logged smoke_torch "$T" -m bench.smoke --framework torch --preset small --out "$OUT"
run_logged smoke_jax   "$J" -m bench.smoke --framework jax   --preset small --out "$OUT"

run_logged allreduce_torch env NCCL_DEBUG=INFO "$TR" --nproc_per_node=2 -m bench.allreduce --backend torch --out "$OUT"
run_logged allreduce_jax   "$J" -m bench.allreduce --backend jax --out "$OUT"

run_logged parity_torch   "$T" -m bench.parity --framework torch --preset "$PRESET" --out "$OUT"
run_logged parity_jax     "$J" -m bench.parity --framework jax   --preset "$PRESET" --out "$OUT"
run_logged parity_compare "$T" -m bench.parity --compare         --preset "$PRESET" --out "$OUT"

run_logged gate "$T" -m bench.gate --preset "$PRESET" --out "$OUT"

run_logged single_torch "$T" -m bench.single --framework torch --preset "$PRESET" --repeats "$REPEATS" --out "$OUT"
run_logged single_jax   "$J" -m bench.single --framework jax   --preset "$PRESET" --repeats "$REPEATS" --out "$OUT"

run_logged scale_weak_torch "$T" -m bench.scale --weak --framework torch --preset "$PRESET" --repeats "$REPEATS" --out "$OUT"
run_logged scale_weak_jax   "$J" -m bench.scale --weak --framework jax   --preset "$PRESET" --repeats "$REPEATS" --out "$OUT"

run_logged scale_strong_torch "$T" -m bench.scale --strong --framework torch --preset "$PRESET" --repeats "$REPEATS" --out "$OUT"
run_logged scale_strong_jax   "$J" -m bench.scale --strong --framework jax   --preset "$PRESET" --repeats "$REPEATS" --out "$OUT"

run_logged report "$T" -m bench.report --results "$OUT" --readme README.md

echo "=== Completed all stages -> $OUT ==="
