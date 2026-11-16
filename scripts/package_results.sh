#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${1:-$(date -u +%Y-%m-%d)}"
TARGET_DIR="results/${DATE_TAG}"
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Results directory not found: $TARGET_DIR" >&2
  exit 1
fi

ARCHIVE="results-${DATE_TAG}.tar.gz"
tar -czf "$ARCHIVE" \
  --exclude="${TARGET_DIR}/traces" \
  --exclude="${TARGET_DIR}/hlo" \
  --exclude="${TARGET_DIR}/parity/*.npy" \
  "$TARGET_DIR" README.md

echo "Packaged $ARCHIVE"
