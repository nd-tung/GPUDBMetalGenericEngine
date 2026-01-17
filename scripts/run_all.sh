#!/bin/bash
cd "$(dirname "$0")/.." || exit

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"
echo "Run identifier: $TIMESTAMP"

echo "=== Running MetalDB ==="
./scripts/run_metal.sh "$RESULTS_DIR"

echo "=== Running DuckDB (Reference) ==="
./scripts/run_duckdb.sh "$RESULTS_DIR"

echo "=== Verifying Results ==="
python3 scripts/verify_result.py "$RESULTS_DIR"
