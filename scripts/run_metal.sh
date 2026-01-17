#!/bin/bash
cd "$(dirname "$0")/.." || exit

if [ -z "$1" ]; then
    TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
    RESULTS_DIR="results/$TIMESTAMP/metal"
else
    RESULTS_DIR="$1"
fi

mkdir -p "$RESULTS_DIR"
echo "Running MetalDB... Output: $RESULTS_DIR"

ENGINE_BIN="./build/bin/MetalGenericDBEngine"
if [ ! -f "$ENGINE_BIN" ]; then
    echo "Error: $ENGINE_BIN not found. Please build first."
    exit 1
fi

for i in {1..22}; do
    q=$(printf "q%02d" $i)
    QUERY_FILE="queries/$q.sql"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "Warning: $QUERY_FILE not found, skipping."
        continue
    fi
    
    echo "Running $q..."
    $ENGINE_BIN sf1 "$QUERY_FILE" > "$RESULTS_DIR/$q.log" 2> "$RESULTS_DIR/$q.err"
    RET=$?
    if [ $RET -ne 0 ]; then
        echo "$q FAIL ($RET)"
    else
        echo "$q PASS"
    fi
done
