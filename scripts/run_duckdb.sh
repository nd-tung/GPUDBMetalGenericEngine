#!/bin/bash
cd "$(dirname "$0")/.." || exit

if [ -z "$1" ]; then
    TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
    RESULTS_DIR="results/$TIMESTAMP/duckdb"
else
    RESULTS_DIR="$1"
fi

mkdir -p "$RESULTS_DIR"
echo "Running DuckDB... Output: $RESULTS_DIR"

for i in {1..22}; do
    q=$(printf "q%02d" $i)
    QUERY_FILE="queries/$q.sql"
    if [ ! -f "$QUERY_FILE" ]; then
        echo "Warning: $QUERY_FILE not found, skipping."
        continue
    fi
    
    echo "Running $q..."
    duckdb data/SF-1/data.duckdb -csv -f "$QUERY_FILE" > "$RESULTS_DIR/$q.duckdb.out" 2> "$RESULTS_DIR/$q.duckdb.err"
    RET=$?
    if [ $RET -ne 0 ]; then
        echo "$q FAIL ($RET)"
    else
        echo "$q PASS"
    fi
done
