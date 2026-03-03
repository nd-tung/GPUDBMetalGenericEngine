#!/bin/bash
# Run all SQL queries through the engine and capture timing + kernel info
# Shows what runs on GPU vs CPU for each query

cd "$(dirname "$0")/.."
ENGINE=./build/bin/MetalGenericDBEngine
RESULTS_DIR=results/test_run
mkdir -p "$RESULTS_DIR"

export GPUDB_KERNEL_DETAIL=1

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
TOTAL=0

echo "================================================================="
echo "  GPU DB Metal Engine - Full Test Run"
echo "================================================================="
echo ""

for sql_file in queries/q*.sql queries/test_*.sql; do
    [ -f "$sql_file" ] || continue
    name=$(basename "$sql_file" .sql)
    TOTAL=$((TOTAL + 1))
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}▶ Running: $name${NC}  ($sql_file)"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    output=$($ENGINE "$sql_file" 2>&1)
    exit_code=$?
    
    # Save full output
    echo "$output" > "$RESULTS_DIR/${name}.txt"
    
    if [ $exit_code -ne 0 ]; then
        echo -e "  ${RED}✗ FAILED (exit code $exit_code)${NC}"
        # Show error
        echo "$output" | grep -i "error\|failed\|exception" | head -5
        FAIL=$((FAIL + 1))
        echo ""
        continue
    fi
    
    echo -e "  ${GREEN}✓ OK${NC}"
    
    # Extract timing info
    gpu_ms=$(echo "$output" | grep "GPU kernels time" | awk '{print $(NF-1)}')
    cpu_ms=$(echo "$output" | grep "CPU postprocess time" | awk '{print $(NF-1)}')
    upload_ms=$(echo "$output" | grep "Data Load Time" | awk '{print $(NF-1)}')
    total_ms=$(echo "$output" | grep "Total Host Execution" | awk '{print $(NF-1)}')
    
    echo "  Timings: Upload=${upload_ms}ms  GPU=${gpu_ms}ms  CPU=${cpu_ms}ms  Total=${total_ms}ms"
    
    # Show kernel breakdown
    kernel_summary=$(echo "$output" | sed -n '/Kernel Timing Summary/,/^$/p' | head -30)
    if [ -n "$kernel_summary" ]; then
        echo "$kernel_summary" | while IFS= read -r line; do
            echo "    $line"
        done
    fi
    
    # Show any CPU fallback messages
    cpu_fallbacks=$(echo "$output" | grep -i "CPU fallback\|on CPU\|host side\|CPU-side\|CPU gather\|CPU postprocess\|CPU sort\|std::sort" | head -10)
    if [ -n "$cpu_fallbacks" ]; then
        echo -e "  ${YELLOW}CPU operations detected:${NC}"
        echo "$cpu_fallbacks" | while IFS= read -r line; do
            echo "    $line"
        done
    fi
    
    PASS=$((PASS + 1))
    echo ""
done

echo ""
echo "================================================================="
echo "  SUMMARY"
echo "================================================================="
echo -e "  Total: $TOTAL   ${GREEN}Pass: $PASS${NC}   ${RED}Fail: $FAIL${NC}"
echo "================================================================="
echo ""
echo "Detailed results saved in: $RESULTS_DIR/"
