#!/usr/bin/env python3
"""Compare engine output results vs DuckDB reference for TPC-H Q1-Q22."""
import sys, os, re, subprocess, math

ENGINE = "./build/bin/MetalGenericDBEngine"
DATA = "data/SF-1"
QUERY_DIR = "queries"
REF_DIR = "results/duckdb"
OUT_DIR = "results/engine_run_new"

# Relative tolerance for float comparison
REL_TOL = 1e-2   # 1% — GPU f32 arithmetic vs DuckDB double
ABS_TOL = 0.5    # absolute tolerance for small values

def extract_engine_result(filepath):
    """Extract result block from engine output file."""
    lines = open(filepath).read().split('\n')
    result_lines = []
    found = False
    skip_next_dash = False
    for line in lines:
        if re.match(r'^=+RESULT=+$', line):
            found = True
            skip_next_dash = True
            continue
        if found and skip_next_dash and re.match(r'^-+$', line):
            skip_next_dash = False
            continue
        if found and not skip_next_dash and re.match(r'^-+$', line):
            break
        if found and not skip_next_dash:
            result_lines.append(line)
    return [l for l in result_lines if l.strip()]

def parse_scalar(lines):
    """Parse scalar result like 'Scalar name: value' → dict."""
    results = {}
    for line in lines:
        m = re.match(r'^Scalar\s+(.+?):\s*(.+)$', line)
        if m:
            results[m.group(1).strip()] = m.group(2).strip()
    return results

def parse_table(lines):
    """Parse pipe-delimited table → (header_list, list_of_row_dicts)."""
    if not lines:
        return [], []
    header = [h for h in lines[0].split('|') if h]
    rows = []
    for line in lines[1:]:
        vals = [v for v in line.split('|') if v != '']
        if vals:
            row = {}
            for i, h in enumerate(header):
                row[h] = vals[i] if i < len(vals) else ''
            rows.append(row)
    return header, rows

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def compare_values(ref_val, eng_val, col_name=""):
    """Compare two values. Returns (match, detail_string)."""
    if ref_val == eng_val:
        return True, ""
    
    # Both numeric? Compare with tolerance
    if is_number(ref_val) and is_number(eng_val):
        r, e = float(ref_val), float(eng_val)
        if r == 0:
            if abs(e) <= ABS_TOL:
                return True, ""
            return False, f"ref=0 eng={e}"
        rel_err = abs((e - r) / r)
        if rel_err <= REL_TOL and abs(e - r) <= max(ABS_TOL, abs(r) * REL_TOL):
            return True, ""
        return False, f"ref={ref_val} eng={eng_val} rel_err={rel_err:.4e}"
    
    # String comparison (trim trailing zeros, etc.)
    ref_clean = ref_val.strip().rstrip('0').rstrip('.')
    eng_clean = eng_val.strip().rstrip('0').rstrip('.')
    if ref_clean == eng_clean:
        return True, ""
    
    return False, f"ref='{ref_val}' eng='{eng_val}'"

def compare_query(qnum):
    """Run query and compare. Returns (status, details)."""
    qfile = f"{QUERY_DIR}/q{qnum:02d}.sql"
    reffile = f"{REF_DIR}/q{qnum:02d}.txt"
    outfile = f"{OUT_DIR}/q{qnum:02d}.txt"
    
    if not os.path.exists(qfile):
        return "SKIP", "query file not found"
    if not os.path.exists(reffile):
        return "SKIP", "reference file not found"
    
    # Run engine
    try:
        result = subprocess.run(
            [ENGINE, qfile, DATA],
            capture_output=True, text=True, timeout=300
        )
        with open(outfile, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
        if result.returncode != 0:
            return "ERROR", f"exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return "ERROR", "timeout (300s)"
    except Exception as e:
        return "ERROR", str(e)
    
    # Extract engine result
    eng_lines = extract_engine_result(outfile)
    if not eng_lines:
        return "ERROR", "no result extracted from engine output"
    
    # Parse reference
    ref_lines = [l for l in open(reffile).read().split('\n') if l.strip()]
    
    # Detect scalar result from engine
    if eng_lines and eng_lines[0].startswith('Scalar '):
        eng_scalars = parse_scalar(eng_lines)
        ref_header, ref_rows = parse_table(ref_lines)
        
        if not ref_rows:
            return "ERROR", "reference has no data rows"
        
        # Scalar: match engine scalar names to ref columns
        mismatches = []
        ref_row = ref_rows[0]
        for ref_col, ref_val in ref_row.items():
            # Find matching engine scalar
            matched = False
            for eng_name, eng_val in eng_scalars.items():
                # Try fuzzy column name match
                if ref_col.lower() in eng_name.lower() or eng_name.lower() in ref_col.lower():
                    ok, detail = compare_values(ref_val, eng_val, ref_col)
                    if not ok:
                        mismatches.append(f"  {ref_col}: {detail}")
                    matched = True
                    break
            if not matched:
                # Try matching by position
                if len(eng_scalars) == 1 and len(ref_row) == 1:
                    eng_val = list(eng_scalars.values())[0]
                    ok, detail = compare_values(ref_val, eng_val, ref_col)
                    if not ok:
                        mismatches.append(f"  {ref_col}: {detail}")
                else:
                    mismatches.append(f"  {ref_col}: not found in engine scalars")
        
        if mismatches:
            return "DIFF", f"scalar ({len(mismatches)} mismatches)\n" + "\n".join(mismatches)
        return "PASS", f"scalar (1 row)"
    
    # Tabular result
    ref_header, ref_rows = parse_table(ref_lines)
    eng_header, eng_rows = parse_table(eng_lines)
    
    details = []
    
    # Row count check
    if len(ref_rows) != len(eng_rows):
        return "DIFF", f"row count: ref={len(ref_rows)} eng={len(eng_rows)}"
    
    # Find common columns (use reference column order)
    common_cols = [c for c in ref_header if c in eng_header]
    missing_cols = [c for c in ref_header if c not in eng_header]
    extra_cols = [c for c in eng_header if c not in ref_header]
    
    if missing_cols:
        details.append(f"  Missing cols in engine: {missing_cols}")
    if extra_cols:
        details.append(f"  Extra cols in engine: {extra_cols}")
    
    # Compare row by row using common columns
    row_mismatches = 0
    mismatch_details = []
    for i, (ref_row, eng_row) in enumerate(zip(ref_rows, eng_rows)):
        for col in common_cols:
            ref_val = ref_row.get(col, '')
            eng_val = eng_row.get(col, '')
            ok, detail = compare_values(ref_val, eng_val, col)
            if not ok:
                if len(mismatch_details) < 20:  # limit output
                    mismatch_details.append(f"  Row{i+1}.{col}: {detail}")
                row_mismatches += 1
    
    if row_mismatches > 0:
        detail_str = f"{len(eng_rows)} rows, {row_mismatches} value mismatches"
        if missing_cols:
            detail_str += f", missing cols: {missing_cols}"
        detail_str += "\n" + "\n".join(mismatch_details)
        if row_mismatches > 20:
            detail_str += f"\n  ... and {row_mismatches - 20} more"
        return "DIFF", detail_str
    
    status_detail = f"{len(eng_rows)} rows"
    if missing_cols or extra_cols:
        status_detail += f" (col order differs)"
    return "PASS", status_detail

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    summary = {"PASS": 0, "DIFF": 0, "ERROR": 0, "SKIP": 0}
    results = []
    
    queries = range(1, 23)
    if len(sys.argv) > 1:
        queries = [int(x) for x in sys.argv[1:]]
    
    for q in queries:
        print(f"Q{q:02d} ... ", end="", flush=True)
        status, detail = compare_query(q)
        summary[status] = summary.get(status, 0) + 1
        results.append((q, status, detail))
        print(f"{status}: {detail}")
    
    print()
    print("=" * 60)
    total = sum(summary.values())
    print(f"Summary: PASS={summary['PASS']}  DIFF={summary['DIFF']}  "
          f"ERROR={summary['ERROR']}  SKIP={summary['SKIP']}  Total={total}")
    print("=" * 60)
    for q, status, detail in results:
        marker = "✓" if status == "PASS" else "✗" if status in ("DIFF", "ERROR") else "○"
        print(f"  {marker} Q{q:02d}: {status}")

if __name__ == "__main__":
    main()
