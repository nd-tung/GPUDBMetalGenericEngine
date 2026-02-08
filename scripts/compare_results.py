#!/usr/bin/env python3
"""
"""Compare GPU engine results against DuckDB ground truth for TPC-H Q1-Q20.
Handles: column order mismatch, row order differences, engine 10-row display cap.
Accepts small floating-point differences (f32 vs decimal).
"""

import duckdb
import os
import re
import sys
import math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "SF-1")
QUERIES_DIR = os.path.join(BASE, "queries")
ENGINE_DIR = os.path.join(BASE, "results", "engine")
DUCKDB_DIR = os.path.join(BASE, "results", "duckdb")

# Relative tolerance for float comparison (0.1% for f32 precision loss)
REL_TOL = 1e-3
# Absolute tolerance for very small values
ABS_TOL = 0.5


def get_duckdb_connection():
    """Create an in-memory DuckDB connection with TPC-H tables loaded from .tbl files."""
    con = duckdb.connect(":memory:")

    tables = {
        'lineitem': "l_orderkey INTEGER, l_partkey INTEGER, l_suppkey INTEGER, l_linenumber INTEGER, l_quantity DECIMAL(12,2), l_extendedprice DECIMAL(12,2), l_discount DECIMAL(12,2), l_tax DECIMAL(12,2), l_returnflag VARCHAR, l_linestatus VARCHAR, l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR, l_shipmode VARCHAR, l_comment VARCHAR",
        'orders': "o_orderkey INTEGER, o_custkey INTEGER, o_orderstatus VARCHAR, o_totalprice DECIMAL(12,2), o_orderdate DATE, o_orderpriority VARCHAR, o_clerk VARCHAR, o_shippriority INTEGER, o_comment VARCHAR",
        'customer': "c_custkey INTEGER, c_name VARCHAR, c_address VARCHAR, c_nationkey INTEGER, c_phone VARCHAR, c_acctbal DECIMAL(12,2), c_mktsegment VARCHAR, c_comment VARCHAR",
        'supplier': "s_suppkey INTEGER, s_name VARCHAR, s_address VARCHAR, s_nationkey INTEGER, s_phone VARCHAR, s_acctbal DECIMAL(12,2), s_comment VARCHAR",
        'part': "p_partkey INTEGER, p_name VARCHAR, p_mfgr VARCHAR, p_brand VARCHAR, p_type VARCHAR, p_size INTEGER, p_container VARCHAR, p_retailprice DECIMAL(12,2), p_comment VARCHAR",
        'partsupp': "ps_partkey INTEGER, ps_suppkey INTEGER, ps_availqty INTEGER, ps_supplycost DECIMAL(12,2), ps_comment VARCHAR",
        'nation': "n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER, n_comment VARCHAR",
        'region': "r_regionkey INTEGER, r_name VARCHAR, r_comment VARCHAR",
    }

    for tbl, cols in tables.items():
        tbl_file = os.path.join(DATA_DIR, tbl + '.tbl')
        col_parts = []
        for c in cols.split(', '):
            parts = c.split()
            col_parts.append(f"'{parts[0]}': '{' '.join(parts[1:])}'")
        col_str = ', '.join(col_parts)
        sql = f"CREATE TABLE {tbl} AS SELECT * FROM read_csv('{tbl_file}', delim='|', header=false, columns={{{col_str}}})"
        con.execute(sql)

    return con


def run_duckdb_query(con, query_num):
    """Run a TPC-H query through DuckDB and return (col_names, list_of_dicts)."""
    qfile = os.path.join(QUERIES_DIR, f"q{query_num:02d}.sql")
    with open(qfile) as f:
        sql = f.read()

    result = con.execute(sql)
    cols = [desc[0] for desc in result.description]
    rows_raw = result.fetchall()

    rows = []
    for row in rows_raw:
        d = {}
        for i, c in enumerate(cols):
            d[c] = row[i]
        rows.append(d)

    return cols, rows


def save_duckdb_result(cols, rows, query_num):
    """Write DuckDB result to text file."""
    os.makedirs(DUCKDB_DIR, exist_ok=True)
    with open(os.path.join(DUCKDB_DIR, f"q{query_num:02d}.txt"), 'w') as f:
        f.write("|".join(cols) + "|\n")
        for row in rows:
            parts = []
            for c in cols:
                v = row[c]
                if v is None:
                    parts.append("NULL")
                elif isinstance(v, float):
                    parts.append(f"{v:.2f}")
                else:
                    parts.append(str(v))
            f.write("|".join(parts) + "|\n")


def parse_engine_output(query_num):
    """Parse engine output file. Returns (col_names, list_of_dicts, is_scalar, scalar_value, scalar_name)."""
    fpath = os.path.join(ENGINE_DIR, f"q{query_num:02d}.txt")
    if not os.path.exists(fpath):
        return None, None, False, 0, ""

    with open(fpath) as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    in_result = False
    header_line = None
    data_lines = []
    separator_count = 0

    # Check if file starts directly with data (no RESULT marker)
    has_result_marker = any("RESULT" in l and "====" in l for l in lines)

    if not has_result_marker:
        # Direct data format: first line is header or "Scalar ..."
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Scalar "):
                m = re.match(r'Scalar\s+(.+?):\s*([-\d.eE+]+)', stripped)
                if m:
                    return None, None, True, float(m.group(2)), m.group(1)
                return None, None, True, 0, stripped
            if header_line is None:
                header_line = stripped
                continue
            if stripped.startswith("..."):
                continue
            data_lines.append(stripped)
    else:
        for line in lines:
            if "RESULT" in line and "====" in line:
                in_result = True
                separator_count = 0
                continue

            if in_result:
                if line.startswith("---"):
                    separator_count += 1
                    if separator_count >= 2:
                        break
                    continue

                if line.startswith("Scalar "):
                    m = re.match(r'Scalar\s+(.+?):\s*([-\d.eE+]+)', line)
                    if m:
                        return None, None, True, float(m.group(2)), m.group(1)
                    return None, None, True, 0, line

                if header_line is None:
                    header_line = line
                    continue

                if line.startswith("Planning") or line.startswith("Data Load") or \
                   line.startswith("GPU") or line.startswith("CPU") or line.startswith("Total"):
                    break

                if line.strip() and not line.startswith("..."):
                    data_lines.append(line)

    if header_line is None:
        return None, None, False, 0, ""

    cols = [c.strip() for c in header_line.split('|') if c.strip()]

    rows = []
    for dl in data_lines:
        parts = dl.split('|')
        # Remove empty trailing element from trailing |
        parts = [p.strip() for p in parts]
        # Filter keeping all values including empty strings between pipes, but drop last empty
        if parts and parts[-1] == '':
            parts = parts[:-1]
        d = {}
        for i, c in enumerate(cols):
            if i < len(parts):
                d[c] = parts[i]
            else:
                d[c] = ""
        rows.append(d)

    return cols, rows, False, 0, ""


def try_float(s):
    """Try to parse a string as float."""
    try:
        return float(str(s))
    except (ValueError, TypeError):
        return None


def normalize_date(s):
    """Convert YYYYMMDD integer string or YYYY-MM-DD to comparable form."""
    s = str(s).strip()
    if re.match(r'^\d{8}$', s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def values_match(eng_val, duck_val, col_name=""):
    """Compare two values with float tolerance. Returns (match, rel_error)."""
    es = str(eng_val).strip()
    ds = str(duck_val).strip()

    # Direct string match
    if es == ds:
        return True, 0

    # Date normalization
    en = normalize_date(es)
    dn = normalize_date(ds)
    if en == dn:
        return True, 0

    # Numeric comparison
    ef = try_float(es)
    df = try_float(ds)

    if ef is not None and df is not None:
        if df == 0 and ef == 0:
            return True, 0
        if df == 0:
            return abs(ef) < ABS_TOL, abs(ef)
        rel_err = abs(ef - df) / max(abs(df), 1e-10)
        abs_err = abs(ef - df)
        if rel_err < REL_TOL or abs_err < ABS_TOL:
            return True, rel_err
        return False, rel_err

    # String comparison (case insensitive)
    if es.lower() == ds.lower():
        return True, 0

    return False, 0


def find_matching_col(eng_col, duck_cols):
    """Find the matching DuckDB column for an engine column name."""
    ec = eng_col.lower().strip()
    for dc in duck_cols:
        if dc.lower().strip() == ec:
            return dc
    return None


def compare_query(con, query_num):
    """Compare engine output with DuckDB for one query."""
    eng_cols, eng_rows, is_scalar, scalar_val, scalar_name = parse_engine_output(query_num)

    # Get DuckDB result
    try:
        duck_cols, duck_rows = run_duckdb_query(con, query_num)
    except Exception as e:
        return False, f"DuckDB error: {e}"

    save_duckdb_result(duck_cols, duck_rows, query_num)

    # Handle scalar results
    if is_scalar:
        dv = float(duck_rows[0][duck_cols[0]])
        match, rel = values_match(scalar_val, dv)
        if match:
            return True, f"Scalar: engine={scalar_val:.2f} duck={dv:.2f} (rel={rel*100:.4f}%)"
        else:
            return False, f"Scalar MISMATCH: engine={scalar_val:.2f} duck={dv:.2f} (rel={rel*100:.4f}%)"

    if eng_cols is None or eng_rows is None:
        return False, "No engine output found"

    # Build column mapping: engine_col -> duck_col
    col_map = {}
    unmapped_eng = []
    for ec in eng_cols:
        dc = find_matching_col(ec, duck_cols)
        if dc:
            col_map[ec] = dc
        else:
            unmapped_eng.append(ec)

    if not col_map:
        return False, f"No columns matched! engine={eng_cols} duck={duck_cols}"

    mapped_duck = set(col_map.values())
    unmapped_duck = [c for c in duck_cols if c not in mapped_duck]

    # -- Match rows --
    # Engine rows are ordered by query's ORDER BY but may only show top 10.
    # DuckDB rows are also ordered by the query. Compare top N positionally.
    eng_count = len(eng_rows)
    duck_count = len(duck_rows)
    compare_count = min(eng_count, duck_count)

    mismatches = []
    max_rel_err = 0.0

    for i in range(compare_count):
        for ec, dc in col_map.items():
            ev = eng_rows[i].get(ec, "")
            dv = duck_rows[i].get(dc, "")
            match, rel = values_match(ev, dv, dc)
            max_rel_err = max(max_rel_err, rel)
            if not match:
                ef = try_float(ev)
                df = try_float(dv)
                if ef is not None and df is not None and df != 0:
                    mismatches.append(
                        f"  Row {i} '{dc}': engine={ev} duck={dv} (err={rel*100:.4f}%)")
                else:
                    mismatches.append(
                        f"  Row {i} '{dc}': engine={ev} duck={dv}")

    # Summary
    parts = [f"rows: engine={eng_count} duck={duck_count}"]
    if eng_count > 0 and duck_count > 0 and eng_count < duck_count:
        parts.append(f"(engine shows top {eng_count})")
    parts.append(f"cols matched: {len(col_map)}/{len(duck_cols)}")

    if unmapped_duck:
        parts.append(f"(duck-only: {unmapped_duck})")
    if unmapped_eng:
        parts.append(f"(engine-only: {unmapped_eng})")

    summary = ", ".join(parts)

    if mismatches:
        detail = "\n".join(mismatches[:15])
        if len(mismatches) > 15:
            detail += f"\n  ... and {len(mismatches) - 15} more"
        return False, f"{summary}\n{detail}"

    extra = ""
    if max_rel_err > 0:
        extra = f", max_rel_err={max_rel_err*100:.4f}%"
    return True, f"{summary}{extra} ✓"


def main():
    print("=" * 70)
    print("GPU Engine vs DuckDB — TPC-H Q1–Q20 Comparison")
    print("=" * 70)
    print(f"Float tolerance: rel={REL_TOL*100:.1f}%, abs={ABS_TOL}")
    print()

    con = get_duckdb_connection()

    passed = 0
    failed = 0

    for q in range(1, 21):
        ok, msg = compare_query(con, q)
        tag = "✅ PASS" if ok else "❌ FAIL"
        print(f"Q{q:02d}: {tag}")
        for line in msg.split('\n'):
            print(f"     {line}")
        print()
        if ok:
            passed += 1
        else:
            failed += 1

    con.close()

    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of 20")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
