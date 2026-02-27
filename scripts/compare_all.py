#!/usr/bin/env python3
import os, re, sys

def parse_result(path):
    with open(path) as f:
        lines = f.readlines()
    rows = []
    header = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('-') or line.startswith('=') or line.startswith('['):
            continue
        # Handle scalar output: "Scalar name: value"
        if line.startswith('Scalar '):
            m = re.match(r'Scalar\s+(.+?):\s+(.+)', line)
            if m:
                name = m.group(1).strip()
                val = m.group(2).strip()
                if header is None:
                    header = [name]
                rows.append([val])
            continue
        if '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if not parts:
                continue
            is_data = any(re.match(r'^-?[\d.]', p) for p in parts)
            if header is None and not is_data:
                header = parts
            elif is_data or header is not None:
                if header is None:
                    header = []
                rows.append(parts)
    return header, rows

def normalize(val):
    try:
        f = float(val)
        return round(f, 2)
    except:
        return val.strip()

def vals_match(a, b, tol=0.02):
    """Compare two normalized values with tolerance for floats."""
    if isinstance(a, float) and isinstance(b, float):
        if b == 0:
            return abs(a) < tol
        return abs(a - b) / max(abs(b), 1) < 0.0001 or abs(a - b) <= tol
    return a == b

def align_rows(header_from, header_to, rows):
    """Reorder columns in rows to match header_to's column order."""
    if not header_from or not header_to:
        return rows
    # Build index mapping: for each col in header_to, find its index in header_from
    mapping = []
    for col in header_to:
        if col in header_from:
            mapping.append(header_from.index(col))
        else:
            return rows  # Can't align, return as-is
    result = []
    for row in rows:
        if len(row) >= len(mapping):
            result.append([row[i] for i in mapping])
        else:
            result.append(row)
    return result

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []
for i in range(1, 23):
    name = f'q{i:02d}'
    eng_path = os.path.join(base, f'results/engine_run/{name}.txt')
    duck_path = os.path.join(base, f'results/duckdb/{name}.txt')
    if not os.path.exists(eng_path) or not os.path.exists(duck_path):
        results.append((name, 'SKIP'))
        continue
    eh, erows = parse_result(eng_path)
    dh, drows = parse_result(duck_path)
    # Align engine columns to DuckDB column order
    if eh and dh and set(eh) == set(dh):
        erows = align_rows(eh, dh, erows)
    if len(erows) != len(drows):
        results.append((name, f'FAIL rows {len(erows)} vs {len(drows)}'))
        continue
    match = True
    first_diff = None
    for ri, (er, dr) in enumerate(zip(erows, drows)):
        en = [normalize(v) for v in er]
        dn = [normalize(v) for v in dr]
        row_match = len(en) == len(dn) and all(vals_match(a, b) for a, b in zip(en, dn))
        if not row_match:
            match = False
            first_diff = f'row {ri}: engine={en} duck={dn}'
            break
    results.append((name, 'PASS' if match else f'FAIL {first_diff}'))

for name, status in results:
    print(f'{name}: {status}')
passed = sum(1 for _, s in results if s == 'PASS')
print(f'\nTotal: {passed}/22 PASS')
