#!/usr/bin/env python3
"""Create a persistent DuckDB database from TPC-H .tbl files."""
import duckdb, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE, "data", "SF-1")
db_path = os.path.join(data_dir, "data.duckdb")

# Remove old db if exists
for f in [db_path, db_path + '.wal']:
    if os.path.exists(f):
        os.remove(f)

con = duckdb.connect(db_path)

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
    tbl_file = os.path.join(data_dir, tbl + '.tbl')
    if not os.path.exists(tbl_file):
        print(f"WARNING: {tbl_file} not found, skipping")
        continue
    col_parts = []
    for c in cols.split(', '):
        parts = c.split()
        col_parts.append(f"'{parts[0]}': '{' '.join(parts[1:])}'")
    col_str = ', '.join(col_parts)
    sql = f"CREATE TABLE {tbl} AS SELECT * FROM read_csv('{tbl_file}', delim='|', header=false, columns={{{col_str}}})"
    print(f"Creating {tbl}...")
    con.execute(sql)
    count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl}: {count} rows")

con.close()
print(f"\nPersistent database created: {db_path}")
print(f"Size: {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")
