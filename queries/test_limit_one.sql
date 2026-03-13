-- Test: LIMIT 1 — returns exactly one row from a larger set
-- Tests minimal result set handling with sort + limit
select
    l_orderkey,
    l_partkey,
    l_extendedprice
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and l_shipdate >= date '1998-01-01'
order by
    l_extendedprice desc
limit 1;
