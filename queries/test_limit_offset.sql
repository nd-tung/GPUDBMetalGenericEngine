-- Test: Large LIMIT with OFFSET on a big table
-- Tests LIMIT/OFFSET when offset > 0 skips rows
select
    l_orderkey,
    l_partkey,
    l_quantity,
    o_totalprice
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and l_shipdate >= date '1997-06-01'
    and l_shipdate < date '1997-07-01'
order by
    l_orderkey
limit 5 offset 10;
