-- Test: Large sort on joined result set
-- Tests sort+limit on large join
select
    l_orderkey,
    l_partkey,
    l_extendedprice,
    o_totalprice
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and l_shipdate >= date '1997-01-01'
    and l_shipdate < date '1997-02-01'
order by
    l_extendedprice desc
limit 25;
