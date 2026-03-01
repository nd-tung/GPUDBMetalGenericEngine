-- Test: Multiple range filters on date columns
-- Tests multiple >=/<= comparisons on joined data
select
    l_orderkey,
    o_totalprice,
    o_orderdate
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and o_orderdate >= date '1995-01-01'
    and o_orderdate <= date '1995-06-30'
    and l_quantity >= 20
    and l_quantity <= 30
    and l_discount >= 0.05
    and l_discount <= 0.07
order by
    o_totalprice desc
limit 20;
