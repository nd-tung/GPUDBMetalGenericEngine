-- Test: Multiple AND filter conditions on join
-- Tests complex predicate evaluation
select
    l_orderkey,
    l_quantity,
    l_extendedprice,
    l_discount
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and l_shipdate >= date '1995-01-01'
    and l_shipdate < date '1996-01-01'
    and l_returnflag = 'A'
order by
    l_extendedprice desc
limit 20;
