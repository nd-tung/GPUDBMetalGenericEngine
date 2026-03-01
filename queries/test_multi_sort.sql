-- Test: ORDER BY multiple columns (ASC + DESC mixed)
-- Tests multi-column sort on join result
select
    l_orderkey,
    l_extendedprice,
    l_discount
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1994-02-01'
    and l_returnflag = 'R'
order by
    l_extendedprice desc,
    l_discount asc
limit 25;
