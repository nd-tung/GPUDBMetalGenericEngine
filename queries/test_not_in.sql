-- Test: Negated string filter + aggregate
-- Tests string not-equal + aggregate on join
select
    l_shipmode,
    count(*) as cnt,
    sum(l_extendedprice) as total_price
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and l_shipinstruct = 'DELIVER IN PERSON'
    and l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1995-01-01'
group by
    l_shipmode
order by
    cnt desc;
