-- Test: Three-table join chain
-- Tests multi-way join (nation -> supplier -> lineitem)
select
    n_name,
    count(*) as line_count,
    sum(l_extendedprice * (1 - l_discount)) as revenue
from
    nation,
    supplier,
    lineitem
where
    n_nationkey = s_nationkey
    and s_suppkey = l_suppkey
    and l_shipdate >= date '1995-01-01'
    and l_shipdate < date '1995-04-01'
group by
    n_name
order by
    revenue desc
limit 10;
