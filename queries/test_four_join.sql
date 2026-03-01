-- Test: 4-table join with filter and aggregation
-- Tests large multi-way join pipeline
select
    r_name as region,
    count(*) as order_count,
    sum(l_extendedprice * (1 - l_discount)) as revenue
from
    region,
    nation,
    supplier,
    lineitem
where
    r_regionkey = n_regionkey
    and n_nationkey = s_nationkey
    and s_suppkey = l_suppkey
    and l_shipdate >= date '1995-01-01'
    and l_shipdate < date '1996-01-01'
group by
    r_name
order by
    revenue desc;
