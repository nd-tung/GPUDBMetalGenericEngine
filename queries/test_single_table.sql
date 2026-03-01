-- Test: Single table scan with filter, aggregation, and sort (no joins)
-- Tests the engine's ability to handle a query on one table only
select
    l_returnflag,
    l_linestatus,
    count(*) as cnt,
    sum(l_quantity) as sum_qty,
    sum(l_extendedprice) as sum_price,
    avg(l_discount) as avg_disc
from
    lineitem
where
    l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1994-04-01'
group by
    l_returnflag,
    l_linestatus
order by
    l_returnflag,
    l_linestatus;
