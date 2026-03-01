-- Test: Multi-key group by with 3 keys
-- Tests group-by with more than 2 keys
select
    l_returnflag,
    l_linestatus,
    l_shipmode,
    count(*) as cnt,
    sum(l_quantity) as total_qty
from
    lineitem
where
    l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1995-01-01'
group by
    l_returnflag,
    l_linestatus,
    l_shipmode
order by
    cnt desc
limit 20;
