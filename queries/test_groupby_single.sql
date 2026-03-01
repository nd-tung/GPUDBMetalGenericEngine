-- Test: Group by with HAVING equivalent (filter on aggregate)
-- Tests group-by followed by filter on aggregated values
select
    l_returnflag,
    count(*) as cnt,
    sum(l_quantity) as total_qty
from
    lineitem
group by
    l_returnflag
order by
    total_qty desc;
