-- Test: Query that returns zero rows
-- Tests empty result handling across filter, join, group by, order by
select
    n_name,
    count(*) as cnt,
    sum(n_regionkey) as total
from
    nation
where
    n_regionkey > 999
group by
    n_name
order by
    n_name;
