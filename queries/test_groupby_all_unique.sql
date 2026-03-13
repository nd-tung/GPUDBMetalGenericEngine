-- Test: GROUP BY where every row is its own group
-- Tests high-cardinality grouping (25 groups from 25 rows)
select
    n_nationkey,
    n_name,
    count(*) as cnt
from
    nation
group by
    n_nationkey,
    n_name
order by
    n_nationkey;
