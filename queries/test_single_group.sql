-- Test: GROUP BY that produces a single group
-- Tests aggregation collapsing all rows into one group
select
    r_regionkey,
    count(*) as cnt,
    min(r_regionkey) as min_key,
    max(r_regionkey) as max_key
from
    region
where
    r_regionkey = 2
group by
    r_regionkey;
