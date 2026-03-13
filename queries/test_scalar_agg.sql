-- Test: Aggregate without GROUP BY (scalar aggregation)
-- Tests global reduction on full table
select
    count(*) as total_nations,
    sum(n_regionkey) as sum_keys,
    min(n_regionkey) as min_key,
    max(n_regionkey) as max_key,
    avg(n_regionkey) as avg_key
from
    nation;
