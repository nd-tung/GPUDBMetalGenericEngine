-- Test: Two-table join with group by and order
-- Tests simple equi-join + aggregation
select
    n_name,
    count(*) as supplier_count,
    sum(s_acctbal) as total_acctbal
from
    supplier,
    nation
where
    s_nationkey = n_nationkey
group by
    n_name
order by
    supplier_count desc,
    n_name
limit 10;
