-- Test: IN list filter on join
-- Tests equality with string matching + join
select
    n_name,
    count(*) as num_suppliers,
    sum(s_acctbal) as total_bal
from
    nation,
    supplier
where
    n_nationkey = s_nationkey
    and n_name = 'FRANCE'
group by
    n_name;
