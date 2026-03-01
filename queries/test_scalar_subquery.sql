-- Test: Join with aggregate in projection  
-- Tests sum/count in join context
select
    n_name,
    sum(ps_supplycost * ps_availqty) as total_value
from
    nation,
    supplier,
    partsupp
where
    n_nationkey = s_nationkey
    and s_suppkey = ps_suppkey
    and n_name = 'GERMANY'
group by
    n_name;
