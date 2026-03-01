-- Test: Join partsupp with supplier and part
-- Tests three-way join with filter and limit
select
    p_name,
    s_name,
    ps_supplycost
from
    partsupp,
    supplier,
    part
where
    ps_suppkey = s_suppkey
    and ps_partkey = p_partkey
    and ps_supplycost < 10
    and p_size = 1
order by
    ps_supplycost asc
limit 20;
