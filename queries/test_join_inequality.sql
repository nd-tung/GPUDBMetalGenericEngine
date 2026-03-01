-- Test: Join with inequality filter on joined columns
-- Tests post-join filter on columns from both sides
select
    s_name,
    p_name,
    ps_supplycost,
    p_retailprice,
    p_retailprice - ps_supplycost as margin
from
    partsupp,
    supplier,
    part
where
    ps_suppkey = s_suppkey
    and ps_partkey = p_partkey
    and ps_supplycost < 50
    and p_retailprice > 1500
order by
    margin desc
limit 15;
