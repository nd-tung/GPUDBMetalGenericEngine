-- Test: String LIKE filter on joined tables
-- Tests GPU string prefix matching
select
    p_partkey,
    p_name,
    p_type,
    l_extendedprice
from
    lineitem,
    part
where
    l_partkey = p_partkey
    and p_type like 'PROMO%'
    and l_shipdate >= date '1995-01-01'
    and l_shipdate < date '1995-04-01'
order by
    l_extendedprice desc
limit 20;
