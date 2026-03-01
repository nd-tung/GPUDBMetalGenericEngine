-- Test: CASE WHEN expression via Q14-style conditional aggregate
-- Tests conditional logic in aggregate
select
    100.00 * sum(case
        when p_type like 'PROMO%'
        then l_extendedprice * (1 - l_discount)
        else 0
    end) / sum(l_extendedprice * (1 - l_discount)) as promo_pct
from
    lineitem,
    part
where
    l_partkey = p_partkey
    and l_shipdate >= date '1996-01-01'
    and l_shipdate < date '1996-04-01';
