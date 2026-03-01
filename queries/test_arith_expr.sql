-- Test: Arithmetic expressions in projection
-- Tests computed columns with math operations
select
    l_orderkey,
    l_extendedprice * (1 - l_discount) as net_price,
    l_extendedprice * l_discount as discount_amount,
    l_extendedprice * (1 - l_discount) * (1 + l_tax) as total_charge
from
    lineitem,
    orders
where
    l_orderkey = o_orderkey
    and l_shipdate >= date '1995-01-01'
    and l_shipdate < date '1995-04-01'
order by
    total_charge desc
limit 15;
