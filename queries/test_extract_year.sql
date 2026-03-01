-- Test: Year extraction + group by year
-- Tests EXTRACT(YEAR FROM ...) function
select
    extract(year from o_orderdate) as order_year,
    count(*) as num_orders,
    sum(o_totalprice) as total_revenue
from
    orders
group by
    extract(year from o_orderdate)
order by
    order_year;
