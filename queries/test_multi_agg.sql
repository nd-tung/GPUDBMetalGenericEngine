-- Test: Single table aggregate with multiple aggregate functions
-- Tests COUNT, SUM, AVG, MIN, MAX on one table
select
    count(*) as total_parts,
    sum(p_retailprice) as total_price,
    avg(p_retailprice) as avg_price,
    min(p_size) as min_size,
    max(p_size) as max_size
from
    part;
