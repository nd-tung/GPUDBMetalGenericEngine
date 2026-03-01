-- Test: COUNT DISTINCT equivalent via subquery
-- Tests distinct count pattern
select
    count(*) as n_distinct_brands
from (
    select distinct p_brand
    from part
) t;
