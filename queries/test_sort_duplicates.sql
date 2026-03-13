-- Test: ORDER BY with many duplicate sort keys
-- Tests sort stability / handling of all-identical keys
select
    n_regionkey,
    n_name
from
    nation
order by
    n_regionkey;
