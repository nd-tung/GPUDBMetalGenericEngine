-- Test: Filter that matches every row (no filtering effect)
-- Tests pass-through filter optimization
select
    r_regionkey,
    r_name
from
    region
where
    r_regionkey >= 0
order by
    r_regionkey;
