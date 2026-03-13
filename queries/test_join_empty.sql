-- Test: Join that produces zero matches
-- Tests handling of empty join result
select
    n_name,
    r_name
from
    nation,
    region
where
    n_regionkey = r_regionkey
    and r_regionkey > 999;
