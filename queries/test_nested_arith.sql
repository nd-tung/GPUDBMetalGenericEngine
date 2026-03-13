-- Test: Nested arithmetic expressions with mixed types
-- Tests expression evaluation depth and type casting
select
    n_nationkey,
    (n_nationkey * 2 + 1) * (n_regionkey - 1) as complex_expr,
    n_nationkey * 1.0 / (n_regionkey + 1) as div_expr
from
    nation
where
    n_regionkey > 0
order by
    n_nationkey;
