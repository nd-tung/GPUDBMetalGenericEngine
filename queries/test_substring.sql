-- Test: SUBSTRING function
-- Tests string substring extraction
select
    substring(c_phone from 1 for 2) as country_code,
    count(*) as cnt,
    avg(c_acctbal) as avg_bal
from
    customer
where
    c_acctbal > 0
group by
    substring(c_phone from 1 for 2)
order by
    cnt desc;
