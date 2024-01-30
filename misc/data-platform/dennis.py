
import pandas as pd
from lytx import get_conn


conn = get_conn('snowflake')
query = """
    select *
    from GPS.GPS_ENRICHED_TBL
    where
        companyid is not null
        AND companyid in ('7495', '8035', '7978')
        AND tssec BETWEEN extract(epoch from '2021-06-01 00:00:00+00'::timestamp) AND extract(epoch from '2021-08-01 00:00:00+00'::timestamp)
    order by companyid asc, tssec asc
    """
df = pd.read_sql_query(query, conn)
