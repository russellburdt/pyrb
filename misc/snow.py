
import pandas as pd
from snowflake import connector

query = f"""
    SELECT VEHICLE_ID, TS_SEC, LATITUDE, LONGITUDE
    FROM GPS.GPS_ENRICHED
    WHERE VEHICLE_ID='9100FFFF-48A9-CB63-A305-A8A3E03F0000'
    AND TS_SEC BETWEEN
        EXTRACT(EPOCH FROM '2021-09-23 12:13:02'::timestamp) AND
        EXTRACT(EPOCH FROM '2021-10-23 12:13:02'::timestamp)
    ORDER BY TS_SEC"""
con0 = connector.connect(
        user='SVC_REST_API',
        database='dp_prod_db',
        warehouse='APP_REST_PROD_VWH',
        role='LYTX_REST_APP_INTERNAL_PROD',
        password='!G3titT0W0rk!',
        account='lytx')
pd.read_sql_query('USE WAREHOUSE \"APP_REST_PROD_VWH\"', con0)
con1 = connector.connect(
        user='SVC_REST_API',
        database='dp_prod_db',
        warehouse='LABS_PROD_VWH_2XL',
        role='LYTX_REST_APP_INTERNAL_PROD',
        password='!G3titT0W0rk!',
        account='lytx')
pd.read_sql_query('USE WAREHOUSE \"APP_REST_PROD_VWH_2XL\"', con1)
df0 = pd.read_sql_query(sql=query, con=con0)
df1 = pd.read_sql_query(sql=query, con=con1)
