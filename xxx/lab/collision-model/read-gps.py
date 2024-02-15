
"""
validate speed of gps query via Athena
- result is awswrangler much faster than sqlalchemy
"""

import os
import lytx
import pandas as pd
import numpy as np
import sqlalchemy as sa
import awswrangler as wr
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


# event recorder associations for vehicle population and time-window
population = 'dist300'
t0 = pd.Timestamp(datetime.strptime('6/1/2023', '%m/%d/%Y'))
t1 = pd.Timestamp(datetime.strptime('7/1/2023', '%m/%d/%Y'))
te0, te1 = int(t0.timestamp()), int(t1.timestamp())
dx = lytx.event_recorder_associations_window(population=lytx.population_dict(population), time0=t0, time1=t1)
dx = dx.loc[(dx['CreationDate'] < t0) & (dx['DeletedDate'] > t1)].reset_index(drop=True).copy()
assert pd.unique(dx['VehicleId']).size == dx.shape[0]

# build gps query for nx random vehicles
def gps_query(nx):

    # vehicle-ids and vstr
    vids = dx.loc[np.random.choice(dx.index, size=nx, replace=False), 'VehicleId'].values
    vstr = ','.join([f"""'{x}'""" for x in vids])

    # gps query
    query = f"""
        SELECT
            vehicleid AS VehicleId, tssec AS epoch, tsusec, latitude, longitude, speed, speedsourceid, heading,
            hdop, weather, serialnumber, companyid, state_province, posted_speed, timezone, numsats, horizontalaccuracy
        FROM gps_prod
        WHERE tssec >= {te0} AND tssec < {te1}
        AND vehicleId IN ({vstr})"""
    td = pd.DataFrame(data=[(x.day, x.month, x.year) for x in pd.date_range(t0, t1, freq='D')[:-1]], columns=['day', 'month', 'year'])
    td = td.groupby(['year', 'month'])['day'].unique().reset_index(drop=False).squeeze()
    days = ','.join([f"""'{x}'""" for x in td['day']])
    query += f"""
        AND (year = '{td['year']}' AND month = '{td['month']}' AND day IN ({days}))"""
    return query

# athena connection
# cstr = 'awsathena+rest://athena.us-west-2.amazonaws.com:443/russell_athena_db?s3_staging_dir=s3://russell-athena'
# conn = sa.create_engine(cstr).connect()
# region name for awswrangler
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

# query parameters
nxs = np.tile(np.array([3, 10, 50, 100, 200, 500, 1000]), 5)
# nxs = np.tile(np.array([3]), 5)
np.random.shuffle(nxs)
df = defaultdict(list)
for nx in tqdm(nxs, desc='gps queries'):
    now = datetime.now()
    query = gps_query(nx)
    dv = wr.athena.read_sql_query(database='russell_athena_db', ctas_approach=False, sql=query)
    # dv0 = pd.read_sql_query(sa.text(query), conn)
    sec = (datetime.now() - now).total_seconds()
    df['vehicles'].append(nx)
    df['sec'].append(sec)
    df['records'].append(dv.shape[0])
    print(f'{nx} vehicles, {(te1 - te0) / (60*60*24):.0f} days, query time {(datetime.now() - now).total_seconds():.1f} sec, {dv.shape[0]} records')

# """
# methods to query GPS data
# """

# import os
# import lytx
# import pandas as pd
# import sqlalchemy as sa
# import awswrangler as wr
# import boto3
# from pyspark import SparkConf
# from pyspark.sql import SparkSession

# # gps query via snowflake, old and new sources
# cstr = 'snowflake://SVC_LABS_USER:4:^A]N>N#eH=p&Qp@lytx/dp_prod_db?warehouse=LABS_PROD_VWH_XL'
# conn = sa.create_engine(cstr).connect()
# pd.read_sql_query(sa.text('USE WAREHOUSE \"LABS_PROD_VWH_XL\"'), conn)
# query1 = f"""
#     SELECT ts_sec, ts_usec, latitude, longitude, speed, heading
#     FROM GPS.GPS_ENRICHED
#     WHERE vehicle_id = '9100FFFF-48A9-D563-C54B-9543E7480000'
#     AND ts_sec BETWEEN 1697337800 AND 1697337900
#     ORDER BY ts_sec"""
# df1 = pd.read_sql_query(con=conn, sql=sa.text(query1))
# query2 = f"""
#     SELECT ts_sec, ts_usec, latitude, longitude, speed, heading
#     FROM GPS.GPS_ENRICHED_REPORTING_VIEW
#     WHERE vehicle_id = '9100FFFF-48A9-D563-C54B-9543E7480000'
#     AND ts_sec BETWEEN 1697337800 AND 1697337900
#     ORDER BY ts_sec"""
# df2 = pd.read_sql_query(con=conn, sql=sa.text(query2))

# # # gps parquet dataset in s3 (cross-account, permissions defined in policy) query via spark
# # spark = lytx.spark_session(memory='32g', cores='*', jars=['hadoop-aws-3.3.2.jar', 'aws-java-sdk-bundle-1.12.587.jar'])
# # gps = spark.read.parquet(f's3a://lytx-gps-kafka-prod-003/refined/gps_enriched/1.1')
# # gps.createOrReplaceTempView('gps')
# # df3 = spark.sql(f"""
# #     SELECT tsSec, tsUsec, latitude, longitude, speed, heading
# #     FROM gps
# #     WHERE year=2023 AND month=10 AND day=15
# #     AND vehicleId='9100FFFF-48A9-D563-C54B-9543E7480000'
# #     AND tsSec BETWEEN 1697337800 AND 1697337900
# #     ORDER BY tsSec""").toPandas()

# # gps query via athena + sqlalchemy (after setting up ddl)
# cstr = 'awsathena+rest://athena.us-west-2.amazonaws.com:443/russell_athena_db?s3_staging_dir=s3://russell-athena'
# conn = sa.create_engine(cstr).connect()
# df4 = pd.read_sql_query(con=conn, sql=f"""
#     SELECT tssec, tsusec, latitude, longitude, speed, heading
#     FROM russell_athena_db.gps_prod
#     WHERE year='2023' AND month='10' AND day='15'
#     AND vehicleid='9100FFFF-48A9-D563-C54B-9543E7480000'
#     AND tssec BETWEEN 1697337800 AND 1697337900
#     ORDER BY tssec""")

# # gps query via athena + awswrangler (may be much faster than sqlalchemy)
# os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
# df5 = wr.athena.read_sql_query(database='russell_athena_db', ctas_approach=False, sql=f"""
#     SELECT tssec, tsusec, latitude, longitude, speed, heading
#     FROM gps_prod
#     WHERE year='2023' AND month='10' AND day='15'
#     AND vehicleid='9100FFFF-48A9-D563-C54B-9543E7480000'
#     AND tssec BETWEEN 1697337800 AND 1697337900
#     ORDER BY tssec""")

# first
# CREATE DATABASE russell_athena_db
# then
# CREATE EXTERNAL TABLE `russell-athena`(
#   `tssec` bigint,
#   `tsusec` int,
#   `latitude` float,
#   `longitude` float,
#   `speed` float,
#   `heading` float,
#   `devicesourceid` int,
#   `speedsourceid` int,
#   `modifiedbyer` int,
#   `gpsfix` int,
#   `speedaccuracy` int,
#   `hdop` float,
#   `numsats` int,
#   `horizontalaccuracy` float,
#   `serialnumber` string,
#   `modelnumber` string,
#   `companyid` string,
#   `stackid` string,
#   `correlationid` string,
#   `rootgroupid` string,
#   `groupid` string,
#   `eventrecorderid` string,
#   `vehicleid` string,
#   `timezone` string,
#   `posted_speed` float,
#   `is_moving` int,
#   `trip_info` string,
#   `country` string,
#   `state_province` string,
#   `vehicle_movement_algo_id` string,
#   `trip_algo_id` string,
#   `weather` string,
#   `video_download_url` string,
#   `er_reg_ts` string,
#   `gps_ts` string,
#   `schemaversion` string,
#   `avggpsspeed` float,
#   `mingpsspeed` float,
#   `maxgpsspeed` float,
#   `avgecmspeed` float,
#   `minecmspeed` float,
#   `maxecmspeed` float,
#   `device_ts` string,
#   `trackpoint_id` string,
#   `kafka_ts` string,
#   `prev_trackpoint_id` string,
#   `time_interval_from_prev_trackpoint_sec` bigint,
#   `distance_from_prev_trackpoint_mtr` float)
# PARTITIONED BY (
#   `year` string,
#   `month` string,
#   `day` string)
# ROW FORMAT SERDE
#   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
# STORED AS INPUTFORMAT
#   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat'
# OUTPUTFORMAT
#   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
# LOCATION
#   's3://lytx-gps-kafka-prod-003/refined/gps_enriched/1.1'
# TBLPROPERTIES (
#   'classification'='parquet',
#   'compressionType'='none',
#   'transient_lastDdlTime'='1653049497')
# then
# AmazonAthenaFullAccess in EC2 role

# # gps query via athena (cross-account)
# from pyathena import connect
# access = 'ASIAV54V4OEUBACKIR72'
# secret = '150/NwZe6Thd0rgYaoLmzEuUNN+0nFbNgfNmRp2r'
# session = 'IQoJb3JpZ2luX2VjECwaCXVzLXdlc3QtMiJGMEQCIBvmhT9PrIdliUSY+Z7D9MaY7vDh5dRdIS1SvGJM0L+zAiB1zy/KspU7xApcob4aCYAIWc+duqKPYktVQeK2kVugLCrTAwiV//////////8BEAIaDDQwNzc5OTM2MTgzMiIMonOr+M6Gd6+gAi3FKqcDWC8nQgpUb4FV391EEUZieLht/BiMvTdrD9qVcCqK6PvKyRzJhJnpVzGL1YlAiSkh5+G7L2hA0uzl4aPJtkN/IPZNLJ+VZCG5entZMHTchFCp2KD3eS6xA4r91TPnHnvm4KAAK4u2Y4bbMZ+bOxsOKHDK7L9xPKoZvVdDjiWEj959fynFE19sFIaYiHBMqs9hjNg+PnJODrF/HI1hhFEn3KTX4W48NHU6vR3hKYG+Q3dKRSWoTAGPqhk2VO2ZOcOYohwiu/A5CFz0oNNVNJzWz1ncVBcJixJnwl72u4CN3YAeyE37vaqoSsFZ8JX8yR9HxHUgD+54Q7mkbytPvCJiUcKBvObPiFokbmN2qW624uQLD0gyujKUuK2V/WJPlAf59sU8WCWPsxvRn1ZohIboC4mGSSo2SyQRiXMxhjSNSSftIUzUQmEkLzUxfFh6KRo1ZYStTMwNDJX+gaUWThUiyyVSs5pEYnc40MikCNPw+EtWn9+PdsIBJbS+hEWse8W1hVZx0t9Vd6Y1JfRwy+gpxkXPHMRYEBvviRgE3cWU6D1oiq74xAAyMIehw6sGOqcBiJkRHwmSNVFUfFsg3mYWC8J41zLon79UN0zUrr7mla+2jKrhGxulJ/l6BcZZ8prguU7KG+1nwXtmV6VfZlb0qDqtXfy/+Vs498JGc1j4Q/5iEF0y05FljPkxOi2qBB2nQ11edcCEXTXjh7QJ29qVvF1k/FrLUrvE5KXNqyp8VQpVFplXDFKMv9pHWfqEEMiPszYPYZeeYsxouQBrbFfDbqjkNHI9z8E='
# conn = connect(
#     aws_access_key_id=access,
#     aws_secret_access_key=secret,
#     aws_session_token=session,
#     s3_staging_dir='s3://lytx-athena-result-prod-003/raw/',
#     region_name='us-west-2')
# sql = f"""
#     SELECT * FROM dp_prod_db_gps.gps_enriched_1_1
#     WHERE vehicleid = '9100FFFF-48A9-D563-C54B-9543E7480000' AND tssec BETWEEN 1697337800 AND 1697337900"""
# dt = pd.read_sql_query(sql, conn)

# # gps parquet dataset in s3 (cross-account, without permissions) query via spark
# access, secret, session = 'ASIAV54V4OEUP242KTNS', 'u9B26brgrlcCQww324j326Kcd6tmxJIHd74Fbvsg', 'IQoJb3JpZ2luX2VjEA0aCXVzLXdlc3QtMiJIMEYCIQDbGlFYfzAu886HW2GJNprPHjXz1asXhKR5fb1KFrWDZQIhAJDgeCUSqQWzI0DitQb8EYz30IeiN60tmj4nDKofth04KsoDCFYQAhoMNDA3Nzk5MzYxODMyIgy0F/GK+UBMR/4l6lEqpwOj4QLQ1HX/D0w7l2Fw0MXmW0qnfY2/gNOigOa0SoDFvV+k7soanZF2jF8v68cs+82BZTLc5liFvLldM2ygn7foNwWkkTKyEAH9nJeLPw/wKhlbVD7Y2HkU0UIv7+Aw8udIUf7zEMaX1AAlFg9FWGKN1aW3jPwv9W5Di5jX8JX08Y0X1sRjvV6WFyhZ2zKBEVNyY/cz+Uko/4CemvTlsw5ZOiiF7ubY5BYdY0ziMB0VWvEXj5Ko0hzWIIqkXXNoisxumIVgYIS7NNOqnzzQxbCFOnd/0JDC1RdUfJl00PJMhM6x4IiYIMLxb9SBSYrCk3ExIJFTYoUJ7D8OuAM64S/fDrs5IXI0H98xQMlfsF9aWNkZPSSbhl1fp0fPZGP4QfP6oxgKVAqCgCycVyIlQqpdBVXWcq0psOACU7RW582CPTvI54PwZb9B+oSR0XO6/oofqnMJBZIHDb05cGS2gkMlbBsvteEInIqtuOWQ2aHJIIEfuL5CX5YRH5a5bRKBnJN4Vtc60m2xJsqR9hMdGOLNs7lhWOm78XNfvDWA5qIczhhbVxyGbD4wtPHLqgY6pQExRdoFsw1FRgrJWQkGEYIaHNMIusipR3jZ1EvEsQmCLMJQSZ8Njh6ih+u1rZ1N7/bJC9p4sJw845Plptz91NCd1htpQAnBWKy0DTxPK573Jl90vZvy4PwA/fr9Bw8ocu5huhdBDMhLtZJJfuJJ7sX51up23Dj+0umec00DbJNBV2pqXDNapCB8gXddnVPsrz3iDvuT7VcrNZNrEmnw0ynpjeZkCfc='
# client = boto3.client('s3', aws_access_key_id=access, aws_secret_access_key=secret, aws_session_token=session)
# objects = client.list_objects_v2(Bucket='lytx-gps-kafka-prod-003', Prefix='refined/gps_enriched/1.1')['Contents']
# spark.conf.set('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider')
# spark.conf.set('spark.hadoop.fs.s3a.access.key', access)
# spark.conf.set('spark.hadoop.fs.s3a.secret.key', secret)
# spark.conf.set('spark.hadoop.fs.s3a.session.token', session)
# gps = spark.read.parquet(f's3a://lytx-gps-kafka-prod-003/refined/gps_enriched/1.1')
# gps.createOrReplaceTempView('gps')
# dx = spark.sql(f"""
#     SELECT * FROM gps WHERE year = 2023 AND month = 10 AND day = 15
#     AND vehicleId = '9100FFFF-48A9-D563-C54B-9543E7480000'
#     AND tsSec BETWEEN 1697337800 AND 1697337900""").toPandas()
