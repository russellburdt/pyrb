
# """
# GPS data via S3
# """

# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from datetime import datetime

# # spark conf standard settings
# conf = SparkConf()
# conf.set('spark.driver.memory', '8g')
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.session.timeZone', 'UTC')
# # spark conf to read avro dataset
# conf.set('spark.jars', '/mnt/home/russell.burdt/spark-avro_2.12-3.1.2.jar')
# # spark conf settings to read from s3, credentials are for S3 bucket account, not EC2 instance account!
# conf.set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.2.0')
# conf.set('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider')
# conf.set('spark.hadoop.fs.s3a.access.key', 'ASIAV54V4OEUNOEZL67H')
# conf.set('spark.hadoop.fs.s3a.secret.key', 'qExr8s38i7GOxHNdYotJZgETUAmOdpDwhUWLQboX')
# conf.set('spark.hadoop.fs.s3a.session.token', 'IQoJb3JpZ2luX2VjEAoaCXVzLXdlc3QtMiJHMEUCIQD9G93L8dVAg+uiiGspiL5vPCbdsluasjyxaBBxMKOREQIgO8i4jkWoEQ6lW1jeHKXD8hc636VJQ9jQTkTcWyPeg2QqlgMIMxABGgw0MDc3OTkzNjE4MzIiDCycB28tDCTbjGQ+mCrzApAoPYlDw0eC9DgZvNCVQ7bhgUzMkTUiLNalADaB9R1BYBM5qoFbtTx5Lv72EMp/JnQojnJsBxCxG2Uaz0s9W4bMR9rZ0oOarvLAx193an7zEMiqSg2dFwwQu6+4hQFxjufPKzvR+BTWS6LcMhCH4hCzjNBgiLGi3anFZMOq1TKDSTLzza3N/LJEt5SsLHd46gcE08/HhWz0ege19yLNrtpNtrHd58mI/6F+ZW95PxIOGBQMbZq+POZUuU5FaIhMnXgf3mxxJ/BiwtO/h/X8n23VIlPA6ZnTdTVBWXqKblnji04rI0v2j1Ajab1mB+WbvJ5rWQm/W0lUDs5iY0Hw2EvWhZ85bJPtg2h2oaGyjR+hJJw4/+DaL5VeH2LAJwOOPLCJm8yV4Wiuv/FhZ8ffO5qRfLVfccrVrkpHyPvR9NGVKz3x0Ako/jrlV35lbUESTuVix7j8sdEcbbEAs/K/GHudzMfQHGjyHtj2RMUZdpgzLg2SMJTZq48GOqYBnRpA+kQy+TRVJ8c3NS4PmCDHuOQoKbw/C+b7oUWX7z/hYu7yd8aJeec9tmHlRa6jiKRm2Nq3HRakFcizYE7rzRJU25xcKxLABx1sqjZCsT7Df8LjDPecM0MsA0SUkZcCTekM0+Bmpylt9GlhQZ8X3s3DSQ4HaZP0S+ngRDkl/RAOYpNNo8ikdam4m4UFWVVm1nxreTdU+KtMdUw1xlsvYxnS1DVDSA==')
# # spark session object
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')
# # get GPS DataFrame object from S3 avro source
# now = datetime.now()
# loc = r's3a://lytx-gps-hadoop-prod-003/raw'  # r's3a://lytx-gps-kafka-prod-003/raw'
# gps = spark.read.format('avro').load(loc)
# gps.createOrReplaceTempView('gps')
# print(f'read GPS data at {loc}, {(datetime.now() - now).total_seconds():.1f}sec')
# # simple query
# now = datetime.now()
# query = f'SELECT * FROM gps LIMIT 1000'
# da = spark.sql(query).toPandas()
# print(f'{query}, {(datetime.now() - now).total_seconds():.1f}sec')


# """
# GPS data via Snowflake, available warehouses
# small, 'LABS_PROD_VWH'
# medium, 'LABS_PROD_VWH_M'
# large, 'LABS_PROD_VWH_L'
# xl, 'LABS_PROD_VWH_XL'
# 2xl, 'LABS_PROD_VWH_2XL'
# """

# import pandas as pd
# from snowflake import connector

# conn = connector.connect(
#     user='SVC_REST_API',
#     database='dp_prod_db',
#     warehouse='LABS_PROD_VWH_M',
#     role='LYTX_REST_APP_INTERNAL_PROD',
#     password='n33d!T0k#n!AuTh!%',
#     account='lytx')
# pd.read_sql_query(f'use schema LABS;', conn)

# query = f"""
#     copy into @STAGE/dir from (
#     select * from DP_PROD_DB.GPS.GPS_ENRICHED limit 1000)"""
# df = pd.read_sql_query(query, conn)

# pd.read_sql_query(f"""USE WAREHOUSE \"APP_REST_PROD_VWH_M\"""", conn)
# direct query
# query = f'SELECT * FROM GPS.GPS_ENRICHED LIMIT 1000'
# df = pd.read_sql_query(query, conn)
# parquet method
# pd.read_sql_query(f'use database SANDBOX;', conn)
# pd.read_sql_query(f'use schema LABS;', conn)
# query = f"""
#     copy into @LABS_PARQUET_EXPORT_STAGE/labs/ParquetTest from (
#     SELECT * FROM GPS.GPS_ENRICHED LIMIT 1000);"""
# df = pd.read_sql_query(query, conn)

# more complex query
# now = datetime.now()
# query = f'SELECT DISTINCT(VEHICLE_ID) AS vids FROM GPS.GPS_ENRICHED'
# db = pd.read_sql_query(query, conn)
# print(f'{query}, {(datetime.now() - now).total_seconds():.1f}sec')
# ...
# query = f'COPY INTO @LABS_PARQUET_EXPORT_STAGE/labs/ParquetTest FROM ({query})'
# pd.read_sql_query(f'use database SANDBOX;', conn)
# pd.read_sql_query(f'use schema LABS;', conn)
# dx = pd.read_sql_query(query, conn)
"""copy into @LABS_PARQUET_EXPORT_STAGE/labs/ParquetTest from (
select TS_SEC, LATITUDE, LONGITUDE from GPS_ENRICHED__INSURANCE_MODEL__EDW_ALL_VEHICLE_HISTORY limit 10);"""
