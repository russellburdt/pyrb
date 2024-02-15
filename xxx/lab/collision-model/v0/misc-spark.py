
"""
...
"""

import os
import pandas as pd
from pyspark.sql import SparkSession

pdf = pd.DataFrame({
    'x': [1, 1, 2, 2],
    'y': ['a', 'b', None, None]})
datadir = r'/mnt/home/russell.burdt/data/so.parquet'
assert not os.path.isdir(datadir)
pdf.to_parquet(path=datadir, engine='pyarrow', compression='snappy', index=False, partition_cols=['x'])

spark = SparkSession.builder.getOrCreate()
sdf = spark.read.parquet(datadir)
sdf.createOrReplaceTempView('sdf')
ds = spark.sql('DESC sdf').toPandas()
# dx = spark.sql(f"""SELECT * FROM sdf WHERE x == 1""").toPandas()


# """
# tests on EMR
# """

# import os
# import boto3
# import numpy as np
# import pandas as pd
# from shutil import rmtree
# from glob import glob
# from pyspark.sql import SparkSession
# from datetime import datetime
# from socket import gethostname
# from ipdb import set_trace
# from time import sleep

# # spark session on EC2
# from pyspark import SparkConf
# conf = SparkConf()
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.shuffle.partitions', 20000)
# spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')

# # spark session on EMR
# spark = SparkSession.builder.getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')
# spark.conf.set('spark.sql.shuffle.partitions', 20000)

# def func(pdf):
#     assert pdf.shape[0] == 1
#     x = pdf['id'].values[0]
#     with open(os.path.join('/mnt/home/russell.burdt/s3', f'f{x:02d}'), 'w') as fid:
#         pass
#     return pdf

# vx = spark.range(start=0, end=100, step=1, numPartitions=100)
# dx = vx.groupby('id').applyInPandas(func, schema=vx.schema).toPandas()
# vx = vx.toPandas()
# func(vx.loc[vx.index == 11])



# """
# connect to database on workers
# """

# import pyodbc
# from pyspark.sql import SparkSession
# from time import sleep

# spark = SparkSession.builder.getOrCreate()

# def func(x):
#     sleep(1)
#     cstr = f"""Driver=PostgreSQL;"""
#     cstr += """ServerName=dev-labs-aurora-postgres-instance-1.cctoq0yyopdx.us-west-2.rds.amazonaws.com;"""
#     cstr += """Database=labs;UserName=postgres;Password=uKvzYu0ooPo4Cw9Jvo7b;Port=5432"""
#     conn = pyodbc.connect(cstr)
#     return x

# sdf = spark.range(start=0, end=8, step=1, numPartitions=8)
# pdf = sdf.groupby('id').applyInPandas(func, schema=sdf.schema).toPandas()




# """
# pyspark tests
# """

# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# import numpy as np
# import pandas as pd
# from time import sleep
# from datetime import datetime
# from tqdm import tqdm

# # spark configuration and session
# # conf = SparkConf()
# # conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# # conf.set('spark.sql.session.timeZone', 'UTC')
# # conf.set('spark.sql.shuffle.partitions', 6000)
# # conf.set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.2.0')
# # spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()
# # spark.sparkContext.setLogLevel('ERROR')
# spark = SparkSession.builder.getOrCreate()

# # num of supposed cores
# cores = 32

# # read and count data in parquet on s3
# now = datetime.now()
# sdf = spark.read.parquet(r's3a://russell-s3/gps.parquet')
# sc = sdf.count()
# sec = (datetime.now() - now).total_seconds()
# print(f'read and count {sc} records in parquet on s3, {sec:.1f}sec')

# # measure processing overhead from distributed sleep function
# cx = 10
# def func(df):
#     sleep(1)
#     return df
# sdf = spark.range(start=0, end=cores * cx, step=1, numPartitions=cores * cx)
# sdf = sdf.groupby('id').applyInPandas(func, schema=sdf.schema)
# now = datetime.now()
# dx = sdf.toPandas()
# sec = (datetime.now() - now).total_seconds()
# print(f'distributed sleep function overhead, expected {cx:.1f}sec, actual {sec:.1f}sec, overhead={100 * ((sec / cx) - 1):.1f}%')

# # distributed prime-count function
# def isprime(num):
#     for x in range(2, num):
#         if num % x == 0:
#             return False
#     return True
# xmax = 100000
# rdd = spark.sparkContext.parallelize(np.arange(xmax))
# now = datetime.now()
# primes = rdd.filter(isprime).collect()
# sec = (datetime.now() - now).total_seconds()
# print(f'{len(primes)} prime numbers counted up to {xmax} in {sec:.1f} sec')



# """
# geocode Parquet datasets
# """

# import os
# import config
# import utils
# import numpy as np
# import pandas as pd
# from shutil import rmtree
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from pyspark.sql.types import StructType, StructField, StringType, TimestampType, BooleanType


# # population and spark session
# population = 'nst'
# conf = SparkConf()
# conf.set('spark.driver.memory', '16g')
# conf.set('spark.driver.maxResultSize', 0)
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.session.timeZone', 'UTC')
# conf.set('spark.sql.shuffle.partitions', 2000)
# conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')

# # scan data sources
# for src in ['events.parquet', 'behaviors.parquet', 'gps.parquet']: #'triggers.parquet']:
#     print(f'distributed geocoding, {src}')

#     # read and validate Parquet dataset
#     loc = os.path.join(config.DATADIR, population, src)
#     sdf = spark.read.parquet(loc)
#     assert all([x in sdf.columns for x in ['latitude_gps', 'longitude_gps', 'TS_SEC']])
#     sdf = sdf.select(*[x for x in sdf.columns if x not in ['timezone', 'localtime', 'day_of_week', 'weekday', 'state', 'county', 'country']])

#     # get geocoded Spark DataFrame
#     schema = StructType(sdf.schema.fields + [
#         StructField('timezone', StringType(), True),
#         StructField('localtime', TimestampType(), True),
#         StructField('day_of_week', StringType(), True),
#         StructField('weekday', BooleanType(), True),
#         StructField('state', StringType(), True),
#         StructField('county', StringType(), True),
#         StructField('country', StringType(), True)])
#     sdf = sdf.mapInPandas(utils.distributed_geocode, schema=schema)

#     # write df as tmp.parquet, then replace existing gps.parquet dataset
#     tmp = os.path.join(config.DATADIR, population, 'tmp.parquet')
#     delete_me = os.path.join(config.DATADIR, population, 'delete_me')
#     assert not os.path.isdir(tmp)
#     assert not os.path.isdir(delete_me)
#     partitionBy = 'VehicleId' if 'VehicleId' in sdf.columns else 'EventRecorderId'
#     sdf.write.parquet(path=tmp, mode='error', partitionBy=partitionBy, compression='snappy')
#     os.rename(loc, delete_me)
#     os.rename(tmp, loc)
#     rmtree(delete_me)



# def distributed_geocode_old(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
#     """
#     distributed function to update Parquet dataset with geocoded columns
#     """

#     # initialize object for geo-coding
#     tzf = TimezoneFinder()
#     rg = reverse_geocoder.RGeocoder()

#     # pyspark boilerplate
#     for dx in iterator:

#         # timestamp objects and valid lat/lon
#         dt = np.array([pd.Timestamp(datetime.utcfromtimestamp(x)) for x in dx['TS_SEC']])
#         lat = dx['latitude_gps'].values
#         lon = dx['longitude_gps'].values
#         ok = np.logical_and(~np.isnan(lat), ~np.isnan(lon))
#         lat, lon, dt = lat[ok], lon[ok], dt[ok]

#         # timezone, localtime, day-of-week, weekday
#         dx['timezone'] = None
#         dx['localtime'] = pd.NaT
#         dx['day_of_week'] = None
#         dx['weekday'] = None
#         if (lat.size > 0) and (lon.size > 0):
#             timezone = np.array([tzf.timezone_at(lng=a, lat=b) for a, b in zip(lon, lat)])
#             localtime = np.array([a.tz_localize('UTC').astimezone(b).tz_localize(None) for a, b in zip(dt, timezone)])
#             dow = np.array([x.strftime('%a') for x in localtime])
#             weekday = np.array([False if x in ['Sat', 'Sun'] else True for x in dow])
#             dx.loc[ok, 'timezone'] = timezone
#             dx.loc[ok, 'localtime'] = localtime
#             dx.loc[ok, 'day_of_week'] = dow
#             dx.loc[ok, 'weekday'] = weekday

#         # state, county, country
#         dx['state'] = None
#         dx['county'] = None
#         dx['country'] = None
#         if (lat.size > 0) and (lon.size > 0):
#             locations = rg.query([(a, b) for a, b in zip(lat, lon)])
#             dx.loc[ok, 'state'] = np.array([x['admin1'] for x in locations])
#             dx.loc[ok, 'county'] = np.array([x['admin2'] for x in locations])
#             dx.loc[ok, 'country'] = np.array([x['cc'] for x in locations])
#         yield dx



# """
# pyspark serialization Stack Overflow question
# """

# import numpy as np
# import pandas as pd
# from pyspark import SparkConf
# from pyspark.sql import SparkSession

# conf = SparkConf()
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# sdf = spark.createDataFrame(pd.DataFrame({
#     'lat': np.tile([37, 42, 35, -22], 100),
#     'lng': np.tile([-113, -107, 127, 34], 100)}))

# from typing import Iterator
# from timezonefinder import TimezoneFinder

# instance = [None]
# def func(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
#     if instance[0] is None:
#         instance[0] = TimezoneFinder()
#     else:
#         print('re-used')
#     tzf = instance[0]
#     for dx in iterator:
#         dx['timezone'] = [tzf.timezone_at(lng=a, lat=b) for a, b in zip(dx['lng'], dx['lat'])]
#         yield dx
# pdf = sdf.mapInPandas(func, schema='lat double, lng double, timezone string').toPandas()

# from pyspark.sql.types import StringType
# from pyspark.sql.functions import pandas_udf
# from timezonefinder import TimezoneFinder

# sdf.createOrReplaceTempView('df')
# @pandas_udf(returnType=StringType())
# def outer(lng, lat):
#     tzf = TimezoneFinder()
#     def func(lng: pd.Series, lat: pd.Series) -> pd.Series:
#         return pd.Series([tzf.timezone_at(lng=a, lat=b) for a, b in zip(lng, lat)])
#     return func(lng, lat)
# spark.udf.register('func', outer)
# pdf = spark.sql(f'SELECT lng, lat, func(lng, lat) AS timezone FROM df').toPandas()


# """
# pandas udf with spark
# """
# import pandas as pd
# from pyspark.sql.functions import pandas_udf
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from pyspark.sql.types import LongType, DoubleType, StringType, IntegerType
# from typing import Iterator

# # spark session
# conf = SparkConf()
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# spark = SparkSession.builder.config(conf=conf).getOrCreate()

# # data
# pdf = pd.DataFrame({'x': [1, 1, 2, 2], 'y': [3, 4, 5, 6]})
# sdf = spark.createDataFrame(pdf)
# sdf.createOrReplaceTempView('df')

# # series to series
# @pandas_udf(returnType=IntegerType())
# def func0(a: pd.Series, b: pd.Series) -> pd.Series:
#     return a * b
# spark.udf.register('func0', func0)
# p0 = spark.sql(f'SELECT x, y, func0(x, y) FROM df').toPandas()

# # groupby series to float
# p1a = spark.sql(f'SELECT x, AVG(Y) AS yavg FROM df GROUP BY x').toPandas()
# @pandas_udf(DoubleType())
# def func1(x: pd.Series) -> float:
#     return x.mean()
# spark.udf.register('func1', func1)
# p1b = spark.sql(f'SELECT x, func1(y) AS yavg FROM df GROUP BY x').toPandas()

# # groupby series to series (window function)
# p2a = spark.sql(f'SELECT x, y, SUM(y) OVER (PARTITION BY x ORDER BY y) AS ysum FROM df').toPandas()
# def func2(pdf):
#     pdf['ysum'] = pdf['y'].cumsum()
#     return pdf
# p2b = sdf.groupby('x').applyInPandas(func2, schema='x long, y long, ysum long').toPandas()

# # DataFrame to DataFrame
# pdf = pd.DataFrame({'animal': ['cow', 'zebra', 'lion', 'dog']})
# sdf = spark.createDataFrame(pdf)
# sdf.createOrReplaceTempView('df')
# def func3(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
#     for pdf in iterator:
#         pdf['size'] = [sizes.get(x, -1) for x in pdf['animal']]
#         yield pdf
# sizes = {
#     'cow': 4,
#     'lion': 7,
#     'dog': 2}
# p3 = sdf.mapInPandas(func3, schema='animal string, size long').toPandas()

# # DataFrame to DataFrame
# pdf = pd.DataFrame({'animal': ['elephant+bear', 'cat+bear']})
# sdf = spark.createDataFrame(pdf)
# amap = {'elephant': 1, 'bear': 4, 'cat': 9}
# def addition(astr):
#     return sum([amap.get(x, 0) for x in astr.split('+')])
# def func(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
#     for dx in iterator:
#         dx['size'] = [addition(x) for x in dx['animal']]
#         yield dx
# p3 = sdf.mapInPandas(func, schema='animal string, size long').toPandas()


# def enrich_gps_with_intervals_old(gps_loc, spark):
#     """
#     enrich gps Parquet dataset with metrics derived from gps record intervals
#     """

#     # validation
#     assert 'gps' in [x.name for x in spark.catalog.listTables()]

#     # get gps intervals and validate
#     intervals = get_intervals_by_vid_sid_old(spark)
#     intervals.createOrReplaceTempView('intervals')
#     assert sorted(intervals.columns) == ['TS_SEC', 'VehicleId', 'distance_interval_miles', 'segmentId', 'time_interval_sec']

#     # enriched intervals
#     intervals = spark.sql(f"""
#         SELECT
#             TS_SEC,
#             VehicleId,
#             segmentId,
#             distance_interval_miles,
#             distance_interval_miles / (time_interval_sec / 3600) AS mph,
#             SUM(distance_interval_miles) OVER (PARTITION BY VehicleId ORDER BY TS_SEC) AS cumulative_distance_miles,
#             SUM(distance_interval_miles) OVER (PARTITION BY VehicleId, segmentId ORDER BY TS_SEC) AS cumulative_segment_distance_miles,
#             time_interval_sec,
#             SUM(time_interval_sec) OVER (PARTITION BY VehicleId ORDER BY TS_SEC) / (60*60*24) AS cumulative_time_days,
#             SUM(time_interval_sec) OVER (PARTITION BY VehicleId, segmentId ORDER BY TS_SEC) / (60*60*24) AS cumulative_segment_time_days
#         FROM intervals""")
#     intervals.createOrReplaceTempView('intervals')

#     # get pyspark DataFrame object of enriched gps with intervals
#     gps = spark.sql(f'SELECT * FROM gps')
#     common = ['VehicleId', 'TS_SEC']
#     assert all([(x in gps.columns) and (x in intervals.columns) for x in common])
#     a = ',\n'.join([f'gps.{x}' for x in common])
#     b = ',\n'.join([f'gps.{x}' for x in set(gps.columns).difference(intervals.columns)])
#     c = ',\n'.join([f'intervals.{x}' for x in set(intervals.columns).difference(common)])
#     d = ' AND '.join([f'gps.{x}=intervals.{x}' for x in common])
#     query = f"""SELECT {a},\n{b},\n{c}\nFROM gps \nLEFT JOIN intervals\nON {d} ORDER BY gps.VehicleId, gps.TS_SEC"""
#     enriched = spark.sql(query)

#     # write enriched as tmp.parquet, then replace existing gps.parquet dataset
#     tmp = os.path.join(os.path.split(gps_loc)[0], 'tmp.parquet')
#     enriched.write.parquet(path=tmp, mode='error', partitionBy='VehicleId', compression='snappy')
#     delete_me = os.path.join(os.path.split(gps_loc)[0], 'delete_me')
#     os.rename(gps_loc, delete_me)
#     os.rename(tmp, gps_loc)
#     rmtree(delete_me)

# def reset_segment_id_old(gps_loc, spark, method='dask'):
#     """
#     reset segmentId column in gps.parquet dataset
#     """

#     # validation
#     assert 'gps' in [x.name for x in spark.catalog.listTables()]

#     # will create .crc files and may create more than one .parquet file by partition
#     if method == 'spark':

#         # pyspark DataFrame with reset segmentId column
#         cols = spark.sql(f'SHOW COLUMNS IN gps').toPandas().values.flatten()
#         cols = [x for x in cols if x != 'segmentId']
#         df = spark.sql(f"""SELECT {','.join(cols)},CAST(0 AS DOUBLE) AS segmentId FROM gps ORDER BY VehicleId, TS_SEC""")

#         # write df as tmp.parquet, then replace existing gps.parquet dataset
#         tmp = os.path.join(os.path.split(gps_loc)[0], 'tmp.parquet')
#         df.write.parquet(path=tmp, mode='error', partitionBy='VehicleId', compression='snappy')
#         delete_me = os.path.join(os.path.split(gps_loc)[0], 'delete_me')
#         os.rename(gps_loc, delete_me)
#         os.rename(tmp, gps_loc)
#         rmtree(delete_me)

#     # assert existing single parquet file and overwrite directly
#     if method == 'dask':

#         # reset segmentId for an individual vehicle-id
#         def func(vid):

#             # identify parquet file for vid
#             fn = glob(os.path.join(gps_loc, f'VehicleId={vid}', '*.parquet'))
#             assert len(fn) == 1
#             fn = fn[0]

#             # reset segmentId column and rewrite
#             df = pq.ParquetFile(fn).read()
#             if 'segmentId' in df.column_names:
#                 df = df.drop(['segmentId'])
#             df = df.append_column('segmentId', pa.array([0] * df.num_rows, pa.float64()))
#             pq.write_table(df, fn)

#         # get vids as an iterable
#         vids = spark.sql(f'SELECT DISTINCT(VehicleId) AS vids FROM gps').toPandas().values.flatten()

#         # run distributed function to reset segmentId
#         print('reset segmentId')
#         with ProgressBar():
#             compute(*[delayed(func)(vid) for vid in vids])

# def get_intervals_by_vid_sid_old(spark, method='sql'):
#     """
#     return Spark DataFrame representing time/distance intervals between gps records, partitioned by VehicleId and segmentId
#     (methods are SQL window function or applyInPandas)
#     """

#     # validation
#     assert 'gps' in [x.name for x in spark.catalog.listTables()]
#     assert len(set(['VehicleId', 'segmentId']).intersection(spark.sql(f"""SHOW COLUMNS IN gps""").toPandas().values.flatten())) == 2

#     if method == 'sql':

#         # register pandas udf to get geo-distance if not already registered
#         if 'get_distance_miles' not in [x.name for x in spark.catalog.listFunctions()]:
#             geod = Geod(ellps='WGS84')
#             @pandas_udf(returnType=DoubleType())
#             def get_distance_miles(a: pd.Series, b: pd.Series, c: pd.Series, d: pd.Series) -> pd.Series:
#                 # in meters
#                 _, _, distance = geod.inv(lons1=a, lats1=b, lons2=c, lats2=d)
#                 # in miles
#                 return pd.Series(0.000621371 * distance)
#             spark.udf.register('get_distance_miles', get_distance_miles)

#         # intervals and mph partitioned by VehicleId and segmentId
#         return spark.sql(f"""
#             WITH
#                 tmp AS (
#                     SELECT
#                         VehicleId,
#                         segmentId,
#                         latitude_gps,
#                         LAG(latitude_gps) OVER (PARTITION BY VehicleId, segmentId ORDER BY TS_SEC) as prev_latitude_gps,
#                         longitude_gps,
#                         LAG(longitude_gps) OVER (PARTITION BY VehicleId, segmentId ORDER BY TS_SEC) as prev_longitude_gps,
#                         TS_SEC,
#                         LAG(TS_SEC) OVER (PARTITION BY VehicleId, segmentId ORDER BY TS_SEC) AS prev_time_sec
#                     FROM gps
#                     WHERE segmentId IS NOT NULL)
#             SELECT
#                 VehicleId,
#                 segmentId,
#                 TS_SEC,
#                 get_distance_miles(longitude_gps, latitude_gps, prev_longitude_gps, prev_latitude_gps) as distance_interval_miles,
#                 (TS_SEC - prev_time_sec) AS time_interval_sec
#             FROM tmp
#             ORDER BY VehicleId, TS_SEC""")

#     elif method == 'pandas':

#         # intervals schema from gps schema and new fields
#         gps = spark.sql(f'SELECT * FROM gps')
#         schema = StructType(
#             [x for x in gps.schema.fields if x.name in ['VehicleId', 'segmentId', 'TS_SEC']] + [
#             StructField('distance_interval_miles', DoubleType(), True),
#             StructField('time_interval_sec', DoubleType(), True)])

#         # distributed function
#         def func(pdf):
#             geod = Geod(ellps='WGS84')
#             del pdf['latitude']
#             del pdf['longitude']
#             lon = pdf.pop('longitude_gps').values
#             lat = pdf.pop('latitude_gps').values
#             # distance interval in meters
#             _, _, distance = geod.inv(lons1=lon[1:], lats1=lat[1:], lons2=lon[:-1], lats2=lat[:-1])
#             # distance interval in miles
#             pdf['distance_interval_miles'] = np.hstack((np.nan, 0.000621371 * distance))
#             # time interval in sec
#             pdf['time_interval_sec'] = np.hstack((np.nan, np.diff(pdf['TS_SEC'].values)))
#             if pdf.shape[1] != 5:
#                 raise ValueError(pdf['VehicleId'].iloc[0])
#             return pdf

#         # return intervals as Spark Dataframe
#         return gps.groupby(['VehicleId', 'segmentId']).applyInPandas(func, schema=schema)


# """
# reverse geocode via pyspark
# """
# import numpy as np
# import pandas as pd
# import reverse_geocoder
# from timezonefinder import TimezoneFinder
# from datetime import datetime
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from pyspark.sql.types import StructType, StructField, StringType, TimestampType
# from typing import Iterator

# conf = SparkConf()
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.shuffle.partitions', 2000)
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')

# loc = r'/mnt/home/russell.burdt/data/collision-model/v2/dft/events.parquet'
# sdf = spark.read.parquet(loc)
# sdf.createOrReplaceTempView('df')
# vids = spark.sql(f'SELECT DISTINCT(VehicleId) FROM df').toPandas().values.flatten()
# vstr = ','.join([f"""'{x}'""" for x in vids[:20000]])
# sdf = spark.sql(f"""SELECT * FROM df WHERE VehicleId IN ({vstr})""")
# sdf.createOrReplaceTempView('df')

# schema = StructType(sdf.schema.fields + [
#     StructField('timezone', StringType(), True),
#     StructField('localtime', TimestampType(), True),
#     StructField('state', StringType(), True),
#     StructField('county', StringType(), True),
#     StructField('country', StringType(), True)])
# import utils
# pdf = sdf.mapInPandas(utils.func, schema=schema).toPandas()


# def reset_gps(iterator, cols):
#     """
#     distributed function to reset gps Parquet dataset
#     - remove enriched columns
#     - reset segmentId column
#     """
#     def func(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
#         for dx in iterator:
#             dx = dx[cols]
#             dx['segmentId'] = 0
#             yield dx
#     return func(iterator)

# def spark_dataframe_to_parquet(sdf, dst, tmp, clean_up=True, spark=None):
#     """
#     write Spark DataFrame to a Parquet dataset
#     - dst is path of Parquet dataset, and also source dataset for sdf
#     - sdf first written to tmp, then moved to dst
#     - clean_up=True means to remove crc files and merge parquet files
#     """

#     # validate
#     assert isinstance(sdf, DataFrame)
#     assert os.path.isdir(dst)
#     assert not os.path.isdir(tmp)
#     if clean_up:
#         assert isinstance(spark, SparkSession)

#     # write sdf as a Parquet dataset at tmp
#     sdf.write.parquet(path=tmp, mode='error', partitionBy='VehicleId', compression='snappy')

#     # move source dataset to 'delete_me' folder
#     delete_me = os.path.join(os.path.split(dst)[0], 'delete_me')
#     os.rename(dst, delete_me)

#     # move Parqet dataset at tmp to dst, then remove source dataset
#     os.rename(tmp, dst)
#     rmtree(delete_me)

#     # clean up dataset
#     if clean_up:
#         clean_up_crc(dst)
#         merge_gps_parquet(spark=spark, loc=dst)

# def clean_up_crc(loc):
#     """
#     remove _SUCCESS and .crc files in a Parquet dataset to avoid Checksum exceptions
#     """
#     if os.path.isfile(os.path.join(loc, '_SUCCESS')):
#         os.remove(os.path.join(loc, '_SUCCESS'))
#     xs = glob(os.path.join(loc, '*'))
#     for x in xs:
#         crc = glob(os.path.join(x, '.*.crc'))
#         [os.remove(x) for x in crc]

# def merge_gps_parquet(spark, loc, num_per_iteration=1, method='spark'):
#     """
#     merge more than one parquet files created during gps enrichment to a single file
#     - faster than using repartition(1)
#     - always rewrite even if single file already (tested)
#     """

#     # get partitions as a 2d numpy array
#     vids = np.array(glob(os.path.join(loc, '*'))).astype('object')
#     assert all([(os.path.split(x)[1][:10] == 'VehicleId=') and (':' not in x) for x in vids])
#     assert vids.size > 1
#     rows = int(np.ceil(vids.size / num_per_iteration))
#     vids = np.hstack((vids, np.tile(None, rows * num_per_iteration - vids.size)))
#     vids = np.reshape(vids, (rows, num_per_iteration))

#     # merge gps parquet files in a single partition
#     def func(dx):
#         fns = glob(os.path.join(dx, '*'))
#         assert len(fns) > 0
#         dfs = [pq.ParquetFile(fn).read().to_pandas() for fn in fns]
#         df = pd.concat(dfs).sort_values('TS_SEC').reset_index(drop=True)
#         assert df.duplicated().sum() == 0
#         df['VehicleId'] = os.path.split(dx)[1][10:]
#         df.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'], flavor='spark')
#         [os.remove(fn) for fn in fns]

#     # using dask - does not seem to scale with CPUs, eg consistently 1 hour execution time with 48-128 CPUs for dft population
#     if method == 'dask':
#         def outer(row):
#             row = [x for x in row if x is not None]
#             for dx in row:
#                 func(dx)
#         print('merge gps parquet files')
#         with ProgressBar():
#             compute(*[delayed(outer)(row) for row in vids])

#     # using spark - scales with CPUs
#     if method == 'spark':
#         def outer(pdf):
#             row = [x for x in pdf['vids'].values[0] if x is not None]
#             for dx in row:
#                 func(dx)
#             return pdf
#         sdf = spark.createDataFrame(pd.DataFrame({'group': range(vids.shape[0]), 'vids': vids.tolist()}))
#         now = datetime.now()
#         sdf.groupby('group').applyInPandas(outer, schema=sdf.schema).toPandas()
#         print(f'merge gps parquet files, {(1 / 60) * (datetime.now() - now).total_seconds():.1f} minutes')

# def gps_segmentation_intervals(spark, time_interval_sec=61, distance_interval_miles=0.1):
#     """
#     return a DataFrame of VehicleId and TS_SEC that complete intervals exceeding limits in args
#     """

#     # validation
#     assert 'gps' in [x.name for x in spark.catalog.listTables()]
#     assert len(set(['VehicleId', 'segmentId']).intersection(spark.sql(f"""SHOW COLUMNS IN gps""").toPandas().values.flatten())) == 2

#     # intervals schema from gps schema and new fields
#     gps = spark.sql(f'SELECT * FROM gps')
#     schema = StructType([x for x in gps.schema.fields if x.name in ['VehicleId', 'TS_SEC']])

#     # distributed function
#     def func(pdf):
#         geod = Geod(ellps='WGS84')
#         lon = pdf['longitude_gps'].values
#         lat = pdf['latitude_gps'].values
#         # distance interval in meters
#         _, _, dx = geod.inv(lons1=lon[1:], lats1=lat[1:], lons2=lon[:-1], lats2=lat[:-1])
#         # distance interval in miles
#         dx = np.hstack((np.nan, 0.000621371 * dx))
#         # time interval in sec
#         tx = np.hstack((np.nan, np.diff(pdf['TS_SEC'].values)))
#         # return segmentation intervals only
#         return pdf.loc[np.logical_and(tx > time_interval_sec, dx > distance_interval_miles), ['VehicleId', 'TS_SEC']]

#     # segmentation intervals as Spark Dataframe
#     return gps.groupby(['VehicleId', 'segmentId']).applyInPandas(func, schema=schema)

# def gps_data_segmentation(spark, nok):
#     """
#     update segmentId column based on GPS data segmentation algorithm
#     """

#     # validation
#     assert 'gps' in [x.name for x in spark.catalog.listTables()]
#     assert spark.sql(f"""SELECT DISTINCT(segmentId) FROM gps""").toPandas().values.flatten() == 0

#     # update segmentId for an individual vehicle-id
#     def func(df):

#         # get nok rows for vid, return if None
#         vid = pd.unique(df['VehicleId'])
#         assert vid.size == 1
#         vid = vid[0]
#         nok_vid = nok.loc[nok['VehicleId'] == vid]
#         if nok_vid.size == 0:
#             return df

#         # update segmentId
#         df = df.sort_values('TS_SEC').reset_index(drop=True)
#         for x in np.sort(np.where(df['TS_SEC'].isin(nok_vid['TS_SEC']).values)[0]):

#             # standard update for all nok rows
#             df.loc[x, 'segmentId'] = None

#             # increment segmentId if subsequent rows
#             if x < df.shape[0] - 2:
#                 df.loc[x + 1:, 'segmentId'] += 1

#             # correct 1-row segment at start of df
#             if (x == 1) and (~np.isnan(df.loc[0, 'segmentId'])):
#                 df.loc[0, 'segmentId'] = None

#             # correct 1-row segments after start of df
#             elif (x > 1) and (~np.isnan(df.loc[x - 1, 'segmentId'])) and (np.isnan(df.loc[x - 2, 'segmentId'])):
#                 df.loc[x - 1, 'segmentId'] = None

#         # correct 1-row segments at end of df
#         if (np.isnan(df.loc[df.shape[0] - 2, 'segmentId'])) and (~np.isnan(df.loc[df.shape[0] - 1, 'segmentId'])):
#             df.loc[df.shape[0] - 1, 'segmentId'] = None

#         # validate no 1-row segments
#         if any(~pd.isnull(df['segmentId'])):
#             assert df['segmentId'].value_counts().min() > 1

#         # validate and return
#         assert np.all(np.sort(df['TS_SEC'].values) == df['TS_SEC'].values)
#         return df

#     # return gps Spark DataFrame with updated segmentId
#     gps = spark.sql(f'SELECT * FROM gps')
#     return gps.groupby('VehicleId').applyInPandas(func, schema=gps.schema)

# def gps_interval_metrics(spark):
#     """
#     enrich gps dataset with interval based metrics
#     """

#     # validation
#     assert 'gps' in [x.name for x in spark.catalog.listTables()]
#     gps = spark.sql(f'SELECT * FROM gps')

#     # first updated schema
#     schema = StructType(gps.schema.fields + [
#         StructField('distance_interval_miles', DoubleType(), True),
#         StructField('cumulative_segment_distance_miles', DoubleType(), True),
#         StructField('time_interval_sec', DoubleType(), True),
#         StructField('cumulative_segment_time_days', DoubleType(), True),
#         StructField('mph', DoubleType(), True)])

#     # first distributed function
#     def func(df):
#         geod = Geod(ellps='WGS84')
#         df = df.sort_values('TS_SEC')
#         lon = df['longitude_gps'].values
#         lat = df['latitude_gps'].values
#         assert (np.isnan(lon).sum() == 0) and (np.isnan(lat).sum() == 0) and (np.isnan(df['TS_SEC']).sum() == 0)
#         # distance interval in meters
#         _, _, dx = geod.inv(lons1=lon[1:], lats1=lat[1:], lons2=lon[:-1], lats2=lat[:-1])
#         # distance interval in miles
#         dx = np.hstack((np.nan, 0.000621371 * dx))
#         df['distance_interval_miles'] = dx
#         df['cumulative_segment_distance_miles'] = np.hstack((np.nan, np.cumsum(dx[1:])))
#         # time interval in sec
#         tx = np.hstack((np.nan, np.diff(df['TS_SEC'].values)))
#         df['time_interval_sec'] = tx
#         df['cumulative_segment_time_days'] = np.hstack((np.nan, (1 / 86400) * np.cumsum(tx[1:])))
#         df['mph'] = dx / (tx / 3600)
#         # return enriched DataFrame
#         return df

#     # first enriched Spark DataFrame
#     gps = gps.groupby(['VehicleId', 'segmentId']).applyInPandas(func, schema=schema)

#     # second updated schema
#     schema = StructType(gps.schema.fields + [
#         StructField('cumulative_distance_miles', DoubleType(), True),
#         StructField('cumulative_time_days', DoubleType(), True)])

#     # second distributed function
#     def func(df):
#         df = df.sort_values('TS_SEC')
#         df['cumulative_distance_miles'] = np.nancumsum(df['distance_interval_miles'])
#         df['cumulative_time_days'] = (1 / 86400) * np.nancumsum(df['time_interval_sec'])
#         return df

#     # return second enriched Spark DataFrame
#     return gps.groupby('VehicleId').applyInPandas(func, schema=schema)


# # reset gps dataset (remove enriched columns, reset segmentId column)
# cols = ['TS_SEC', 'longitude', 'latitude', 'longitude_gps', 'latitude_gps', 'VehicleId']
# schema = StructType([x for x in gps.schema.fields if x.name in cols] + [StructField('segmentId', DoubleType(), True)])
# utils.spark_dataframe_to_parquet(sdf=gps.mapInPandas(partial(utils.reset_gps, cols=cols), schema=schema),
#     dst=gps_loc, tmp=os.path.join(os.path.split(gps_loc)[0], 'tmp.parquet'), clean_up=True, spark=spark)
# gps = reload_and_validate_gps()

# # apply gps data segmentation
# nok = utils.gps_segmentation_intervals(spark, time_interval_sec=61, distance_interval_miles=0.1).toPandas()
# utils.spark_dataframe_to_parquet(sdf=utils.gps_data_segmentation(spark, nok),
#     dst=gps_loc, tmp=os.path.join(os.path.split(gps_loc)[0], 'tmp.parquet'), clean_up=True, spark=spark)
# gps = reload_and_validate_gps()
# g0 = gps.toPandas()

# # enrich gps based on interval metrics
# utils.spark_dataframe_to_parquet(sdf=utils.gps_interval_metrics(spark),
#     dst=gps_loc, tmp=os.path.join(os.path.split(gps_loc)[0], 'tmp.parquet'), clean_up=True, spark=spark)
# gps = reload_and_validate_gps()
