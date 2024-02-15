
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType, DoubleType

@pandas_udf(returnType=IntegerType())
def func(a: pd.Series, b: pd.Series) -> pd.Series:
    return a * b
spark.udf.register('func', func)
dx = spark.sql(f'SELECT x, y, func(x, y) AS product FROM df').toPandas()

@pandas_udf(DoubleType())
def func(x: pd.Series) -> float:
    return x.mean()
spark.udf.register('func', func)
dx = spark.sql(f'SELECT x, func(y) AS ymean FROM df GROUP BY x').toPandas()

spark.sql(f'SELECT x, y, SUM(y) OVER (PARTITION BY x ORDER BY y) AS ysum FROM df').toPandas()
def func(pdf):
    pdf['ysum'] = pdf['y'].cumsum()
    return pdf
dx = sdf.groupby('x').applyInPandas(func, schema='x long, y long, ysum long').toPandas()

from typing import Iterator

def func(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    constant = 77.3
    for df in iterator:
        df['operation'] = df['x'] + df['y'] + constant
        yield df
dx = sdf.mapInPandas(func, schema='x long, y long, operation double').toPandas()




# import os
# import pyodbc
# import numpy as np
# import pandas as pd

# path = r'/mnt/home/russell.burdt/data.parquet'

# vids = [
#     '9100FFFF-48A9-D463-7F25-3A63F36F0000',
#     '9100FFFF-48A9-D463-FF09-3A63F3FF0000',
#     '9100FFFF-48A9-CB63-325D-A8A3E3070000',
#     'AAB20D06-C6C8-E411-9747-E61F13277AAB']
# edw = pyodbc.connect('DSN=EDW')
# for vid in vids:
#     query = f"""
#         SELECT VehicleId, RecordDate, Latitude, Longitude, EventTriggerTypeId AS Id
#         FROM flat.Events
#         WHERE VehicleId = '{vid}'
#         AND RecordDate BETWEEN '2021-10-01' AND '2021-12-31'
#         """
#     df = pd.read_sql_query(query, edw)
#     df.to_parquet(
#         path=path, engine='pyarrow', compression='snappy', index=False,
#         partition_cols=['VehicleId'], flavor='spark')

# from pyspark import SparkConf
# from pyspark.sql import SparkSession

# conf = SparkConf()
# # memory available for objects returned by Spark
# conf.set('spark.driver.memory', '2g')
# # enable Apache Arrow
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# # no implicit timezone conversions
# conf.set('spark.sql.session.timeZone', 'UTC')
# # directory for temporary files
# conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
# # 200 is default, critical to increase proportionally to data volume
# conf.set('spark.sql.shuffle.partitions', 200)
# spark = SparkSession.builder.config(conf=conf).getOrCreate()

# sdf = spark.read.parquet(path)
# sdf.createOrReplaceTempView('df')
# spark.sql(f'SELECT COUNT(*) FROM df').toPandas()
# sdf.count()
# spark.sql(f"""
#     SELECT VehicleId, COUNT(*) AS records
#     FROM df
#     GROUP BY VehicleId
#     ORDER BY VehicleId""").toPandas()
