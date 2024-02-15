
"""
mapInPandas for dataframe enrichment
"""

import pandas as pd
import numpy as np
from typing import Iterator
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

conf = SparkConf()
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
spark = SparkSession.builder.master(f'local[*]').config(conf=conf).getOrCreate()

pdf = pd.DataFrame(data={'a':[1,2,3], 'b':[4*np.pi,5*np.pi,6*np.pi], 'c':['xa','xb','xc']})
sdf = spark.createDataFrame(pdf)

def func(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for df in iterator:
        df['d'] = 'xxx'
        df['e'] = df['a'].values * df['b'].values
        yield df

schema = StructType()
[schema.add(x) for x in sdf.schema]
schema.add(StructField('d', StringType(), False))
schema.add(StructField('e', DoubleType(), False))
dx = sdf.mapInPandas(func, schema).toPandas()
