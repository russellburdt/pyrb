"""
Minimal examples for Stack Overflow questions
"""

import pandas as pd
from pyspark.sql import SparkSession

df = pd.DataFrame({
    'id': [1, 1, 1, 1, 2, 2, 2, 2],
    'day': [1, 2, 3, 4, 1, 2, 3, 4],
    'value': [12, 15, 8, 6, 9, 22, 43, 8]})
days = pd.DataFrame({
    'id': [1, 2],
    'dmin': [2, 3],
    'dmax': [3, 4]})

spark = SparkSession.builder.getOrCreate()
spark.createDataFrame(df).createOrReplaceTempView('df')
spark.createDataFrame(days).createOrReplaceTempView('days')

q0 = ''
for _, row in days.iterrows():
    q0 += f"""SELECT id, day, value\nFROM df\nWHERE id={row['id']}\nAND day BETWEEN {row['dmin']} AND {row['dmax']}\nUNION ALL\n"""
q0 = q0[:-11]

q1 = f"""
    SELECT df.id, df.day, df.value
    FROM df
        INNER JOIN days
        ON df.id = days.id
        AND df.day BETWEEN days.dmin AND days.dmax
    """


# import pandas as pd
# df = pd.DataFrame({
#         'id': ['a','a','a','b','b','b','b','c','c'],
#         'value': [0,1,2,3,4,5,6,7,8]})
# path = r'c:/data.parquet'
# df.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['id'], flavor='spark')

# # pyspark view
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()
# spark.read.parquet(path).createTempView('data')
# sf = spark.sql(f"""SELECT id, value, 0 AS segment FROM data""")

# # overwrite parquet dataset
# sf.write.partitionBy('id').mode('append').parquet(path)
# sf.write.partitionBy('id').mode('overwrite').parquet(path)


# spark DataFrame and write to partitioned parquet dataset
#
# sf = spark.createDataFrame(df)


# import pandas as pd
# df = pd.DataFrame({
#         'id': ['a','a','a','b','b','b','b','c','c'],
#         'name': ['up','down','left','up','down','left','right','up','down'],
#         'count': [6,7,5,3,4,2,9,12,4]})
# #   id   name  count
# # 0  a     up      6
# # 1  a   down      7
# # 2  a   left      5
# # 3  b     up      3
# # 4  b   down      4
# # 5  b   left      2
# # 6  b  right      9
# # 7  c     up     12
# # 8  c   down      4
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()
# ds = spark.createDataFrame(df)
# dp = ds.groupBy('id').pivot('name').max().toPandas()
# #   id  down  left  right  up
# # 0  c     4   NaN    NaN  12
# # 1  b     4   2.0    9.0   3
# # 2  a     7   5.0    NaN   6
# ds.createOrReplaceTempView('ds')
# spark.sql(f"""
#     SELECT * FROM ds
#     PIVOT (
#         MAX(count)
#         FOR name in ('up','down','left','right'))""").toPandas()

# spark.sql(f"""
#     SELECT * FROM ds
#     PIVOT
#     (MAX(count)
#      FOR
#      ...)""").toPandas()


# import pandas as pd
# import numpy as np
# from math import modf
# from datetime import datetime, timedelta
# from bokeh.plotting import figure
# from bokeh.models import ColumnDataSource
# from bokeh.models import CrosshairTool, HoverTool
# from bokeh.events import MouseMove
# from bokeh.io import curdoc


# n = 100
# data = ColumnDataSource({
#     'x': pd.date_range(start='06/01/2021', end='06/02/2021', periods=n),
#     'y': np.random.rand(n)})

# crosshair = CrosshairTool(dimensions='height', line_width=3)
# hover = HoverTool()
# fig = figure(width=600, height=300, tools=[crosshair, hover], x_axis_type='datetime')
# fig.circle('x', 'y', source=data)
# fig.line('x', 'y', source=data)
# hover.tooltips = [('x', '@x{%d %b %Y %H:%M:%S}'), ('y', '@y')]
# hover.formatters = {'@x': 'datetime'}

# def mousemove(event):
#     sec, epoch = modf((1e-3) * event.x)
#     print(pd.Timestamp(datetime.utcfromtimestamp(int(epoch)) + timedelta(seconds=sec)))

# fig.on_event(MouseMove, mousemove)
# curdoc().add_root(fig)


# import matplotlib.pyplot as plt
# from bokeh.plotting import figure, show

# # data
# x = [1 + 1.3e6, 2 + 1.3e6, 3 + 1.3e6]
# y = [1, 2, 3]

# # matplotlib
# fig = plt.figure(figsize=(4, 2))
# plt.plot(x, y)
# fig.tight_layout()
# plt.show()

# # bokeh
# fig = figure(width=400, height=200, tools='')
# fig.line(x, y)
# show(fig)
