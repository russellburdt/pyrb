
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
plt.style.use('bmh')

ds = pd.read_pickle(r'c:\Users\russell.burdt\Downloads\ds.p')
fig, ax = open_figure('utilization prediction model results for individual vehicle', figsize=(16, 6))
ax.plot(ds['target day'], ds['actual'], 'x-', label='actual miles per day')
ax.plot(ds['target day'], ds['baseline'], 'x-', label='baseline model\n(predict same as yesterday)')
ax.plot(ds['target day'], ds['dl'], 'x-', label='LSTM model')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, handlelength=4)
format_axes('', 'miles', 'utilization prediction model results for individual vehicle', ax, apply_concise_date_formatter=True)
largefonts(18)
fig.tight_layout()

plt.show()



# """
# vehicle state chart from gps data
# """
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from ipdb import set_trace
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# plt.style.use('bmh')

# vs = {None: 0, 'motion': 1, 'idle': 2, 'off': 3}

# df = pd.DataFrame(columns=['localtime', 'vehicle_state'], data=
#     ((pd.Timestamp('2022-8-26 10:16'), None),
#     (pd.Timestamp('2022-8-26 10:28'), 'motion'),
#     (pd.Timestamp('2022-8-26 10:43'), 'off'),
#     (pd.Timestamp('2022-8-26 10:56'), 'idle'),
#     (pd.Timestamp('2022-8-26 11:09'), 'idle'),
#     (pd.Timestamp('2022-8-26 11:19'), 'motion'),
#     (pd.Timestamp('2022-8-26 11:29'), 'off'),
#     (pd.Timestamp('2022-8-26 11:41'), 'motion')))
# x = np.vstack((df['localtime'].values[:-1], df['localtime'].values[1:])).flatten(order='F')
# y = np.vstack((df['vehicle_state'].values[1:], df['vehicle_state'].values[1:])).flatten(order='F')
# y = np.array([vs[yi] for yi in y])

# fig, ax = open_figure('vehicle state vs time', figsize=(14, 6))
# ax.plot(x, y, 'x-', ms=10, mew=2, lw=4, label='manual data')
# ax.set_yticks([0, 1, 2, 3])
# ax.set_yticklabels(['None', 'motion', 'idle', 'off'])
# format_axes('localtime', '', 'vehicle state vs time', apply_concise_date_formatter=True)
# ax.set_xticks(np.unique(x))

# df = pd.read_pickle(r'c:\Users\russell.burdt\Downloads\gps.p')
# x = np.vstack((df['localtime'].values[:-1], df['localtime'].values[1:])).flatten(order='F')
# y = np.vstack((df['vehicle_state'].values[1:], df['vehicle_state'].values[1:])).flatten(order='F')
# y = np.array([vs[yi] for yi in y])
# ax.plot(x, y, '-', ms=10, mew=2, lw=4, alpha=0.4, label='device data')
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, shadow=True, fancybox=True, handlelength=4)
# ax.set_xlim(pd.Timestamp('8-26-2022 10:10'), pd.Timestamp('8-26-2022 11:45'))

# largefonts(20)
# fig.tight_layout()
# plt.show()



# """
# own device test
# """
# import os
# import lytx
# import pandas as pd
# from functools import partial
# from pyspark import SparkConf
# from pyspark.sql import SparkSession

# # extract GPS data for own device (MV01000865, 9100FFFF-48A9-CC63-7A15-A8A3E03F0000)
# datadir = r'/mnt/home/russell.burdt/data/utilization/russell'
# assert os.path.isdir(datadir)
# dp = pd.DataFrame({
#     'VehicleId': ['9100FFFF-48A9-CC63-7A15-A8A3E03F0000'],
#     'IndustryDesc': ['test'],
#     'CompanyName': ['test'],
#     'time0': [pd.Timestamp('2022-8-20')],
#     'time1': [pd.Timestamp('2022-9-1')]})
# dp.to_pickle(os.path.join(datadir, 'dp.p'))
# lytx.distributed_data_extraction(dataset='gps', datadir=datadir, df=dp, xid='VehicleId', n=1, distributed=False, assert_new=True)

# # load GPS data for own device
# conf = SparkConf()
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.session.timeZone', 'UTC')
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')
# loc = os.path.join(datadir, f'gps.parquet')
# gps = spark.read.parquet(loc)
# gps.createOrReplaceTempView('gps')

# # clean GPS data
# lytx.gps_segmentation(spark, loc)
# gps = spark.read.parquet(loc)
# gps.createOrReplaceTempView('gps')
# os.mkdir(os.path.join(datadir, 'coverage'))
# ds = lytx.gps_segmentation_metrics(dp, spark)
# ds.to_pickle(os.path.join(datadir, 'coverage', 'gps_segmentation_metrics.p'))
# lytx.interpolate_gps(spark, loc)
# gps = spark.read.parquet(loc)
# gps.createOrReplaceTempView('gps')
# lytx.gps_interval_metrics(spark, loc)
# gps = spark.read.parquet(loc)
# gps.createOrReplaceTempView('gps')
# xs = spark.sql(f'SELECT DISTINCT VehicleId FROM gps')
# xs.groupby('VehicleId').applyInPandas(partial(lytx.distributed_geocode, loc=loc), schema=xs.schema).toPandas()
# gps = spark.read.parquet(loc)
# gps.createOrReplaceTempView('gps')

# # GPS data to memory
# dx = gps.toPandas()
# dx = dx.loc[~dx['interpolated']].reset_index(drop=True)


# """
# identify specific vehicle state patterns
# """

# import os
# import numpy as np
# import pandas as pd
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from skimage.util.shape import view_as_windows
# from ipdb import set_trace


# # load gps data, get unique vehicle-ids
# loc = r'/mnt/home/russell.burdt/data/utilization/amt/gps.parquet'
# conf = SparkConf()
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.session.timeZone', 'UTC')
# conf.set('spark.driver.memory', '32g')
# conf.set('spark.driver.maxResultSize', 0)
# conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')
# gps = spark.read.parquet(loc)
# gps.createOrReplaceTempView('gps')
# vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps').toPandas()

# # identify vehicle-id and data indices matching pattern
# while True:

#     # load data for a random vid
#     vid = np.random.choice(vids['VehicleId'].values)
#     print(f'checking {vid}')
#     dx = spark.sql(f"""SELECT * FROM gps WHERE VehicleId='{vid}'""").toPandas()
#     vs = dx['vehicle_state'].values

#     # continue if no matches to vehicle-state-pattern (vsp)
#     # vsp = np.array(['motion', 'motion', 'motion', 'idle', 'idle', 'idle', 'off', 'idle', 'idle', 'idle', 'motion', 'motion', 'motion'])
#     vsp = np.array(['motion', 'motion', 'motion', None, None, 'motion', 'motion', 'motion'])
#     windows = view_as_windows(vs, vsp.size)
#     matches = np.where(np.all(windows == vsp, axis=1))[0]
#     if matches.size == 0:
#         continue
#     # x = np.random.choice(matches)
#     # dv = dx.loc[x : x + vsp.size - 1]
#     # break

#     # continue if no matches to time_interval_sec pattern
#     for x in matches:
#         dv = dx.loc[x : x + vsp.size - 1]
#         assert all(dv['vehicle_state'].values == vsp)
#         if not any(dv.loc[pd.isnull(dv['vehicle_state']), 'all_time_interval_sec'] > 32000):
#             continue
#         # if (dv.loc[dv['vehicle_state'] == 'off', 'time_interval_sec'] < 7200).values[0]:
#         #     continue
#         # if (dv.loc[dv['vehicle_state'] == 'idle', 'time_interval_sec'] > 500).sum() < 3:
#         #     continue
#         assert False


