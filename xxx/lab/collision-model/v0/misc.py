
"""
figs for c-suite slide
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrb.mpl import metric_distribution, open_figure, format_axes, largefonts, save_pngs
from functools import reduce
from ipdb import set_trace
plt.style.use('bmh')


# pitch-deck data
yp3 = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/artifacts-00/model-prediction-probabilities.p')
dp3 = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/artifacts-00/population-data.p')
assert all(dp3['collision-47'] == yp3['actual outcome'])
ok = dp3['IndustryDesc'] == 'Freight/Trucking'
yp3 = yp3.loc[ok].reset_index(drop=True)
dp3 = dp3.loc[ok].reset_index(drop=True)

# gwcc data
yp5 = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/artifacts-05/model-prediction-probabilities.p')
dp5 = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/artifacts-05/population-data.p')
dx5 = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/artifacts-05/ml-data.p')
df5 = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/artifacts-05/feature-importance.p')
assert all(np.logical_or(dp5['collision-gwcc'].values, dp5['collision-47'].values) == yp5['actual outcome'])
assert dx5.shape == (yp5.shape[0], df5['features'].size)
yp5['miles'] = dx5[:, df5['features'] == 'gps_miles'].flatten()

# segmentation data, prediction probability
yp3 = yp3.sort_values('prediction probability', ascending=False).reset_index(drop=True)
x3 = 100 * np.arange(1, yp3.shape[0] + 1) / yp3.shape[0]
y3 = 100 * yp3['actual outcome'].values.cumsum() / yp3['actual outcome'].sum()
yp5 = yp5.sort_values('prediction probability', ascending=False).reset_index(drop=True)
x5 = 100 * np.arange(1, yp5.shape[0] + 1) / yp5.shape[0]
y5 = 100 * yp5['actual outcome'].values.cumsum() / yp5['actual outcome'].sum()

# segmentation data, miles
yp5 = yp5.sort_values('miles', ascending=False).reset_index(drop=True)
xm5 = 100 * np.arange(1, yp5.shape[0] + 1) / yp5.shape[0]
ym5 = 100 * yp5['actual outcome'].values.cumsum() / yp5['actual outcome'].sum()

# segmentation curve
fig, ax = open_figure('segmentation curves', figsize=(13, 7))
ax.plot(x3, y3, '-', lw=5, label='Pitch Deck, segmentation by\nprediction probability', color='darkblue')
ax.plot(x5, y5, '-', lw=5, label='GWCC, segmentation by\nprediction probability', color='seagreen')
ax.plot(xm5, ym5, '--', lw=5, label='GWCC, segmentation by\nmiles-driven', color='seagreen')
ax.legend(loc='lower right', handlelength=5, fancybox=True, shadow=True, labelspacing=1.6)
ax.set_xlim(-1, 101)
ax.set_ylim(-1, 101)
ax.set_xticks(np.arange(0, 101, 10))
ax.set_yticks(np.arange(0, 101, 10))
format_axes('percentage of vehicle evaluations', 'cumulative percentage of collisions', 'vehicle evaluation segmentation curves', ax)
largefonts(22)
fig.tight_layout()

# miles-driven distribution
x = yp5['miles'].values
bins = np.linspace(0, 72000, 100)
metric_distribution(x, bins, title='distribution of miles-driven, GWCC Test Population', xlabel='miles', size=22, figsize=(13, 7))

plt.show()


# """
# process additional data for gwcc 8-10-23,
# reproduce charts from presentation
# """

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pyrb.mpl import metric_distribution
# from functools import reduce
# from ipdb import set_trace
# plt.style.use('bmh')


# # read and validate data wrt metrics from presentation
# fn = r'c:/Users/russell.burdt/Downloads/gwcc-lytx-data.parquet'
# df = pd.read_parquet(fn)
# assert df.shape[0] == 64858
# assert pd.unique(df['VehicleId']).size == 5776
# assert pd.unique(df['CompanyName']).size == 43
# assert df['collision-lytx'].sum() == 390
# assert df['collision-gwcc'].sum() == 205
# assert np.logical_or(df['collision-lytx'], df['collision-gwcc']).sum() == 562


# # slide-5 data
# df = df.sort_values('prediction probability', ascending=False).reset_index(drop=True)
# d5 = pd.DataFrame({
#     'x-axis, cumulative percentage of vehicle evals': 100 * np.arange(1, df.shape[0] + 1) / df.shape[0],
#     'y1, cumulative percentage of lytx collisions': 100 * df['collision-lytx'].values.cumsum() / df['collision-lytx'].sum(),
#     'y2, cumulative percentage of gwcc collisions': 100 * df['collision-gwcc'].values.cumsum() / df['collision-gwcc'].sum(),
#     'y3, cumulative percentage of lytx+gwcc collisions': 100 * np.logical_or(df['collision-lytx'], df['collision-gwcc']).values.cumsum() / np.logical_or(df['collision-lytx'], df['collision-gwcc']).sum()})
# d5.to_csv(r'c:/Users/russell.burdt/Downloads/slide5.csv', index=False)
# fig, ax = plt.subplots(figsize=(9, 6))
# title = 'vehicle-eval segmentation curves'
# x = d5['x-axis, cumulative percentage of vehicle evals'].values
# ax.plot(x, d5['y1, cumulative percentage of lytx collisions'].values, '-', lw=4, label='lytx collisions')
# ax.plot(x, d5['y3, cumulative percentage of lytx+gwcc collisions'].values, '-', lw=4, label='lytx+gwcc collisions')
# ax.plot(x, d5['y2, cumulative percentage of gwcc collisions'].values, '-', lw=4, label='gwcc collisions')
# fig.canvas.manager.set_window_title(title)
# ax.legend(loc='lower right', fontsize=20)
# plt.xlabel('percentage of vehicle evals', fontsize=20)
# plt.ylabel('cumulative percentage of collisions', fontsize=20)
# plt.title(title, fontsize=20)
# fig.tight_layout()
# assert False

# # slide-6 data
# d6 = df[['prediction probability', 'miles-driven']].copy()
# d6['collision'] = np.logical_or(df['collision-lytx'], df['collision-gwcc'])
# d6['collision'] = d6['collision'].astype('int')
# d6.to_csv(r'c:/Users/russell.burdt/Downloads/slide6.csv', index=False)

# # vehicle-eval segmentation curves
# fig, ax = plt.subplots(figsize=(9, 6))
# title = 'vehicle-eval segmentation curves'
# df = df.sort_values('prediction probability', ascending=False).reset_index(drop=True)
# x = 100 * np.arange(1, df.shape[0] + 1) / df.shape[0]
# # lytx collisions
# nc = df['collision-lytx'].sum()
# y = 100 * df['collision-lytx'].values.cumsum() / nc
# ax.plot(x, y, '-', lw=4, label=f'{nc} lytx collisions')
# # lytx + gwcc collisions
# nc = np.logical_or(df['collision-lytx'], df['collision-gwcc']).sum()
# y = 100 * np.logical_or(df['collision-lytx'], df['collision-gwcc']).values.cumsum() / nc
# ax.plot(x, y, '-', lw=4, label=f'{nc} lytx+gwcc collisions')
# # gwcc collisions
# nc = df['collision-gwcc'].sum()
# y = 100 * df['collision-gwcc'].values.cumsum() / nc
# ax.plot(x, y, '-', lw=4, label=f'{nc} gwcc collisions')
# # clean up
# fig.canvas.manager.set_window_title(title)
# ax.legend(loc='lower right', fontsize=20)
# plt.xlabel('percentage of vehicle evals', fontsize=20)
# plt.ylabel('cumulative percentage of collisions', fontsize=20)
# plt.title(title, fontsize=20)
# fig.tight_layout()

# # model perfomance as compared to sorting vehicles by miles-driven
# fig, ax = plt.subplots(figsize=(9, 6))
# title = 'segmentation curve vs baseline miles-driven'
# # lytx + gwcc collisions, sorted by descending prediction probability
# df = df.sort_values('prediction probability', ascending=False).reset_index(drop=True)
# nc = np.logical_or(df['collision-lytx'], df['collision-gwcc']).sum()
# x = 100 * np.arange(1, df.shape[0] + 1) / df.shape[0]
# y = 100 * np.logical_or(df['collision-lytx'], df['collision-gwcc']).values.cumsum() / nc
# ax.plot(x, y, '-', lw=4, label=f'prediction probability', color='darkblue')
# # miles-driven
# df = df.sort_values('miles-driven', ascending=False).reset_index(drop=True)
# y = 100 * np.logical_or(df['collision-lytx'], df['collision-gwcc']).values.cumsum() / nc
# ax.plot(x, y, '-', lw=4, label=f'miles-driven', color='lightblue')
# # clean up
# fig.canvas.manager.set_window_title(title)
# ax.legend(loc='upper left', fontsize=20)
# plt.xlabel('percentage of vehicle evals', fontsize=20)
# plt.ylabel('cumulative percentage of collisions', fontsize=20)
# plt.title(title, fontsize=20)
# fig.tight_layout()

# # miles-driven metric distribution
# metric_distribution(x=df['miles-driven'], bins=np.linspace(0, 72000, 100))

# # collision prediction model performance by company, slide 7 data
# dg = df.groupby('CompanyName')
# dc0 = dg['VehicleId'].count().reset_index(drop=False).rename(columns={'VehicleId': 'num vehicle evaluations'})
# dc1 = dg['prediction probability'].mean().reset_index(drop=False).rename(columns={'prediction probability': 'mean collision prediction'})
# def func(dx):
#     assert (~pd.isnull(dx['Tier_Mod'])).sum() == dx['collision-gwcc'].sum()
#     return pd.Series({
#         'gwcc collision associations': dx['collision-gwcc'].sum(),
#         'gwcc tier_mod': ','.join([str(x) for x in dx.loc[dx['collision-gwcc'], 'Tier_Mod'].values]),
#         'gwcc miles-driven': ','.join([str(x) for x in dx.loc[dx['collision-gwcc'], 'miles-driven'].values])})
# dc2 = dg.apply(func).reset_index(drop=False)
# d7 = reduce(lambda a, b: pd.merge(a, b, on='CompanyName', how='inner'), (dc0, dc1, dc2))
# d7 = d7.sort_values('mean collision prediction').reset_index(drop=True)
# d7['CompanyName'] = [f'Company{x+1}' for x in range(d7.shape[0])]
# d7.to_csv(r'c:/Users/russell.burdt/Downloads/slide7.csv', index=False)


# """
# additional data for gwcc, 8-10-23
# """

# import os
# import pickle
# import numpy as np
# import pandas as pd
# from scipy.interpolate import interp1d
# from ipdb import set_trace

# # gwcc raw data
# d0 = pd.read_csv(r'c:/Users/russell.burdt/Downloads/gwcc-loss-records.csv')
# d0 = d0[['Company_Name', 'ACCD_DATM', 'Tier_Mod', 'TERR_CODE', 'TERR_FAC']].copy().rename(columns={'Company_Name': 'GW-companyname', 'ACCD_DATM': 'GW-datetime'})
# d0['GW-datetime'] = [pd.Timestamp(x) for x in d0['GW-datetime']]
# d0 = d0.loc[~d0.duplicated()].reset_index(drop=True)
# d1 = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/gwcc-lytx-loss-records.p')
# d2 = pd.merge(left=d1, right=d0, on=['GW-companyname', 'GW-datetime'], how='left')
# assert d1.shape[0] == d2.shape[0]
# assert all(d1[['GW-companyname', 'GW-datetime']] == d2[['GW-companyname', 'GW-datetime']])

# # model artifacts
# adir = r'c:/Users/russell.burdt/Downloads/artifacts-05'
# assert os.path.isdir(adir)
# yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
# dp = pd.read_pickle(os.path.join(adir, 'population-data.p'))
# df = pd.read_pickle(os.path.join(adir, 'ml-data.p'))
# with open(os.path.join(adir, 'feature-importance.p'), 'rb') as fid:
#     dfm = pickle.load(fid)
# features = dfm['features']
# assert (df.shape[0] == dp.shape[0] == yp.shape[0]) and (df.shape[1] == features.size)

# # combined dataset
# dx = dp[['VehicleId', 'CompanyName', 'time0', 'time1', 'time2', 'collision-47', 'collision-gwcc', 'collision-gwcc-idx']].copy()
# assert dx.duplicated().sum() == 0
# dx = dx.sort_values(['VehicleId', 'time1'])
# yp = yp.loc[dx.index].reset_index(drop=True)
# miles = df[dx.index, features == 'gps_miles']
# dx = dx.reset_index(drop=True)
# ypx = interp1d((yp['prediction probability'].min(), yp['prediction probability'].max()), (0, 100))
# dx['prediction probability'] = ypx(yp['prediction probability'].values)
# dx = dx.rename(columns={'collision-47': 'collision-lytx'})
# dx['miles-driven'] = miles
# dx['GW-companyname'] = None
# dx['GW-datetime'] = None
# dx['VIN'] = None
# dx['Tier_Mod'] = None
# dx['TERR_CODE'] = None
# dx['TERR_FAC'] = None
# for x, row in dx.loc[dx['collision-gwcc']].iterrows():
#     c0 = isinstance(row['collision-gwcc-idx'], int)
#     c1 = isinstance(row['collision-gwcc-idx'], str)
#     c2 = c1 and (',' in row['collision-gwcc-idx'])
#     if c0 or (c1 and not c2):
#         if c0:
#             dx.loc[x, 'collision-gwcc-idx'] = str(dx.loc[x, 'collision-gwcc-idx'])
#         xx = int(dx.loc[x, 'collision-gwcc-idx'])
#         gwcc = d2.loc[xx]
#         assert row['VehicleId'] == gwcc['VehicleId']
#         assert row['time1'] < gwcc['GW-datetime'] < row['time2']
#         for col in ['GW-companyname', 'VIN', 'Tier_Mod', 'TERR_CODE', 'TERR_FAC']:
#             dx.loc[x, col] = gwcc[col]
#         dx.loc[x, 'GW-datetime'] = gwcc['GW-datetime'].strftime('%Y-%m-%d %H:%M:%S')
#         continue
#     assert c2
#     xx = np.array([int(xi) for xi in row['collision-gwcc-idx'].split(',')])
#     gwcc = d2.loc[xx]
#     for col in ['GW-companyname', 'VIN', 'Tier_Mod', 'TERR_CODE', 'TERR_FAC']:
#         assert pd.unique(gwcc[col]).size == 1
#         dx.loc[x, col] = gwcc.iloc[0][col]
#     assert pd.unique(gwcc['GW-datetime']).size > 1
#     dx.loc[x, 'GW-datetime'] = ','.join([xi.strftime('%Y-%m-%d %H:%M:%S') for xi in gwcc['GW-datetime']])

# # anonymize company-name and vehicle-id
# vxs = pd.unique(dx['VehicleId'])
# vxs = {a: b for a, b in zip(vxs, [f'Vehicle{x + 1:04d}' for x in range(vxs.size)])}
# dx['VehicleId'] = [vxs[x] for x in dx['VehicleId']]
# cxs = pd.unique(dx['CompanyName'])
# cxs = {a: b for a, b in zip(cxs, [f'Company{x + 1:02d}' for x in range(cxs.size)])}
# dx['CompanyName'] = [cxs[x] for x in dx['CompanyName']]

# # validate wrt '2023-8-2 Collision Prediction Model for GWCC'
# assert dx.shape[0] == 64858
# assert pd.unique(dx['VehicleId']).size == 5776
# assert pd.unique(dx['CompanyName']).size == 43
# assert dx['collision-lytx'].sum() == 390
# assert dx['collision-gwcc'].sum() == 205
# assert np.logical_or(dx['collision-lytx'], dx['collision-gwcc']).sum() == 562
# dx.to_parquet(r'c:/Users/russell.burdt/Downloads/gwcc-lytx-data.parquet', index=False)




# """
# prediction metrics by company
# """

# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from functools import reduce
# from pyrb.mpl import metric_distribution, open_figure, format_axes, largefonts, save_pngs
# from scipy.interpolate import interp1d
# plt.style.use('bmh')

# adir = r'c:/Users/russell.burdt/Downloads/artifacts-07'
# assert os.path.isdir(adir)
# yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
# ypx = interp1d((yp['prediction probability'].min(), yp['prediction probability'].max()), (0, 100))
# yp['prediction probability'] = ypx(yp['prediction probability'].values)
# dp = pd.read_pickle(os.path.join(adir, 'population-data.p'))
# dp = dp.loc[~dp['oversampled']]
# assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))
# df = pd.merge(yp, dp, left_index=True, right_index=True, how='inner')
# assert all(df['actual outcome'] == df['outcome'])
# del df['actual outcome']

# dg = df.groupby('CompanyName')
# d0 = dg['outcome'].count().to_frame().reset_index(drop=False).rename(columns={'outcome': 'num evals'})
# d1 = dg['outcome'].sum().to_frame().reset_index(drop=False).rename(columns={'outcome': 'num collisions'})
# d2 = dg['prediction probability'].mean().to_frame().reset_index(drop=False).rename(columns={'prediction probability': 'mean'})
# d3 = dg['prediction probability'].std().to_frame().reset_index(drop=False).rename(columns={'prediction probability': 'stdev'})
# d4 = dg['prediction probability'].min().to_frame().reset_index(drop=False).rename(columns={'prediction probability': 'min'})
# d5 = dg['prediction probability'].max().to_frame().reset_index(drop=False).rename(columns={'prediction probability': 'max'})
# d6 = dg['prediction probability'].quantile(0.10).to_frame().reset_index(drop=False).rename(columns={'prediction probability': 'q10'})
# d7 = dg['prediction probability'].quantile(0.90).to_frame().reset_index(drop=False).rename(columns={'prediction probability': 'q90'})
# dx = reduce(lambda a, b: pd.merge(a, b, on='CompanyName', how='inner'), (d0, d1, d2, d3, d4, d5, d6, d7))
# dx['positive class, %'] = 100 * dx['num collisions'] / dx['num evals']
# dx = dx.sort_values('mean', ascending=True).reset_index(drop=True)

# fig, ax = open_figure('prediction metrics by company', figsize=(16, 9))
# kws = {'color': 'darkblue'}
# for x, row in dx.iterrows():
#     assert row['min'] < row['q10'] < row['q90'] < row['max']
#     ax.plot(np.array([row['min'], row['max']]), np.tile(x, 2), '-', color='darkblue', lw=1, label=None)
#     l0 = ax.plot(row['mean'], x, ms=10, marker='s', **kws)[0]
#     l1 = ax.plot(np.array([row['mean'] - row['stdev'], row['mean'] + row['stdev']]), np.tile(x, 2), ms=8, marker='o', **kws)[0]
#     l2 = ax.plot(row['min'], x, ms=12, marker='>', **kws)[0]
#     l3 = ax.plot(row['max'], x, ms=12, marker='<', **kws)[0]
#     l4 = ax.plot(row['q10'], x, ms=14, marker='4', **kws)[0]
#     l5 = ax.plot(row['q90'], x, ms=14, marker='3', **kws)[0]
#     if x == 0:
#         l0.set_label('mean')
#         l1.set_label('mean +-1 stdev')
#         l2.set_label('min')
#         l3.set_label('max')
#         l4.set_label('10-percentile')
#         l5.set_label('90-percentile')
#     format_axes('collision prediction', '', 'collision prediction model metrics by company-name (anonymous)', ax)
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=3, shadow=True, fancybox=True)
#     ax.set_xlim(-1, 100)
#     ax.set_xticks(np.arange(0, 101, 10))
#     ax.set_ylim(-0.5, dx.shape[0] - 0.5)
#     ax.set_yticks(np.arange(dx.shape[0]))
#     # ax.set_yticklabels(dx['CompanyName'].values)
#     ax.set_yticklabels([f'Company{x + 1}' for x in range(dx.shape[0])])
#     largefonts(18)
#     fig.tight_layout()

# fig, ax = open_figure('mean prediction vs percentage of vehicle evals associated with any collision', figsize=(10, 6))
# ax.plot(dx['mean'], dx['positive class, %'], 'x', ms=12, mew=3)
# title = 'percentage of vehicle evals associated with any collision vs collision prediction mean'
# format_axes('collision prediction mean for all vehicle evals by company', '%', title, ax)
# largefonts(18)
# ax.set_xlim(0, 100)
# ax.set_ylim(0, 5)
# fig.tight_layout()



# """
# vehicle-eval-segmentation curves, resolved by industry
# """

# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from functools import reduce
# from pyrb.mpl import metric_distribution, open_figure, format_axes, largefonts, save_pngs
# from scipy.stats import ks_2samp
# plt.style.use('bmh')


# # load data
# xs = [3, 4, 5, 7]
# for x in xs:
#     adir = rf'c:/Users/russell.burdt/Downloads/artifacts-0{x}'
#     assert os.path.isdir(adir)
#     yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
#     dp = pd.read_pickle(os.path.join(adir, 'population-data.p'))
#     dp = dp.loc[~dp['oversampled']]
#     assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))

#     # vehicle-eval segmentation curve
#     title = f'vehicle-eval-segmentation curves'
#     fig, ax = open_figure(title, figsize=(10, 6))

#     # # full dataset
#     # x = 100 * np.arange(1, yp.shape[0] + 1) / yp.shape[0]
#     # y = 100 * yp.sort_values('prediction probability', ascending=False)['actual outcome'].values.cumsum() / yp['actual outcome'].sum()
#     # p = ax.plot(x, y, '-', lw=4, label=f"""full dataset, {yp.shape[0]} evals\nGWCC population""")[0]

#     # freight/trucking
#     ok = dp['IndustryDesc'] == 'Freight/Trucking'
#     ypx = yp.loc[ok].reset_index(drop=True)
#     x = 100 * np.arange(1, ypx.shape[0] + 1) / ypx.shape[0]
#     y = 100 * ypx.sort_values('prediction probability', ascending=False)['actual outcome'].values.cumsum() / ypx['actual outcome'].sum()
#     label = f"""{os.path.split(adir)[1].split('-')[1]}, Freight/Trucking"""
#     ax.plot(x, y, '-', lw=4, label=label)

#     # # not freight/trucking
#     # ok = dp['IndustryDesc'] != 'Freight/Trucking'
#     # ypx = yp.loc[ok].reset_index(drop=True)
#     # x = 100 * np.arange(1, ypx.shape[0] + 1) / ypx.shape[0]
#     # y = 100 * ypx.sort_values('prediction probability', ascending=False)['actual outcome'].values.cumsum() / ypx['actual outcome'].sum()
#     # ax.plot(x, y, linestyle='dotted', lw=4, label=f"""not Freight/Trucking, {ypx.shape[0]} evals""", color=p.get_color(), alpha=0.8)

#     # clean up
#     format_axes('percentage of vehicle evals', 'cumulative percentage of collisions', title, ax)
#     ax.legend(loc='lower right', handlelength=5)
#     largefonts(18)
#     fig.tight_layout()





# """
# filter dv for data already collected - goes in s1
# """
# # filter dv for data already collected
# conf = SparkConf()
# conf.set('spark.driver.memory', '64g')
# conf.set('spark.driver.maxResultSize', 0)
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.debug.maxToStringFields', 500)
# spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')
# v0 = np.array([x.split('=')[1] for x in glob(os.path.join(datadir, 'gps0.parquet', 'VehicleId=*'))])
# dv['ok'] = False
# vx = spark.range(start=0, end=dv.shape[0], step=1, numPartitions=dv.shape[0])
# schema = StructType([StructField('VehicleId', StringType(), nullable=False), StructField('ok', BooleanType(), nullable=False)])
# def func(pdf):

#     # validate and handle null case
#     assert pdf.shape[0] == 1
#     dvx = dv.loc[pdf['id'].values[0]]
#     if dvx['VehicleId'] not in v0:
#         return pd.DataFrame(data={'VehicleId': [dvx['VehicleId']], 'ok': [False]})

#     # min and max time from snowflake data
#     snow = lytx.get_conn('snowflake')
#     time0 = int((dvx['time0'] - datetime(1970, 1, 1)).total_seconds())
#     time1 = int((dvx['time1'] - datetime(1970, 1, 1)).total_seconds())
#     query = f"""
#         SELECT MIN(TS_SEC) AS tmin, MAX(TS_SEC) AS tmax
#         FROM GPS.GPS_ENRICHED
#         WHERE TS_SEC BETWEEN {time0} AND {time1}
#         AND ((VEHICLE_ID = '{dvx['VehicleId'].lower()}') OR (VEHICLE_ID = '{dvx['VehicleId'].upper()}'))"""
#     tmin, tmax = pd.read_sql_query(con=snow, sql=query).values.flatten()

#     # min and max time from data already collected
#     fn = glob(os.path.join(datadir, 'gps0.parquet', f"""VehicleId={dvx['VehicleId']}""", '*'))
#     assert len(fn) == 1
#     dx = pq.ParquetFile(fn[0]).read().to_pandas()
#     ta, tb = dx['TS_SEC'].min(), dx['TS_SEC'].max()

#     # validate if data already collected is complete
#     if (ta == tmin) and (tb == tmax):
#         return pd.DataFrame(data={'VehicleId': [dvx['VehicleId']], 'ok': [True]})
#     return pd.DataFrame(data={'VehicleId': [dvx['VehicleId']], 'ok': [False]})
# dx = vx.groupby('id').applyInPandas(func, schema=schema).toPandas()
# # vx = vx.toPandas()
# # func(vx.loc[vx.index == 0])
# dx = pd.read_pickle(os.path.join(datadir, 'gps_already_collected.p'))
# dv = pd.merge(dv, dx, how='inner', on='VehicleId')
# for _, row in tqdm(dv.loc[dv['ok']].iterrows(), total=dv['ok'].sum()):
#     gps = os.path.join(datadir, 'gps0.parquet', f"""VehicleId={row['VehicleId']}""")
#     assert os.path.isdir(gps)
#     cmd = f"""cp -r {gps} {os.path.join(datadir, 'gps.parquet')}"""
#     os.system(cmd)
# dv = dv.loc[~dv['ok']].reset_index(drop=True)


# """
# Jeremy Corps stored procedure minimal query
# """

# import sqlalchemy as sa
# import pandas as pd

# edw = sa.create_engine('mssql+pyodbc://edw.drivecam.net/EDW?driver=ODBC+Driver+17+for+SQL+Server', isolation_level="AUTOCOMMIT").connect()
# query = f"""
#     EXEC Sandbox.aml.Sel_Triggers
#     @IdString = '1C00FFFF-2688-36E6-3005-4663F0800000,1C00FFFF-2688-35E6-6170-4663F0800000,1C00FFFF-2688-35E6-617F-4663F0800000,1C00FFFF-2688-33E6-97AD-5D43E7800000,1C00FFFF-2688-35E6-6194-4663F0800000,1C00FFFF-2688-35E6-61A3-4663F0800000,1C00FFFF-2688-33E6-A216-5D43E63C0000,1C00FFFF-2688-33E6-A3B6-5D43E63C0000,1C00FFFF-2688-33E6-218C-5D43E7800000,1C00FFFF-2688-33E6-2007-5D43E7800000',
#     @StartDate = '07-03-2021 00:00:00',
#     @EndDate = '10-01-2021 00:00:00',
#     @TimeWindowToProcessInMinutes = 10,
#     @RangeFrontPaddingDays = 7,
#     @RangeBackPaddingDays = 30,
#     @ResumeOperation=0"""
# dx = pd.read_sql_query(sa.text(query), edw)


# """
# evaluate Jeremy Corps stored procedure
# """

# import os
# import lytx
# import sqlalchemy as sa
# import numpy as np
# import pandas as pd
# from pyproj import Transformer
# from datetime import datetime
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from tqdm import tqdm
# from ipdb import set_trace


# # direct query
# fn = r'/mnt/home/russell.burdt/data/dx.p'
# if not os.path.isfile(fn):
#     edw = lytx.get_conn('edw')
#     query = f"""
#         SELECT
#             ERF.EventRecorderId,
#             ERF.EventRecorderFileId,
#             ERF.CreationDate,
#             ERF.FileName,
#             ERF.EventTriggerTypeId,
#             ERF.EDWUpdateDate AS EDW0,
#             ERFT.TriggerTime,
#             ERFT.Position.Lat AS Latitude,
#             ERFT.Position.Long as Longitude,
#             ERFT.ForwardExtremeAcceleration,
#             ERFT.SpeedAtTrigger,
#             ERFT.PostedSpeedLimit,
#             ERFT.EDWUpdateDate AS EDW1
#         FROM hs.EventRecorderFiles AS ERF
#             LEFT JOIN hs.EventRecorderFileTriggers AS ERFT
#             ON ERFT.EventRecorderFileId = ERF.EventRecorderFileId
#         WHERE ERF.EventRecorderId = '1C00FFFF-2688-35E6-79B2-4663F0800000'
#         AND ERFT.TriggerTime BETWEEN '2021-08-22 00:00:00' AND '2021-08-23 00:00:00'"""
#     dx = pd.read_sql_query(sa.text(query), edw)
#     dx.to_pickle(r'/mnt/home/russell.burdt/data/dx.p')
# else:
#     dx = pd.read_pickle(fn).sort_values('TriggerTime').reset_index(drop=True)

# # stored procedure for same data
# query = f"""
#     EXEC Sandbox.aml.Sel_Triggers
#     @IdString = '1C00FFFF-2688-35E6-79B2-4663F0800000',
#     @StartDate = '2021-08-22 00:00:00',
#     @EndDate = '2021-08-22 01:00:00',
#     @TimeWindowToProcessInMinutes = 240,
#     @RangeFrontPaddingDays = 0,
#     @RangeBackPaddingDays = 1"""
# edw = lytx.get_conn('edw')
# # dc = pd.read_sql_query(sa.text(query), edw)


# """
# evaluate Jeremy Corps data collection
# """

# import os
# import lytx
# import sqlalchemy as sa
# import numpy as np
# import pandas as pd
# from pyproj import Transformer
# from datetime import datetime
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from tqdm import tqdm
# from ipdb import set_trace


# # spark session
# conf = SparkConf()
# conf.set('spark.driver.memory', '64g')
# conf.set('spark.driver.maxResultSize', 0)
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.parquet.enableVectorizedReader', 'false')
# conf.set('spark.sql.session.timeZone', 'UTC')
# conf.set('spark.sql.shuffle.partitions', 20000)
# conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
# spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')

# # # read triggers data, scan over ers
# # triggers = spark.read.parquet(r'/mnt/home/russell.burdt/data/collision-model/v2/great-west/t0.parquet')
# # triggers.createOrReplaceTempView('triggers')
# # ers = spark.sql(f'SELECT DISTINCT(EventRecorderId) FROM triggers').toPandas().values.flatten()
# # edw = lytx.get_conn('edw')
# # dm = pd.DataFrame()
# # for er in tqdm(ers, desc='scanning ers'):

# #     # triggers data already collected for er, and on sandbox for same er
# #     df = spark.sql(f"""SELECT * FROM triggers WHERE EventRecorderId='{er}' ORDER BY eventdatetime""").toPandas()
# #     query = f"""SELECT * FROM Sandbox.aml.EventRecorderFileSample WHERE EventRecorderId='{er}' ORDER BY TriggerTime"""
# #     dx = pd.read_sql_query(sa.text(query), edw)

# #     # clean sandbox data
# #     dx['eventdatetime'] = dx.pop('TriggerTime')
# #     dx['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['eventdatetime']]
# #     lon = dx.pop('Longitude').values
# #     lat = dx.pop('Latitude').values
# #     transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
# #     dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
# #     dx['longitude_gps'] = lon
# #     dx['latitude_gps'] = lat

# #     # compare df and dx
# #     assert sorted(df.columns) == sorted(dx.columns)
# #     cok = sorted(df.columns)
# #     df, dx = df[cok], dx[cok]
# #     assert df.duplicated().sum() == 0
# #     dx = dx.loc[dx['TS_SEC'].isin(df['TS_SEC'])].reset_index(drop=True)
# #     dx = dx.loc[~dx.duplicated()].reset_index(drop=True)
# #     if df.shape != dx.shape:
# #         assert df.shape[0] > dx.shape[0]
# #         missing = df.loc[df['TS_SEC'].isin(np.array(list(set(df['TS_SEC'].values).difference(dx['TS_SEC'].values))))]
# #         dm = pd.concat((dm, missing))
# #     else:
# #         assert df.shape == dx.shape
# #         assert all(df == dx)
# # print(f"""sandbox location contains {100 - (100 * dm.shape[0] / triggers.count()):.4f}% of triggers rows""")

# # read dce-scores data, scan over vids
# dce = spark.read.parquet(r'/mnt/home/russell.burdt/data/collision-model/v2/nst/dce_scores.parquet')
# dce.createOrReplaceTempView('dce')
# vids = spark.sql(f'SELECT DISTINCT(VehicleId) FROM dce').toPandas().values.flatten()
# edw = lytx.get_conn('edw')
# dmm = pd.DataFrame()
# for vid in tqdm(vids, desc='scanning vids'):

#     # dce-scores data already collected for vid, and on sandbox for same vid
#     df = spark.sql(f"""SELECT * FROM dce WHERE VehicleId='{vid}' ORDER BY eventdatetime""").toPandas()
#     query = f"""SELECT * FROM Sandbox.aml.[dce-scores] WHERE VehicleId='{vid}' ORDER BY RecordDate"""
#     dx = pd.read_sql_query(sa.text(query), edw)

#     # clean sandbox data
#     dx['eventdatetime'] = dx.pop('RecordDate')
#     dx['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['eventdatetime']]
#     lon = dx.pop('Longitude').values
#     lat = dx.pop('Latitude').values
#     transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
#     dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
#     dx['longitude_gps'] = lon
#     dx['latitude_gps'] = lat

#     # compare df and dx
#     assert sorted(df.columns) == sorted(dx.columns)
#     cok = sorted(df.columns)
#     df, dx = df[cok], dx[cok]
#     assert df.duplicated().sum() == 0
#     dx = dx.loc[dx['TS_SEC'].isin(df['TS_SEC'])].reset_index(drop=True)
#     dx = dx.loc[~dx.duplicated()].reset_index(drop=True)
#     if df.shape != dx.shape:
#         assert df.shape[0] > dx.shape[0]
#         missing = df.loc[df['TS_SEC'].isin(np.array(list(set(df['TS_SEC'].values).difference(dx['TS_SEC'].values))))]
#         dmm = pd.concat((dmm, missing))
#     else:
#         assert df.shape == dx.shape
#         assert all(df == dx)
# print(f"""sandbox location contains {100 - (100 * dmm.shape[0] / dce.count()):.4f}% of dce-score rows""")




# """
# KS-statistic and p-value for all features from artifacts dir
# """
# import os
# import pickle
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from collections import defaultdict
# from scipy.stats import ks_2samp


# # load data
# adir = r'c:/Users/russell.burdt/Downloads/artifacts-08b'
# assert os.path.isdir(adir)
# yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
# dp = pd.read_pickle(os.path.join(adir, 'population-data.p'))
# dp = dp.loc[~dp['oversampled']]
# assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))
# df = pd.read_pickle(os.path.join(adir, 'ml-data.p'))
# with open(os.path.join(adir, 'feature-importance.p'), 'rb') as fid:
#     dfm = pickle.load(fid)
# features = dfm['features']
# assert (df.shape[0] == dp.shape[0]) and (df.shape[1] == features.size)

# # ks-statistic and p-value
# dks = defaultdict(list)
# for feature in tqdm(features, desc='features'):

#     # ks-test for feature
#     pos = df[dp['outcome'].astype('bool').values, features == feature].flatten()
#     neg = df[~dp['outcome'].astype('bool').values, features == feature].flatten()
#     ks = ks_2samp(pos, neg)
#     dks['feature'].append(feature)
#     dks['ks-statistic'].append(ks.statistic)
#     dks['p-value'].append(ks.pvalue)
# dks = pd.DataFrame(dks).sort_values('p-value').reset_index(drop=True)




# """
# pdf / cdf / vehicle-eval-segmentation for prediction probability and one arbitrary feature from artifacts dir
# """
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import metric_distribution, open_figure, format_axes, largefonts, save_pngs
# from scipy.stats import ks_2samp
# plt.style.use('bmh')


# # load data
# adir = r'c:/Users/russell.burdt/Downloads/artifacts-05'
# assert os.path.isdir(adir)
# yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
# dp = pd.read_pickle(os.path.join(adir, 'population-data.p'))
# dp = dp.loc[~dp['oversampled']]
# assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))
# df = pd.read_pickle(os.path.join(adir, 'ml-data.p'))
# with open(os.path.join(adir, 'feature-importance.p'), 'rb') as fid:
#     dfm = pickle.load(fid)
# features = dfm['features']
# assert (df.shape[0] == dp.shape[0]) and (df.shape[1] == features.size)

# # pdfs for prediction probability
# pos = yp.loc[dp['outcome'].astype('bool'), 'prediction probability'].values
# neg = yp.loc[~(dp['outcome'].astype('bool')), 'prediction probability'].values
# bins = np.linspace(0, 1, 60)
# title = f'pdf, prediction probability, {os.path.split(adir)[1]}'
# xlabel = 'collision-prediction model probability'
# kws = {'title': title, 'xlabel': xlabel, 'figsize': (16, 6), 'pdf': True, 'bins': bins, 'size': 18}
# metric_distribution(x=pos, legend=f'{pos.size} collision probabilities', **kws)
# metric_distribution(x=neg, legend=f'{neg.size} non-collision probabilities', **kws)

# # cdfs for prediction probability
# width = np.diff(bins)[0]
# centers = (bins[1:] + bins[:-1]) / 2
# posx = np.digitize(pos, bins)
# negx = np.digitize(neg, bins)
# posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
# negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
# assert (posx[-1] == 0) and (negx[-1] == 0)
# posx, negx = posx[:-1], negx[:-1]
# posx = posx / posx.sum()
# negx = negx / negx.sum()
# posx = np.cumsum(posx)
# negx = np.cumsum(negx)
# ks = ks_2samp(pos, neg)
# title = f'cdf, prediction probability, {os.path.split(adir)[1]}'
# fig, ax = open_figure(title, figsize=(16, 6))
# ax.plot(centers, posx, '-', lw=4, label=f'{pos.size} collision probabilities')
# ax.plot(centers, negx, '-', lw=4, label=f'{neg.size} non-collision probabilities')
# x = np.argmax(np.abs(posx - negx))
# ax.plot(np.tile(centers[x], 2), np.array([negx[x], posx[x]]), '-', lw=4, label='max distance', color='darkgreen')
# title += f'\nKS statistic {ks.statistic:.2f}, KS p-value {ks.pvalue:.2f}'
# format_axes('collision-prediction model probability', 'CDF', title, ax)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=3)
# largefonts(18)
# fig.tight_layout()

# # vehicle-eval segmentation curve
# title = f'vehicle-eval-segmentation curve, {os.path.split(adir)[1]}'
# fig, ax = open_figure(title, figsize=(14, 6))
# x = 100 * np.arange(1, yp.shape[0] + 1) / yp.shape[0]
# y = 100 * yp.sort_values('prediction probability', ascending=False)['actual outcome'].values.cumsum() / yp['actual outcome'].sum()
# ax.plot(x, y, '-', lw=4, label='prediction probability')
# format_axes('percentage of vehicle evals', 'cumulative percentage of collisions', title, ax)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=3)
# largefonts(18)
# fig.tight_layout()

# # pdfs for arbitrary metric
# feature = 'gps_miles'
# # feature = 'gpse_travel_distance_meters_sum'
# assert feature in features
# pos = df[dp['outcome'].astype('bool').values, features == feature].flatten()
# neg = df[~dp['outcome'].astype('bool').values, features == feature].flatten()
# assert np.all(pos >= 0) and np.all(neg >= 0)
# bins = np.linspace(0, 1.1 * df[:, features == feature].flatten().max(), 80)
# title = f'pdf, {feature}, {os.path.split(adir)[1]}'
# xlabel = feature
# kws = {'title': title, 'xlabel': xlabel, 'figsize': (16, 6), 'pdf': True, 'bins': bins, 'size': 18}
# metric_distribution(x=pos, legend=f'{pos.size} collision probabilities', **kws)
# metric_distribution(x=neg, legend=f'{neg.size} non-collision probabilities', **kws)

# # cdfs for arbitrary metric
# width = np.diff(bins)[0]
# centers = (bins[1:] + bins[:-1]) / 2
# posx = np.digitize(pos, bins)
# negx = np.digitize(neg, bins)
# posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
# negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
# assert (posx[-1] == 0) and (negx[-1] == 0)
# posx, negx = posx[:-1], negx[:-1]
# posx = posx / posx.sum()
# negx = negx / negx.sum()
# posx = np.cumsum(posx)
# negx = np.cumsum(negx)
# ks = ks_2samp(pos, neg)
# title = f'cdf, {feature}, {os.path.split(adir)[1]}'
# fig, ax = open_figure(title, figsize=(16, 6))
# ax.plot(centers, posx, '-', lw=4, label=f'{pos.size} collision probabilities')
# ax.plot(centers, negx, '-', lw=4, label=f'{neg.size} non-collision probabilities')
# x = np.argmax(np.abs(posx - negx))
# ax.plot(np.tile(centers[x], 2), np.array([negx[x], posx[x]]), '-', lw=4, label='max distance', color='darkgreen')
# title += f'\nKS statistic {ks.statistic:.2f}, KS p-value {ks.pvalue:.2f}'
# format_axes(feature, 'CDF', title, ax)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=3)
# largefonts(18)
# fig.tight_layout()

# # update vehicle-eval segmentation curve
# title = f'vehicle-eval-segmentation curve, {os.path.split(adir)[1]}'
# fig, ax = open_figure(title, figsize=(14, 6))
# x = 100 * np.arange(1, yp.shape[0] + 1) / yp.shape[0]
# y = 100 * yp.loc[np.argsort(df[:, features == feature].flatten())[::-1], 'actual outcome'].values.cumsum() / yp['actual outcome'].sum()
# ax.plot(x, y, '-', lw=4, label=feature)
# format_axes('percentage of vehicle evals', 'cumulative percentage of collisions', title, ax)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=3)
# largefonts(18)
# fig.tight_layout()




# """
# EDW connection troubleshooting
# """

# import pandas as pd
# from sqlalchemy import create_engine, text

# cstr = 'mssql+pyodbc://edw.drivecam.net/EDW?driver=ODBC+Driver+17+for+SQL+Server'
# conn = create_engine(cstr).connect()

# sql = f'SELECT TOP 10 * FROM flat.Events'
# dx = pd.read_sql_query(text(sql), conn)


# """
# compare gps-enriched metrics between osm220718 and osm221107
# """
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import metric_distribution
# plt.style.use('bmh')

# # load data
# with open(r'c:/Users/russell.burdt/Downloads/common.p', 'rb') as fid:
#     dc = pickle.load(fid)
# da = pd.DataFrame(dc['osm220718-2023-2-v2-dft'])
# db = pd.DataFrame(dc['osm221107-2023-3-v2-dft'])
# assert all(da.columns == db.columns)
# cxs = da.columns.to_numpy()

# # plot data
# for cx in cxs:
#     xmin = min(np.nanmin(da[cx]), np.nanmin(db[cx]))
#     xmax = max(np.nanmax(da[cx]), np.nanmax(db[cx]))
#     bins = np.linspace(xmin, xmax, 100)
#     title = f'distribution of {cx}'
#     kws = {'bins': bins, 'title': title, 'xlabel': cx, 'logscale': True}
#     metric_distribution(x=da[cx].values, legend='osm220718', **kws)
#     metric_distribution(x=db[cx].values, legend='osm221107', **kws)
# plt.show()




# """
# build data to compare gps-enriched metrics between osm220718 and osm221107
# """
# import os
# import boto3
# import pickle
# import numpy as np
# import pandas as pd

# # load enriched gps metrics
# datadir = r'/mnt/home/russell.burdt/data'
# s3a, s3b = 'osm220718-2023-2-v2-dft', 'osm221107-2023-3-v2-dft'
# boto3.client('s3').download_file(Bucket='russell-s3', Key=f'{s3a}/enriched_gps_metrics.p', Filename=os.path.join(datadir, 'da.p'))
# boto3.client('s3').download_file(Bucket='russell-s3', Key=f'{s3b}/enriched_gps_metrics.p', Filename=os.path.join(datadir, 'db.p'))
# da, db = pd.read_pickle(os.path.join(datadir, 'da.p')), pd.read_pickle(os.path.join(datadir, 'db.p'))
# os.remove(os.path.join(datadir, 'da.p'))
# os.remove(os.path.join(datadir, 'db.p'))
# assert len(set(da.columns).intersection(db.columns)) == da.shape[1]
# ca, cb = ~np.all(np.isnan(da), axis=0), ~np.all(np.isnan(db), axis=0)
# ca, cb = ca[ca].index, cb[cb].index
# cok = np.array(list(set(ca).intersection(cb)))
# da, db = da[cok], db[cok]

# # feature data from cross-validated models
# with open(r'/mnt/home/russell.burdt/data/collision-model/app/artifacts-07/feature-importance.p', 'rb') as fid:
#     dfa = pickle.load(fid)
# with open(r'/mnt/home/russell.burdt/data/collision-model/app/artifacts-07b/feature-importance.p', 'rb') as fid:
#     dfb = pickle.load(fid)
# assert len(set(dfa['features']).intersection(dfb['features'])) == dfa['features'].size
# features = np.array(list(set(da.columns).intersection(dfa['features'])))

# # common metrics in top n 'most different' features
# n = 10
# mx = np.array([(da[x].mean() - db[x].mean()) / da[x].mean() for x in features])
# common = features[np.argsort(mx)][-n:]

# # # common metrics in top n features
# # n = 10
# # fa = dfa['features'][np.argsort(dfa['model feature importance'])][::-1]
# # fb = dfb['features'][np.argsort(dfb['model feature importance'])][::-1]
# # common = np.array(list(set(fa[:n]).intersection(fb[:n])))

# # dictionary of common metrics
# dc = {}
# dc['osm220718-2023-2-v2-dft'] = {}
# dc['osm221107-2023-3-v2-dft'] = {}
# for x in common:
#     dc['osm220718-2023-2-v2-dft'][x] = da[x].values
#     dc['osm221107-2023-3-v2-dft'][x] = db[x].values
# with open(r'/mnt/home/russell.burdt/data/common.p', 'wb') as fid:
#     pickle.dump(dc, fid)




# """
# daily evaluation results
# """
# import os
# import utils
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# from tqdm import tqdm
# plt.style.use('bmh')

# # load data
# datadir = r'c:/Users/russell.burdt/Downloads/artifacts-12'
# dcm = pd.read_pickle(os.path.join(datadir, 'population-data.p'))
# dp = pd.read_pickle(os.path.join(datadir, 'collisions.p'))
# yp = pd.read_pickle(os.path.join(datadir, 'model-prediction-probabilities.p'))

# # vehicles with any collision
# vids = pd.unique(dcm.loc[dcm['collision-47'], 'VehicleId'])
# for vid in tqdm(vids, desc='scanning vids'):

#     # predictions for vid
#     dv = dcm.loc[dcm['VehicleId'] == vid]
#     yv = yp.loc[dv.index]
#     assert all(yv['actual outcome'] == dv['collision-47'])

#     # collisions for vid
#     idx = np.unique(np.hstack((dv['collision-47-idx'].values)))
#     dts = dp.loc[idx, 'RecordDate'].values
#     assert dts.size > 0

#     # chart of prediction probability and collision markers
#     title = f'daily prediction probability and collisions for {vid}'
#     fig, ax = open_figure(title, figsize=(14, 6))
#     ax.plot(dv['time1'].values, yv['prediction probability'].values, 'x-', ms=8, lw=3, label='prediction probability')
#     ax.set_ylim(0, 0.7)
#     for x, dt in enumerate(dts):
#         p = ax.plot(np.tile(dt, 2), ax.get_ylim(), '--', color='black', lw=4)[0]
#         if x == 0:
#             p.set_label('collision-47 event')
#     format_axes('', 'prediction probability', title.replace('for ', 'for\nVehicleId '), ax, apply_concise_date_formatter=True)
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, numpoints=3, handlelength=4)
#     largefonts(18)
#     fig.tight_layout()
#     save_pngs(r'c:/Users/russell.burdt/Downloads')


# """
# daily evaluation DataFrame
# """

# import os
# import utils
# import pandas as pd
# import numpy as np


# # model datadir
# datadir = r'/mnt/home/russell.burdt/data/collision-model/v2/nst'
# assert os.path.isdir(datadir)

# # metadata and collision prediction model DataFrame
# dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
# dv = pd.read_pickle(os.path.join(datadir, 'metadata', 'vehicle_metadata.p'))
# dp = pd.read_pickle(os.path.join(datadir, 'metadata', 'positive_instances.p'))
# de = pd.read_pickle(os.path.join(datadir, 'metadata', 'event_recorder_associations.p'))
# dcm0 = pd.read_pickle(os.path.join(datadir, 'dcm.p'))
# utils.validate_dcm_dp(dcm0, dp)
# dm0 = utils.get_population_metadata(dcm0, dc, datadir=None)

# # daily collision prediction model population DataFrame
# dcm = utils.model_population(dc, dv, dp, de, daily=True, oversampled=False)
# dm = utils.get_population_metadata(dcm, dc, datadir=None)
# dcm.to_pickle(os.path.join(datadir, 'dcm-daily.p'))





# """
# model metadata and eval metrics by industry from artifacts dir
# """

# import os
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.metrics import average_precision_score, roc_auc_score

# # data from model artifacts
# adir = r'/mnt/home/russell.burdt/data/collision-model/app/artifacts-04'
# assert os.path.isdir(adir)
# dp = pd.read_pickle(os.path.join(adir, 'population-data.p'))
# df = pd.read_pickle(os.path.join(adir, 'ml-data.p'))
# yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
# with open(os.path.join(adir, 'feature-importance.p'), 'rb') as fid:
#     dfm = pickle.load(fid)
# if 'gps_miles' not in dfm['features']:
#     ok = np.where(dfm['features'] == 'gpse_travel_distance_meters_sum')[0][0]
#     dfm['features'][ok] = 'gps_miles'
#     df[:, ok] = 0.000621371 * df[:, ok]
# assert dp.shape[0] == df.shape[0] == yp.shape[0]
# assert df.shape[1] == dfm['features'].size
# df = pd.DataFrame(data=df, columns=dfm['features'])
# dx = pd.merge(dp[['VehicleId', 'time0', 'time1', 'IndustryDesc', 'outcome']], df[['gps_miles']], left_index=True, right_index=True, how='inner')

# # industries sorted by num of vehicle evals in descending order
# xs = sorted(pd.unique(pd.unique(dp['IndustryDesc'])), key=lambda x: dp.loc[dp['IndustryDesc'] == x].shape[0])[::-1]

# # DataFrame of dataset metadata by industry
# dm = pd.DataFrame({
#     'number of 90-day vehicle evaluations': [dx.loc[dx['IndustryDesc'] == x].shape[0] for x in xs],
#     'mean miles per vehicle eval': [dx.loc[(dx['IndustryDesc'] == x) & (dx['gps_miles'] > 0), 'gps_miles'].mean() for x in xs],
#     'stdev miles per vehicle eval': [dx.loc[(dx['IndustryDesc'] == x) & (dx['gps_miles'] > 0), 'gps_miles'].std() for x in xs],
#     'num of positive outcomes': [dx.loc[dx['IndustryDesc'] == x, 'outcome'].sum() for x in xs],
#     'percentage of positive outcomes': [100 * dx.loc[dx['IndustryDesc'] == x, 'outcome'].sum() / dx.loc[dx['IndustryDesc'] == x].shape[0] for x in xs]}).T
# dm.columns = xs
# dm['all'] = dm.values.sum(axis=1)
# dm.loc[dm.index == 'mean miles per vehicle eval', 'all'] = dx.loc[dx['gps_miles'] > 0, 'gps_miles'].mean()
# dm.loc[dm.index == 'stdev miles per vehicle eval', 'all'] = dx.loc[dx['gps_miles'] > 0, 'gps_miles'].std()
# dm.loc[dm.index == 'percentage of positive outcomes', 'all'] = 100 * dx['outcome'].sum() / dx.shape[0]

# # AUC and AP by industry
# auc = [roc_auc_score(
#     y_true=yp.loc[dx['IndustryDesc'] == x, 'actual outcome'].values,
#     y_score=yp.loc[dx['IndustryDesc'] == x, 'prediction probability'].values)
#     if np.unique(yp.loc[dx['IndustryDesc'] == x, 'actual outcome'].values).size > 1 else np.nan for x in xs]
# auc.append(roc_auc_score(y_true=yp['actual outcome'].values, y_score=yp['prediction probability'].values))
# ap = [average_precision_score(
#     y_true=yp.loc[dx['IndustryDesc'] == x, 'actual outcome'].values,
#     y_score=yp.loc[dx['IndustryDesc'] == x, 'prediction probability'].values)
#     if np.unique(yp.loc[dx['IndustryDesc'] == x, 'actual outcome'].values).size > 1 else np.nan for x in xs]
# ap.append(average_precision_score(y_true=yp['actual outcome'].values, y_score=yp['prediction probability'].values))
# dm = pd.concat((dm, pd.DataFrame(data={'AUC': auc}, index=xs + ['all']).T))
# dm = pd.concat((dm, pd.DataFrame(data={'AP': ap}, index=xs + ['all']).T))

# # format dm
# for col in ['number of 90-day vehicle evaluations', 'num of positive outcomes']:
#     dm.loc[dm.index == col] = [f'{x:.0f}' for x in dm.loc[dm.index == col].values.flat]
# for col in ['mean miles per vehicle eval', 'stdev miles per vehicle eval']:
#     dm.loc[dm.index == col] = [f'{x:.1f}' for x in dm.loc[dm.index == col].values.flat]
# for col in ['percentage of positive outcomes', 'AUC', 'AP']:
#     dm.loc[dm.index == col] = [f'{x:.3f}' for x in dm.loc[dm.index == col].values.flat]



# """
# ROC vs volume of test data
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from ipdb import set_trace
# from tqdm import tqdm
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# from sklearn.metrics import average_precision_score, roc_auc_score
# plt.style.use('bmh')


# # load data
# datadir = r'c:/Users/russell.burdt/Downloads/artifacts-01'
# assert os.path.isdir(datadir)
# dcm = pd.read_pickle(os.path.join(datadir, 'population-data.p'))
# yp = pd.read_pickle(os.path.join(datadir, 'model-prediction-probabilities.p'))
# assert dcm.shape[0] == yp.shape[0]

# # number of vehicle evaluations as array
# xs = np.linspace(1, yp.shape[0], 200).astype('int')
# ok = np.random.choice(yp.index, size=yp.shape[0], replace=False)
# auc, nc = [], []
# for x in tqdm(xs, desc='vehicle-evals'):
#     xk = ok[:x]
#     nc.append(dcm.loc[xk, 'outcome'].sum())
#     y_true = yp.loc[xk, 'actual outcome'].values
#     assert y_true.sum() == nc[-1]
#     y_score = yp.loc[xk, 'prediction probability'].values
#     if np.unique(y_true).size > 1:
#         auc.append(roc_auc_score(y_true=y_true, y_score=y_score))
#     else:
#         auc.append(np.nan)
# nc = np.array(nc)
# pc = 100 * nc / xs
# auc = np.array(auc)

# # results
# fig, ax = open_figure('model evaluation vs num of vehicle evals in test set', 3, 1, figsize=(12, 8))
# ax[0].plot(xs, nc, 'x-', ms=6, lw=3)
# format_axes('num of vehicle evals in test set', 'num', 'num of collisions in test set', ax[0])
# ax[1].plot(xs, pc, 'x-', ms=6, lw=3)
# format_axes('num of vehicle evals in test set', '%', 'percentage of vehicle evals in test set associated with any collision', ax[1])
# ax[2].plot(xs, auc, 'x-', ms=6, lw=3)
# format_axes('num of vehicle evals in test set', 'score', 'AUC test set', ax[2])
# largefonts(18)
# fig.tight_layout()
# plt.show()



# """
# visualize RB metric vs DC metric, for aggegated enriched GPS metrics
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# plt.style.use('bmh')

# dxx = pd.read_pickle(r'c:\Users\russell.burdt\Downloads\dxx.p')
# cols = [x[5:] for x in dxx.columns if x[:5] == 'gpse_']
# for col in cols:
#     x = dxx[f'DC_{col}'].values
#     y = dxx[f'gpse_{col}'].values
#     fig, ax = open_figure(col, figsize=(10, 8))
#     ax.plot(x, y, 'x', ms=8, mew=3)
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     l1 = min((xlim[0], ylim[0]))
#     l2 = max((xlim[1], ylim[1]))
#     ax.set_xlim(l1, l2)
#     ax.set_ylim(l1, l2)
#     format_axes(f'DC_{col}', f'RB_{col}', col, ax)
#     largefonts(16)
#     fig.tight_layout()
#     save_pngs(r'c:\Users\russell.burdt\Downloads')
# plt.show()


# """
# compare dft and dft1b
# """

# import numpy as np
# import pandas as pd

# # read data
# cols = ['VehicleId', 'time0', 'time1']
# dcm0 = pd.read_pickle('/mnt/home/russell.burdt/data/collision-model/v1/dft/dcm.p')[cols]
# df0 = pd.read_pickle('/mnt/home/russell.burdt/data/collision-model/v1/dft/df.p')
# dcm1 = pd.read_pickle('/mnt/home/russell.burdt/data/collision-model/v1/dft1b/dcm.p')[cols]
# df1 = pd.read_pickle('/mnt/home/russell.burdt/data/collision-model/v1/dft1b/df.p')
# # same rows
# x = pd.merge(dcm0.reset_index(drop=False), dcm1, on=cols, how='inner')['index'].values
# assert x.size == dcm1.shape[0]
# dcm0 = dcm0.loc[x].reset_index(drop=True)
# df0 = df0.loc[x].reset_index(drop=True)
# # same columns
# assert all(dcm0.columns == dcm1.columns)
# cols = set(df0.columns).intersection(df1.columns)
# print(f""" in df0, not in df1, {', '.join(list(set(df0.columns).difference(cols)))}""")
# print(f""" in df1, not in df0, {', '.join(list(set(df1.columns).difference(cols)))}""")
# cols = np.array(list(cols))
# df0 = df0[cols]
# df1 = df1[cols]
# assert df0.shape == df1.shape
# # check same values
# same = np.isclose(df0.values, df1.values, equal_nan=True)
# sp = same.sum() / df0.size
# print(f"""{100 * sp:.2f}% of values are same""")
# # percentage of different values by column name
# dcs = cols[~np.all(same, axis=0)]
# for dc in dcs:
#     cs = np.isclose(df0[dc], df1[dc], equal_nan=True)
#     print(f"""- same values for {100 * cs.sum() / df0.shape[0]:.2f}% of {dc}""")

# """
# follow up with AmTrust
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts
# from collections import defaultdict
# plt.style.use('bmh')

# # read and convert data from misc-convert-artifacts
# df = pd.read_pickle(r'/mnt/home/russell.burdt/data/artifacts-12.p')
# df = df.loc[df['CompanyName'] == 'Beacon Transport LLC'].reset_index(drop=True)
# mpp = df.groupby('VehicleId')['prediction probability'].mean().reset_index(drop=False)
# mpp['avg prediction probability'] = mpp.pop('prediction probability')
# sc = df.groupby('VehicleId')['collision'].sum().reset_index(drop=False)
# sc['any collision'] = sc.pop('collision').astype('bool')
# dv = pd.merge(mpp, sc, on='VehicleId', how='inner').sort_values('avg prediction probability').reset_index(drop=True)

# # AmTrust window boundaries
# table = defaultdict(list)
# xt = (0.249, 0.384, 0.443, 0.491, 0.585)
# for x in xt:
#     if xt.index(x) != len(xt) - 1:
#         xa = xt[xt.index(x)]
#         xb = xt[xt.index(x) + 1]
#         table['avg prediction by vehicle'].append(f'({xa}, {xb}]')
#         table['num vehicles'].append(((dv['avg prediction probability'] > xa) & (dv['avg prediction probability'] <= xb)).sum())
#         table['num collisions'].append(dv.loc[(dv['avg prediction probability'] > xa) & (dv['avg prediction probability'] <= xb), 'any collision'].sum())
# table = pd.DataFrame(table)




# # prediction probability distribution
# x = dv['avg prediction probability'].values
# bins = np.linspace(0.249, 0.585, 20)
# title = 'AmTrust collision prediction model results'
# xlabel = 'average prediction probability by vehicle-id'
# legend = 'distribution of model\navg prediction probability'
# size = 18
# fig, ax = metric_distribution(x=x, bins=bins, title=title, xlabel=xlabel, ylabel='count', legend=legend, figsize=(14, 6), size=size)
# label = 'cumulative count of\nany collision in Lytx data'
# ax.plot(dv['avg prediction probability'].values, np.cumsum(dv['any collision'].values), '-', lw=3, label=label)

# # AmTrust window boundaries
# ylim = ax.get_ylim()
# xt = (0.249, 0.384, 0.443, 0.491, 0.585)
# nv = []
# for x in xt:
#     lx = ax.plot(np.tile(x, 2), ylim, '-', color='black', lw=3)[0]
#     if xt.index(x) != len(xt) - 1:
#         xa = xt[xt.index(x)]
#         xb = xt[xt.index(x) + 1]
#         nv.append(((dv['avg prediction probability'] > xa) & (dv['avg prediction probability'] <= xb)).sum())
# lx.set_label(f'AmTrust window boundaries\nappx {int(np.mean(nv))} vehicles per window')

# # clean up
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# ax.set_xlim(xt[0] - 0.002, xt[-1] + 0.002)
# largefonts(size)
# fig.tight_layout()
# plt.show()

# for xa, xb in xs:
#     dx = dv.loc[(dv['avg prediction probability'] > xa) & (dv['avg prediction probability'] <= xb)].reset_index(drop=True)





# """
# matplotlib chart of collision and predictor intervals based on dcm.p
# (similar code as in business-case.py)
# """
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pyrb.processing import int_to_ordinal
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# plt.style.use('bmh')
# dcm = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/dcm.p')
# # dcm['oversampled'] = False
# # dcm['time2'] = dcm['time1'] + pd.Timedelta(days=30)
# dts = dcm.loc[~dcm['oversampled'], ['time0', 'time1', 'time2']]
# # dts = dcm[['time0', 'time1', 'time2']]
# dts = dts.loc[~dts.duplicated()].sort_values('time1').reset_index(drop=True)
# dts.index = range(1, dts.shape[0] + 1)
# predictor_days = pd.unique(dts['time1'] - dts['time0'])
# assert predictor_days.size == 1
# predictor_days = int(predictor_days[0].astype('float') * (1e-9) / (60 * 60 * 24))
# fig, ax = open_figure(f'collision-prediction model definition', figsize=(12, 4))
# # fig, ax = open_figure(f'collision-prediction model definition', figsize=(11, 5))
# for x, (time0, time1, time2) in dts.iterrows():
#     p0 = ax.fill_between(x=np.array([time0, time1]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='darkblue', alpha=0.8)
#     p1 = ax.fill_between(x=np.array([time1, time2]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='orange', alpha=0.2)
#     if x == 1:
#         p0.set_label(f'predictor interval, {predictor_days} days')
#         p1.set_label(f'collision interval')
# # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, shadow=True, fancybox=True)
# ax.legend(loc='upper left', numpoints=3, shadow=True, fancybox=True)
# format_axes('', '', 'predictor and collision intervals in collision-prediction model', ax, apply_concise_date_formatter=True)
# ax.set_yticks(dts.index)
# ax.set_yticklabels([f'{int_to_ordinal(x)} interval' for x in dts.index])
# ax.set_xlim(dts.iloc[0]['time0'] - pd.Timedelta(days=3), dts.iloc[-1]['time2'] + pd.Timedelta(days=3))
# largefonts(18)
# fig.tight_layout()
# plt.show()



# """
# company-aggregated results
# """

# import os
# import numpy as np
# import pandas as pd

# adir = r'/mnt/home/russell.burdt/data/collision-model/app/artifacts-14'
# assert os.path.isdir(adir)
# yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
# dp = pd.read_pickle(os.path.join(adir, 'population-data.p'))
# assert all(yp['actual outcome'] == dp['outcome'])
# del yp['actual outcome']
# df = pd.concat((dp, yp), axis=1)
# df['outcome'] = df['outcome'].astype('bool')

# for company in pd.unique(df['CompanyName']):
#     c0 = df['CompanyName'] == company
#     c1 = df['outcome']
#     c2 = ~c1
#     print(f"""company, {company}""")
#     print(f"""  num of 90-day vehicle evaluations, {c0.sum()}""")
#     print(f"""  mean of {c0.sum()} prediction probabilities, {df.loc[c0, 'prediction probability'].mean():.2f}""")
#     print(f"""  stdev of {c0.sum()} prediction probabilities, {df.loc[c0, 'prediction probability'].std():.2f}\n""")
#     print(f"""  num of positive prediction probabilities, {(c0 & c1).sum()}""")
#     print(f"""  mean of {(c0 & c1).sum()} positive prediction probabilities, {df.loc[c0 & c1, 'prediction probability'].mean():.2f}""")
#     print(f"""  stdev of {(c0 & c1).sum()} positive prediction probabilities, {df.loc[c0 & c1, 'prediction probability'].std():.2f}\n""")
#     print(f"""  num of negative prediction probabilities, {(c0 & c2).sum()}""")
#     print(f"""  mean of {(c0 & c2).sum()} negative prediction probabilities, {df.loc[c0 & c2, 'prediction probability'].mean():.2f}""")
#     print(f"""  stdev of {(c0 & c2).sum()} negative prediction probabilities, {df.loc[c0 & c2, 'prediction probability'].std():.2f}\n""")

# """
# distributions of Lytx Rating Factor
# """
# import pandas as pd
# import matplotlib.pyplot as plt
# from utils import metric_distribution
# from pyrb.mpl import save_pngs

# fn, log_scale = r'c:/Users/russell.burdt/Downloads/artifacts-14.p', True
# df = pd.read_pickle(fn)
# collision = df['collision'].values
# x0 = df['prediction probability'].values
# x1 = df['rating factor, %'].values
# metric_distribution(
#     figsize=(12, 4),
#     title=f'distribution of prediction probablity\nfor {collision.size} vehicle evaluations',
#     x=x0[~collision], xr=(0, 1), xs=0.01, xlabel='collision prediction model probability',
#     legend=f'{(~collision).sum()} vehicle evals with no collision', size=16)
# metric_distribution(
#     figsize=(12, 4),
#     title=f'distribution of prediction probablity\nfor {collision.size} vehicle evaluations',
#     x=x0[collision], xr=(0, 1), xs=0.01, xlabel='collision prediction model probability',
#     legend=f'{collision.sum()} vehicle evals with any collision', size=16)
# metric_distribution(
#     figsize=(12, 4),
#     title=f'distribution of Lytx Rating Factor\nfor {collision.size} vehicle evaluations',
#     x=x1[~collision], xr=(0, 100), xs=1, xlabel='Lytx Rating Factor, %',
#     legend=f'{(~collision).sum()} vehicle evals with no collision', size=16)
# metric_distribution(
#     figsize=(12, 4),
#     title=f'distribution of Lytx Rating Factor\nfor {collision.size} vehicle evaluations',
#     x=x1[collision], xr=(0, 100), xs=1, xlabel='Lytx Rating Factor, %',
#     legend=f'{collision.sum()} vehicle evals with any collision', size=16)

# plt.show()


# # fictional business-case, monthly insurance premium by industry
# BC = {}
# BC['nominal monthly premium'] = {}
# BC['nominal monthly premium']['Distribution'] = 200
# BC['nominal monthly premium']['Freight/Trucking'] = 1000

# # fictional business-case, collision payout by severity and industry
# BC['collision payout'] = {}
# BC['collision payout']
# BC['collision payout']['1-Distribution'] = 1e5
# BC['collision payout']['234-Distribution'] = 3e4
# BC['collision payout']['1-Freight/Trucking'] = 1e6
# BC['collision payout']['234-Freight/Trucking'] = 7e4



# """
# events/triggers troubleshooting
# """

# import os
# import numpy as np
# import pandas as pd
# from pyspark import SparkConf
# from pyspark.sql import SparkSession

# # SparkSession object
# conf = SparkConf()
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.session.timeZone', 'UTC')
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')

# # data objects
# datadir = r'/mnt/home/russell.burdt/data/collision-model/v2/dft'
# assert os.path.isdir(datadir)
# dcm = pd.read_pickle(os.path.join(datadir, 'dcm.p'))
# dcm = dcm.loc[dcm['CompanyName'] == 'Synctruck LLC'].reset_index(drop=True)

# # get single event and associated trigger
# spark.read.parquet(os.path.join(datadir, 'events.parquet')).createOrReplaceTempView('events')
# spark.read.parquet(os.path.join(datadir, 'triggers.parquet')).createOrReplaceTempView('triggers')
# while True:
#     vid = np.random.choice(dcm['VehicleId'].values)
#     event = spark.sql(f"""SELECT * FROM events WHERE VehicleId='{vid}' LIMIT 1""").toPandas().squeeze()
#     if event.size == 0:
#         continue
#     trigger = spark.sql(f"""SELECT * FROM triggers WHERE EventRecorderFileId='{event['EventRecorderFileId']}'""").toPandas().squeeze()
#     if trigger.size == 0:
#         continue
# assert (event['EventRecorderId'] == trigger['EventRecorderId']) and (event['NameId'] == trigger['EventTriggerTypeId'])


# """
# identify common contributors to diffs in prediction probability for Updike Logistics vs all companies
# """

# import os
# import pickle
# import pandas as pd
# import numpy as np
# from scipy.stats import ks_2samp

# # load and validate data for all companies and for Updike Logistics
# datadir = r'/mnt/home/russell.burdt/data/collision-model/archived/v2_models/artifacts-09'
# assert os.path.isdir(datadir)
# dcm = pd.read_pickle(os.path.join(datadir, 'population-data.p'))
# dcm1 = dcm.loc[dcm['CompanyName'] == 'Updike Distribution Logistics']
# yp = pd.read_pickle(os.path.join(datadir, 'model-prediction-probabilities.p'))
# pos = yp.loc[yp['actual outcome'] == 1, 'prediction probability'].values
# neg = yp.loc[yp['actual outcome'] == 0, 'prediction probability'].values
# yp1 = yp.loc[yp.index.isin(dcm1.index)]
# pos1 = yp1.loc[yp['actual outcome'] == 1, 'prediction probability'].values
# neg1 = yp1.loc[yp['actual outcome'] == 0, 'prediction probability'].values
# assert all(dcm['outcome'] == yp['actual outcome'])
# assert all(dcm1['outcome'] == yp1['actual outcome'])
# with open(os.path.join(datadir, 'feature-importance.p'), 'rb') as fid:
#     dfm = pickle.load(fid)
# features = dfm['features']
# with open(os.path.join(datadir, 'shap.p'), 'rb') as fid:
#     ds = pickle.load(fid)
# base = ds['base']
# values = ds['values']
# base1 = ds['base'][dcm1.index]
# values1 = ds['values'][dcm1.index, :]
# with open(os.path.join(datadir, 'ml-data.p'), 'rb') as fid:
#     df = pickle.load(fid)
# df1 = df[dcm1.index, :]
# assert values.shape[1] == values1.shape[1] == df.shape[1] == df1.shape[1] == features.size
# assert base.shape[0] == values.shape[0] == df.shape[0] == dcm.shape[0]
# assert base1.shape[0] == values1.shape[0] == df1.shape[0] == dcm1.shape[0]
# assert all(np.isclose(base + values.sum(axis=1), yp['prediction probability'].values))
# assert all(np.isclose(base1 + values1.sum(axis=1), yp1['prediction probability'].values))

# # run KS test for each feature nominal value and shap value
# ks = np.full((features.size, 2), np.nan)
# shap = np.full((features.size, 2), np.nan)
# for x, feature in enumerate(features):
#     test = ks_2samp(df[:, x], df1[:, x])
#     ks[x, :] = np.array([test.statistic, test.pvalue])
#     test = ks_2samp(values[:, x], values1[:, x])
#     shap[x, :] = np.array([test.statistic, test.pvalue])

# # run KS test for each company neg vs pos prediction probabilities
# companies = pd.unique(dcm['CompanyName'])
# ks = np.full((companies.size, 2), np.nan)
# for x, company in enumerate(companies):
#     neg = yp.loc[(dcm['CompanyName'] == company) & (yp['actual outcome'] == 0), 'prediction probability'].values
#     pos = yp.loc[(dcm['CompanyName'] == company) & (yp['actual outcome'] == 1), 'prediction probability'].values
#     if (neg.size == 0) or (pos.size == 0):
#         continue
#     test = ks_2samp(neg, pos)
#     ks[x, :] = np.array([test.statistic, test.pvalue])


# """
# identify same rows in v1 and v2 populations
# """
# import os
# import utils
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# fmt = r'%d %b %Y'

# # v1 data
# d1 = r'/mnt/home/russell.burdt/data/collision-model/v1/nst'
# assert os.path.isdir(d1)
# dc1 = pd.read_pickle(os.path.join(d1, 'metadata', 'model_params.p'))
# dp1 = pd.read_pickle(os.path.join(d1, 'metadata', 'positive_instances.p'))
# dcm1 = pd.read_pickle(os.path.join(d1, 'dcm.p'))
# df1 = pd.read_pickle(os.path.join(d1, 'df.p'))
# assert len(set(dcm1.columns).intersection(df1.columns)) == 0
# assert dcm1.shape[0] == df1.shape[0]
# dcm1, dp1 = utils.convert_v1_to_v2(dcm1, dp1, dc1)
# dm1 = utils.get_population_metadata(dcm1, dc1)
# window = np.array(
#     [f"""{a.strftime(fmt)}, {b.strftime(fmt)}, {c.strftime(fmt)}""" for a, b, c in zip(dcm1['time0'], dcm1['time1'], dcm1['time2'])])
# dcm1['window'] = None
# # dft
# # dcm1.loc[window == '01 Aug 2021, 30 Oct 2021, 29 Nov 2021', 'window'] = 'w1'
# # dcm1.loc[window == '01 Sep 2021, 30 Nov 2021, 30 Dec 2021', 'window'] = 'w2'
# # nst
# dcm1.loc[window == '03 Jul 2021, 01 Oct 2021, 31 Oct 2021', 'window'] = 'w1'
# dcm1.loc[window == '03 Aug 2021, 01 Nov 2021, 01 Dec 2021', 'window'] = 'w2'
# dcm1.loc[window == '02 Sep 2021, 01 Dec 2021, 31 Dec 2021', 'window'] = 'w3'
# dcm1.loc[window == '03 Oct 2021, 01 Jan 2022, 31 Jan 2022', 'window'] = 'w4'
# dcm1.loc[window == '03 Nov 2021, 01 Feb 2022, 03 Mar 2022', 'window'] = 'w5'
# assert np.all(~pd.isnull(dcm1['window']))

# # v2 data
# d2 = r'/mnt/home/russell.burdt/data/collision-model/v2/nst'
# assert os.path.isdir(d2)
# dc2 = pd.read_pickle(os.path.join(d2, 'metadata', 'model_params.p'))
# dp2 = pd.read_pickle(os.path.join(d2, 'metadata', 'positive_instances.p'))
# dcm2 = pd.read_pickle(os.path.join(d2, 'dcm.p'))
# df2 = pd.read_pickle(os.path.join(d2, 'df.p'))
# assert len(set(dcm2.columns).intersection(df2.columns)) == 0
# assert dcm2.shape[0] == df2.shape[0]
# ok = ~dcm2['oversampled']
# dcm2, df2 = dcm2[ok], df2[ok]
# dm2 = utils.get_population_metadata(dcm2, dc2)
# window = np.array(
#     [f"""{a.strftime(fmt)}, {b.strftime(fmt)}, {c.strftime(fmt)}""" for a, b, c in zip(dcm2['time0'], dcm2['time1'], dcm2['time2'])])
# dcm2['window'] = None
# # dft
# # dcm2.loc[window == '03 Aug 2021, 01 Nov 2021, 01 Dec 2021', 'window'] = 'w1'
# # dcm2.loc[window == '02 Sep 2021, 01 Dec 2021, 01 Jan 2022', 'window'] = 'w2'
# # nst
# dcm2.loc[window == '03 Jul 2021, 01 Oct 2021, 01 Nov 2021', 'window'] = 'w1'
# dcm2.loc[window == '03 Aug 2021, 01 Nov 2021, 01 Dec 2021', 'window'] = 'w2'
# dcm2.loc[window == '02 Sep 2021, 01 Dec 2021, 01 Jan 2022', 'window'] = 'w3'
# dcm2.loc[window == '03 Oct 2021, 01 Jan 2022, 01 Feb 2022', 'window'] = 'w4'
# dcm2.loc[window == '03 Nov 2021, 01 Feb 2022, 01 Mar 2022', 'window'] = 'w5'
# dcm2.loc[window == '01 Dec 2021, 01 Mar 2022, 01 Apr 2022', 'window'] = 'w6'
# assert np.all(~pd.isnull(dcm2['window']))

# # identify same rows by VehicleId and window, filter and save
# on = ['VehicleId', 'window', 'collision-47']
# left = dcm1[on].reset_index(drop=False).rename(columns={'index': 'index1'})
# right = dcm2[on].reset_index(drop=False).rename(columns={'index': 'index2'})
# dcm = pd.merge(left, right, on=on, how='inner')
# for validate in ['EventRecorderId', 'Model', 'CompanyName', 'CompanyId', 'IndustryDesc']:
#     assert all(dcm1.loc[dcm['index1'].values, 'EventRecorderId'].values == dcm2.loc[dcm['index2'].values, 'EventRecorderId'].values)
# dcm1 = pd.read_pickle(os.path.join(d1, 'dcm.p')).loc[dcm['index1'].values].reset_index(drop=True)
# df1 = df1.loc[dcm['index1'].values].reset_index(drop=True)
# dcm2 = dcm2.loc[dcm['index2'].values].reset_index(drop=True)
# df2 = df2.loc[dcm['index2'].values].reset_index(drop=True)
# dcm1.to_pickle(os.path.join(d1, 'dcmx.p'))
# df1.to_pickle(os.path.join(d1, 'dfx.p'))
# dcm2.to_pickle(os.path.join(d2, 'dcmx.p'))
# df2.to_pickle(os.path.join(d2, 'dfx.p'))

# # compare raw data
# assert all(df1.columns == df2.columns)
# diff = np.full(df1.shape[1], np.nan)
# for x, col in enumerate(df1.columns):
#     dx1 = df1[col].values
#     dx2 = df2[col].values
#     diff[x] = np.isnan(dx2).sum() - np.isnan(dx1).sum()


# """
# individual feature vs prediction probability for all vehicle evals
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts
# plt.style.use('bmh')

# df = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/ml-data.p')
# yp = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/model-prediction-probabilities.p')
# yproba = yp['prediction probability'].values
# ytrue = yp['actual outcome'].values.astype('bool')
# xs = 'gps_miles'
# # xs = 'nevents_30_52_all'
# # xs = 'ntriggers_26_all'
# x = df[xs].values
# ok = x > 0
# x, yproba, ytrue = x[ok], yproba[ok], ytrue[ok]
# title = f'{xs} vs prediction probability'
# fig, ax = open_figure(title, figsize=(12, 6))
# ax.plot(x[~ytrue], yproba[~ytrue], 'x', alpha=0.1, label=f'{(~ytrue).sum()} negative instances')
# label = f'{ytrue.sum()} positive instances\n({100 * ytrue.sum() / ytrue.shape[0]:.3f}% of {ytrue.shape[0]} instances)'
# ax.plot(x[ytrue], yproba[ytrue], 'x', label=label)
# format_axes(xs, 'prediction probability', title, ax)
# leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, shadow=True, fancybox=True)
# for x in leg.legendHandles:
#     x._legmarker.set_alpha(1)
# largefonts(16)
# fig.tight_layout()
# plt.show()


# """
# download and process dce
# """
# import os
# import boto3
# from botocore.exceptions import ClientError
# from tempfile import NamedTemporaryFile

# s3 = boto3.resource(service_name='s3')
# uri = r's3://lytx-amlnas-us-west-2/dce-files/acnas0/T90_00_SD06/TrueTransportIn/MV00721033/event_MV00721033_1631852526_491225_6_8_4.DCE'
# uri = [x for x in uri.split(os.sep) if x]
# assert uri[0] == 's3:'
# obj = s3.Object(bucket_name=uri[1], key=os.sep.join(uri[2:]))
# try:
#     obj.load()
#     with NamedTemporaryFile(suffix='.DCE') as fx:
#         obj.download_file(fx.name)
#         # merged audio/video via dce2mkv in ffmpeg3 environment
#         cmd = f'conda run -n ffmpeg3 --cwd {os.path.split(fx.name)[0]} '
#         cmd += 'python /mnt/home/russell.burdt/miniconda3/envs/ffmpeg3/lib/python3.6/site-packages/dceutils/dce2mkv.py '
#         cmd += fx.name
#         os.system(cmd)
#         assert os.path.isfile(fx.name[:-4] + '_merged.mkv')
#         assert os.path.isfile(fx.name[:-4] + '_discrete.mkv')
#         os.remove(fx.name[:-4] + '_discrete.mkv')

# except ClientError as err:
#     print(err)


# """
# reuse triggers data from v1-dft47 for v2-dft
# """
# import os
# import utils
# import pandas as pd
# import numpy as np
# import pyarrow.parquet as pq
# from collections import defaultdict
# from glob import glob
# from tqdm import tqdm

# d1 = r'/mnt/home/russell.burdt/data/collision-model/v1/dft47'
# d2 = r'/mnt/home/russell.burdt/data/collision-model/v2/dft'
# assert os.path.isdir(d1) and os.path.isdir(d2)
# dcm1 = pd.read_pickle(os.path.join(d1, 'dcm.p'))
# dcm2 = pd.read_pickle(os.path.join(d2, 'dcm.p'))
# de1 = utils.time_bounds_dataframe(dcm=dcm1, xid='EventRecorderId')
# de2 = utils.time_bounds_dataframe(dcm=dcm2, xid='EventRecorderId')
# dex = defaultdict(list)
# for x, r2 in tqdm(de2.iterrows(), desc='de2', total=de2.shape[0]):
#     # v2 ER not covered in v1
#     if r2['EventRecorderId'] not in de1['EventRecorderId'].values:
#         for key, value in r2.iteritems():
#             dex[key].append(value)
#         continue
#     r1 = de1.loc[de1['EventRecorderId'] == r2['EventRecorderId']]
#     assert r1.shape[0] == 1
#     r1 = r1.squeeze()
#     # additional data at front
#     if r2['time0'] < r1['time0']:
#         dex['EventRecorderId'].append(r2['EventRecorderId'])
#         dex['time0'].append(r2['time0'])
#         dex['time1'].append(r1['time0'])
#     # additional data at back
#     if r2['time1'] > r1['time1']:
#         dex['EventRecorderId'].append(r2['EventRecorderId'])
#         dex['time0'].append(r1['time1'])
#         dex['time1'].append(r2['time1'])
# dex = pd.DataFrame(dex)
# dex['days'] = dex['time1'] - dex['time0']
# dex = dex.loc[dex['days'] > pd.Timedelta(days=1)].reset_index(drop=True)
# del dex['days']
# dex = utils.time_bounds_dataframe(dcm=dex, xid='EventRecorderId').sort_values('time0').reset_index(drop=True)
# de_indices = utils.reshape_indices(dex, n=1000)
# population = 'dft'
# # utils.distributed_data_extraction('triggers', population, dex, de_indices, distributed=False)
# assert os.path.isdir(os.path.join(d2, 'extra.parquet'))
# assert os.path.isdir(os.path.join(d2, 'triggers.parquet'))
# loc = os.path.join(d2, 'triggers.parquet')
# assert not glob(os.path.join(loc, '*'))
# for x, r2 in tqdm(de2.iterrows(), desc='de2', total=de2.shape[0]):
#     fn1 = glob(os.path.join(d1, 'triggers.parquet', f"""EventRecorderId={r2['EventRecorderId']}""", '*'))
#     assert len(fn1) < 2
#     fn2 = glob(os.path.join(d2, 'extra.parquet', f"""EventRecorderId={r2['EventRecorderId']}""", '*'))
#     assert len(fn2) < 2
#     if (len(fn1) == 0) and (len(fn2) == 0):
#         continue
#     elif (len(fn1) == 0) and (len(fn2) == 1):
#         df = pq.ParquetFile(fn2[0]).read().to_pandas()
#         df['EventRecorderId'] = r2['EventRecorderId']
#         assert df.duplicated().sum() == 0
#     elif (len(fn1) == 1) and (len(fn2) == 0):
#         df = pq.ParquetFile(fn1[0]).read().to_pandas()
#         df['EventRecorderId'] = r2['EventRecorderId']
#         assert df.duplicated().sum() == 0
#     elif (len(fn1) == 1) and (len(fn2) == 1):
#         df = pd.concat((
#             pq.ParquetFile(fn1[0]).read().to_pandas(),
#             pq.ParquetFile(fn1[0]).read().to_pandas()))
#         df['EventRecorderId'] = r2['EventRecorderId']
#         df = df[~df.duplicated()].reset_index(drop=True)
#     df.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=['EventRecorderId'], flavor='spark')


# """
# clean dcm based on oversampled rows and raw data availability
# """
# import os
# import utils
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from ipdb import set_trace

# datadir = r'/mnt/home/russell.burdt/data/collision-model/v2/dft'
# dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
# dp = pd.read_pickle(os.path.join(datadir, 'metadata', 'positive_instances.p'))
# dcm = pd.read_pickle(os.path.join(datadir, 'dcm.p'))
# dm = utils.get_population_metadata(dcm, dc)
# bounds = pd.read_pickle(os.path.join(datadir, 'coverage', 'bounds.p'))
# del bounds['tmin_epoch']
# del bounds['tmax_epoch']

# xs = pd.unique(dcm.loc[dcm['oversampled'], 'oversampled index']).astype('int')
# assert np.isnan(xs).sum() == 0
# for x in tqdm(xs, desc='oversampled rows'):

#     # original row in dcm (not oversampled)
#     dcm0 = dcm.loc[x]
#     assert not dcm0['oversampled']
#     assert dcm0['collision-47'] and (dcm0['collision-47-idx'].size > 0)

#     # corresponding oversampled rows
#     dcm1 = dcm.loc[dcm['oversampled index'] == x]
#     assert all(dcm1['oversampled'])

#     # time bounds for VehicleId / EventRecorderId (unused)
#     vbs = pd.concat((
#         bounds.loc[bounds['VehicleId'] == dcm0['VehicleId']],
#         bounds.loc[bounds['EventRecorderId'] == dcm0['EventRecorderId']])).reset_index(drop=True)

#     # check if other original rows in dcm extend beyond the oversampled rows
#     if dcm.loc[(~dcm['oversampled']) & (dcm['VehicleId'] == dcm0['VehicleId']), 'time1'].max() > dcm1['time1'].max():
#         pass
#     else:
#         set_trace()


# """
# total gps records in population
# """
# import os
# import config
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import broadcast

# population = 'dft'
# conf = SparkConf()
# conf.set('spark.driver.memory', '32g')
# conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
# conf.set('spark.sql.session.timeZone', 'UTC')
# conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
# conf.set('spark.sql.shuffle.partitions', 10000)
# spark = SparkSession.builder.config(conf=conf).getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')
# gps = spark.read.parquet(os.path.join(config.DATADIR, population, 'gps.parquet')).createOrReplaceTempView('gps')
# dc = pd.read_pickle(os.path.join(config.DATADIR, population, 'metadata', 'model_params.p'))
# dcm = pd.read_pickle(os.path.join(config.DATADIR, population, 'dcm_oversampled.p'))
# def time_bounds_dataframe(xid):
#     dx = pd.merge(
#         left=dcm.groupby(xid)['time0'].min().to_frame().reset_index(),
#         right=dcm.groupby(xid)['time1'].max().to_frame().reset_index(),
#         on=xid, how='inner')
#     sx = np.argsort([x.total_seconds() for x in (dx['time1'] - dx['time0'])])
#     return dx.loc[sx].reset_index(drop=True)
# def sort_time_bounds_dataframe(dx):

#     # case where all row durations are same as nominal window duration
#     if np.all((dx['time1'] - dx['time0']) == pd.Timedelta(days=int(dc['HISTORY_DAYS']))):
#         return dx.sort_values('time0').reset_index(drop=True)

#     # identify where window size first exceeds nominal window size
#     x0 = np.where((dx['time1'] - dx['time0']) > pd.Timedelta(days=int(dc['HISTORY_DAYS'])))[0][0]

#     # split into nominal and greater-than-nominal window size
#     dx0 = dx.loc[: x0 - 1].reset_index(drop=True)
#     dx1 = dx.loc[x0 :].reset_index(drop=True)
#     assert np.all((dx0['time1'] - dx0['time0']) == pd.Timedelta(days=int(dc['HISTORY_DAYS'])))
#     assert np.all((dx1['time1'] - dx1['time0']) > pd.Timedelta(days=int(dc['HISTORY_DAYS'])))

#     # sort by start time and concat
#     dx0 = dx0.sort_values('time0')
#     dx1 = dx1.sort_values('time0')
#     return pd.concat((dx0, dx1)).reset_index(drop=True)
# dv = sort_time_bounds_dataframe(time_bounds_dataframe('VehicleId'))
# dv['time0'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dv['time0']]
# dv['time1'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dv['time1']]
# dv = spark.createDataFrame(dv)
# dv = broadcast(dv)
# dv.createOrReplaceTempView('dv')
# dx = spark.sql(f"""
#     SELECT
#         dv.VehicleId,
#         COUNT(*) AS n_records,
#         COUNT(DISTINCT(gps.TS_SEC)) AS n_ts_sec,
#         MIN(gps.TS_SEC) AS tmin,
#         MAX(gps.TS_SEC) AS tmax,
#         (MAX(gps.TS_SEC) - MIN(gps.TS_SEC)) / (24*60*60) AS days
#     FROM gps JOIN dv
#         ON gps.VehicleId = dv.VehicleId
#         AND gps.TS_SEC >= dv.time0
#         AND gps.TS_SEC <= dv.time1
#     GROUP BY dv.VehicleId, dv.time0, dv.time1""")
# dx = dx.toPandas()

# """
# representation of National-Interstate population in dft population
# """
# import utils
# import numpy as np
# import pandas as pd

# # load data for both populations
# dm0 = pd.read_pickle(r'/mnt/home/russell.burdt/data/collision-model/dft/metadata/model_params.p')
# dm1 = pd.read_pickle(r'/mnt/home/russell.burdt/data/collision-model/national-interstate/metadata/model_params.p')
# dcm0 = pd.read_pickle(r'/mnt/home/russell.burdt/data/collision-model/dft/dcm.p')
# dcm1 = pd.read_pickle(r'/mnt/home/russell.burdt/data/collision-model/national-interstate/dcm.p')
# dp0 = utils.get_population_metadata(dcm0, dm0)
# dp1 = utils.get_population_metadata(dcm1, dm1)

# dc0 = dcm0.groupby('IndustryDesc')['collision'].sum()
# dc1 = dcm1.groupby('IndustryDesc')['collision'].sum()

# dcm2 = dcm1.loc[dcm1['IndustryDesc'].isin(['Distribution', 'Freight/Trucking'])].reset_index(drop=True)
# dcm3 = dcm0.loc[dcm0['VehicleId'].isin(pd.unique(dcm2['VehicleId']))].reset_index(drop=True)


# """
# parallelize pandas groupby apply via dask
# """
# import pandas as pd
# from dask import delayed, compute
# from dask.diagnostics import ProgressBar

# """
# plot results of dce model score coverage
# """

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# plt.style.use('bmh')

# # # load results
# # r0 = pd.DataFrame([
# #     ('10-01-2021', '11-01-2021', 94.58),
# #     ('09-01-2021', '10-01-2021', 93.71),
# #     ('08-01-2021', '09-01-2021', 71.47),
# #     ('07-01-2021', '08-01-2021', 71.31),
# #     ('06-01-2021', '07-01-2021', 68.39),
# #     ('05-01-2021', '06-01-2021', 62.14),
# #     ('04-01-2021', '05-01-2021', 46.01),
# #     ('03-01-2021', '04-01-2021', 12.81),
# #     ('02-01-2021', '03-01-2021', 10.17),
# #     ('01-01-2021', '02-01-2021', 8.94),
# #     ('12-01-2020', '01-01-2021', 4.95),
# #     ('11-01-2020', '12-01-2020', 3.82),
# #     ('10-01-2020', '11-01-2020', 1.01)], columns=['time0', 'time1', 'coverage'])
# # r0['time0'] = [pd.Timestamp(x) for x in r0['time0']]
# # r0['time1'] = [pd.Timestamp(x) for x in r0['time1']]
# r1 = pd.read_pickle(r'c:/Users/russell.burdt/Data/collision-model/coverage.p')
# r1['coverage'] = 100 * r1['coverage']

# # # # plot results
# # # fig, ax = open_figure('dce model coverage', figsize=(10, 4))
# # # ax.plot(r0['time1'], r0['coverage'], '.-', lw=3, ms=10, label='all SF300 devices in 30 day window')
# # # ax.plot(r1['time1'], r1['coverage'], '.-', lw=3, ms=10, label='collision model population in 30 day window')
# # # format_axes('', '%', 'Coverage of dce-model to recorded (27,30,31) events', ax, apply_concise_date_formatter=True)
# # # ax.legend(loc='upper left', numpoints=3, handlelength=4)
# # # ax.set_ylim(0, 101)
# # # ax.set_yticks(np.arange(0, 101, 10))
# # # largefonts(16)
# # # fig.tight_layout()
# # # plt.show()

# # plot results
# fig, ax = open_figure('dce model coverage', figsize=(10, 4))
# label = '30-day window for all SF300 devices\nin the collision model population'
# ax.plot(r1['time1'], r1['coverage'], '.-', lw=3, ms=10, label=label)
# format_axes('', '%', 'Coverage of dce-model to recorded (27,30,31) events', ax, apply_concise_date_formatter=True)
# ax.legend(loc='upper left', numpoints=3, handlelength=4)
# ax.set_ylim(0, 101)
# ax.set_yticks(np.arange(0, 101, 10))
# largefonts(16)
# fig.tight_layout()
# plt.show()


# """
# coverage of dce model scores for SF300 events, results from Jan 2022
# """

# import pandas as pd
# from lytx import get_conn, get_columns

# results = [
#     ('11-01-2021', '12-01-2021', 95.13),
#     ('10-01-2021', '11-01-2021', 94.58),
#     ('09-01-2021', '10-01-2021', 93.71),
#     ('08-01-2021', '09-01-2021', 71.47),
#     ('07-01-2021', '08-01-2021', 71.31),
#     ('06-01-2021', '07-01-2021', 68.39),
#     ('05-01-2021', '06-01-2021', 62.14),
#     ('04-01-2021', '05-01-2021', 46.01),
#     ('03-01-2021', '04-01-2021', 12.81),
#     ('02-01-2021', '03-01-2021', 10.17),
#     ('01-01-2021', '02-01-2021', 8.94),
#     ('12-01-2020', '01-01-2021', 4.95),
#     ('11-01-2020', '12-01-2020', 3.82),
#     ('10-01-2020', '11-01-2020', 1.01)]

# edw = get_conn('edw')
# events = get_columns(edw, 'flat.Events')
# devices = get_columns(edw, 'flat.Devices')
# dce = get_columns(edw, 'ml.Dce')
# re = get_columns(edw, 'ml.ModelResponse')
# rq = get_columns(edw, 'ml.ModelRequest')
# query1 = f"""
#     SELECT COUNT(DISTINCT(E.EventId)) AS n
#     FROM flat.Events AS E
#         LEFT JOIN flat.Devices AS D
#         ON D.VehicleId = E.VehicleId
#     WHERE E.Deleted = 0
#     AND D.Model = 'ER-SF300'
#     AND E.VehicleId <> '00000000-0000-0000-0000-000000000000'
#     AND E.EventTriggerTypeId IN (27,30,31)
#     AND E.RecordDate BETWEEN '10-01-2020' AND '11-01-2020'"""
# dx1 = pd.read_sql_query(query1, edw)
# query2 = f"""
#     SELECT COUNT(DISTINCT(E.EventId)) AS n
#     FROM flat.Events AS E
#         LEFT JOIN flat.Devices AS D
#         ON D.VehicleId = E.VehicleId
#         INNER JOIN ml.Dce AS DCE
#         ON DCE.EventId = E.EventId
#         INNER JOIN ml.ModelRequest AS RQ
#         ON RQ.DceId = DCE.DceId
#         INNER JOIN ml.ModelResponse AS RE
#         ON RE.ModelRequestId = RQ.ModelRequestId
#         AND RE.ModelKey = 'collision'
#         AND RQ.ModelId = 4
#     WHERE E.Deleted = 0
#     AND D.Model = 'ER-SF300'
#     AND E.VehicleId <> '00000000-0000-0000-0000-000000000000'
#     AND E.EventTriggerTypeId IN (27,30,31)
#     AND E.RecordDate BETWEEN '10-01-2020' AND '11-01-2020'"""
# dx2 = pd.read_sql_query(query2, edw)

# """
# Dan Lambert, Pulling every trip with associated driver for a time interval
# """

# declare @CompanyRootGroupId uniqueidentifier = '5100FFFF-60B6-E4CD-7A6E-7D43E4A30000' --SD Deliveries Plus
# declare @StartTimeAnalysis datetime = '2021-10-01 00:00:00'
# declare @StopTimeAnalysis datetime = '2021-11-01 00:00:00'

# select
#     [ct].[CompanyName]                    as [TripCompanyName],
#     [gt].[Name]                           as [TripGroupName],
#     [c].[CompanyName]                     as [ActiveERCompanyName],
#     [g].[Name]                            as [ActiveERGroupName],
#     [t].[TripDurationInSeconds],
#     [t].[TripId],
#     [er].[SerialNumber],
#     [era].[VehicleId],
#     [v].[VehicleName],
#     [dvs].[DriverId],
#     [u].[LastName],
#     [u].[FirstName],
#     [u].[EmployeeNum],
#     [g].[FullPath]                        as [GroupFullPath],
#     datepart(week, [t].[TripStartTime])   as [WeekTrip],
#     datepart(year, [t].[TripStartTime])   as [YearTrip]
# from [EDW].[distraction].[Trip] as [t]
#     inner join [EDW].[hs].[Groups] as [gt] --Trip GroupId
#         on [t].[GroupId] = [gt].[Id]
#     inner join [EDW].[hub].[Companies] as [ct]
#         on [gt].[CompanyId] = [ct].[Id]
#     inner join [EDW].[hs].[EventRecorders] as [er]
#         on [t].[EventRecorderId] = [er].[Id]
#     left outer join [EDW].[hs].[EventRecorderAssociations] as [era]
#         on [er].[Id] = [era].[EventRecorderId]
#            and [era].[DeletedDate] = '9999-01-01 00:00:00.0000000'
#     left outer join [EDW].[hs].[Groups] as [g] -- Active event recorder GroupId
#         on [era].[GroupId] = [g].[Id]
#     left outer join [EDW].[hub].[Companies] as [c]
#         on [g].[CompanyId] = [c].[Id]
#     left outer join [EDW].[hs].[Vehicles] as [v]
#         on [era].[VehicleId] = [v].[Id]
#     left outer join [EDW].[hs].[DriverVehicleSchedule] [dvs]
#         on [dvs].[VehicleId] = [era].[VehicleId]
#            and [t].[TripStartTime] > [dvs].[ScheduleStart]
#            and [t].[TripStartTime] < [dvs].[ScheduleEnd]
#     left outer join [EDW].[flat].[Users] as [u]
#         on [dvs].[DriverId] = [u].[UserId]
# where [t].[TripStartTime] >= @StartTimeAnalysis
#       and [t].[TripStartTime] < @StopTimeAnalysis
#       and[g].[RootGroupId] = @CompanyRootGroupId