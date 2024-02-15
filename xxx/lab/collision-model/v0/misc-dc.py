
"""
new gps enrichment functions
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd
from sqlalchemy import text as sqltext
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql.functions import broadcast
from pyspark.sql import SparkSession
from ipdb import set_trace
from tqdm import tqdm


# spark session
conf = SparkConf()
conf.set('spark.driver.memory', '64g')
conf.set('spark.driver.maxResultSize', 0)
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.parquet.enableVectorizedReader', 'false')
conf.set('spark.sql.session.timeZone', 'UTC')
conf.set('spark.sql.shuffle.partitions', 20000)
conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# gps data
datadir = r'/mnt/home/russell.burdt/data/collision-model/v2/nst'
assert os.path.isdir(datadir)
gps = spark.read.parquet(os.path.join(datadir, f'gps2.parquet'))
gps.createOrReplaceTempView('gps')

# load data for vid, prepare dx for enrichment
vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps').toPandas()
vid = np.random.choice(vids['VehicleId'].values)
df = lytx.spark_etl_load_gps(datadir, src='gps2.parquet', vid=vid, service='EC2')
assert np.any(~pd.isnull(df['COMPANY_ID']))
df.loc[pd.isnull(df['COMPANY_ID']), 'COMPANY_ID'] = df.loc[~pd.isnull(df['COMPANY_ID']), 'COMPANY_ID'].iloc[0]
df['COMPANY_ID'] = df['COMPANY_ID'].astype('int')
segments = pd.unique(df['segmentId'])
segment = np.random.choice(segments[~np.isnan(segments)])
c0 = ['TS_SEC', 'TS_USEC', 'COMPANY_ID', 'HEADING', 'SERIAL_NUMBER', 'longitude_gps', 'latitude_gps', 'SPEED', 'VehicleId']
da = df.loc[df['segmentId'] == segment].reset_index(drop=True)
dx = df.loc[df['segmentId'] == segment, c0].reset_index(drop=True)
for col in ['TS_SEC', 'TS_USEC', 'HEADING', 'SPEED', 'SERIAL_NUMBER']:
    dx[col.lower()] = dx.pop(col)
dx['vehicle_id'] = [x.lower() for x in dx.pop('VehicleId')]
dx['longitude'] = dx.pop('longitude_gps')
dx['latitude'] = dx.pop('latitude_gps')
dx['company_id'] = dx.pop('COMPANY_ID').astype('int')
dx['timestamp'] = [datetime.utcfromtimestamp(x) for x in dx['ts_sec']]

# db connection and write dx to database
conn, schema = lytx.gps_enrichment_dbs(rc=0)
name = f"""deleteme{vid.replace('-','_').lower()}"""
dx.to_sql(name=name, con=conn.engine, schema=schema, if_exists='replace', index=False)

# osm220718.lytxlab_riskcore_enrichgps_trip
ds0 = lytx.database_function_schema(conn, 'osm220718', 'lytxlab_riskcore_enrichgps_trip')
sql0 = f"""SELECT * FROM osm220718.lytxlab_riskcore_enrichgps_trip('{schema}.{name}', null, '{vid.lower()}')"""
de0 = pd.read_sql_query(con=conn, sql=sqltext(sql0)).sort_values('ts_sec').reset_index(drop=True)
de0 = lytx.align_dataframe_datatypes_sql(de0, ds0)
dx0 = pd.DataFrame([(col, de0[col].dtype) for col in de0.columns], columns=['column', 'datatype'])
assert sorted(ds0['column'].values) == sorted(dx0['column'].values)
dc0 = pd.merge(left=ds0, right=dx0, on='column', how='inner', suffixes=('_src', '_python'))
dc0['conversion'] = [f"""{row['datatype_src']} to {row['datatype_python']}""" for _, row in dc0.iterrows()]

# osm221107.lytxlab_riskcore_enrichgps_trip
ds1 = lytx.database_function_schema(conn, 'osm221107', 'lytxlab_riskcore_enrichgps_trip')
sql1 = f"""SELECT * FROM osm221107.lytxlab_riskcore_enrichgps_trip('{schema}.{name}', null, '{vid.lower()}')"""
de1 = pd.read_sql_query(con=conn, sql=sqltext(sql1)).sort_values('ts_sec').reset_index(drop=True)
de1 = lytx.align_dataframe_datatypes_sql(de1, ds1)
dx1 = pd.DataFrame([(col, de1[col].dtype) for col in de1.columns], columns=['column', 'datatype'])
assert sorted(ds1['column'].values) == sorted(dx1['column'].values)
dc1 = pd.merge(left=ds1, right=dx1, on='column', how='inner', suffixes=('_src', '_python'))
dc1['conversion'] = [f"""{row['datatype_src']} to {row['datatype_python']}""" for _, row in dc1.iterrows()]

# compare results from osm220718.lytxlab_riskcore_enrichgps_trip and osm221107.lytxlab_riskcore_enrichgps_trip
assert de0.shape == de1.shape
assert de0.shape[0] == dx.shape[0]
assert np.all(np.sort(de0.columns) == np.sort(de1.columns))
assert all(ds0.sort_values('column').reset_index(drop=True) == ds1.sort_values('column').reset_index(drop=True))
assert all(dx0.sort_values('column').reset_index(drop=True) == dx1.sort_values('column').reset_index(drop=True))

# osm221107.lytxlab_riskcore_enrichgps_trip_core
ds2 = lytx.database_function_schema(conn, 'osm221107', 'lytxlab_riskcore_enrichgps_trip_core')
sql2 = f"""SELECT * FROM osm221107.lytxlab_riskcore_enrichgps_trip_core('{schema}.{name}', null, '{vid.lower()}')"""
de2 = pd.read_sql_query(con=conn, sql=sqltext(sql2)).sort_values('ts_sec').reset_index(drop=True)
de2 = lytx.align_dataframe_datatypes_sql(de2, ds2)
assert de2.shape[0] == de0.shape[0]
dx2 = pd.DataFrame([(col, de2[col].dtype) for col in de2.columns], columns=['column', 'datatype'])
assert sorted(ds2['column'].values) == sorted(dx2['column'].values)
dc2 = pd.merge(left=ds2, right=dx2, on='column', how='inner', suffixes=('_src', '_python'))
dc2['conversion'] = [f"""{row['datatype_src']} to {row['datatype_python']}""" for _, row in dc2.iterrows()]

# write de2 to database
name = name + '_core'
de2.to_sql(name=name, con=conn.engine, schema=schema, if_exists='replace', index=False)

# osm221107.lytxlab_riskcore_normalize_and_enrich_gps_segments
ds3 = lytx.database_function_schema(conn, 'osm221107', 'lytxlab_riskcore_normalize_and_enrich_gps_segments')
sql3 = f"""SELECT * FROM osm221107.lytxlab_riskcore_normalize_and_enrich_gps_segments('{schema}.{name}', null, '{vid.lower()}')"""
de3 = pd.read_sql_query(con=conn, sql=sqltext(sql3))
de3 = lytx.align_dataframe_datatypes_sql(de3, ds3)
dx3 = pd.DataFrame([(col, de3[col].dtype) for col in de3.columns], columns=['column', 'datatype'])
assert sorted(ds3['column'].values) == sorted(dx3['column'].values)
dc3 = pd.merge(left=ds3, right=dx3, on='column', how='inner', suffixes=('_src', '_python'))
dc3['conversion'] = [f"""{row['datatype_src']} to {row['datatype_python']}""" for _, row in dc3.iterrows()]

# write de3 to database
name = name + '_enriched'
de3.to_sql(name=name, con=conn.engine, schema=schema, if_exists='replace', index=False)

# clean up
conn.close()

# initialize df3
df3 = pd.DataFrame()
df3['seg_x0'] = [x.lower for x in de3['segment__enrichgps_trip_gps_segment_id_range']]
df3['seg_x1'] = [x.upper for x in de3['segment__enrichgps_trip_gps_segment_id_range']]
df3['seg_meters'] = de3['segment__length_meters']
df3['grp_index'] = de3['segmentgroup__index']
df3['grp_maneuver'] = de3['segmentgroup__maneuver']
df3['grp_segment_count'] = de3['segmentgroup__segment_count']
df3['grp_length_meters'] = de3['segmentgroup__length_meters']



# """
# plot metrics from gps enrichment functions
# """

# import os
# import lytx
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# from sqlalchemy import text as sqltext
# plt.style.use('bmh')

# # db connection
# conn, schema = lytx.gps_enrichment_dbs(rc=0)

# # load data
# df = pd.read_sql_query(con=conn, sql=sqltext(f"""
#     SELECT * FROM {schema}.deleteme1c00ffff_59ae_4bc9_9ce6_4663f0800000"""))
# dc = pd.read_sql_query(con=conn, sql=sqltext(f"""
#     SELECT * FROM {schema}.deleteme1c00ffff_59ae_4bc9_9ce6_4663f0800000_core"""))
# de = pd.read_sql_query(con=conn, sql=sqltext(f"""
#     SELECT * FROM {schema}.deleteme1c00ffff_59ae_4bc9_9ce6_4663f0800000_core_enriched"""))

# # plot data
# fig, ax = open_figure('from gps enrichment functions', 2, 1, figsize=(14, 8))

# s0 = np.cumsum(dc['gps_segment_length_meters'].values[1:])
# ax[0].plot(s0, 'x-')
# title = 'cumulative sum of gps_segment_length_meters\nosm221107.lytxlab_riskcore_enrichgps_trip_core'
# format_axes('row index', '', title, ax[0])

# s1 = np.cumsum(de['segment__length_meters'].values)
# ax[1].plot(s1, 'x-')
# title = 'cumulative sum of segment__length_meters\nosm22107.lytxlab_riskcore_normalize_and_enrich_gps_segments'
# format_axes('row index', '', title, ax[1])

# largefonts(18)
# fig.tight_layout()
# plt.show()




# """
# chart of GPS enrichment metrics
# """
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
# plt.style.use('bmh')

# dx = pd.read_pickle(r'c:\Users\russell.burdt\Downloads\gps_enrichment_metrics.p')

# fig, ax = open_figure('enrichment time vs records count', figsize=(12, 6))
# ax.plot(dx['n_records'].values, dx['minutes'].values, 'x', ms=8, mew=3, alpha=0.6, label='7B GPS records')
# format_axes('num records by vehicle-id', 'GPS enrichment time, minutes', 'enrichment time vs num records by vehicle-id', ax)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=1)
# largefonts(20)
# fig.tight_layout()

# plt.show()



# """
# pre-process enriched gps metrics
# """

# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from glob import glob
# from lytx import get_conn

# # population data and metrics
# datadir = r'/mnt/home/russell.burdt/data/collision-model/v1/dft'
# assert os.path.isdir(datadir)
# dcm = pd.read_pickle(os.path.join(datadir, r'dcm.p'))
# df = pd.read_pickle(os.path.join(datadir, 'df.p'))
# assert dcm.shape[0] == df.shape[0]
# assert len(set(dcm.columns).intersection(df.columns)) == 0

# # enriched gps metrics
# fn = r'/mnt/home/russell.burdt/data/collision-model/dc/tripreport20220401.p'
# if os.path.isfile(fn):
#     dc = pd.read_pickle(fn)
# else:
#     lab = get_conn('lytx-lab')
#     dc = pd.read_sql_query(f'SELECT * FROM insurance_model.distfreighttruckingcrash_08012021_11302021_tripreport20220401', lab)
#     dc['VehicleId'] = dc.pop('vehicle_id')
#     dc['EventRecorderId'] = dc.pop('eventrecorderid')
#     dc['Model'] = dc.pop('model')
#     dc['CompanyName'] = dc.pop('companyname')
#     dc['IndustryDesc'] = dc.pop('industrydesc')
#     for col in [
#             'creationdate', 'deleteddate', 'company_id', 'num_data_points',
#             'timestamp_max', 'timestamp_min', 'timestamp_total_duration',
#             'flat_companies_channelsubsidyaccountname', 'flat_companies_companyname', 'flat_companies_industrydesc',
#             'flat_companies_industrysectordesc', 'flat_companies_parentcompanyname']:
#         del dc[col]
#     dc.to_pickle(fn)

# # remove timestamp columns
# for x in [x for x in dc.columns if 'timestamp' in x]:
#     del dc[x]

# # common and other features in dc
# common = sorted(list(set(dc.columns).intersection(dcm.columns)))
# others = sorted(list(set(dc.columns).difference(dcm.columns)))
# dc = dc[common + others]

# # modify column names to be easily identifed later on
# dc.columns = [f'DC_{x}' if x in others else x for x in dc.columns]
# others = sorted(list(set(dc.columns).difference(dcm.columns)))
# dc.columns = [x.replace('gps', 'gxs') if x in others else x for x in dc.columns]
# others = sorted(list(set(dc.columns).difference(dcm.columns)))
# dc.columns = [x.replace('__', '_') if x in others else x for x in dc.columns]
# others = sorted(list(set(dc.columns).difference(dcm.columns)))
# keys = ['trip', 'events', 'behaviors', 'dce_model', 'triggers', 'industry', 'gps']
# assert len([x for x in others if any([key in x.lower() for key in keys])]) == 0

# # remove rows with null or duplicates in metadata columns
# nok = pd.isnull(dc[common]).any(axis=1)
# dc = dc.loc[~nok].reset_index(drop=True)
# nok = dc.duplicated(subset=common, keep=False)
# dc = dc.loc[~nok].reset_index(drop=True)

# # remove rows in dc not in dcm (should be none or less than 1%)
# ok = pd.concat((dcm[common], dc[common])).duplicated()[-dc.shape[0]:]
# dc = dc.loc[ok].reset_index(drop=True)
# assert pd.concat((dcm[common], dc[common])).duplicated().sum() == dc.shape[0]

# # remove columns in dc with one unique value
# nok = dc[others].nunique() == 1
# nok = nok.loc[nok.values].index.to_numpy()
# for x in nok:
#     del dc[x]
# others = sorted(list(set(dc.columns).difference(dcm.columns)))

# # remove columns in dc with all null values
# nok = pd.isnull(dc).all(axis=0)
# nok = nok.loc[nok.values].index.to_numpy()
# for x in nok:
#     del dc[x]
# others = sorted(list(set(dc.columns).difference(dcm.columns)))

# # validate unique columns in dc, and no all-null or single-value columns
# assert pd.unique(dc.columns).size == dc.shape[1]
# assert len(set(dc.columns).intersection(df.columns)) == 0
# assert pd.isnull(dc).all(axis=0).values.sum() == 0
# assert dc[others].nunique().min() > 1

# # filter dcm and df by rows in dc
# ok = pd.concat((dcm[common], dc[common])).duplicated(keep=False)[:dcm.shape[0]]
# assert ok.sum() == dc.shape[0]
# dcm = dcm[ok].reset_index(drop=True)
# df = df[ok].reset_index(drop=True)
# assert dcm.shape[0] == df.shape[0] == dc.shape[0]
# assert all(pd.concat((dcm[common], dc[common])).duplicated(keep=False))

# # join dc with df
# dx = pd.merge(dcm[common], dc[common].reset_index(drop=False), on=common, how='left')
# assert all(dcm[common] == dx[common])
# dc = dc.loc[dx['index'].values].reset_index(drop=True)
# assert all(dcm[common] == dc[common])
# dc = dc[others]

# # # smaller population
# # vids = np.random.choice(pd.unique(dcm['VehicleId']), size=100, replace=False)
# # vids = np.array([vid for vid in tqdm(vids, desc='vids') if os.path.isdir(os.path.join(datadir, 'gps0.parquet', f'VehicleId={vid}'))])
# # dcm.loc[dcm['VehicleId'].isin(vids)].reset_index(drop=True)
# # for vid in tqdm(vids, desc='vids'):
# #     src = os.path.join(datadir, 'gps0.parquet', f'VehicleId={vid}')
# #     fn = glob(os.path.join(src, '*.parquet'))
# #     assert len(fn) == 1
# #     fn = fn[0]
# #     dst = os.path.join(os.path.split(datadir)[0], 'dft20v', 'gps.parquet', f'VehicleId={vid}', os.path.split(fn)[1])
# #     os.mkdir(os.path.join(os.path.split(datadir)[0], 'dft20v', 'gps.parquet', f'VehicleId={vid}'))
# #     os.system(f'cp {fn} {dst}')

# # save data
# dcm.to_pickle(os.path.join(datadir, 'enriched_dcm.p'))
# df.to_pickle(os.path.join(datadir, 'enriched_df_core.p'))
# pd.concat((df, dc), axis=1).to_pickle(os.path.join(datadir, 'enriched_df_full.p'))
# dc.to_pickle(os.path.join(datadir, 'enriched_df_new.p'))




# """
# query insurance_model tables directly
# """

# # database connection objects, create smaller population table
# lab = get_conn('lytx-lab')
# cstr = f"""postgresql://postgres:uKvzYu0ooPo4Cw9Jvo7b@dev-labs-aurora-postgres-instance-1.cctoq0yyopdx.us-west-2.rds.amazonaws.com/labs"""
# engine = create_engine(cstr)
# # dcm = pd.read_sql_query(f'SELECT * FROM insurance_model.distfreighttruckingcrash_08012021_11302021', lab)
# # dcm = dcm.loc[np.random.choice(dcm.index, 10, replace=False)].reset_index(drop=True)
# # dcm.to_sql(name='distfreighttruckingcrash_08012021_11302021_subset', con=engine, schema='insurance_model', if_exists='fail', index=False)

# query = f"""
#     SELECT
#         tGPS.vehicle_id,
#         tVehicle.time0,
#         tVehicle.time1,
#         -- count of GPS points
#         count(*) as num_data_points
#     FROM
#         insurance_model.distfreighttruckingcrash_08012021_11302021_gps as tGPS
#         -- only include vehicles within timeframe
#         INNER JOIN insurance_model.distfreighttruckingcrash_08012021_11302021_subset as tVehicle
#             ON tVehicle.vehicleid::uuid = tGPS.vehicle_id
#             AND tVehicle.time0 <= tGPS."timestamp"
#             AND tVehicle.time1 > tGPS."timestamp"
#     WHERE
#         -- limit to north america where we have map data
#         tGPS.longitude BETWEEN -175 AND -49
#         AND tGPS.latitude BETWEEN 10 AND 75
#     GROUP BY tGPS.vehicle_id, tVehicle.time0, tVehicle.time1"""
# dx = pd.read_sql_query(query, lab)
