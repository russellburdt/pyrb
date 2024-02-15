
"""
process munich-re poc dataset
"""

import os
import lytx
import utils
import pickle
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pyrb.bokeh import MapInterface
from pyrb.mpl import metric_distribution, open_figure, largefonts, format_axes, save_pngs
from bokeh.io import show, output_file, curdoc
from bokeh.models import Title
from sqlalchemy import create_engine
from datetime import datetime
from glob import glob
from snowflake import connector
from sklearn.metrics import average_precision_score, roc_auc_score
from pyproj import Geod, Transformer
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType
from pyspark.sql.functions import broadcast
from collections import defaultdict
from lytx import get_conn
from tqdm import tqdm
from ipdb import set_trace


# spark session
conf = SparkConf()
conf.set('spark.driver.memory', '64g')
conf.set('spark.driver.maxResultSize', 0)
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.debug.maxToStringFields', 500)
spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()

# write raw dataset to labs database (very memory intensive, should be refactored)
if False:

    # connection to labs database
    engine = create_engine("""postgresql://{username}:{password}@{server}/{database}""".format(
        username='postgres',
        password='uKvzYu0ooPo4Cw9Jvo7b',
        server='dev-labs-aurora-postgres-instance-1.cctoq0yyopdx.us-west-2.rds.amazonaws.com',
        database='labs'))

    # scan provided data by company and dataset
    # for company in ['Lytx', 'Geotab']:
    for company in ['Geotab']:
        datadir = os.path.join(r'/mnt/home/russell.burdt/data/munich-re/12-27-2022', company)
        assert os.path.isdir(datadir)
        for ds in sorted(glob(os.path.join(datadir, '*'))):

            # dataset metadata
            ds = glob(os.path.join(ds, '*'))
            assert len(ds) == 1
            ds = ds[0]
            name, ext = os.path.split(ds)[1].split('.')
            assert (name[:len(company)] == company.lower()) and (ext == 'parquet')

            # read to local memory
            print(f'read {name} to local memory')
            sdf = spark.read.parquet(ds)
            dx = sdf.toPandas()

            # write to database
            print(f'write {name} to db')
            name = 'geotablogrecord_20221227'
            dx['geom4326'] = None
            dx['geom3857'] = None
            dx.to_sql(name=name, con=engine, schema='insurance_model_munich_re', if_exists='fail', index=False)

# create and validate gps.parquet from raw geotab dataset
if False:

    # raw gps and device datasets
    gps = spark.read.parquet(r'/mnt/home/russell.burdt/data/munich-re/12-27-2022/Geotab/Geotab_Log_Record/geotablogrecord.parquet')
    gps.createOrReplaceTempView('gps')
    device = spark.read.parquet(r'/mnt/home/russell.burdt/data/munich-re/12-12-2022/Geotab/Geotab_Device/geotabdevice.parquet')
    device.createOrReplaceTempView('device')

    # combined Spark DataFrame representing raw gps data for valid / active devices
    sdf = spark.sql(f"""
        SELECT gps.*, device.vin AS VehicleId, device.activefrom, device.activeto
        FROM gps
            INNER JOIN device
            ON gps.device_id = device.id
            AND gps.datetime BETWEEN device.activefrom AND device.activeto
        WHERE device.vin IS NOT NULL""")
    sdf.createOrReplaceTempView('sdf')

    # modify column names, rewrite gps dataset
    loc = r'/mnt/home/russell.burdt/data/munich-re/poc/gps.parquet'
    assert not os.path.isdir(loc)
    sdf = sdf.withColumnRenamed('device_id', 'DeviceId')
    sdf = sdf.withColumnRenamed('speed', 'SPEED')
    sdf = sdf.drop('id').drop('version')
    sdf.write.parquet(path=loc, partitionBy='VehicleId', compression='snappy')

    # clean up parquet dataset, read gps data
    lytx.clean_up_crc(loc)
    lytx.merge_parquet(spark=spark, loc=loc, num_per_iteration=1, xid='VehicleId', ts='datetime')
    gps = spark.read.parquet(loc)
    gps.createOrReplaceTempView('gps')

    # convert lat/lon, create TS_SEC
    def convert(pdf):

        # read data for DeviceId
        assert pdf.shape[0] == 1
        vid = pdf['VehicleId'].values[0]
        fn = glob(os.path.join(loc, f'VehicleId={vid}', '*.parquet'))
        assert len(fn) == 1
        fn = fn[0]
        df = pq.ParquetFile(fn).read().to_pandas().sort_values('datetime').reset_index(drop=True)
        df['VehicleId'] = vid

        # conver lat/lon
        transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
        lon = df.pop('longitude').values
        lat = df.pop('latitude').values
        df['longitude'], df['latitude'] = transform(xx=lon, yy=lat)
        df['longitude_gps'] = lon
        df['latitude_gps'] = lat

        # remove full duplicate rows, create TS_SEC
        df = df.loc[~df.duplicated()].reset_index(drop=True)
        df['TS_SEC'] = [int((pd.Timestamp(x) - datetime(1970, 1, 1)).total_seconds()) for x in df['datetime'].values]
        df['TS_SEC'] = df['TS_SEC'].astype('int')

        # filter partial duplicates (same TS_SEC only)
        nok = df['TS_SEC'].value_counts()
        nok = nok[nok > 1].index.to_numpy()
        if nok.size > 0:
            df0 = df[~df['TS_SEC'].isin(nok)].copy()
            ok = pd.DataFrame()
            for tx in nok:
                assert df.loc[df['TS_SEC'] == tx].shape[0] > 1
                ok = pd.concat((ok, df.loc[df['TS_SEC'] == tx].iloc[0].to_frame().T))
            df = pd.concat((df0, ok)).copy().sort_values('TS_SEC').reset_index(drop=True)
            assert df.duplicated().sum() == 0
            assert df['TS_SEC'].value_counts().max() == 1

        # modify df for compatibility with gps enrichment function
        df['SPEED'] = df['SPEED'].astype('float')
        df['TS_USEC'] = 0.0
        df['COMPANY_ID'] = -11111
        df['HEADING'] = np.nan
        df['SERIAL_NUMBER'] = df['DeviceId'].values

        # update data
        df.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'], flavor='spark')
        os.remove(fn)
        return pdf

    # distribute convert function
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps')
    dx = vids.groupby('VehicleId').applyInPandas(convert, schema=vids.schema).toPandas()
    # debug
    # vids = vids.toPandas()
    # convert(vids.loc[vids.index == 17])

    # validate schema
    gps = spark.read.parquet(loc)
    gps.createOrReplaceTempView('gps')
    lytx.validate_consistent_parquet_schema(spark=spark, loc=loc, src='gps', xid='VehicleId')

# event recorder associations from raw lytx dataset
if False:

    # points Parquet dataset
    sdf = spark.read.parquet(r'/mnt/home/russell.burdt/data/munich-re/12-12-2022/Lytx/Lytx_Points/lytxpoints.parquet')
    sdf.createOrReplaceTempView('sdf')

    def convert(pdf):

        # validate pdf
        assert pd.unique(pdf['erserialnumber']).size == 1
        er = pdf['erserialnumber'].iloc[0]

        # convert occurrencedate to datetime, get time bounds
        pdf['datetime'] = [datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p') for x in pdf['occurrencedate'].values]

        # save to parquet
        pdf.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=['erserialnumber'], flavor='spark')

        # erserialnumber as a pandas DataFrame
        return pd.DataFrame(data={'erserialnumber': [er]})

    # distributed convert and repartition at loc
    loc = r'/mnt/home/russell.burdt/data/munich-re/lytx/points.parquet'
    schema = StructType([StructField('erserialnumber', StringType(), nullable=False)])
    ers = sdf.groupby('erserialnumber').applyInPandas(convert, schema=schema).toPandas()
    # debug
    # pdf = spark.sql(f"""SELECT * FROM sdf WHERE erserialnumber='MV00557454'""").toPandas()
    # convert(pdf)

    # read converted dataset
    points = spark.read.parquet(loc)
    points.createOrReplaceTempView('points')

    # validate same ers
    ers_sdf = np.sort(spark.sql(f'SELECT DISTINCT(erserialnumber) FROM sdf').toPandas().values.flatten())
    ers_points = np.sort(spark.sql(f'SELECT DISTINCT(erserialnumber) FROM points').toPandas().values.flatten())
    assert np.all(ers_sdf == ers_points) & np.all(np.sort(ers_sdf) == np.sort(ers.values.flatten()))

    def valid_ers(pdf):

        # validate pdf
        assert pd.unique(pdf['erserialnumber']).size == 1
        er = pdf['erserialnumber'].iloc[0]

        # event-recorder-ids associated with erserialnumber
        query = f"""
            SELECT SerialNumber, Id
            FROM hs.EventRecorders
            WHERE SerialNumber='{er}'"""
        ers = pd.read_sql_query(sql=query, con=get_conn('edw-cloud'))
        assert ers.shape[0] > 0

        # ers as pandas DataFrame
        return pd.DataFrame(data={
            'SerialNumber': ers['SerialNumber'].values.tolist(),
            'EventRecorderId': ers['Id'].values.tolist()})

    # pandas DataFrame of valid event recorder ids
    schema = StructType([
        StructField('SerialNumber', StringType(), nullable=False),
        StructField('EventRecorderId', StringType(), nullable=False)])
    ers = points.groupby('erserialnumber').applyInPandas(valid_ers, schema=schema).toPandas()
    # debug
    # pdf = spark.sql(f"""SELECT * FROM points WHERE erserialnumber='MV00557454'""").toPandas()
    # valid_ers(pdf)

    def valid_eras(pdf):

        # validate pdf
        assert pd.unique(pdf['erserialnumber']).size == 1
        er = pdf['erserialnumber'].iloc[0]

        # convert occurrencedate to datetime, get time bounds
        tmin = pdf['datetime'].min()
        tmax = pdf['datetime'].max()
        assert tmin < tmax

        # event-recorder-ids associated with erserialnumber
        erx = ers.loc[ers['SerialNumber'] == er, 'EventRecorderId'].values
        assert erx.size > 0
        erxs = ','.join(["""'{}'""".format(er) for er in erx])

        # valid event recorder associations
        query = f"""
            SELECT EventRecorderId, VehicleId, GroupId, CreationDate, DeletedDate
            FROM hs.EventRecorderAssociations
            WHERE EventRecorderId IN ({erxs})
            AND VehicleId <> '00000000-0000-0000-0000-000000000000'
            ORDER BY CreationDate"""
        era = pd.read_sql_query(sql=query, con=get_conn('edw-cloud'))
        era['CreationDate'] = [pd.Timestamp(x) for x in era['CreationDate'].values]
        if era['DeletedDate'].dtype.type == np.object_:
            era['DeletedDate'] = [pd.Timestamp(x) if x.strftime(r'%Y-%m-%d') != '9999-01-01' else pd.Timestamp('2262-04-11 23:47:16.854775807') for x in era['DeletedDate'].values]
        assert era['CreationDate'].dtype.type == np.datetime64
        assert era['DeletedDate'].dtype.type == np.datetime64
        era = era.loc[np.array([x.total_seconds() for x in era['DeletedDate'] - era['CreationDate']]) > 0].reset_index(drop=True)
        era = era.loc[(era['CreationDate'] < tmax) & (era['DeletedDate'] > tmin)].reset_index(drop=True)

        # time delta between consecutive event recorder associations
        td = era['CreationDate'].values[1:] - era['DeletedDate'].values[:-1]
        td = (1e-9) * td.astype('float')
        assert all(td > -1)
        era['tds'] = np.hstack((np.nan, td))

        # additional metadata from pdf
        era['SerialNumber'] = er
        era['tmin'] = tmin
        era['tmax'] = tmax

        return era[['SerialNumber', 'tmin', 'tmax', 'EventRecorderId', 'VehicleId', 'GroupId', 'CreationDate', 'DeletedDate', 'tds']]

    # pandas DataFrame of valid event recorder associations
    schema = StructType([
        StructField('SerialNumber', StringType(), nullable=False),
        StructField('tmin', TimestampType(), nullable=False),
        StructField('tmax', TimestampType(), nullable=False),
        StructField('EventRecorderId', StringType(), nullable=False),
        StructField('VehicleId', StringType(), nullable=False),
        StructField('GroupId', StringType(), nullable=False),
        StructField('CreationDate', TimestampType(), nullable=False),
        StructField('DeletedDate', TimestampType(), nullable=False),
        StructField('tds', DoubleType(), nullable=True)])
    dex = points.groupby('erserialnumber').applyInPandas(valid_eras, schema=schema).toPandas()
    # debug
    # pdf = spark.sql(f"""SELECT * FROM points WHERE erserialnumber='MV00534035'""").toPandas()
    # valid_eras(pdf)
    dex.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/lytx/metadata/event_recorder_associations.p')

# collision prediction model population for lytx dataset
if False:

    # event recorder associations
    dex = pd.read_pickle(r'/mnt/home/russell.burdt/data/munich-re/lytx/metadata/event_recorder_associations.p')

    # valid collision / predictor intervals
    dcm = defaultdict(list)
    vids = pd.unique(dex['VehicleId'])
    for vid in tqdm(vids, desc='scanning vids'):

        # scan over event recorder associations for vid
        for _, era in dex.loc[dex['VehicleId'] == vid].iterrows():

            # validate
            assert era['DeletedDate'] > era['CreationDate']
            assert era['tmax'] > era['tmin']
            assert (era['CreationDate'] < era['tmax']) & (era['DeletedDate'] > era['tmin'])

            # time bounds for vehicle evals
            tmin = max((era['tmin'], era['CreationDate']))
            tmax = min((era['tmax'], era['DeletedDate']))
            assert tmax > tmin

            # first valid predictor / collision intervals
            month, year = tmax.strftime('%m-%Y').split('-')
            time2 = pd.Timestamp(f'{year}-{month}-01 00:00:00')
            assert tmax > time2
            month, year = (time2 - pd.Timedelta(days=1)).strftime('%m-%Y').split('-')
            time1 = pd.Timestamp(f'{year}-{month}-01 00:00:00')
            time0 = time1 - pd.Timedelta(days=90)

            # record valid intervals
            while time0 > tmin:

                # metadata from era
                for item in ['VehicleId', 'EventRecorderId', 'GroupId', 'CreationDate', 'DeletedDate']:
                    dcm[item].append(era[item])

                # predictor and collision intervals
                dcm['time0'].append(time0)
                dcm['time1'].append(time1)
                dcm['time2'].append(time2)

                # update predictor and collision intervals
                time2 = time1
                month, year = (time2 - pd.Timedelta(days=1)).strftime('%m-%Y').split('-')
                time1 = pd.Timestamp(f'{year}-{month}-01 00:00:00')
                time0 = time1 - pd.Timedelta(days=90)
    dcm = pd.DataFrame(dcm)

    # validate
    assert all(dcm['time0'] > dcm['CreationDate'])
    assert all(dcm['time2'] < dcm['DeletedDate'])

    def collision_records(pdf):

        # validate
        assert pdf.shape[0] == 1
        assert pd.unique(pdf['index']).size == 1

        # identify collisions
        query = f"""
            SELECT
                E.VehicleId,
                E.RecordDate,
                E.Latitude,
                E.Longitude,
                E.EventId,
                E.CustomerEventIdString,
                value AS BehaviorId
            FROM flat.Events AS E
                CROSS APPLY STRING_SPLIT(COALESCE(E.BehaviourStringIds, '-1'), ',')
            WHERE E.RecordDate > '{pdf.iloc[0]['time1'].strftime('%m-%d-%Y %H:%M:%S')}'
            AND E.RecordDate < '{pdf.iloc[0]['time2'].strftime('%m-%d-%Y %H:%M:%S')}'
            AND E.VehicleId = '{pdf.iloc[0]['VehicleId']}'
            AND E.Deleted=0
            AND value IN (45,46,47)"""
        dx = pd.read_sql_query(con=get_conn('edw-cloud'), sql=query)
        dx['BehaviorId'] = dx['BehaviorId'].astype('int')

        # return DataFrame with consistent schema
        if dx.size > 0:
            return dx
        else:
            return pd.DataFrame(data={
                'VehicleId': np.array([]).astype('object'),
                'RecordDate': np.array([]).astype(np.datetime64),
                'Latitude': np.array([]).astype('double'),
                'Longitude': np.array([]).astype('double'),
                'EventId': np.array([]).astype('object'),
                'CustomerEventIdString': np.array([]).astype('object'),
                'BehaviorId': np.array([]).astype('int')})

    # valid collision records
    sdf = spark.createDataFrame(dcm.reset_index(drop=False))
    schema = StructType([
        StructField('VehicleId', StringType(), nullable=False),
        StructField('RecordDate', TimestampType(), nullable=False),
        StructField('Latitude', DoubleType(), nullable=False),
        StructField('Longitude', DoubleType(), nullable=False),
        StructField('EventId', StringType(), nullable=False),
        StructField('CustomerEventIdString', StringType(), nullable=False),
        StructField('BehaviorId', IntegerType(), nullable=False)])
    dp = sdf.groupby('index').applyInPandas(collision_records, schema=schema).toPandas()
    # debug
    # dx = dcm.reset_index(drop=False)
    # collision_records(dx.loc[dx['index'] == 2])
    dp.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/lytx/metadata/positive_instances.p')

    # join dcm and dp
    dpm = defaultdict(list)
    for _, row in tqdm(dcm.iterrows(), desc='collision intervals', total=dcm.shape[0]):

        # valid collisions
        dpx = dp.loc[(dp['VehicleId'] == row['VehicleId']) & (dp['RecordDate'] > row['time1']) & (dp['RecordDate'] < row['time2'])]

        # append collisions
        for cx in [45, 46, 47]:
            dpm[f'collision-{cx}'].append(np.any(dpx['BehaviorId'] == cx))
            dpm[f'collision-{cx}-idx'].append(dpx.index[(dpx['BehaviorId'] == cx)].to_numpy())
    dpm = pd.DataFrame(dpm)
    dcm = pd.concat((dcm, dpm), axis=1)
    dcm['oversampled'] = False

    def industry_and_company_name(pdf):

        # validate
        assert pdf.shape[0] == 1

        # identify IndustryDesc and CompanyName
        query = f"""
            SELECT D.VehicleId, C.CompanyName, C.IndustryDesc
            FROM flat.Devices AS D
                INNER JOIN flat.Companies AS C
                ON D.CompanyId = C.CompanyId
            WHERE D.DeviceId = '{pdf.iloc[0]['EventRecorderId']}'"""
        dx = pd.read_sql_query(sql=query, con=get_conn('edw-cloud'))

        # validate and return
        assert dx.shape[0] == 1
        return pd.DataFrame(data={
            'index': pdf['index'].tolist(),
            'IndustryDesc': dx['IndustryDesc'].tolist(),
            'CompanyName': dx['CompanyName'].tolist()})

    # IndustryDesc and CompanyName
    sdf = spark.createDataFrame(dcm.reset_index(drop=False))
    schema = schema = StructType([
        StructField('index', IntegerType(), nullable=False),
        StructField('IndustryDesc', StringType(), nullable=False),
        StructField('CompanyName', StringType(), nullable=False)])
    dx = sdf.groupby('index').applyInPandas(industry_and_company_name, schema=schema).toPandas()
    # debug
    # dx = dcm.reset_index(drop=False)
    # industry_and_company_name(dx.loc[dx['index'] == 2])

    # merge dx and dcm
    dcm = pd.merge(left=dcm, right=dx, left_index=True, right_on='index', how='inner').reset_index(drop=True)
    dcm = dcm.loc[dcm['CompanyName'].isin(['Hi Pro, Inc', 'Synctruck LLC', 'L&L Redi-Mix, Inc.', 'TCA Logistics Corp'])].reset_index(drop=True)

    # save dcm
    dcm.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/lytx/dcm.p')

    # model params and save
    dc = pd.Series({
        'desc': 'Munich-Re POC',
        'predictor interval days': 90,
        'collision intervals': 'misc'})
    dc.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/lytx/metadata/model_params.p')

    # validate
    dm = utils.get_population_metadata(dcm, dc, datadir=None)
    utils.validate_dcm_dp(dcm, dp)

# gps data for Lytx population provided by Munich-Re (needs review)
if False:

    def gps_data(pdf):

        # validate pdf, get erserialnumber
        assert pdf.shape[0] == 1
        erx = ers.loc[pdf['id'].values[0], 'erserialnumber']

        # gps raw data provided by Munich-Re for erserialnumber
        assert os.path.isdir(os.path.join(loc, f'erserialnumber={erx}'))
        fn = glob(os.path.join(loc, f'erserialnumber={erx}', '*.parquet'))
        assert len(fn) == 1
        fn = fn[0]
        df = pq.ParquetFile(fn).read().to_pandas().sort_values('datetime').reset_index(drop=True)

        # TS_SEC and convert gps coords
        df['TS_SEC'] = [int((pd.Timestamp(x) - datetime(1970, 1, 1)).total_seconds()) for x in df['datetime'].values]
        transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
        lon = df.pop('positionlongitude').values
        lat = df.pop('positionlatitude').values
        df['longitude'], df['latitude'] = transform(xx=lon, yy=lat)
        df['longitude_gps'] = lon
        df['latitude_gps'] = lat

        # snowflake connection object
        snow = connector.connect(
            user='SVC_LABS_USER',
            database='dp_prod_db',
            warehouse='LABS_PROD_VWH_XL',
            password='4:^A]N>N#eH=p&Qp',
            account='lytx')
        pd.read_sql_query('USE WAREHOUSE \"LABS_PROD_VWH_XL\"', snow)

        # scan over valid event-recorder-associations
        dev = dex.loc[dex['SerialNumber'] == erx]
        assert dev.shape[0] > 0
        for _, era in dev.iterrows():

            # time bounds
            tmin = max((era['tmin'], era['CreationDate']))
            tmax = min((era['tmax'], era['DeletedDate']))
            assert (tmax - tmin).total_seconds() > 0

            # gps data provided by Munich-Re and clean up
            gps0 = df.loc[(df['datetime'] >= tmin) & (df['datetime'] <= tmax)].reset_index(drop=True)
            gps0['VehicleId'] = era['VehicleId']
            gps0 = gps0.loc[~gps0.duplicated(['TS_SEC'])].reset_index(drop=True)
            gps0 = gps0.sort_values('datetime').reset_index(drop=True)

            # gps data from Snowflake and clean up
            time0 = int((tmin - datetime(1970, 1, 1)).total_seconds())
            time1 = int((tmax - datetime(1970, 1, 1)).total_seconds())
            query = f"""
                SELECT
                    UPPER(VEHICLE_ID) AS VEHICLEID,
                    TS_SEC,
                    TS_USEC,
                    LATITUDE,
                    LONGITUDE,
                    SPEED,
                    HEADING,
                    HDOP,
                    SERIAL_NUMBER,
                    COMPANY_ID
                FROM GPS.GPS_ENRICHED
                WHERE TS_SEC BETWEEN {time0} AND {time1}
                AND (VEHICLE_ID = '{era['VehicleId'].lower()}' OR VEHICLE_ID = '{era['VehicleId'].upper()}')"""
            gps1 = pd.read_sql_query(query, snow)
            gps1['datetime'] = [pd.Timestamp(datetime.utcfromtimestamp(x)) for x in gps1['TS_SEC']]
            gps1['VehicleId'] = gps1.pop('VEHICLEID')
            transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
            lon = gps1.pop('LONGITUDE').values
            lat = gps1.pop('LATITUDE').values
            gps1['longitude'], gps1['latitude'] = transform(xx=lon, yy=lat)
            gps1['longitude_gps'] = lon
            gps1['latitude_gps'] = lat
            gps1 = gps1.loc[~gps1.duplicated(['TS_SEC'])].reset_index(drop=True)
            gps1 = gps1.sort_values('datetime').reset_index(drop=True)

            # write gps0 to parquet dataset
            if gps0.size > 0:
                d0 = os.path.join(os.path.split(loc)[0], 'gps-mre.parquet')
                if os.path.isdir(os.path.join(d0, f"""VehicleId={era['VehicleId']}""")):
                    count = len(glob(os.path.join(d0, f"""VehicleId={era['VehicleId']}""", '*.parquet')))
                else:
                    count = 0
                gps0.to_parquet(path=d0, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'], flavor='spark')
                assert len(glob(os.path.join(d0, f"""VehicleId={era['VehicleId']}""", '*.parquet'))) == count + 1

            # write gps1 to parquet dataset
            if gps1.size > 0:
                d1 = os.path.join(os.path.split(loc)[0], 'gps.parquet')
                if os.path.isdir(os.path.join(d1, f"""VehicleId={era['VehicleId']}""")):
                    count = len(glob(os.path.join(d1, f"""VehicleId={era['VehicleId']}""", '*.parquet')))
                else:
                    count = 0
                gps1.to_parquet(path=d1, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'], flavor='spark')
                assert len(glob(os.path.join(d1, f"""VehicleId={era['VehicleId']}""", '*.parquet'))) == count + 1

        return pdf

    # distributed gps extraction by erserialnumber
    ers = spark.sql(f'SELECT DISTINCT erserialnumber FROM points').toPandas()
    vx = spark.range(start=0, end=ers.shape[0], step=1, numPartitions=int(1.5 * ers.shape[0]))
    dx = vx.groupby('id').applyInPandas(gps_data, schema=vx.schema).toPandas()
    # debug
    # vx = vx.toPandas()
    # gps_data(vx.loc[vx.index == 11])
    # gps_data(vx.loc[ers['erserialnumber'] == ''])

    # merge parquet files by vehicle-id
    lytx.merge_parquet(spark, loc='/mnt/home/russell.burdt/data/munich-re/lytx0/gps-mre.parquet')
    lytx.merge_parquet(spark, loc='/mnt/home/russell.burdt/data/munich-re/lytx0/gps.parquet')

        # gps datasets
    mre = spark.read.parquet(r'/mnt/home/russell.burdt/data/munich-re/lytx0/gps-mre.parquet')
    mre.createOrReplaceTempView('mre')
    gps = spark.read.parquet(r'/mnt/home/russell.burdt/data/munich-re/lytx0/gps.parquet')
    gps.createOrReplaceTempView('gps')

    # count records by vehicle-id
    ngps = spark.sql(f'SELECT VehicleId, COUNT(*) AS n_records FROM gps GROUP BY VehicleId').toPandas()
    nmre = spark.sql(f'SELECT VehicleId, COUNT(*) AS n_records FROM mre GROUP BY VehicleId').toPandas()
    dc = pd.merge(ngps, nmre, on='VehicleId', how='inner', suffixes=['_gps', '_mre'])
    dc['percent diff'] = 100 * (dc['n_records_mre'] - dc['n_records_gps']) / dc['n_records_gps']
    dc = dc.sort_values('percent diff').reset_index(drop=True)

    # validate time bounds by vehicle-id
    tgps = spark.sql(f'SELECT VehicleId, MIN(datetime) AS tmin, MAX(datetime) AS tmax FROM gps GROUP BY VehicleId').toPandas()
    tmre = spark.sql(f'SELECT VehicleId, MIN(datetime) AS tmin, MAX(datetime) AS tmax FROM mre GROUP BY VehicleId').toPandas()
    tx = pd.merge(
        left=dex.groupby('VehicleId')['tmin'].min().to_frame(),
        right=dex.groupby('VehicleId')['tmax'].max().to_frame(),
        left_index=True, right_index=True, how='inner').reset_index(drop=False)
    dts = pd.merge(tgps, tmre, on='VehicleId', how='inner', suffixes=['_gps', '_mre'])
    dts = pd.merge(dts, tx, on='VehicleId', how='left')
    assert all(dts['tmin_gps'] >= dts['tmin'])
    assert all(dts['tmin_mre'] >= dts['tmin'])
    assert all(dts['tmax_gps'] <= dts['tmax'])
    assert all(dts['tmax_mre'] <= dts['tmax'])

# save raw gps data gps trails as a png file
if False:

    # read gps data
    loc = r'/mnt/home/russell.burdt/data/munich-re/poc/gps.parquet'
    assert os.path.isdir(loc)
    gps = spark.read.parquet(loc)
    gps.createOrReplaceTempView('gps')
    vids = spark.sql(f"""
        SELECT VehicleId, COUNT(*) AS n_records
        FROM gps
        GROUP BY VehicleId ORDER BY n_records ASC""").toPandas()

    def process_vehicle(pdf):
        """
        chart of gps raw data trails for individual vehicle id
        """

        # validate
        assert pd.unique(pdf['VehicleId']).size == 1
        pdf = pdf.sort_values('datetime').reset_index(drop=True)
        lon = pdf['longitude'].values
        lat = pdf['latitude'].values
        assert (np.isnan(lon).sum() == 0) and (np.isnan(lat).sum() == 0)
        vid = pdf.iloc[0]['VehicleId']
        records = pdf.shape[0]
        days = (pdf['datetime'].max() - pdf['datetime'].min()).total_seconds() / (3600*24)
        t0 = pdf['datetime'].min().strftime('%m/%d/%Y %H:%M:%S')
        t1 = pdf['datetime'].max().strftime('%m/%d/%Y %H:%M:%S')

        # map object
        mapx = MapInterface(width=800, height=400)
        mapx.path.data = {'lon': lon, 'lat': lat}
        if (lon.min() < lon.max()) and (lat.min() < lat.max()):
            mapx.reset_map_view(lon0=lon.min(), lon1=lon.max(), lat0=lat.min(), lat1=lat.max(), convert=False)
        mapx.fig.add_layout(Title(text=f'min datetime {t0}, max datetime {t1}, {days:.1f} days',
            text_font_style='italic', text_font_size='16pt'), 'above')
        mapx.fig.add_layout(Title(text=f'raw gps for vehicle {vid}, {records} records',
            text_font_style='bold', text_font_size='16pt'), 'above')

        # export to png
        fn = rf'/mnt/home/russell.burdt/data/munich-re/map_html/{vid}.html'
        output_file(fn)
        show(mapx.fig)

        from html2image import Html2Image
        hti = Html2Image(
            browser='chrome',
            browser_executable=r'/usr/bin/google-chrome',
            output_path=os.path.split(fn)[0],
            size=(900, 600))
        hti.screenshot(html_file=fn, save_as=f'{vid}.png')
        os.remove(fn)

        # return pandas DataFrame
        return pd.DataFrame({'VehicleId': [vid], 'n_records': [records]})

    # distributed processing by device_id
    geod = Geod(ellps='WGS84')
    schema = StructType([
        StructField('VehicleId', StringType(), False),
        StructField('n_records', IntegerType(), False)])
    dx = gps.groupby('VehicleId').applyInPandas(process_vehicle, schema=schema).toPandas()
    # debug
    # pdf = spark.sql(f"""SELECT * FROM gps WHERE VehicleId='{vids.loc[11, 'VehicleId']}'""").toPandas()
    # process_vehicle(pdf)

# bokeh app to view gps data from specific vehicle
if False:

    # read gps data
    loc = r'/mnt/home/russell.burdt/data/munich-re/poc/gps.parquet'
    assert os.path.isdir(loc)
    gps = spark.read.parquet(loc)
    gps.createOrReplaceTempView('gps')
    vids = spark.sql(f"""
        SELECT VehicleId, COUNT(*) AS n_records
        FROM gps
        GROUP BY VehicleId ORDER BY n_records ASC""").toPandas()

    # specific vehicle
    # vehicle = 'b158'
    # records = dcs.loc[dcs['DeviceId'] == device, 'n_records'].values[0]
    # pdf = spark.sql(f"""SELECT * FROM gps WHERE DeviceId='{device}' ORDER BY datetime""").toPandas()
    # assert pdf.shape[0] == records
    # assert all(np.sort(pdf['datetime'].values) == pdf['datetime'].values)
    # assert all([pd.unique(pdf[x]).size == 1 for x in ['vin', 'activefrom', 'activeto']])
    # vin, activefrom, activeto = pdf.loc[0, ['vin', 'activefrom', 'activeto']]
    # days = (activeto - activefrom).total_seconds() / (3600 * 24)
    # activefrom = activefrom.strftime('%m/%d/%Y %H:%M:%S')
    # activeto = activeto.strftime('%m/%d/%Y %H:%M:%S')
    # lon, lat = pdf['longitude'].values, pdf['latitude'].values
    # assert (np.isnan(lon).sum() == 0) and (np.isnan(lat).sum() == 0)

    # map interface and data
    # gps = MapInterface(width=800, height=400)
    # gps.path.data = {'lon': lon, 'lat': lat}
    # if (lon.min() < lon.max()) and (lat.min() < lat.max()):
    #     gps.reset_map_view(lon0=lon.min(), lon1=lon.max(), lat0=lat.min(), lat1=lat.max(), convert=False)
    # gps.fig.add_layout(Title(text=f'activefrom {activefrom}, activeto {activeto}, {days:.1f} days',
    #     text_font_style='italic', text_font_size='16pt'), 'above')
    # gps.fig.add_layout(Title(text=f'raw gps for device {device}, vin {vin}, {records} records',
    #     text_font_style='bold', text_font_size='16pt'), 'above')

    # # deploy app
    # doc = curdoc()
    # doc.add_root(gps.fig)

# collision prediction model population for geotab dataset
if False:

    # metadata folder
    if not os.path.isdir(r'/mnt/home/russell.burdt/data/munich-re/poc/metadata'):
        os.mkdir(r'/mnt/home/russell.burdt/data/munich-re/poc/metadata')

    # combined gps and device dataset
    gps = spark.read.parquet(r'/mnt/home/russell.burdt/data/munich-re/poc/gps.parquet')
    gps.createOrReplaceTempView('gps')

    # read and save collision data
    fns = glob(os.path.join(r'/mnt/home/russell.burdt/data/munich-re/12-12-2022', '*loss*.xlsx'))
    dps = [pd.read_excel(fn) for fn in fns]
    common = set.intersection(*(set(dp.columns) for dp in dps))
    dp = pd.concat(dps)[common].reset_index(drop=True)
    dp = dp.loc[~dp.duplicated()].reset_index(drop=True)
    dp = dp.loc[~pd.isnull(dp['Vehicle Vin'])].reset_index(drop=True)
    cok = np.all(~pd.isnull(dp), axis=0)
    cok = cok[cok].index.to_numpy()
    dp = dp[cok]
    dp['VehicleId'] = dp.pop('Vehicle Vin').values
    dp['RecordDate'] = dp.pop('Loss Date').values
    dp = dp[['VehicleId', 'RecordDate'] + [x for x in sorted(dp.columns) if x not in ['VehicleId', 'RecordDate']]]
    dp['BehaviorId'] = 47
    dp.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/poc/metadata/positive_instances.p')

    # create and save 'model_params'
    dc = pd.Series({
        'desc': 'Munich-Re POC',
        'predictor interval days': 90,
        'collision intervals': 'misc'})
    dc.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/poc/metadata/model_params.p')

    # time bounds by vehicle-id / device-id
    dts = spark.sql(f"""
        SELECT VehicleId, COUNT(*) AS n_records, MIN(datetime) AS tmin, MAX(datetime) AS tmax, DeviceId
        FROM gps
        GROUP BY VehicleId, DeviceId""").toPandas()
    dts['days'] = [x.total_seconds() / (3600 * 24) for x in dts['tmax'] - dts['tmin']]
    dts = dts.sort_values('days').reset_index(drop=True)
    assert pd.isnull(dts).values.sum() == 0

    # build collision-prediction model population DataFrame
    dcm = defaultdict(list)
    for _, row in tqdm(dts.iterrows(), desc='scanning vids', total=dts.shape[0]):

        # collisions for vin
        collisions = dp.loc[dp['VehicleId'].str.lower() == row['VehicleId'].lower()]

        # first valid predictor / collision intervals
        month, year = row['tmax'].strftime('%m-%Y').split('-')
        time2 = pd.Timestamp(f'{year}-{month}-01 00:00:00')
        assert row['tmax'] > time2
        month, year = (time2 - pd.Timedelta(days=1)).strftime('%m-%Y').split('-')
        time1 = pd.Timestamp(f'{year}-{month}-01 00:00:00')
        time0 = time1 - pd.Timedelta(days=90)

        # record valid intervals
        while time0 > row['tmin']:
            dcm['VehicleId'].append(row['VehicleId'])
            dcm['DeviceId'].append(row['DeviceId'])
            dcm['time0'].append(time0)
            dcm['time1'].append(time1)
            dcm['time2'].append(time2)

            # collisions for vehicle-id
            if any((collisions['RecordDate'] > time1) & (collisions['RecordDate'] < time2)):
                dcm['collision'].append(True)
                idx = collisions.loc[(collisions['RecordDate'] > time1) & (collisions['RecordDate'] < time2)].index.to_numpy()
                dcm['collision-idx'].append(idx)
            else:
                dcm['collision'].append(False)
                dcm['collision-idx'].append(np.array([]).astype('int'))

            # update predictor and collision intervals
            time2 = time1
            month, year = (time2 - pd.Timedelta(days=1)).strftime('%m-%Y').split('-')
            time1 = pd.Timestamp(f'{year}-{month}-01 00:00:00')
            time0 = time1 - pd.Timedelta(days=90)
    dcm = pd.DataFrame(dcm)
    # modify for compatibility
    dcm['IndustryDesc'] = 'na'
    dcm['CompanyName'] = 'na'
    dcm['oversampled'] = False
    dcm['collision-45'] = False
    dcm['collision-46'] = False
    dcm['collision-47'] = dcm.pop('collision').values
    dcm['collision-45-idx'] = [np.array([]).astype('int') for _ in range(dcm.shape[0])]
    dcm['collision-46-idx'] = [np.array([]).astype('int') for _ in range(dcm.shape[0])]
    dcm['collision-47-idx'] = dcm.pop('collision-idx').values
    dcm.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/poc/dcm.p')

    # validate methods do not raise errors
    utils.get_population_metadata(dcm, dc)
    utils.validate_dcm_dp(dcm, dp)

# gps data bounds and coverage for geotab dataset
if False:

    # dcm and gps dataset
    dcm = pd.read_pickle(r'/mnt/home/russell.burdt/data/munich-re/poc/dcm.p')
    gps = spark.read.parquet(r'/mnt/home/russell.burdt/data/munich-re/poc/gps.parquet')
    gps.createOrReplaceTempView('gps')
    r0 = lytx.count_all(spark, 'gps')

    # gps time bounds
    bounds = lytx.get_time_bounds(spark, src='gps', xid='VehicleId', ts_min='TS_SEC', ts_max='TS_SEC')
    if not os.path.isdir('/mnt/home/russell.burdt/data/munich-re/poc/coverage'):
        os.mkdir('/mnt/home/russell.burdt/data/munich-re/poc/coverage')
    bounds.to_pickle('/mnt/home/russell.burdt/data/munich-re/poc/coverage/bounds.p')

    # gps daily record count
    drc = lytx.records_per_day_xid(spark, dcm, src='gps', xid='VehicleId')
    drc.to_pickle(r'/mnt/home/russell.burdt/data/munich-re/poc/coverage/gps_coverage.p')
