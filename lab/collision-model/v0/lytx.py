
"""
common data processing
"""

import os
import pytz
import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sqlalchemy as sa
from glob import glob
from shutil import rmtree
from itertools import chain
from functools import reduce
from socket import gethostname
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, TimestampType
from pyspark.sql.functions import broadcast
from pyproj import Transformer, Geod
from geopandas.array import from_wkb
from collections import defaultdict
from datetime import datetime
from time import sleep
from ipdb import set_trace
from tqdm import tqdm


# vehicle populations
def get_population(population):
    """
    return dict representing a vehicle population, defined as a dict with
    - desc - population description
    - one or both of ['CompanyName', 'IndustryDesc'] as key - list of values for population
    - all values must exist in flat.Companies
    - special case is 'vehicle-id'
    """

    # validate population type
    assert isinstance(population, str)

    # define population
    if population == 'munich-re':
        population = {
            'desc': 'companies insured by Munich-Re',
            'CompanyName': [
                """Hi Pro, Inc""",
                """L&L Redi-Mix, Inc.""",
                """Synctruck LLC""",
                """TCA Logistics Corp"""]}

    elif population == 'gwcc':
        population = {
            'desc': 'companies insured by GWCC',
            'CompanyName': [
                """Ag Express Inc""",
                """Anderson Trucking Service""",
                """Arizona Milk Transport""",
                """Best Overnite Express, Inc.""",
                """Cox Petroleum Transport""",
                """Diamond Line Delivery Systems, Inc.""",
                """Express Lease LLC""",
                """First Choice Transport, Inc.""",
                """Gary Amoth Trucking, Inc.""",
                """Gruhn Transport""",
                """HUELLINGHOFF BROS INC""",
                """Haney and White Cartage Inc""",
                """Hogan Transports, Inc.""",
                """KW Rantz Trucking""",
                """Key Trucking, Inc.""",
                """Miller Trucking, Ltd.""",
                """Mutual Express Company""",
                """Northwest Carriers Inc""",
                """Redbird Trucking, LLC""",
                """Silica Transport Inc""",
                """Twin Lake Trucking, Ltd."""]}

    elif population == 'synctruck':
        population = {
            'desc': 'Synctruck LLC',
            'CompanyName': ["""Synctruck LLC"""]}

    elif population == 'nst':
        population = {
            'desc': 'companies insured by National-Interstate',
            'CompanyName': [
                """A/T Transportation""",
                """Ace Intermountain Recycling Center""",
                """Air Capitol Delivery & Whse LLC""",
                """Air Ground Xpress, Inc. - NIIC""",
                """Antonini Freight Express Inc""",
                """Appalachian Freight Carriers, Inc.""",
                """Apple Towing Co""",
                """Arlo Lott - NIIC""",
                """Arrow Limousine Worldwide""",
                """Atlas Disposal Ind LLC""",
                """Bayview Limousine""",
                """Bedore Tours & Charter Inc""",
                """Bellavance Trucking""",
                """Big M Transportation, Inc. - Canal""",
                """Black Gold Express, Inc.""",
                """Brady Trucking Inc""",
                """Bugman & Sons, Inc.""",
                """C&L Transport, Inc.""",
                """California Materials, Inc.""",
                """California Wine Tours & Transportation""",
                """CKJ Transport, LP""",
                """Custom Commodities, Inc. - NIIC""",
                """D L Landis Trucking - NIIC""",
                """Duncan & Son Lines""",
                """Edwards Moving & Rigging Inc""",
                """EmpireCLS WW Chauffeured Services""",
                """Eppler""",
                """Euless B&B Wrecker Service""",
                """FMC Transport""",
                """Go Riteway""",
                """Great Lakes Cold Logistics, Inc""",
                """Green Lines Transportation - NIIC""",
                """Hallcon Corp""",
                """Hilltrux Tank Lines Inc""",
                """I.B. Dickinson & Sons, Inc.""",
                """Illinois Central School Bus, LLC""",
                """Island Transportation Corp. - NIIC""",
                """JIT-EX LLC""",
                """KeyStops LLC""",
                """Knight Brothers, LLC""",
                """LGA Trucking""",
                """Livingston Trucking, Inc.""",
                """MedicOne Medical Response""",
                """Medstar LLC""",
                """MID Cities, Inc""",
                """Motorcoach Class A Transportation, Inc. - NIIC""",
                """Mountain Crane""",
                """Nagle Toledo, Inc.""",
                """Neal Pool Rekers""",
                """OBriens Moving and Storage""",
                """Oneida Warehousing""",
                """Owen Transport Services""",
                """Pahoa Express, Inc.""",
                """Pavlich, Inc. - NIIC""",
                """Pneumatic Trucking Inc""",
                """Presidential Worldwide Transportation""",
                """PTG Logistics - NIIC""",
                """R M Trucking , Inc.""",
                """Regent Coach Line""",
                """Ridge Ambulance Svc Inc""",
                """Rock Transfer & Storage Inc""",
                """Samson Heavy Haul""",
                """Sharp Transportation""",
                """Siedhoff Distributing Co - NIIC""",
                """Signature Transportation Group""",
                """Slay Industries""",
                """Southwest Dedicated Transport LLC""",
                """Suburban Disposal Corp.""",
                """Super T Transport""",
                """Tate Transportation, Inc.""",
                """Terminal Consolidation Co.""",
                """Terry Hill, Inc.""",
                """Thompson & Harvey Transportation - NIIC""",
                """TMH Transport LLC""",
                """Total Luxury Limousine - NIIC""",
                """Tramcor Corp.""",
                """Tri-County Ambulance Service, Inc.""",
                """Updike Distribution Logistics""",
                """Van Lingen""",
                """Vito Trucking LLC""",
                """Williams Dairy Trucking, Inc.""",
                """WL Logan Trucking Co""",
                """Young's Commercial Transfer"""]}

    elif population == 'amt':
        population = {
            'desc': 'companies insured by AmTrust',
            'CompanyName': [
                """Beacon Transport LLC""",
                """Chizek Elevator & Transport""",
                """Carolina Logistic Inc."""]}

    elif population == 'dtft':
        population = {
            'desc': 'all of Distribution, Transit, Freight/Trucking',
            'IndustryDesc': ['Distribution', 'Transit', 'Freight/Trucking']}

    elif population == 'dft':
        population = {
            'desc': 'all of Distribution and Freight/Trucking',
            'IndustryDesc': ['Distribution', 'Freight/Trucking']}

    elif population == 'dist':
        population = {
            'desc': 'all of Distribution Industry',
            'IndustryDesc': ['Distribution']}

    elif population == 'ftc':
        population = {
            'desc': 'few Freight/Trucking companies',
            'CompanyName': [
                """Eagle Express Lines Inc""",
                """Autobahn Freight Lines Ltd.""",
                """McLane Company""",
                """Hogan Transports, Inc.""",
                """NFI""",
                """Arrow Transport Inc""",
                """Dan S Romero And Son Inc""",
                """Rayne Logsitics Inc"""]}

    else:
        assert ValueError('population not recognized')

    # validate all fields are columns of flat.Companies and valid metadata fields
    edw = get_conn('edw')
    columns = get_columns(edw, 'flat.Companies')['column'].values
    assert all([(x in columns) and (x in ['CompanyName', 'IndustryDesc']) for x in population if x != 'desc'])

    # validate items in each field as existing in flat.Companies
    for field, xf in population.items():
        if field == 'desc':
            continue
        sf = ','.join(["""'{}'""".format(x) for x in [x.replace("""'""", """''""") for x in xf]])
        query = f"""
            SELECT {field}
            FROM flat.Companies
            WHERE {field} IN ({sf})"""
        df = pd.read_sql_query(sa.text(query), edw)
        assert pd.unique(df[field]).size == len(xf)

    return population

# database connections and nominal queries
def get_conn(server):
    """
    db connection object by name, currently supports
    - edw
    - lab
    - snowflake
    """

    # edw
    if server == 'edw':
        # cstr = 'mssql+pyodbc://orp0v-dwhsql07.drivecam.net/EDW?driver=ODBC+Driver+17+for+SQL+Server'
        cstr = 'mssql+pyodbc://edw.drivecam.net/EDW?driver=ODBC+Driver+17+for+SQL+Server'
        return sa.create_engine(cstr, isolation_level="AUTOCOMMIT").connect()

    # lytx-lab
    elif server == 'lab':
        cstr = 'postgresql://postgres:uKvzYu0ooPo4Cw9Jvo7b@dev-labs-aurora-postgres-instance-1.cctoq0yyopdx.us-west-2.rds.amazonaws.com/labs'
        return sa.create_engine(cstr).connect()

    # unified-map
    elif server == 'unified-map':
        cstr = 'postgresql://osm_limited:27f90d43a35596ca930fef872a5db4a1@dev-unified-map-mapping-domain.cluster-custom-cctoq0yyopdx.us-west-2.rds.amazonaws.com/services'
        return sa.create_engine(cstr).connect()

    # snowflake
    elif server == 'snowflake':
        cstr = 'snowflake://SVC_LABS_USER:4:^A]N>N#eH=p&Qp@lytx/dp_prod_db?warehouse=LABS_PROD_VWH_XL'
        conn = sa.create_engine(cstr).connect()
        pd.read_sql_query(sa.text('USE WAREHOUSE \"LABS_PROD_VWH_XL\"'), conn)
        return conn
        # from snowflake import connector
        # conn = connector.connect(
        #     user='SVC_LABS_USER',
        #     database='dp_prod_db',
        #     warehouse='LABS_PROD_VWH_XL',
        #     password='4:^A]N>N#eH=p&Qp',
        #     account='lytx')
        # pd.read_sql_query('USE WAREHOUSE \"LABS_PROD_VWH_XL\"', conn)
        # return conn
        # Okta authentication
        # return connector.connect(
        #     user='russell.burdt',
        #     database='dp_prod_db',
        #     warehouse='labs_prod_vwh',
        #     role='lytx_labs_engineer_prod',
        #     authenticator='externalbrowser',
        #     account='lytx')
        # available warehouses
        #     LABS_PROD_EXPORT_VWH_L
        #     LABS_PROD_VWH
        #     LABS_PROD_VWH_2XL
        #     LABS_PROD_VWH_L
        #     LABS_PROD_VWH_M
        #     LABS_PROD_VWH_XL
        # USE WAREHOUSE <name> same as warehouse name above

    # data-platform
    elif server == 'dataplatform':
        from pyathena import connect
        return connect(
            aws_access_key_id='',
            aws_secret_access_key='',
            aws_session_token='',
            s3_staging_dir='s3://lytx-athena-result-prod-003/raw/',
            region_name='us-west-2')

def get_columns(conn, table):
    """
    return columns in table from database connection object
    """

    # sqlalchemy tables
    if isinstance(conn, sa.Connection):
        schema, table = table.split('.')
        table = sa.Table(table, sa.MetaData(), schema=schema, autoload_with=conn)
        return pd.DataFrame({
            'column': [x.name for x in table.c],
            'dtype': [x.type.python_type.__name__ for x in table.c]})

    # Snowflake
    else:
        from snowflake import connector
        assert isinstance(conn, connector.SnowflakeConnection)
        cursor = conn.cursor()
        desc = cursor.describe(f'SELECT * FROM {table}')
        return pd.DataFrame({
            'column': [x.name for x in desc],
            'dtype': [x.type_code for x in desc]})

def edw_nominal_query(query, xid=None):
    """
    nominal queries to EDW, xid may or may not be used
    """

    # all groups
    if query == 'groups':
        return pd.read_sql_query(
            con=get_conn('edw'),
            sql=sa.text(f"""
                SELECT
                    G.GroupId,
                    G.Name AS GroupName,
                    G.CompanyId,
                    C.CompanyName
                FROM flat.Groups AS G
                    LEFT JOIN flat.Companies AS C
                    ON C.CompanyId = G.CompanyId"""))

    # all companies
    if query == 'companies':
        return pd.read_sql_query(
            con=get_conn('edw'),
            sql=sa.text(f"""
                SELECT
                    C.CompanyId,
                    C.CompanyName,
                    MIN(C.FleetSizeCount) AS FleetSizeCount,
                    MIN(C.FleetTypeDesc) AS FleetTypeDesc,
                    MIN(C.IndustryDesc) AS IndustryDesc,
                    COUNT(D.DeviceId) AS n_devices
                FROM flat.Companies AS C
                    LEFT JOIN flat.Devices AS D
                    ON C.CompanyId = D.CompanyId
                GROUP BY C.CompanyId, C.CompanyName"""))

    # all devices
    if query == 'devices':
        return pd.read_sql_query(
            con=get_conn('edw'),
            sql=sa.text(f"""
                SELECT D.DeviceId, D.Model, D.SerialNumber, D.VehicleId, D.CompanyId, C.CompanyName
                FROM flat.Devices AS D
                    LEFT JOIN flat.Companies AS C
                    ON C.CompanyId = D.CompanyId"""))

    # single device info by SN, eg MV01210685
    if query == 'sn':
        return pd.read_sql_query(
            con=get_conn('edw'),
            sql=sa.text(f"""
                SELECT
                    D.SerialNumber, D.Model,
                    ER.Build, ER.FirmwareVersion, ER.HardwareVersion,
                    D.DeviceId, D.VehicleId, D.GroupId,
                    V.VehicleName, VT.Description AS VehicleDesc,
                    C.CompanyId, C.CompanyName, G.Name AS GroupName,
                    C.StatusDesc, C.FleetTypeDesc, C.IndustryDesc,
                    EC.GpsTrailEnable, EC.GpsTrailTrackInterval,
                    EC.HibernateEnable, EC.HibernateDelay
                FROM flat.Devices AS D
                LEFT JOIN flat.Companies AS C ON D.CompanyId = C.CompanyId
                LEFT JOIN flat.Groups AS G ON G.GroupId = D.GroupId
                LEFT JOIN hs.EventRecorders AS ER ON ER.Id = D.DeviceId
                LEFT JOIN flat.EventRecorderConfigs AS EC ON EC.EventRecorderId = D.DeviceId
                LEFT JOIN hs.Vehicles AS V ON V.Id = D.VehicleId
                LEFT JOIN hs.VehicleTypes_i18n AS VT ON VT.Id = V.VehicleTypeId
                WHERE D.SerialNumber = '{xid}'
                AND VT.LocaleId = 9""")).squeeze()

def database_function_schema(conn, schema, function):
    """
    return database schema from function info
    """

    # query provided by Dennis Cheng
    query = f"""
        with cteGetFunctions as (
            select
                row_number()over(winByFunction) as row_by_function,
                np.nspname,
                proname as function_name,
                unnest(p.proargnames) as proargname,
                unnest(p.proallargtypes)::integer as proallargtype,
                unnest(p.proargmodes) as proargmode
        --      ,*
            from pg_proc p left join pg_namespace np on p.pronamespace=np.oid
            where
                nspname = '{schema}'
                and proname = '{function}'
            window winByFunction as (partition by nspname, proname)
        )
        select
            tFunction.proargname AS column,
            tType.typname AS datatype
        from cteGetFunctions as tFunction left join pg_type as tType on tFunction.proallargtype=tType.oid
        where proargmode='t'
        window byFunctionParam as (partition by nspname, function_name)"""

    # validate and return
    ds = pd.read_sql_query(sa.text(query), conn).sort_values(['column', 'datatype']).reset_index(drop=True)
    assert pd.unique(ds['column']).size == ds.shape[0]
    return ds

def align_dataframe_datatypes_sql(df, ds):
    """
    convert datatypes in pandas DataFrame df according to schema in DataFrame ds
    - df is typically pulled from SQL
    - ds may be created by database_function_schema
    """
    assert sorted(df.columns) == sorted(ds['column'].values)

    # columns and datatypes in df
    dx = pd.DataFrame([(col, df[col].dtype) for col in df.columns], columns=['column', 'datatype'])
    assert ds.shape == dx.shape

    # merged columns and datatypes
    dc = pd.merge(left=ds, right=dx, on='column', how='inner', suffixes=('_sql', '_pandas'))

    # convert
    for _, (col, ts, tx) in dc.iterrows():

        if ts in ['text', 'uuid', 'geometry']:
            df[col] = df[col].astype(pd.StringDtype())

        elif ts == 'int4':
            df[col] = df[col].astype(pd.Int32Dtype())

        elif ts == 'int8':
            df[col] = df[col].astype(pd.Int64Dtype())

        elif ts in ['numeric', 'float8']:
            df[col] = df[col].astype(pd.Float64Dtype())

        elif ts == 'interval':
            df[col] = np.array([pd.Timedelta(x).total_seconds() for x in df[col].values], dtype=np.float64)

        elif ts in ['_int4', '_int8', 'int8range', '_float8', '_numeric', 'tsrange']:
            df[col] = df[col].astype('object')

        elif ts == 'bool':
            df[col] = df[col].astype(pd.BooleanDtype())

        else:
            assert ts == 'timestamp'
            df[col] = df[col].astype('datetime64[ns]')

    return df

# distributed queries
def distributed_data_extraction(dataset, datadir, df, xid, n, distributed=True, assert_new=True):
    """
    extract raw data to a Parquet dataset using distributed queries
    dataset - Parquet dataset name
    datadir - Parquet dataset location
    df - DataFrame with columns time0, time1, and <xid> (eg VehicleId)
    xid - Parquet dataset partition
    n - number of partitions to include in a single query
    distributed - run queries in parallel
    assert_new - assert that datadir does not already exist, may be False to update an existing dataset
    """

    from dask import delayed, compute
    from dask.diagnostics import ProgressBar

    # validate
    assert isinstance(dataset, str)
    assert os.path.isdir(datadir)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(xid, str) and all([x in df.columns for x in [xid, 'time0', 'time1']])
    assert isinstance(n, int)

    # create Parqet dataset path
    path = os.path.join(datadir, f'{dataset}.parquet')
    if assert_new:
        assert not os.path.isdir(path)
        os.mkdir(path)

    # reshape indices of df as a 2d-array with n rows
    rows = int(np.ceil(df.shape[0] / n))
    xs = np.hstack((df.index, np.tile(None, rows * n - df.shape[0]))).reshape(rows, n)

    # split df by xs
    dxs = []
    for x in xs:
        x = np.array([xi for xi in x if xi is not None])
        dxs.append(df.loc[x].reset_index(drop=True))

    # distributed data extraction
    if distributed:
        print(f'distributed {dataset} data extraction')
        with ProgressBar():
            compute(*[delayed(globals()[f'process_{dataset}'])(dx, path) for dx in dxs])
    else:
        [globals()[f'process_{dataset}'](dx, path) for dx in tqdm(dxs, desc=f'{dataset} data extraction')]

def time_bounds_dataframe(df, xid):
    """
    min/max time bounds grouped by 'xid', sorted by window duration
    """

    # validate
    assert xid in ['VehicleId', 'EventRecorderId', 'DeviceId']
    assert all([x in df.columns for x in [xid, 'time0', 'time1']])

    # DataFrame based on full window by xid
    dx = pd.merge(
        left=df.groupby(xid, group_keys=False)['time0'].min().to_frame().reset_index(),
        right=df.groupby(xid, group_keys=False)['time1'].max().to_frame().reset_index(),
        on=xid, how='inner')

    # sort by window duration and return
    sx = np.argsort([x.total_seconds() for x in (dx['time1'] - dx['time0'])])
    return dx.loc[sx].reset_index(drop=True)

def process_events(df, path):
    """
    extract raw events data for a set of vehicles to local memory, then write to Parquet dataset
    """

    # EDW connection and GPS processing objects
    edw = get_conn('edw')
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform

    # build query for raw events over full time range
    vstr = ','.join([f"""'{x}'""" for x in df['VehicleId']])
    query = f"""
        SELECT
            E.VehicleId,
            E.RecordDate,
            E.Latitude,
            E.Longitude,
            E.EventId,
            E.EventRecorderId,
            E.EventRecorderFileId,
            E.SpeedAtTrigger,
            E.EventTriggerTypeId AS NameId,
            E.EventTriggerSubTypeId AS SubId,
            T.Name,
            E.EventFilePath,
            E.EventFileName,
            CASE WHEN E.EventFileName LIKE 'event%'
                THEN 'lytx-amlnas-us-west-2'
                ELSE NULL
            END AS s3_bucket_name,
            CASE WHEN E.EventFileName LIKE 'event%'
                THEN REPLACE(REPLACE(CONCAT(E.EventFilePath, '/', CAST(CONCAT('<t>', REPLACE(E.EventFileName, '_', '</t><t>'), '</t>') AS XML).value('/t[2]', 'varchar(100)'), '/', E.EventFileName), '\\\\drivecam.net', 'dce-files'), '\\', '/')
                ELSE NULL
            END AS s3_key,
            CASE WHEN E.EventFileName LIKE 'event%'
                THEN
                REPLACE(REPLACE(CONCAT(E.EventFilePath, '/', CAST(CONCAT('<t>', REPLACE(E.EventFileName, '_', '</t><t>'), '</t>') AS XML).value('/t[2]', 'varchar(100)'), '/', E.EventFileName), '\\\\drivecam.net', 's3://lytx-amlnas-us-west-2/dce-files'), '\\', '/')
            ELSE NULL
            END AS s3_uri
        FROM flat.Events AS E
            LEFT JOIN hs.EventTriggerTypes_i18n AS T
            ON T.Id = E.EventTriggerTypeId
        WHERE E.Deleted=0
        AND E.RecordDate BETWEEN '{df['time0'].min()}' AND '{df['time1'].max()}'
        AND E.VehicleId IN ({vstr})"""

    # query events data for all vehicles in df over full time range
    dx = pd.read_sql_query(sa.text(query), edw).sort_values(['VehicleId', 'RecordDate']).reset_index(drop=True)
    if dx.size == 0:
        return

    # enforce datatypes
    dx['SpeedAtTrigger'] = dx['SpeedAtTrigger'].astype('float')

    # create 'eventdatetime' and 'TS_SEC'
    if dx['RecordDate'].dtype == np.object_:
        dx['eventdatetime'] = [datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f') for x in dx.pop('RecordDate')]
    else:
        dx['eventdatetime'] = dx.pop('RecordDate')
    dx['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['eventdatetime']]

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['VehicleId'] == row['VehicleId']
        c1 = dx['eventdatetime'] >= row['time0']
        c2 = dx['eventdatetime'] <= row['time1']
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    dx = dx.loc[keep].reset_index(drop=True)

    # convert GPS coords
    lon = dx.pop('Longitude').values
    lat = dx.pop('Latitude').values
    dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
    dx['longitude_gps'] = lon
    dx['latitude_gps'] = lat

    # validation
    assert dx.duplicated().values.sum() == 0
    assert all(dx.groupby('VehicleId', group_keys=False)['TS_SEC'].apply(lambda x: all(np.sort(x) == x)))

    # write to Parquet dataset partitioned by vehicle-id
    dx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

def process_gps(df, path):
    """
    extract raw gps data for a set of vehicles to local memory, then write to Parquet dataset
    """

    # snowflake connection and GPS processing objects
    snow = get_conn('snowflake')
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform

    # build query for gps data over full time range
    vstr_lower = ','.join([f"""'{x.lower()}'""" for x in df['VehicleId']])
    vstr_upper = ','.join([f"""'{x.upper()}'""" for x in df['VehicleId']])
    time0 = int((df['time0'].min() - datetime(1970, 1, 1)).total_seconds())
    time1 = int((df['time1'].max() - datetime(1970, 1, 1)).total_seconds())
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
        AND (VEHICLE_ID IN ({vstr_lower}) OR VEHICLE_ID IN ({vstr_upper}))"""

    # query gps data
    dx = pd.read_sql_query(query, snow).sort_values(['VEHICLEID', 'TS_SEC']).reset_index(drop=True)
    if dx.size == 0:
        return

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['VEHICLEID'] == row['VehicleId']
        c1 = dx['TS_SEC'] >= (row['time0'] - datetime(1970, 1, 1)).total_seconds()
        c2 = dx['TS_SEC'] <= (row['time1'] - datetime(1970, 1, 1)).total_seconds()
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    dx = dx.loc[keep].reset_index(drop=True)

    # convert GPS coords
    lon = dx.pop('LONGITUDE').values
    lat = dx.pop('LATITUDE').values
    dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
    dx['longitude_gps'] = lon
    dx['latitude_gps'] = lat

    # remove full and partial duplicate rows
    dx = dx.loc[~dx.duplicated()].reset_index(drop=True)
    dx = dx.loc[~dx.duplicated(['TS_SEC', 'VEHICLEID'])].reset_index(drop=True)

    # drop vehicles with one gps record only
    ok = dx.groupby('VEHICLEID')['TS_SEC'].count()
    ok = ok.loc[ok > 1].index.to_frame().reset_index(drop=True)
    dx = pd.merge(left=dx, right=ok, on='VEHICLEID', how='inner')

    # validation and convert datatypes
    assert all(dx.groupby('VEHICLEID')['TS_SEC'].apply(lambda x: pd.unique(x).size == x.size))
    assert all(dx.groupby('VEHICLEID')['TS_SEC'].apply(lambda x: all(np.sort(x) == x)))
    assert all(dx.groupby('VEHICLEID')['TS_SEC'].count() > 1)
    dx['TS_USEC'] = dx['TS_USEC'].astype('float')

    # write to Parquet dataset partitioned by vehicle-id
    dx['VehicleId'] = dx.pop('VEHICLEID').values
    dx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

def process_behaviors(df, path):
    """
    extract raw behaviors data for a set of vehicles to local memory, then write to Parquet dataset
    """

    # EDW connection and GPS processing objects
    edw = get_conn('edw')
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform


    # build query for raw behaviors over full time range
    vstr = ','.join([f"""'{x}'""" for x in df['VehicleId']])
    query = f"""
        WITH behaviors AS (
            SELECT
                VehicleId,
                RecordDate,
                Latitude,
                Longitude,
                EventId,
                EventRecorderId,
                value AS BehaviorId,
                CASE WHEN EventFileName LIKE 'event%'
                    THEN 'lytx-amlnas-us-west-2'
                    ELSE NULL
                END AS s3_bucket_name,
                CASE WHEN EventFileName LIKE 'event%'
                    THEN REPLACE(REPLACE(CONCAT(EventFilePath, '/', CAST(CONCAT('<t>', REPLACE(EventFileName, '_', '</t><t>'), '</t>') AS XML).value('/t[2]', 'varchar(100)'), '/', EventFileName), '\\\\drivecam.net', 'dce-files'), '\\', '/')
                    ELSE NULL
                END AS s3_key,
                CASE WHEN EventFileName LIKE 'event%'
                    THEN
                    REPLACE(REPLACE(CONCAT(EventFilePath, '/', CAST(CONCAT('<t>', REPLACE(EventFileName, '_', '</t><t>'), '</t>') AS XML).value('/t[2]', 'varchar(100)'), '/', EventFileName), '\\\\drivecam.net', 's3://lytx-amlnas-us-west-2/dce-files'), '\\', '/')
                ELSE NULL
                END AS s3_uri
            FROM flat.Events
                CROSS APPLY STRING_SPLIT(COALESCE(BehaviourStringIds, '-1'), ',')
            WHERE value <> -1
            AND Deleted = 0
            AND RecordDate BETWEEN '{df['time0'].min()}' AND '{df['time1'].max()}'
            AND VehicleId IN ({vstr}))
        SELECT
            behaviors.VehicleId,
            behaviors.RecordDate,
            behaviors.Latitude,
            behaviors.Longitude,
            behaviors.EventId,
            behaviors.EventRecorderId,
            behaviors.BehaviorId AS NameId,
            hsb.Name,
            behaviors.s3_bucket_name,
            behaviors.s3_key,
            behaviors.s3_uri
        FROM behaviors
            LEFT JOIN hs.Behaviors_i18n AS hsb
            ON behaviors.BehaviorId = hsb.Id"""

    # query behaviors data for all vehicles in df over full time range
    dx = pd.read_sql_query(sa.text(query), edw).sort_values(['VehicleId', 'RecordDate']).reset_index(drop=True)
    if dx.size == 0:
        return

    # create 'eventdatetime' and 'TS_SEC'
    if dx['RecordDate'].dtype == np.object_:
        dx['eventdatetime'] = [datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f') for x in dx.pop('RecordDate')]
    else:
        dx['eventdatetime'] = dx.pop('RecordDate')
    dx['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['eventdatetime']]

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['VehicleId'] == row['VehicleId']
        c1 = dx['eventdatetime'] >= row['time0']
        c2 = dx['eventdatetime'] <= row['time1']
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    dx = dx.loc[keep].reset_index(drop=True)

    # convert GPS coords
    lon = dx.pop('Longitude').values
    lat = dx.pop('Latitude').values
    dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
    dx['longitude_gps'] = lon
    dx['latitude_gps'] = lat

    # validation
    assert dx.duplicated().values.sum() == 0
    assert all(dx.groupby('VehicleId')['TS_SEC'].apply(lambda x: all(np.sort(x) == x)))

    # write to Parquet dataset partitioned by vehicle-id
    dx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

def process_trips(df, path):
    """
    extract raw trips data for a set of vehicles to local memory, then write to Parquet dataset
    """

    # EDW connection and GPS processing objects
    edw = get_conn('edw')
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform

    # build query for raw trips over full time range
    vstr = ','.join([f"""'{x}'""" for x in df['VehicleId']])
    query = f"""
        SELECT
            VehicleId,
            StartPositionTimeUTC AS time0,
            EndPositionTimeUTC AS time1,
            Distance,
            TripPointCount
        FROM gps.Trips
        WHERE StartPositionTimeUTC > '{df['time0'].min()}'
        AND EndPositionTimeUTC < '{df['time1'].max()}'
        AND VehicleId IN ({vstr})"""

    # query trips
    dx = pd.read_sql_query(sa.text(query), edw).sort_values(['VehicleId', 'time0']).reset_index(drop=True)
    if dx.size == 0:
        return

    # format 'time0'
    if dx['time0'].dtype == np.object_:
        dx['time0'] = [datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f') for x in dx.pop('time0')]
    else:
        dx['time0'] = dx.pop('time0')
    dx['TS_SEC0'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['time0']]

    # format 'time1'
    if dx['time1'].dtype == np.object_:
        dx['time1'] = [datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f') for x in dx.pop('time1')]
    else:
        dx['time1'] = dx.pop('time1')
    dx['TS_SEC1'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['time1']]

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['VehicleId'] == row['VehicleId']
        c1 = dx['time1'] > row['time0']
        c2 = dx['time0'] < row['time1']
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    dx = dx.loc[keep].reset_index(drop=True)
    dx = dx.loc[~dx.duplicated()].reset_index(drop=True)

    # validation
    assert dx.duplicated().values.sum() == 0
    assert all(dx.groupby('VehicleId')['TS_SEC0'].apply(lambda x: all(np.sort(x) == x)))

    # write to Parquet dataset partitioned by vehicle-id
    dx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

def process_dce_scores(df, path):
    """
    extract raw dce-scores data for a set of vehicles to local memory, then write to Parquet dataset
    """

    # EDW connection and GPS processing objects
    edw = get_conn('edw')
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform

    # build query for dce scores over full time range
    vstr = ','.join([f"""'{x}'""" for x in df['VehicleId']])
    # query = f"""
    #     SELECT
    #         E.VehicleId,
    #         E.RecordDate,
    #         E.EventId,
    #         E.Latitude,
    #         E.Longitude,
    #         RE.ModelValue
    #     FROM flat.Events AS E
    #         INNER JOIN ml.Dce AS DCE
    #         ON DCE.EventId = E.EventId
    #         INNER JOIN ml.ModelRequest AS RQ
    #         ON RQ.DceId = DCE.DceId
    #         INNER JOIN ml.ModelResponse AS RE
    #         ON RE.ModelRequestId = RQ.ModelRequestId
    #         AND RE.ModelKey = 'collision'
    #         AND RQ.ModelId = 4
    #     WHERE E.Deleted=0
    #     AND E.EventTriggerTypeId in (27,30,31)
    #     AND E.RecordDate BETWEEN '{df['time0'].min()}' AND '{df['time1'].max()}'
    #     AND E.VehicleId IN ({vstr})"""
    query = f"""
        SELECT *
        FROM Sandbox.aml.[dce-scores]
        WHERE VehicleId IN ({vstr})"""

    # query dce scores data for all vehicles in df over full time range
    dx = pd.read_sql_query(sa.text(query), edw).sort_values(['VehicleId', 'RecordDate']).reset_index(drop=True)
    if dx.size == 0:
        return

    # create 'eventdatetime' and 'TS_SEC'
    if dx['RecordDate'].dtype == np.object_:
        dx['eventdatetime'] = [datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f') for x in dx.pop('RecordDate')]
    else:
        dx['eventdatetime'] = dx.pop('RecordDate')
    dx['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['eventdatetime']]

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['VehicleId'] == row['VehicleId']
        c1 = dx['eventdatetime'] >= row['time0']
        c2 = dx['eventdatetime'] <= row['time1']
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    dx = dx.loc[keep].reset_index(drop=True)

    # convert GPS coords
    lon = dx.pop('Longitude').values
    lat = dx.pop('Latitude').values
    dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
    dx['longitude_gps'] = lon
    dx['latitude_gps'] = lat

    # validation
    dx = dx.loc[~dx.duplicated()].reset_index(drop=True)
    assert all(dx.groupby('VehicleId')['TS_SEC'].apply(lambda x: all(np.sort(x) == x)))

    # write to Parquet dataset partitioned by vehicle-id
    dx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

def process_triggers(df, path):
    """
    extract raw triggers data for a set of vehicles to local memory, then write to Parquet dataset
    """

    # EDW connection and GPS processing objects
    edw = get_conn('edw')
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform

    # build query for triggers over full time range using stored procedure
    ers = ','.join([f"""{x}""" for x in df['EventRecorderId'].values])
    tmin = df['time0'].min().strftime('%m-%d-%Y %H:%M:%S')
    tmax = df['time1'].max().strftime('%m-%d-%Y %H:%M:%S')
    query = f"""
        EXEC Sandbox.aml.Sel_Triggers
        @IdString = '{ers}',
        @StartDate = '{tmin}',
        @EndDate = '{tmax}',
        @TimeWindowToProcessInMinutes = 60,
        @RangeFrontPaddingDays = 7,
        @RangeBackPaddingDays = 30,
        @ResumeOperation=0"""

    # build query for triggers over full time range
    # ers = ','.join([f"""'{x}'""" for x in df['EventRecorderId'].values])
    # tmin = df['time0'].min().strftime('%m-%d-%Y %H:%M:%S')
    # tmax = df['time1'].max().strftime('%m-%d-%Y %H:%M:%S')
    # query = f"""
    #     SELECT
    #         ERF.EventRecorderId,
    #         ERF.EventRecorderFileId,
    #         ERF.CreationDate,
    #         ERF.FileName,
    #         ERF.EventTriggerTypeId,
    #         ERFT.TriggerTime,
    #         ERFT.Position.Lat AS Latitude,
    #         ERFT.Position.Long as Longitude,
    #         ERFT.ForwardExtremeAcceleration,
    #         ERFT.SpeedAtTrigger,
    #         ERFT.PostedSpeedLimit
    #     FROM hs.EventRecorderFiles AS ERF
    #         LEFT JOIN hs.EventRecorderFileTriggers AS ERFT
    #         ON ERFT.EventRecorderFileId = ERF.EventRecorderFileId
    #     WHERE ERF.EventRecorderId IN ({ers})
    #     AND ERFT.TriggerTime BETWEEN '{tmin}' AND '{tmax}'"""

    # run query and handle null case
    dx = pd.read_sql_query(sa.text(query), edw).sort_values(['EventRecorderId', 'TriggerTime']).reset_index(drop=True)
    if dx.size == 0:
        return

    # create 'eventdatetime' and 'TS_SEC'
    if dx['TriggerTime'].dtype == np.object_:
        dx['eventdatetime'] = [datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f') for x in dx.pop('TriggerTime')]
    else:
        dx['eventdatetime'] = dx.pop('TriggerTime')
    dx['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['eventdatetime']]

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['EventRecorderId'] == row['EventRecorderId']
        c1 = dx['eventdatetime'] >= row['time0']
        c2 = dx['eventdatetime'] <= row['time1']
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    dx = dx.loc[keep].reset_index(drop=True)

    # convert GPS coords
    lon = dx.pop('Longitude').values
    lat = dx.pop('Latitude').values
    dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
    dx['longitude_gps'] = lon
    dx['latitude_gps'] = lat

    # validation and convert datatypes
    dx = dx.loc[~dx.duplicated()].reset_index(drop=True)
    assert all(dx.groupby('EventRecorderId')['TS_SEC'].apply(lambda x: all(np.sort(x) == x)))
    dx['PostedSpeedLimit'] = dx['PostedSpeedLimit'].astype('float')

    # write to Parquet dataset partitioned by EventRecorderId
    ers = pd.unique(dx['EventRecorderId'])
    rows = int(np.ceil(ers.size / 400))
    xs = np.hstack((ers, np.tile(None, rows * 400 - ers.size))).reshape(rows, 400)
    for row in xs:
        xr = np.array([x for x in row if x is not None])
        xx = dx.loc[dx['EventRecorderId'].isin(xr)].reset_index(drop=True)
        xx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['EventRecorderId'])

def process_nodes(df, path):
    """
    extract raw IMN nodes data to local memory then to a Parquet dataset
    """

    # labs postgres database connection
    lab = get_conn('lytx-lab')

    # build query for nodes over full time range
    vstr = ','.join([f"""'{x}'""" for x in df['VehicleId']])
    query = f"""
        SELECT
            vehicle_id,
            node_id,
            node_timestamp__min - node_min_time_interval_from_prev_gps_point_for_idle AS time0,
            node_timestamp__max AS time1,
            ST_ASTEXT(geomnodearea3857) AS polygon,
            ST_ASTEXT(geomnodearea) AS polygon_gps,
            ST_Y(ST_CENTROID(geomnodeareapoints)) AS latitude,
            ST_X(ST_CENTROID(geomnodeareapoints)) AS longitude,
            node_gps_point_count AS gps_records
        FROM mobility_network.distfreighttruckingcrash_08012021_11302021_mobnet_node
        WHERE node_timestamp__min BETWEEN '{df['time0'].min()}' AND '{df['time1'].max()}'
        AND node_timestamp__max BETWEEN '{df['time0'].min()}' AND '{df['time1'].max()}'
        AND vehicle_id IN ({vstr})"""

    # query for all vehicles in df over full time range
    dx = pd.read_sql_query(query, lab)
    dx['VehicleId'] = dx.pop('vehicle_id')
    dx['NodeId'] = dx.pop('node_id')
    dx = dx.sort_values(['VehicleId', 'time0']).reset_index(drop=True)[
        ['VehicleId', 'NodeId', 'time0', 'time1', 'latitude', 'longitude', 'polygon', 'polygon_gps', 'gps_records']]
    if dx.size == 0:
        return

    # create TS_SEC columns
    dx['TS_SEC0'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['time0']]
    dx['TS_SEC1'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['time1']]

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['VehicleId'] == row['VehicleId']
        c1 = dx['time1'] > row['time0']
        c2 = dx['time0'] < row['time1']
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    keep = np.unique(keep)
    dx = dx.loc[keep].reset_index(drop=True)
    dx = dx.loc[~dx.duplicated()].reset_index(drop=True)

    # convert latitude and longitude
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
    lon = dx.pop('longitude').values
    lat = dx.pop('latitude').values
    dx['longitude'], dx['latitude'] = transform(xx=lon, yy=lat)
    dx['longitude_gps'] = lon
    dx['latitude_gps'] = lat

    # validation
    assert dx.duplicated().values.sum() == 0
    assert all(dx.groupby('VehicleId')['TS_SEC0'].apply(lambda x: all(np.sort(x) == x)))

    # write to Parquet dataset partitioned by vehicle-id
    dx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

def process_links(df, path):
    """
    extract raw IMN links data to local memory then to a Parquet dataset
    """

    # labs postgres database connection
    lab = get_conn('lytx-lab')

    # build query for links over full time range
    vstr = ','.join([f"""'{x}'""" for x in df['VehicleId']])
    query = f"""
        SELECT
            link_id,
            vehicle_id,
            ST_X(prevnode_onenternode_geomcurrposition) AS lon0,
            ST_Y(prevnode_onenternode_geomcurrposition) AS lat0,
            ST_X(currnode_onenternode_geomcurrposition) AS lon1,
            ST_Y(currnode_onenternode_geomcurrposition) AS lat1,
            prevnode_onenternode_node_id AS node0,
            currnode_onenternode_node_id AS node1,
            prevnode_onenternode_timestamp AS time0,
            currnode_onenternode_timestamp AS time1,
            link_gps_segment_length_meters AS link_meters
        FROM mobility_network.distfreighttruckingcrash_08012021_11302021_mobnet_link
        WHERE vehicle_id IN ({vstr})
        AND link_gps_segment_length_meters > 0
        AND prevnode_onenternode_node_id <> currnode_onenternode_node_id"""

    # query for all vehicles in df over full time range
    dx = pd.read_sql_query(query, lab)
    dx['VehicleId'] = dx.pop('vehicle_id')
    dx = dx.sort_values(['VehicleId', 'time0']).reset_index(drop=True)
    if dx.size == 0:
        return

    # create TS_SEC columns
    dx['TS_SEC0'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['time0']]
    dx['TS_SEC1'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in dx['time1']]

    # filter dx by time bounds in df
    keep = np.array([])
    for _, row in df.iterrows():
        c0 = dx['VehicleId'] == row['VehicleId']
        c1 = dx['time1'] > row['time0']
        c2 = dx['time0'] < row['time1']
        keep = np.hstack((keep, dx.loc[c0 & c1 & c2].index.values))
    keep = np.unique(keep)
    dx = dx.loc[keep].reset_index(drop=True)
    dx = dx.loc[~dx.duplicated()].reset_index(drop=True)

    # convert latitude and longitude
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
    lon0 = dx.pop('lon0').values
    lon1 = dx.pop('lon1').values
    lat0 = dx.pop('lat0').values
    lat1 = dx.pop('lat1').values
    dx['lon0'], dx['lat0'] = transform(xx=lon0, yy=lat0)
    dx['lon1'], dx['lat1'] = transform(xx=lon1, yy=lat1)
    dx['lon0_gps'] = lon0
    dx['lon1_gps'] = lon1
    dx['lat0_gps'] = lat0
    dx['lat1_gps'] = lat1

    # validation
    assert dx.duplicated().values.sum() == 0
    assert all(dx.groupby('VehicleId')['TS_SEC0'].apply(lambda x: all(np.sort(x) == x)))

    # write to Parquet dataset partitioned by vehicle-id
    dx.to_parquet(path=path, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

# data coverage metrics
def get_time_bounds(spark, dcm, src, xid='VehicleId', ts_min='TS_SEC', ts_max='TS_SEC'):
    """
    return DataFrame of time bounds for data src
    """

    # validate
    assert src in [x.name for x in spark.catalog.listTables()]

    # bounds for data source
    bounds = spark.sql(f"""
        SELECT
            {xid},
            COUNT(*) AS n_records,
            MIN({ts_min}) AS tmin_epoch,
            MAX({ts_max}) AS tmax_epoch
        FROM {src}
        GROUP BY {xid}""").toPandas()
    bounds['src'] = src
    bounds['tmin'] = [datetime.utcfromtimestamp(x) for x in bounds['tmin_epoch']]
    bounds['tmax'] = [datetime.utcfromtimestamp(x) for x in bounds['tmax_epoch']]
    bounds['days'] = (bounds['tmax_epoch'] - bounds['tmin_epoch']) / (3600 * 24)
    assert pd.unique(bounds[xid]).size == bounds.shape[0]
    assert len(set(dcm[xid].values).intersection(bounds[xid].values)) <= pd.unique(dcm[xid]).size

    # merge with time boundaries in dcm and validate
    dxs = pd.merge(
        left=dcm.groupby(xid)['time0'].min().to_frame().reset_index(drop=False),
        right=dcm.groupby(xid)['time1'].max().to_frame().reset_index(drop=False),
        on=xid, how='inner')
    bounds = pd.merge(dxs, bounds, on=xid, how='inner')

    return bounds.sort_values('days').reset_index(drop=True)

def records_per_day_vid(datadir, spark, dcm, src, xid='VehicleId', ts='TS_SEC'):
    """
    record count per day per xid based on dcm and data source
    """

    # validate
    assert src in [x.name for x in spark.catalog.listTables()]
    assert xid in ['VehicleId', 'EventRecorderId']

    # daily evaluation DataFrame
    sdf = spark.createDataFrame(dcm.loc[~dcm['oversampled'], [xid, 'time0', 'time1']])
    sdf.createOrReplaceTempView('sdf')
    def dev_xid(pdf):

        # validate
        assert pd.unique(pdf[xid]).size == 1
        xv = pd.unique(pdf[xid])[0]
        pdf = pdf.sort_values('time0').reset_index(drop=True)

        # unique daily evaluations as a DataFrame for xid
        dev = pd.DataFrame(data={'day': np.sort(np.unique(np.hstack((
            [pd.date_range(start=time0, end=time1, freq='D')[:-1] for _, (_, time0, time1) in pdf.iterrows()]))))})
        dev['xid'] = xv

        # source data for xid, null case
        fn = glob(os.path.join(datadir, f'{src}.parquet', f'{xid}={xv}', '*.parquet'))
        assert len(fn) <= 1
        if len(fn) == 0:
            dev['rc'] = 0
            dev['rc'] = dev['rc'].astype('int')
            return dev

        # non-null case, count records by day in source data based on dev
        fn = fn[0]
        df = pq.ParquetFile(fn).read().to_pandas().sort_values(ts).reset_index(drop=True)
        time0 = np.array([(pd.Timestamp(x) - pd.Timestamp('1970-1-1')).total_seconds() for x in dev['day'].values])
        time1 = np.array([(pd.Timestamp(x) + pd.Timedelta(days=1) - pd.Timestamp('1970-1-1')).total_seconds() for x in dev['day'].values])
        time = df[ts].values
        dev['rc'] = np.array([np.logical_and(time >= t0, time <= t1).sum() for (t0, t1) in zip(time0, time1)])
        dev['rc'] = dev['rc'].astype('int')
        return dev

    schema = StructType([
        StructField('xid', StringType(), nullable=False),
        StructField('day', TimestampType(), nullable=False),
        StructField('rc', DoubleType(), nullable=False)])
    dev = sdf.groupby(xid).applyInPandas(dev_xid, schema=schema)
    dev.createOrReplaceTempView('dev')
    # debug
    # pdf = spark.sql(f"""SELECT * FROM sdf WHERE EventRecorderId='3500FFFF-6B4A-A8A7-F4FB-8E63F0480000'""").toPandas()
    # dev_xid(pdf)

    # records per xid per day DataFrame
    dx = spark.sql(f"""SELECT day, COUNT(*) AS nx, SUM(rc) AS nr FROM dev GROUP BY day ORDER BY day""").toPandas()
    assert pd.isnull(dx).values.sum() == 0
    assert dx['nx'].max() <= pd.unique(dcm['VehicleId']).size
    dx['nr'] = dx['nr'].astype('int')
    dx['nrx'] = dx['nr'] / dx['nx']

    return dx

def get_dce_scores_events_coverage_dataframe(spark):
    """
    evaluate coverage of dce scores relative to accelerometer events
    - based on a 30-day window, evaluated at 7-day intervals
    """

    # validate
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('events' in tables) and ('dce_scores' in tables)

    # build evaluation table covering span of events
    dw = spark.sql(f'SELECT MIN(eventdatetime) AS tmin, MAX(eventdatetime) AS tmax FROM events').toPandas()
    tmin = pd.Timestamp(dw['tmin'].loc[0].date())
    tmax = pd.Timestamp(dw['tmax'].loc[0].date()) - pd.Timedelta(days=30)
    time0 = pd.date_range(tmin, tmax, freq=pd.Timedelta(days=7))
    dw = pd.DataFrame({'time0': time0, 'time1': time0 + pd.Timedelta(days=30)}).reset_index(drop=False)
    spark.createDataFrame(dw).createOrReplaceTempView('dw')

    # count of accelerometer events for each row of evaluation table
    df0 = spark.sql(f"""
        SELECT
            dw.index,
            dw.time0,
            dw.time1,
            COUNT(*) AS n_events
        FROM events
            INNER JOIN dw
            ON events.eventdatetime BETWEEN dw.time0 AND dw.time1
        WHERE events.NameId IN (27,30,31)
        GROUP BY dw.index, dw.time0, dw.time1
        ORDER BY dw.index""").toPandas()

    # count of dce-scores for each row of evaluation table
    df1 = spark.sql(f"""
        SELECT
            dw.index,
            dw.time0,
            dw.time1,
            COUNT(*) AS n_scores
        FROM dce_scores
            INNER JOIN dw
            ON dce_scores.eventdatetime BETWEEN dw.time0 AND dw.time1
        GROUP BY dw.index, dw.time0, dw.time1
        ORDER BY dw.index""").toPandas()

    # merge and get coverage
    df = pd.merge(df0, df1, on=['index', 'time0', 'time1'], how='inner')
    del df['index']
    df['coverage'] = 100 * df['n_scores'] / df['n_events']
    return df

def get_triggers_events_coverage_dataframe(spark):
    """
    get subset of events that are also triggers, resolved by TriggerTypeId
    """
    nte = spark.sql(f"""
        WITH
            event_counts AS (
                SELECT
                    NameId,
                    COUNT(*) as ne
                FROM events
                GROUP BY NameId),
            event_trigger_counts AS (
                SELECT
                    E.NameId,
                    COUNT(*) AS nte
                FROM events AS E
                    INNER JOIN triggers AS T
                    ON E.EventRecorderFileId = T.EventRecorderFileId
                GROUP BY E.NameId)
            SELECT
                EC.NameId,
                EC.ne,
                ETC.nte
            FROM event_counts AS EC
                INNER JOIN event_trigger_counts AS ETC
                ON EC.NameId = ETC.NameId
            ORDER BY EC.NameId""").toPandas()
    nte['frac'] = nte['nte'] / nte['ne']
    return nte

# parquet dataset processing
def clean_up_crc(loc):
    """
    remove _SUCCESS and .crc files in a Parquet dataset to avoid Checksum exceptions
    """
    if os.path.isfile(os.path.join(loc, '_SUCCESS')):
        os.remove(os.path.join(loc, '_SUCCESS'))
    if glob(os.path.join(loc, '.*.crc')):
        assert len(glob(os.path.join(loc, '.*.crc'))) == 1
        os.remove(glob(os.path.join(loc, '.*.crc'))[0])
    xs = glob(os.path.join(loc, '*'))
    for x in xs:
        crc = glob(os.path.join(x, '.*.crc'))
        [os.remove(x) for x in crc]

def merge_parquet(spark, loc, num_per_iteration=1, xid='VehicleId', ts='TS_SEC'):
    """
    merge more than one parquet files to a single file by partition
    - faster than using repartition(1)
    """

    # get partitions as a 2d numpy array
    xids = np.array(glob(os.path.join(loc, '*'))).astype('object')
    assert all([(os.path.split(x)[1][:len(xid) + 1] == f'{xid}=') and (':' not in x) for x in xids])
    assert xids.size > 1
    rows = int(np.ceil(xids.size / num_per_iteration))
    xids = np.hstack((xids, np.tile(None, rows * num_per_iteration - xids.size)))
    xids = np.reshape(xids, (rows, num_per_iteration))

    # merge gps parquet files in a single partition
    def func(dx):
        fns = glob(os.path.join(dx, '*'))
        assert len(fns) > 0
        dfs = [pq.ParquetFile(fn).read().to_pandas() for fn in fns]
        df = pd.concat(dfs).sort_values(ts).reset_index(drop=True)
        df = df.loc[~df.duplicated(keep='first')].reset_index(drop=True)
        assert df.duplicated().sum() == 0
        df[xid] = os.path.split(dx)[1][len(xid) + 1:]
        df.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=[xid])
        [os.remove(fn) for fn in fns]

    # distribute func
    def outer(pdf):
        row = [x for x in pdf['xids'].values[0] if x is not None]
        for dx in row:
            func(dx)
        return pdf
    sdf = spark.createDataFrame(pd.DataFrame({'group': range(xids.shape[0]), 'xids': xids.tolist()}))
    # debug
    # sdf = sdf.toPandas()
    # outer(sdf.loc[sdf.index == 0])
    sdf.groupby('group').applyInPandas(outer, schema=sdf.schema).toPandas()

def count_all(spark, table):
    """
    return count of all records in table
    """
    assert table in [x.name for x in spark.catalog.listTables()]
    return spark.sql(f'SELECT COUNT(*) FROM {table}').toPandas().values.flatten()[0]

def reset_columns_parquet(spark, loc, keep=['TS_SEC', 'longitude', 'latitude', 'longitude_gps', 'latitude_gps']):
    """
    distributed reset of columns in a parquet dataset by VehicleId
    """

    def reset_vehicle(pdf):
        """
        reset columns for individual vehicle-id
        """

        # load data from parquet file
        assert pdf.shape[0] == 1
        vid = pdf['VehicleId'].values[0]
        fn = glob(os.path.join(loc, f'VehicleId={vid}', '*.parquet'))
        assert len(fn) == 1
        fn = fn[0]
        df = pq.ParquetFile(fn).read().to_pandas().sort_values('TS_SEC').reset_index(drop=True)
        df['VehicleId'] = vid

        # reset columns
        df = df[keep].copy()

        # rewrite data and remove original data, return pdf (Spark boilerplate)
        df.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])
        os.remove(fn)
        return pdf

    # distributed reset by vehicle-id
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps')
    dx = vids.groupby('VehicleId').applyInPandas(reset_vehicle_gps, schema=vids.schema).toPandas()

def validate_consistent_parquet_schema(spark, loc, src, xid):
    """
    distributed validation of consistent schema for a parquet dataset
    """

    def get_schema(pdf):
        """
        schema of parquet file
        """

        # load gps data from parquet file
        assert pdf.shape[0] == 1
        vid = pdf[xid].values[0]
        fn = glob(os.path.join(loc, f'{xid}={vid}', '*.parquet'))
        assert len(fn) == 1
        fn = fn[0]
        df = pq.ParquetFile(fn).read().to_pandas()

        # return schema as a DataFrame
        pdf = df.dtypes.to_frame().reset_index(drop=False)
        pdf.columns = ['column', 'datatype']
        pdf[xid] = vid
        pdf['datatype'] = [x.name for x in pdf['datatype']]

        return pdf[[xid, 'column', 'datatype']]

    # Spark DataFrame of schema by vehicle-id
    xids = spark.sql(f'SELECT DISTINCT {xid} FROM {src}')
    schema = StructType([
        StructField(xid, StringType(), True),
        StructField('column', StringType(), True),
        StructField('datatype', StringType(), True)])
    sdf = xids.groupby(xid).applyInPandas(get_schema, schema=schema)
    sdf.createOrReplaceTempView('sdf')
    # debug
    # xids = xids.toPandas()
    # get_schema(xids.loc[xids.index == 0])

    # validate consistent schema
    ds = spark.sql(f"""
        SELECT
            column,
            COUNT(DISTINCT(datatype)) AS num_types
            FROM sdf GROUP BY column""").toPandas()
    assert np.all(ds['num_types'] == 1)
    print(f'consistent schema for {src}')

# aggregated metrics from parquet datasets
def event_count_metrics(spark, span='all', suffix=None):
    """
    event count metrics
    - span expresses aggregation over full window, weekday/weekend/ or day of week
    - may not contain nulls
    """

    # validation
    assert span in ['all', 'weekday', 'weekend', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('events' in tables)

    # count of all events
    query0 = f"""
        SELECT
            dfv.rid,
            COUNT(*) AS nevents
        FROM dfv JOIN events
            ON events.VehicleId = dfv.VehicleId
            AND events.TS_SEC >= dfv.time0
            AND events.TS_SEC <= dfv.time1"""
    if span == 'weekday':
        query0 += f"""\nWHERE weekday"""
    elif span == 'weekend':
        query0 += f"""\nWHERE NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query0 += f"""\nWHERE day_of_week = '{span}'"""
    query0 += f"""\nGROUP BY dfv.rid"""
    df0 = spark.sql(query0).toPandas()

    # count of individual events other than accelerometer events
    query1 = f"""
        SELECT
            CAST(dfv.rid AS STRING) AS rid,
            CAST(events.NameId AS STRING) AS NameId,
            COUNT(*) AS nevents
        FROM dfv JOIN events
            ON events.VehicleId = dfv.VehicleId
            AND events.TS_SEC >= dfv.time0
            AND events.TS_SEC <= dfv.time1
        WHERE events.NameId <> 30"""
    if span == 'weekday':
        query1 += f"""\nAND weekday"""
    elif span == 'weekend':
        query1 += f"""\nAND NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query1 += f"""\nAND day_of_week = '{span}'"""
    query1 += f"""\nGROUP BY dfv.rid, events.NameId"""
    df1 = spark.sql(query1).groupBy('rid').pivot('NameId').max().toPandas()

    # count of accelerometer events resolved by subtype
    query2 = f"""
        SELECT
            CAST(dfv.rid AS STRING) AS rid,
            CONCAT(CAST(events.NameId AS STRING), '_', CAST(events.SubId AS STRING)) AS NameSubId,
            COUNT(*) AS nevents
        FROM dfv JOIN events
            ON events.VehicleId = dfv.VehicleId
            AND events.TS_SEC >= dfv.time0
            AND events.TS_SEC <= dfv.time1
        WHERE events.NameId = 30"""
    if span == 'weekday':
        query2 += f"""\nAND weekday"""
    elif span == 'weekend':
        query2 += f"""\nAND NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query2 += f"""\nAND day_of_week = '{span}'"""
    query2 += f"""\nGROUP BY dfv.rid, events.NameId, events.SubId"""
    df2 = spark.sql(query2).groupBy('rid').pivot('NameSubId').max().toPandas()

    # clean up
    subs = list(set([39, 51, 52, 53, 57]).intersection([int(x.split('30_')[1]) for x in df2.columns if x != 'rid']))
    subs = ['rid'] + [f'30_{x}' for x in subs]
    others = df2[[x for x in df2.columns if x not in subs and x != 'rid']].sum(axis=1).values
    df2 = df2[subs].copy()
    df2['30_others'] = others
    df1 = pd.merge(df1, df2, on='rid', how='outer').reset_index(drop=True)
    df1.columns = [x if x == 'rid' else f'nevents_{x}' for x in df1.columns]
    assert df0.shape[0] == df1.shape[0]
    df1['rid'] = df1['rid'].astype('int')
    df = pd.merge(left=df0, right=df1, on='rid', how='inner')
    df = df.fillna(0)
    assert all(df[[x for x in df.columns if '_' in x]].sum(axis=1).values == df['nevents'].values)
    assert pd.isnull(df).values.sum() == 0

    # update via suffix and return
    if suffix is not None:
        df.columns = [x if x == 'rid' else f'{x}_{suffix}' for x in df.columns]
    return df

def event_speed_metrics(spark, span='all', suffix=None):
    """
    event speed metrics
    - span expresses aggregation over full window, weekday/weekend/ or day of week
    - all columns except rid may contain nulls
    - relevant events include 12,15,16,18,26,30
    """

    # validation
    assert span in ['all', 'weekday', 'weekend', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('events' in tables)

    # average speed of all relevant events
    query0 = f"""
        SELECT
            dfv.rid,
            AVG(events.SpeedAtTrigger) AS avg_speed_events
        FROM dfv JOIN events
            ON events.VehicleId = dfv.VehicleId
            AND events.TS_SEC >= dfv.time0
            AND events.TS_SEC <= dfv.time1
        WHERE events.NameId IN (12,15,16,18,26,30)"""
    if span == 'weekday':
        query0 += f"""\nAND weekday"""
    elif span == 'weekend':
        query0 += f"""\nAND NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query0 += f"""\nAND day_of_week = '{span}'"""
    query0 += """\nGROUP BY dfv.rid"""
    df0 = spark.sql(query0).toPandas()

    # average speed by individual relevant event
    query1 = f"""
        SELECT
            dfv.rid,
            events.NameId,
            AVG(events.SpeedAtTrigger) AS avg_speed_events
        FROM dfv JOIN events
            ON events.VehicleId = dfv.VehicleId
            AND events.TS_SEC >= dfv.time0
            AND events.TS_SEC <= dfv.time1
        WHERE events.NameId IN (12,15,16,18,26,30)"""
    if span == 'weekday':
        query1 += f"""\nAND weekday"""
    elif span == 'weekend':
        query1 += f"""\nAND NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query1 += f"""\nAND day_of_week = '{span}'"""
    query1 += """\nGROUP BY dfv.rid, events.NameId"""
    df1 = spark.sql(query1).toPandas().pivot(index='rid', columns='NameId', values='avg_speed_events').reset_index(drop=False)

    # clean and merge
    df1.columns.name = None
    df1.columns = [f'avg_speed_events_{x}' if x != 'rid' else x for x in df1.columns]
    assert df0.shape[0] == df1.shape[0]
    df = pd.merge(left=df0, right=df1, on='rid', how='inner')
    assert pd.isnull(df['rid']).values.sum() == 0

    # update via suffix and return
    if suffix is not None:
        df.columns = [x if x == 'rid' else f'{x}_{suffix}' for x in df.columns]
    return df

def behavior_count_metrics(spark, span='all', suffix=None):
    """
    behavior count metrics
    - span expresses aggregation over full window, weekday/weekend/ or day of week
    - may not contain nulls
    """

    # validation
    assert span in ['all', 'weekday', 'weekend', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('behaviors' in tables)

    # count of all behaviors
    query0 = f"""
        SELECT
            dfv.rid,
            COUNT(*) AS nbehaviors
        FROM dfv JOIN behaviors
            ON behaviors.VehicleId = dfv.VehicleId
            AND behaviors.TS_SEC >= dfv.time0
            AND behaviors.TS_SEC <= dfv.time1"""
    if span == 'weekday':
        query0 += f"""\nWHERE weekday"""
    elif span == 'weekend':
        query0 += f"""\nWHERE NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query0 += f"""\nWHERE day_of_week = '{span}'"""
    query0 += f"""\nGROUP BY dfv.rid"""
    df0 = spark.sql(query0).toPandas()

    # count of individual behaviors
    query1 = f"""
        SELECT
            CAST(dfv.rid AS STRING) AS rid,
            CAST(behaviors.NameId AS STRING) AS NameId,
            COUNT(*) AS nbehaviors
        FROM dfv JOIN behaviors
            ON behaviors.VehicleId = dfv.VehicleId
            AND behaviors.TS_SEC >= dfv.time0
            AND behaviors.TS_SEC <= dfv.time1"""
    if span == 'weekday':
        query1 += f"""\nAND weekday"""
    elif span == 'weekend':
        query1 += f"""\nAND NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query1 += f"""\nAND day_of_week = '{span}'"""
    query1 += """\nGROUP BY dfv.rid, behaviors.NameId"""
    df1 = spark.sql(query1).groupBy('rid').pivot('NameId').max().toPandas()

    # clean up
    df1.columns = [x if x == 'rid' else f'nbehaviors_{x}' for x in df1.columns]
    assert df0.shape[0] == df1.shape[0]
    df1['rid'] = df1['rid'].astype('int')
    df = pd.merge(left=df0, right=df1, on='rid', how='inner')
    df = df.fillna(0)
    for col in [x for x in df1.columns if x != 'rid']:
        df.loc[pd.isnull(df[col]), col] = 0
    assert all(df[[x for x in df.columns if '_' in x]].sum(axis=1).values == df['nbehaviors'].values)
    assert pd.isnull(df).values.sum() == 0

    # update via suffix and return
    if suffix is not None:
        df.columns = [x if x == 'rid' else f'{x}_{suffix}' for x in df.columns]
    return df

def trip_metrics(spark):
    """
    trip metrics
    - may contain nulls in stddev column only
    """

    # validation
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('trips' in tables)

    # query and return
    query = f"""
        SELECT
            dfv.rid,
            COUNT(*) AS ntrips,
            SUM(trips.Distance) AS trips_distance_sum,
            AVG(trips.Distance) AS trips_distance_mean,
            STDDEV(trips.Distance) AS trips_distance_std
        FROM dfv JOIN trips
            ON trips.VehicleId = dfv.VehicleId
            AND trips.TS_SEC0 >= dfv.time0
            AND trips.TS_SEC1 <= dfv.time1
        GROUP BY dfv.rid"""
    df = spark.sql(query).toPandas()
    assert pd.isnull(df[['rid', 'ntrips', 'trips_distance_sum', 'trips_distance_mean']]).values.sum() == 0

    return df

def dce_score_metrics(spark):
    """
    dce score metrics
    - may contain nulls in coverage and stddev columns only
    """

    # validation
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('dce_scores' in tables)

    # query and return
    query = f"""
        WITH
            q1 AS (
                SELECT
                    dfv.rid,
                    COUNT(*) AS n_dce_scores,
                    SUM(dce_scores.ModelValue) AS dce_model_values_sum,
                    AVG(dce_scores.ModelValue) AS dce_model_values_avg,
                    MAX(dce_scores.ModelValue) AS dce_model_values_max,
                    STDDEV(dce_scores.ModelValue) AS dce_model_values_stddev
                FROM dfv JOIN dce_scores
                    ON dce_scores.VehicleId = dfv.VehicleId
                    AND dce_scores.TS_SEC >= dfv.time0
                    AND dce_scores.TS_SEC <= dfv.time1
                GROUP BY dfv.rid),
            q2 AS (
                SELECT
                    dfv.rid,
                    COUNT(*) AS n_events_accel
                FROM dfv JOIN events
                    ON events.VehicleId = dfv.VehicleId
                    AND events.TS_SEC >= dfv.time0
                    AND events.TS_SEC <= dfv.time1
                WHERE events.NameId IN (27,30,31)
                GROUP BY dfv.rid)
        SELECT
            q1.rid,
            q1.n_dce_scores / q2.n_events_accel AS dce_model_coverage,
            q1.dce_model_values_sum,
            q1.dce_model_values_avg,
            q1.dce_model_values_max,
            q1.dce_model_values_stddev
        FROM q1
            LEFT JOIN q2
            ON q1.rid = q2.rid
        """
    df = spark.sql(query).toPandas()
    assert pd.isnull(df[['rid', 'dce_model_values_sum', 'dce_model_values_avg', 'dce_model_values_max']]).values.sum() == 0
    return df

def gps_metrics(spark):
    """
    gps core usage metrics
    - all columns except rid may contain nulls
    """

    # validation
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('gps' in tables)

    # total days from first to last gps trackpoint
    vdg = spark.sql(f"""
        SELECT
            dfv.rid,
            dfv.VehicleId,
            (MAX(gps.TS_SEC) - MIN(gps.TS_SEC)) / (24*60*60) AS days
        FROM gps JOIN dfv
            ON gps.VehicleId = dfv.VehicleId
            AND gps.TS_SEC >= dfv.time0
            AND gps.TS_SEC <= dfv.time1
        GROUP BY dfv.rid, dfv.VehicleId, dfv.time0, dfv.time1
        ORDER BY dfv.rid""")
    vdg.createOrReplaceTempView('vdg')

    # days and miles covered by gps segments
    segments0 = spark.sql(f"""
        SELECT
            dfv.rid,
            gps.segmentId,
            SUM(gps.time_interval_sec) / (24*60*60) AS days,
            SUM(gps.distance_interval_miles) AS miles
        FROM gps JOIN dfv
            ON gps.VehicleId = dfv.VehicleId
            AND gps.TS_SEC >= dfv.time0
            AND gps.TS_SEC <= dfv.time1
        WHERE gps.segmentId IS NOT NULL
        GROUP BY dfv.rid, dfv.time0, dfv.time1, gps.segmentId""")
    segments0.createOrReplaceTempView('segments0')
    segments = spark.sql(f"""
        SELECT
            rid,
            SUM(days) AS days,
            SUM(miles) AS miles
        FROM segments0
        GROUP BY rid""")
    segments.createOrReplaceTempView('segments')

    # days and miles in motion covered by gps segments
    motion0 = spark.sql(f"""
        SELECT
            dfv.rid,
            gps.segmentId,
            SUM(gps.time_interval_sec) / (24*60*60) AS days,
            SUM(gps.distance_interval_miles) AS miles
        FROM gps JOIN dfv
            ON gps.VehicleId = dfv.VehicleId
            AND gps.TS_SEC >= dfv.time0
            AND gps.TS_SEC <= dfv.time1
        WHERE gps.segmentId IS NOT NULL
        AND gps.mph > 0.1
        GROUP BY dfv.rid, dfv.time0, dfv.time1, gps.segmentId""")
    motion0.createOrReplaceTempView('motion0')
    motion = spark.sql(f"""
        SELECT
            rid,
            SUM(days) AS days,
            SUM(miles) AS miles
        FROM motion0
        GROUP BY rid""")
    motion.createOrReplaceTempView('motion')

    # summary gps data
    df = spark.sql(f"""
        SELECT
            vdg.rid,
            segments.days AS gps_segments_days,
            motion.days AS gps_motion_days,
            segments.miles AS gps_miles,
            motion.miles AS gps_miles_motion,
            100 * motion.days / segments.days AS gps_percent_motion,
            segments.miles / segments.days AS gps_mpd
        FROM vdg
            LEFT JOIN segments
            ON segments.rid = vdg.rid
            LEFT JOIN motion
            ON motion.rid = vdg.rid
        ORDER BY vdg.rid""").toPandas()
    assert pd.isnull(df['rid']).values.sum() == 0

    return df

def gpse_metrics(spark, prefix=None):
    """
    enriched gps metrics
    """

    # validation
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('gps' in tables)

    def complete_query(query):
        query = query.rstrip(',')
        query += f"""
            FROM gps JOIN dfv
                ON gps.VehicleId = dfv.VehicleId
                AND gps.TS_SEC >= dfv.time0
                AND gps.TS_SEC <= dfv.time1
            WHERE gps.segmentId IS NOT NULL
            GROUP BY dfv.rid"""
        return query

    # travel duration metrics
    query1 = f"""SELECT dfv.rid,
        SUM(gps.gpse_time_interval_from_prev_gps_trackpoint / -3600) AS travel_duration_hours_sum,
        SUM(CASE WHEN
            ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS travel_duration_moving_hours_sum,
        SUM(CASE WHEN
            ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS travel_duration_idle_hours_sum"""

    # hotspot metrics
    query2 = f"""SELECT dfv.rid,
        SUM(gps.gpse_segment_hotspotcount) AS all_hotspots_entered_sum,
        SUM(gps.gpse_segment_hotspot_intersectioncount) AS hotspots_entered_intersection_sum,
        AVG(gps.gpse_segment_hotspot_incidentrate) AS hotspots_incidentrate_intersection_avg,
        AVG(gps.gpse_segment_hotspot_intersection_compexity_avg) AS hotspots_entered_intersection_complexity_avg,
        AVG(gps.gpse_crash_hotspots_rankgroup_severity_index_avg) AS hotspots_severity_index_avg,
        SUM(gps.gpse_crash_all_hotspots_entered_sum) AS hotspots_incidents_sum,"""
    for hs in [
            'animal',
            'bicyclist',
            'lowclearance',
            'intersection',
            'pedestrian',
            'slowing_traffic',
            'train',
            'turn_curve']:
        query2 += f"""
            SUM(gps.gpse_crash_{hs}_hotspots_entered) AS {hs}_hotspots_entered_sum,"""
    for hs in [
            'injury_incidents',
            'fatal_incidents',
            'pedestriansinvolved',
            'pedestriansinvolvedunder18',
            'cyclistsinvolved',
            'cyclistsinvolvedunder18']:
        query2 += f"""
            SUM(gps.gpse_crash_hotspots_{hs}_sum) AS hotspots_{hs}_sum,"""

    # urban density metrics
    query3 = f"""SELECT dfv.rid,
        AVG(gps.gpse_roaddensity_km_of_road_per_sq_km) AS urban_density_km_of_road_per_sq_km_avg,
        SUM(CASE WHEN
            gps.gpse_roaddensity_km_of_road_per_sq_km > 30 AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS urban_density_30_plus_km_sq_km_idle_hours_sum,
        SUM(CASE WHEN
            gps.gpse_roaddensity_km_of_road_per_sq_km > 30 AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS urban_density_30_plus_km_sq_km_moving_hours_sum,"""
    for r0, r1 in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)):
        query3 += f"""
            SUM(CASE WHEN
                gps.gpse_roaddensity_km_of_road_per_sq_km BETWEEN {r0} AND {r1} AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS urban_density_{r0}_to_{r1}_km_sq_km_idle_hours_sum,
            SUM(CASE WHEN
                gps.gpse_roaddensity_km_of_road_per_sq_km BETWEEN {r0} AND {r1} AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS urban_density_{r0}_to_{r1}_km_sq_km_moving_hours_sum,"""

    # speed metrics
    query4 = f"""SELECT dfv.rid,
        AVG(gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit) AS delta_speed_mph_vs_speed_limit_avg,
        AVG(CASE
            WHEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit < 1
            THEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit ELSE NULL END)
            AS delta_speed_mph_vs_speed_limit_under_avg,
        SUM(CASE
            WHEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit < 1
            THEN 1 ELSE 0 END)
            AS delta_speed_mph_vs_speed_limit_under_count,
        SUM(CASE
            WHEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit < 1
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS delta_speed_mph_vs_speed_limit_under_hours,
        AVG(CASE
            WHEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit > 1
            THEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit ELSE NULL END)
            AS delta_speed_mph_vs_speed_limit_over_avg,
        SUM(CASE
            WHEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit > 1
            THEN 1 ELSE 0 END)
            AS delta_speed_mph_vs_speed_limit_over_count,
        SUM(CASE
            WHEN gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit > 1
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS delta_speed_mph_vs_speed_limit_over_hours,
        SUM(CASE
            WHEN COALESCE(gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit, 0) BETWEEN -1 AND 1
            THEN 1 ELSE 0 END)
            AS speed_vs_road_speed_limit_equal_count,
        SUM(CASE
            WHEN COALESCE(gps.gpse_speed_mph_delta_gps_speed_vs_road_speed_limit, 0) BETWEEN -1 AND 1
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS speed_vs_road_speed_limit_equal_hours,"""

    # distance and duration by road type
    query5 = f"""SELECT dfv.rid,
        SUM(gps.gpse_segment_length_meters) AS travel_distance_meters_sum,
        SUM(CASE
            WHEN gps.gpse_public_road_code IS NULL THEN gpse_segment_length_meters ELSE 0 END)
            AS travel_distance_meters_roadcode_null_sum,
        SUM(CASE
            WHEN gps.gpse_public_road_code IS NULL
            AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS moving_duration_hours_roadcode_null_sum,
        SUM(CASE
            WHEN gps.gpse_public_road_code IS NULL
            AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS idle_duration_hours_roadcode_null_sum,"""
    for rc in ['LocalRoad', 'HighwayRamp', 'HighwayRoad']:
        query5 += f"""
            SUM(CASE
                WHEN gps.gpse_roadclass='{rc}' THEN gpse_segment_length_meters ELSE 0 END)
                AS travel_distance_meters_roadclass_{rc.lower()}_sum,
            SUM(CASE
                WHEN gps.gpse_roadclass='{rc}'
                AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS moving_duration_hours_roadclass_{rc.lower()}_sum,
            SUM(CASE
                WHEN gps.gpse_roadclass='{rc}'
                AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS idle_duration_hours_roadclass_{rc.lower()}_sum,"""
    for rc, desc in (
            (5111, 'motorway'),
            (5112, 'trunk'),
            (5113, 'primary'),
            (5114, 'secondary'),
            (5115, 'tertiary'),
            (5121, 'unclassified'),
            (5122, 'residential'),
            (5123, 'living_street'),
            (5131, 'motorway_link'),
            (5132, 'trunk_link'),
            (5133, 'primary_link'),
            (5134, 'secondary_link'),
            (5135, 'tertiary_link')):
        query5 += f"""
            SUM(CASE
                WHEN gps.gpse_public_road_code={rc} THEN gpse_segment_length_meters ELSE 0 END)
                AS travel_distance_meters_roadcode_{rc}_{desc}_sum,
            SUM(CASE
                WHEN gps.gpse_public_road_code={rc}
                AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS moving_duration_hours_roadcode_{rc}_{desc}_sum,
            SUM(CASE
                WHEN gps.gpse_public_road_code={rc}
                AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS idle_duration_hours_roadcode_{rc}_{desc}_sum,"""

    # road properties
    query6 = f"""SELECT dfv.rid,
        AVG(gps.gpse_maxspeedestimate) AS road_speed_limit_estimate_avg,
        AVG(gps.gpse_roadangledegrees) AS road_angle_degrees_avg,
        SUM(gps.gpse_private_road_count) AS private_road_sum,
        SUM(CASE
            WHEN gps.gpse_public_road_code IS NULL
            AND gps.gpse_private_road_count IS NULL THEN 1 ELSE 0 END)
            AS road_code_null_off_road_sum,"""
    for rc in ['LocalRoad', 'HighwayRamp', 'HighwayRoad']:
        query6 += f"""
            SUM(CASE WHEN gps.gpse_roadclass='{rc}' THEN 1 ELSE 0 END) AS roadclass_{rc.lower()}_sum,"""
    for rc, desc in (
            (5111, 'motorway'),
            (5112, 'trunk'),
            (5113, 'primary'),
            (5114, 'secondary'),
            (5115, 'tertiary'),
            (5121, 'unclassified'),
            (5122, 'residential'),
            (5123, 'living_street'),
            (5131, 'motorway_link'),
            (5132, 'trunk_link'),
            (5133, 'primary_link'),
            (5134, 'secondary_link'),
            (5135, 'tertiary_link')):
        query6 += f"""
            SUM(CASE WHEN gps.gpse_public_road_code={rc} THEN 1 ELSE 0 END) AS roadcode_{rc}_{desc}_sum,"""

    # average annual daily traffic by road type
    query7 = f"""SELECT dfv.rid,"""
    for rc in ['LocalRoad', 'HighwayRamp', 'HighwayRoad']:
        query7 += f"""
            AVG(CASE WHEN gps.gpse_roadclass='{rc}' THEN gpse_public_road_aadt_current ELSE NULL END)
            AS roadclass_{rc.lower()}_aadt_avg,"""
    for rc, desc in (
            (5111, 'motorway'),
            (5112, 'trunk'),
            (5113, 'primary'),
            (5114, 'secondary'),
            (5115, 'tertiary'),
            (5121, 'unclassified'),
            (5122, 'residential'),
            (5123, 'living_street'),
            (5131, 'motorway_link'),
            (5132, 'trunk_link'),
            (5133, 'primary_link'),
            (5134, 'secondary_link'),
            (5135, 'tertiary_link')):
        query7 += f"""
            AVG(CASE WHEN gps.gpse_public_road_code={rc} THEN gpse_public_road_aadt_current ELSE NULL END)
            AS roadcode_{rc}_{desc}_aadt_avg,"""

    # trip duration based on average annual daily traffic by road type
    query8 = f"""SELECT dfv.rid,"""
    for r0, r1 in ((0, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 3e5), (3e5, 'plus')):
        assert isinstance(r0, (int, float)) and (isinstance(r1, (int, float)) or (r1 == 'plus'))
        if r1 != 'plus':
            if r1 <= 1e3:
                wa = f'BETWEEN {int(r0)} AND {int(r1)}'
                wb = f'{int(r0)}_{int(r1)}'
            else:
                assert r0 >= 1e3
                wa = f'BETWEEN {int(r0)} AND {int(r1)}'
                wb = f'{r0 / 1000:.0f}k_{r1 / 1000:.0f}k'
        else:
            assert r0 >= 1e3
            wa = f'> {int(r0)}'
            wb = f'{r0 / 1000:.0f}k_plus'

        for rc in ['LocalRoad', 'HighwayRamp', 'HighwayRoad']:
            query8 += f"""
                SUM(CASE WHEN gps.gpse_roadclass='{rc}' AND gpse_public_road_aadt_current {wa}
                    THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                    AS roadclass_{rc.lower()}_aadt_{wb}_hours_sum,"""
        for rc, desc in (
                (5111, 'motorway'),
                (5112, 'trunk'),
                (5113, 'primary'),
                (5114, 'secondary'),
                (5115, 'tertiary'),
                (5121, 'unclassified'),
                (5122, 'residential'),
                (5123, 'living_street'),
                (5131, 'motorway_link'),
                (5132, 'trunk_link'),
                (5133, 'primary_link'),
                (5134, 'secondary_link'),
                (5135, 'tertiary_link')):
            query8 += f"""
                SUM(CASE WHEN gps.gpse_public_road_code={rc} AND gpse_public_road_aadt_current {wa}
                    THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                    AS roadcode_{rc}_{desc}_aadt_{wb}_hours_sum,"""

    # road feature metrics
    query9 = f"""SELECT dfv.rid,"""
    for road_type in ['public_road_intersection', 'highway_ramp_junction', 'private_road_intersection']:
        for distance in [0, 10, 20, 30, 60]:
            query9 += f"""
                AVG(gps.gpse_{road_type}_within_0_meters)
                    AS {road_type}_within_{distance}_meters_avg,
                AVG(gps.gpse_{road_type}_complexity_within_{distance}_meters)
                    AS {road_type}_complexity_within_{distance}_meters_avg,"""

    # road corridor metrics
    query10 = f"""SELECT dfv.rid,
        SUM(gps.gpse_private_roads_length_in_corridor_meters) AS private_roads_length_in_corridor_meters_sum,
        SUM(gps.gpse_local_road_length_in_corridor_meters) AS local_road_length_in_corridor_meters_sum,
        SUM(gps.gpse_local_road_width_avg_in_corridor_meters) as local_road_width_avg_in_corridor_meters,
        SUM(gps.gpse_local_road_maxspeedestimate_kmh_avg_in_corridor_meters) as local_road_maxspeedestimate_kmh_avg_in_corridor_meters,
        SUM(gps.gpse_local_road_aadt_avg_in_corridors) as local_road_aadt_avg_in_corridor,"""
    for cx in ['public_road_intersection', 'highway_ramp_junction', 'private_road_intersection']:
        query10 += f"""
            SUM(gps.gpse_{cx}_in_corridor) AS {cx}_in_corridor_sum,
            SUM(gps.gpse_{cx}_complexity_in_corridor) AS {cx}_complexity_in_corridor_sum,"""
    for cx in ['public_roads', 'highway_road', 'highway_ramp']:
        query10 += f"""
            SUM(gps.gpse_{cx}_length_in_corridor_meters) AS {cx}_length_in_corridor_meters_sum,
            SUM(gps.gpse_{cx}_width_avg_in_corridor_meters) as {cx}_width_avg_in_corridor_meters,
            SUM(gps.gpse_{cx}_maxspeedestimate_kmh_avg_in_corridor_meters) as {cx}_maxspeedestimate_kmh_avg_in_corridor_meters,
            SUM(gps.gpse_{cx}_aadt_avg_in_corridor) as {cx}_aadt_avg_in_corridor,"""

    # road features traversed on route
    query11 = f"""SELECT dfv.rid,
        SUM(gps.gpse_intersectioncount) as all_intersections_traversed_sum,
        AVG(gps.gpse_intersectioncomplexity_avg) AS all_intersections_complexity_avg,
        AVG(gps.gpse_intersectioncomplexity_min) AS all_intersections_complexity_min,
        AVG(gps.gpse_intersectioncomplexity_max) AS all_intersections_complexity_max,
        SUM(CASE WHEN gps.gpse_intersectioncount > 0
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS all_intersections_traversed_duration_hours_sum,
        SUM(CASE WHEN gps.gpse_intersectioncount > 0
            AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS all_intersections_traversed_moving_hours_sum,
        SUM(CASE WHEN gps.gpse_intersectioncount > 0
            AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
            THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
            AS all_intersections_traversed_idle_hours_sum,"""
    for ca, cb in (
            ('isintersection', 'public_intersections'),
            ('isserviceroadintersection', 'private_intersections'),
            ('isrampjunction', 'ramp_junctions'),
            ('istrafficsignal', 'traffic_signals'),
            ('iscrosswalk', 'crosswalks'),
            ('israilwaylevelcrossing', 'railway_crossings'),
            ('isyieldsign', 'yieldsign'),
            ('isstopsign', 'stopsign'),
            ('issign_allwaystop', 'allstop')):
        query11 += f"""
            SUM(CASE WHEN gps.gpse_{ca}=1 THEN gps.gpse_intersectioncount ELSE NULL END)
                AS {cb}_traversed_sum,
            AVG(CASE WHEN gps.gpse_{ca}=1 THEN gps.gpse_intersectioncomplexity_avg ELSE NULL END)
                AS {cb}_complexity_avg,
            AVG(CASE WHEN gps.gpse_{ca}=1 THEN gps.gpse_intersectioncomplexity_min ELSE NULL END)
                AS {cb}_complexity_min,
            AVG(CASE WHEN gps.gpse_{ca}=1 THEN gps.gpse_intersectioncomplexity_max ELSE NULL END)
                AS {cb}_complexity_max,
            SUM(CASE WHEN gps.gpse_{ca}=1
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS {cb}_traversed_duration_hours_sum,
            SUM(CASE WHEN gps.gpse_{ca}=1
                AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) >= 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS {cb}_traversed_moving_hours_sum,
            SUM(CASE WHEN gps.gpse_{ca}=1
                AND ABS(gps.gpse_distance_meters_from_prev_gps_trackpoint) < 3
                THEN gps.gpse_time_interval_from_prev_gps_trackpoint / -3600 ELSE NULL END)
                AS {cb}_traversed_idle_hours_sum,"""

    # maneuver count by road type
    query12 = f"""SELECT dfv.rid,"""
    for maneuver in ['straight', 'left', 'leftextreme', 'leftshallow', 'right', 'rightextreme', 'rightshallow']:
        query12 += f"""
            SUM(CASE WHEN gps.gpse_ismaneuver{maneuver}=1 AND gps.gpse_public_road_code IS NOT NULL THEN 1 ELSE 0 END)
                AS public_roads_maneuver_{maneuver}_sum,
            SUM(CASE WHEN gps.gpse_ismaneuver{maneuver}=1 AND gps.gpse_public_road_code IS NULL THEN 1 ELSE 0 END)
                AS private_roads_maneuver_{maneuver}_sum,"""

    # intersection count by maneuver and intersection type
    query13 = f"""SELECT dfv.rid,"""
    for maneuver in ['straight', 'left', 'leftextreme', 'leftshallow', 'right', 'rightextreme', 'rightshallow']:
        for rc in [
                'intersection',
                'serviceroadintersection',
                'rampjunction',
                'trafficsignal',
                'crosswalk',
                'railwaylevelcrossing',
                'yieldsign',
                'stopsign',
                'sign_allwaystop']:
            query13 += f"""
                SUM(CASE
                    WHEN gps.gpse_ismaneuver{maneuver}=1 AND gps.gpse_is{rc}=1
                    THEN gps.gpse_intersectioncount ELSE 0 END)
                    AS intersection_count_{rc}_maneuver_{maneuver}_sum,"""

    # average intersection complexity metrics by maneuver and intersection type
    query14 = f"""SELECT dfv.rid,"""
    for maneuver in ['straight', 'left', 'leftextreme', 'leftshallow', 'right', 'rightextreme', 'rightshallow']:
        for rc in [
                'intersection',
                'serviceroadintersection',
                'rampjunction',
                'trafficsignal',
                'stopsign',
                'sign_allwaystop']:
            query14 += f"""
                AVG(CASE
                    WHEN gps.gpse_ismaneuver{maneuver}=1 AND gps.gpse_is{rc}=1
                    THEN gps.gpse_intersectioncomplexity_avg ELSE NULL END)
                    AS intersection_complexity_avg_{rc}_maneuver_{maneuver}_avg,
                AVG(CASE
                    WHEN gps.gpse_ismaneuver{maneuver}=1 AND gps.gpse_is{rc}=1
                    THEN gps.gpse_intersectioncomplexity_min ELSE NULL END)
                    AS intersection_complexity_min_{rc}_maneuver_{maneuver}_avg,
                AVG(CASE
                    WHEN gps.gpse_ismaneuver{maneuver}=1 AND gps.gpse_is{rc}=1
                    THEN gps.gpse_intersectioncomplexity_max ELSE NULL END)
                    AS intersection_complexity_max_{rc}_maneuver_{maneuver}_avg,"""

    # create list of gps enriched metrics by category
    dfx = [
        # travel duration metrics
        spark.sql(complete_query(query1)).toPandas(),
        # hotspot metrics
        spark.sql(complete_query(query2)).toPandas(),
        # urban density metrics
        spark.sql(complete_query(query3)).toPandas(),
        # speed metrics
        spark.sql(complete_query(query4)).toPandas(),
        # distance and duration by road type
        spark.sql(complete_query(query5)).toPandas(),
        # road properties
        spark.sql(complete_query(query6)).toPandas(),
        # average annual daily traffic by road type
        spark.sql(complete_query(query7)).toPandas(),
        # trip duration based on average annual daily traffic by road type
        spark.sql(complete_query(query8)).toPandas(),
        # road feature metrics
        spark.sql(complete_query(query9)).toPandas(),
        # road corridor metrics
        spark.sql(complete_query(query10)).toPandas(),
        # road features traversed on route
        spark.sql(complete_query(query11)).toPandas(),
        # maneuver count by road type
        spark.sql(complete_query(query12)).toPandas(),
        # intersection count by maneuver and intersection type
        spark.sql(complete_query(query13)).toPandas(),
        # average intersection complexity metrics by maneuver and intersection type
        spark.sql(complete_query(query14)).toPandas()]

    # validate and return merged gps enriched metrics
    assert np.unique([x.shape[0] for x in dfx]).size == 1
    assert np.unique([np.sort(x['rid'].values) for x in dfx]).size == dfx[0].shape[0]
    assert all([pd.isnull(x['rid'].values).sum() == 0 for x in dfx])
    cols = list(chain(*[[x for x in dx.columns if x != 'rid'] for dx in dfx]))
    assert np.unique(cols).size == len(cols)
    df = reduce(lambda d1, d2: pd.merge(d1, d2, on='rid', how='inner'), dfx)
    if prefix is not None:
        assert isinstance(prefix, str)
        df.columns = [x if x == 'rid' else f'{prefix}_{x}' for x in df.columns]
    return df

def gpsn_metrics(spark, prefix=None):
    """
    aggregated metrics based on distance-normalized gps enrichment
    """

    # validation
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfv' in tables) and ('gpse' in tables) and ('gpsm' in tables)

    def complete_query_gpse(query):
        query = query.rstrip(',')
        query += f"""
            FROM gpse JOIN dfv
                ON gpse.VehicleId = dfv.VehicleId
                AND gpse.TS_SEC >= dfv.time0
                AND gpse.TS_SEC <= dfv.time1
            GROUP BY dfv.rid"""
        return query

    # core duration and distance
    query1 = f"""SELECT dfv.rid,
        SUM(gpse.time_interval_sec) / (60 * 60 * 24) AS total_days,
        SUM(CASE WHEN gpse.mph > 0.1 THEN gpse.time_interval_sec ELSE NULL END) / (60 * 60 * 24) AS motion_days,
        SUM(CASE WHEN gpse.mph <= 0.1 THEN gpse.time_interval_sec ELSE NULL END) / (60 * 60 * 24) AS idle_days,
        0.000621371 * SUM(gpse.distance_meters) AS distance_miles"""

    # hotspot metrics
    query2 = f"""SELECT dfv.rid,
        SUM(gpse.segment__crash_all_hotspots_entered__sum) AS num_hotspots_entered"""

    set_trace()

    # # hotspot metrics
    # query2 = f"""SELECT dfv.rid,
    #     SUM(gps.gpse_segment_hotspotcount) AS all_hotspots_entered_sum,
    #     SUM(gps.gpse_segment_hotspot_intersectioncount) AS hotspots_entered_intersection_sum,
    #     AVG(gps.gpse_segment_hotspot_incidentrate) AS hotspots_incidentrate_intersection_avg,
    #     AVG(gps.gpse_segment_hotspot_intersection_compexity_avg) AS hotspots_entered_intersection_complexity_avg,
    #     AVG(gps.gpse_crash_hotspots_rankgroup_severity_index_avg) AS hotspots_severity_index_avg,
    #     SUM(gps.gpse_crash_all_hotspots_entered_sum) AS hotspots_incidents_sum,"""
    # for hs in [
    #         'animal',
    #         'bicyclist',
    #         'lowclearance',
    #         'intersection',
    #         'pedestrian',
    #         'slowing_traffic',
    #         'train',
    #         'turn_curve']:
    #     query2 += f"""
    #         SUM(gps.gpse_crash_{hs}_hotspots_entered) AS {hs}_hotspots_entered_sum,"""
    # for hs in [
    #         'injury_incidents',
    #         'fatal_incidents',
    #         'pedestriansinvolved',
    #         'pedestriansinvolvedunder18',
    #         'cyclistsinvolved',
    #         'cyclistsinvolvedunder18']:
    #     query2 += f"""
    #         SUM(gps.gpse_crash_hotspots_{hs}_sum) AS hotspots_{hs}_sum,"""

    # run queries by category
    dfx = [
        # core duration and distance
        spark.sql(complete_query_gpse(query1)).toPandas(),
        # hotspot metrics
        spark.sql(complete_query_gpse(query2)).toPandas()]

    # validate and return merged gps enriched metrics
    assert np.unique([x.shape[0] for x in dfx]).size == 1
    assert np.unique([np.sort(x['rid'].values) for x in dfx]).size == dfx[0].shape[0]
    assert all([pd.isnull(x['rid'].values).sum() == 0 for x in dfx])
    cols = list(chain(*[[x for x in dx.columns if x != 'rid'] for dx in dfx]))
    assert np.unique(cols).size == len(cols)
    df = reduce(lambda d1, d2: pd.merge(d1, d2, on='rid', how='inner'), dfx)
    if prefix is not None:
        assert isinstance(prefix, str)
        df.columns = [x if x == 'rid' else f'{prefix}_{x}' for x in df.columns]
    return df

def trigger_count_metrics(spark, span='all', suffix=None):
    """
    trigger count metrics
    - span expresses aggregation over full window, weekday/weekend/ or day of week
    - may not contain nulls
    """

    # validation
    assert span in ['all', 'weekday', 'weekend', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfe' in tables) and ('triggers' in tables)

    # count of all triggers
    query0 = f"""
        SELECT
            dfe.rid,
            COUNT(*) AS ntriggers
        FROM dfe JOIN triggers
            ON triggers.EventRecorderId = dfe.EventRecorderId
            AND triggers.CreationDate >= dfe.time0
            AND triggers.CreationDate < dfe.time1"""
    if span == 'weekday':
        query0 += f"""\nWHERE weekday"""
    elif span == 'weekend':
        query0 += f"""\nWHERE NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query0 += f"""\nWHERE day_of_week = '{span}'"""
    query0 += """\nGROUP BY dfe.rid"""
    df0 = spark.sql(query0).toPandas()

    # count of individual triggers
    query1 = f"""
        SELECT
            CAST(dfe.rid AS STRING) AS rid,
            CAST(triggers.EventTriggerTypeId AS STRING) AS NameId,
            COUNT(*) AS nevents
        FROM dfe JOIN triggers
            ON triggers.EventRecorderId = dfe.EventRecorderId
            AND triggers.CreationDate >= dfe.time0
            AND triggers.CreationDate < dfe.time1"""
    if span == 'weekday':
        query1 += f"""\nWHERE weekday"""
    elif span == 'weekend':
        query1 += f"""\nWHERE NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query1 += f"""\nWHERE day_of_week = '{span}'"""
    query1 += """\nGROUP BY dfe.rid, triggers.EventTriggerTypeId"""
    df1 = spark.sql(query1).groupBy('rid').pivot('NameId').max().toPandas()
    df1.columns = [x if x == 'rid' else f'ntriggers_{x}' for x in df1.columns]

    # merge and clean
    assert df0.shape[0] == df1.shape[0]
    df1['rid'] = df1['rid'].astype('int')
    df = pd.merge(left=df0, right=df1, on='rid', how='inner')
    df = df.fillna(0)
    assert pd.isnull(df).values.sum() == 0

    # update via suffix and return
    if suffix is not None:
        df.columns = [x if x == 'rid' else f'{x}_{suffix}' for x in df.columns]
    return df

def trigger_speed_metrics(spark, span='all', suffix=None):
    """
    trigger speed and accel metrics
    - span expresses aggregation over full window, weekday/weekend/ or day of week
    - all columns except rid may contain nulls
    - relevant events include 12,15,16,18,26,30
    """

    # validation
    assert span in ['all', 'weekday', 'weekend', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tables = [x.name for x in spark.catalog.listTables()]
    assert ('dfe' in tables) and ('triggers' in tables)

    # average speed and accel of all relevant triggers
    query0 = f"""
        SELECT
            dfe.rid,
            AVG(triggers.SpeedAtTrigger) AS avg_speed_triggers,
            AVG(triggers.SpeedAtTrigger - triggers.PostedSpeedLimit) AS avg_speed_above_posted_triggers,
            AVG(triggers.ForwardExtremeAcceleration) AS avg_accel_triggers
        FROM dfe JOIN triggers
            ON triggers.EventRecorderId = dfe.EventRecorderId
            AND triggers.CreationDate >= dfe.time0
            AND triggers.CreationDate < dfe.time1
        WHERE triggers.EventTriggerTypeId IN (12,15,16,18,26,30)"""
    if span == 'weekday':
        query0 += f"""\nAND weekday"""
    elif span == 'weekend':
        query0 += f"""\nAND NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query0 += f"""\nAND day_of_week = '{span}'"""
    query0 += """\nGROUP BY dfe.rid"""
    df0 = spark.sql(query0).toPandas()

    # average speed and accel of individual relevant triggers
    query1 = f"""
        SELECT
            dfe.rid,
            triggers.EventTriggerTypeId AS NameId,
            AVG(triggers.SpeedAtTrigger) AS avg_speed_triggers,
            AVG(triggers.SpeedAtTrigger - triggers.PostedSpeedLimit) AS avg_speed_above_posted_triggers,
            AVG(triggers.ForwardExtremeAcceleration) AS avg_accel_triggers
        FROM dfe JOIN triggers
            ON triggers.EventRecorderId = dfe.EventRecorderId
            AND triggers.CreationDate >= dfe.time0
            AND triggers.CreationDate < dfe.time1
        WHERE triggers.EventTriggerTypeId IN (12,15,16,18,26,30)"""
    if span == 'weekday':
        query1 += f"""\nAND weekday"""
    elif span == 'weekend':
        query1 += f"""\nAND NOT weekday"""
    elif span in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        query1 += f"""\nAND day_of_week = '{span}'"""
    query1 += """\nGROUP BY dfe.rid, triggers.EventTriggerTypeId"""
    df1 = spark.sql(query1).toPandas()
    df1 = df1.pivot(index='rid', columns='NameId', values=['avg_speed_triggers', 'avg_speed_above_posted_triggers', 'avg_accel_triggers'])

    # clean and merge
    df1.columns = [f'{a}_{b}' for a, b in df1.columns]
    df1 = df1.reset_index(drop=False)
    assert df0.shape[0] == df1.shape[0]
    df = pd.merge(left=df0, right=df1, on='rid', how='inner')
    assert pd.isnull(df['rid']).values.sum() == 0

    # update via suffix and return
    if suffix is not None:
        df.columns = [x if x == 'rid' else f'{x}_{suffix}' for x in df.columns]
    return df

# GPS parquet dataset enrichment
def spark_etl_load_gps(datadir, src, vid, service):
    """
    load gps DataFrame based on service
    """
    assert service in ['EC2', 'EMR']

    # EC2
    if service == 'EC2':
        assert datadir is not None

        # gps filename
        fn = glob(os.path.join(datadir, src, f'VehicleId={vid}', '*.parquet'))
        try:
            assert len(fn) == 1
        except AssertionError:
            raise ValueError(f'{gethostname()}, {vid}')
        fn = fn[0]

    # EMR
    else:
        assert service == 'EMR'
        assert datadir is None

        # validate one object at gps raw data folder
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket='russell-s3', Prefix=os.path.join(src, f'VehicleId={vid}'))
        if ('Contents' not in response.keys()) or (len(response['Contents']) != 1):
            raise ValueError(f'{gethostname()}, {vid}')
        assert len(response['Contents']) == 1
        key = response['Contents'][0]['Key']

        # create data folder for vid
        try:
            assert not os.path.isdir(f'/mnt1/s3/VehicleId={vid}')
        except AssertionError:
            raise ValueError(f'{gethostname()}, {vid}')
        os.mkdir(f'/mnt1/s3/VehicleId={vid}')
        assert os.path.isdir(f'/mnt1/s3/VehicleId={vid}')

        # download parquet file from s3 to data folder for vid
        fn = f'/mnt1/s3/VehicleId={vid}/{os.path.split(key)[1]}'
        assert not os.path.isfile(fn)
        s3.download_file(Bucket='russell-s3', Key=key, Filename=fn)
        assert os.path.isfile(fn)

    # load gps data from parquet filename and return
    assert fn[-8:] == '.parquet'
    df = pq.ParquetFile(fn).read().to_pandas().sort_values('TS_SEC').reset_index(drop=True)
    df['VehicleId'] = vid
    return df

def spark_etl_save_gps(df, datadir, dst, vid, service):
    """
    save gps DataFrame based on service
    """
    assert service in ['EC2', 'EMR']

    # EC2
    if service == 'EC2':
        assert datadir is not None

        # write df to parquet at dst
        df.to_parquet(path=os.path.join(datadir, dst), engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

    # EMR
    else:
        assert service == 'EMR'
        assert datadir is None

        # identify src file at /mnt1/s3
        fn = glob(os.path.join(f'/mnt1/s3/VehicleId={vid}', '*'))
        assert len(fn) == 1
        fn = fn[0]

        # write df to parquet at /mnt1/s3
        df.to_parquet(path='/mnt1/s3', engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])

        # identify src and dst files at /mnt1/s3
        fxs = glob(os.path.join(f'/mnt1/s3/VehicleId={vid}', '*'))
        assert len(fxs) == 2
        fx = [x for x in fxs if x != fn]
        assert len(fx) == 1
        fx = fx[0]

        # upload to s3 at dst
        s3 = boto3.client('s3')
        s3.upload_file(Filename=fx, Bucket='russell-s3', Key=f'{dst}/VehicleId={vid}/{os.path.split(fx)[1]}')

        # remove data folder for vid
        rmtree(f'/mnt1/s3/VehicleId={vid}')
        assert not os.path.isdir(f'/mnt1/s3/VehicleId={vid}')

def gps_test_etl_pattern(spark, datadir, src, dst, service):
    """
    test pattern for distributed processing of gps data
    """
    assert service in ['EC2', 'EMR']

    def gps_test_etl_pattern_vid(pdf):

        # validate pdf, get vehicle-id
        assert pdf.shape[0] == 1
        vid = vids.loc[pdf['id'].values[0], 'VehicleId']

        # load gps data
        df = spark_etl_load_gps(datadir, src, vid, service)

        # dummy modification of gps data
        df['abc'] = 7

        # save modified gps data and clean up
        spark_etl_save_gps(df, datadir, dst, vid, service)

        return pdf

    # pyspark code to distribute gps_test_etl_pattern_vid
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps').toPandas()
    vx = spark.range(start=0, end=vids.shape[0], step=1, numPartitions=int(1.5 * vids.shape[0]))
    dx = vx.groupby('id').applyInPandas(gps_test_etl_pattern_vid, schema=vx.schema).toPandas()
    # debug
    # vx = vx.toPandas()
    # gps_test_etl_pattern_vid(vx.loc[vx.index == 12])
    # gps_test_etl_pattern_vid(vx.loc[vids['VehicleId'] == '9100FFFF-48A9-CB63-77D6-A8A3E0CF0000'])

def gps_segmentation(spark, datadir, src, dst, service, time_interval_sec=61, distance_interval_miles=1.0, mph_max=200):
    """
    distributed gps segmentation
    """
    assert service in ['EC2', 'EMR']
    geod = Geod(ellps='WGS84')

    def gps_segmentation_vid(pdf):

        # validate pdf, get vehicle-id
        assert pdf.shape[0] == 1
        vid = vids.loc[pdf['id'].values[0], 'VehicleId']

        # load gps data
        df = spark_etl_load_gps(datadir, src, vid, service)

        # validate columns, create utc and initialize sid
        cok = ['VehicleId', 'TS_SEC', 'longitude', 'latitude', 'longitude_gps', 'latitude_gps']
        assert all([x in df.columns for x in cok])
        assert all([x not in df.columns for x in ['segmentId', 'utc']])
        others = sorted([x for x in df.columns if x not in cok])
        df['utc'] = [datetime.utcfromtimestamp(x) for x in df['TS_SEC']]
        df['segmentId'] = np.zeros(df.shape[0]).astype('float')
        df = df[cok + others + ['utc', 'segmentId']]

        # distance interval in miles, time interval in sec
        lon = df['longitude_gps'].values
        lat = df['latitude_gps'].values
        assert all(~np.isnan(lon)) and all(~np.isnan(lat))
        _, _, dx = geod.inv(lons1=lon[1:], lats1=lat[1:], lons2=lon[:-1], lats2=lat[:-1])
        dx = np.hstack((np.nan, 0.000621371 * dx))
        tx = np.hstack((np.nan, np.diff(df['TS_SEC'].values)))

        # debug
        # df['distance_interval_miles'] = dx
        # df['time_interval_sec'] = tx
        # df['mph'] = dx / (tx / 3600)

        # records that complete intervals exceeding limits in args
        mph = dx / (tx / 3600)
        nok = np.logical_or(np.logical_and(dx > distance_interval_miles, tx > time_interval_sec), mph > mph_max)

        # data segmentation algorithm for vehicles with any nok intervals
        if nok.sum() > 0:
            nok = np.sort(np.where(nok)[0])
            for x in nok:

                # standard update for all nok rows
                df.loc[x, 'segmentId'] = None

                # increment segmentId if subsequent rows
                if x < df.shape[0] - 2:
                    df.loc[x + 1:, 'segmentId'] += 1

                # correct 1-row segment at start of dx
                if (x == 1) and (~np.isnan(df.loc[0, 'segmentId'])):
                    df.loc[0, 'segmentId'] = None

                # correct 1-row segments after start of dx
                elif (x > 1) and (~np.isnan(df.loc[x - 1, 'segmentId'])) and (np.isnan(df.loc[x - 2, 'segmentId'])):
                    df.loc[x - 1, 'segmentId'] = None

            # correct 1-row segments at end of dx
            if (np.isnan(df.loc[df.shape[0] - 2, 'segmentId'])) and (~np.isnan(df.loc[df.shape[0] - 1, 'segmentId'])):
                df.loc[df.shape[0] - 1, 'segmentId'] = None

            # validate no 1-row segments
            if any(~pd.isnull(df['segmentId'])):
                assert df['segmentId'].value_counts().min() > 1

            # validate
            assert np.all(np.sort(df['TS_SEC'].values) == df['TS_SEC'].values)

        # save modified gps data and clean up
        spark_etl_save_gps(df, datadir, dst, vid, service)

        return pdf

    # pyspark code to distribute gps_segmentation_vid
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps').toPandas()
    vx = spark.range(start=0, end=vids.shape[0], step=1, numPartitions=int(1.5 * vids.shape[0]))
    dx = vx.groupby('id').applyInPandas(gps_segmentation_vid, schema=vx.schema).toPandas()
    # debug
    # vx = vx.toPandas()
    # gps_segmentation_vid(vx.loc[vx.index == 33])
    # gps_segmentation_vid(vx.loc[vids['VehicleId'] == '9100FFFF-48A9-CC63-7A15-A8A3E03F0000'])

def gps_segmentation_metrics(dcm, spark):
    """
    get pandas DataFrame of gps segmentation metrics
    """

    # validate
    assert 'gps' in [x.name for x in spark.catalog.listTables()]

    # time bounds as a Spark DataFrame object
    dx = dcm[['VehicleId', 'time0', 'time1']].copy()
    dx['time0'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dx['time0']]
    dx['time1'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dx['time1']]
    assert np.unique(dx['time1'] - dx['time0']).size == 1
    days = np.unique(dx['time1'] - dx['time0'])[0] / (60*60*24)
    dx = broadcast(spark.createDataFrame(dx))
    dx.cache()
    dx.createOrReplaceTempView('dx')

    # days between window boundaries and records by vehicle-id / time0 / time1
    s0 = spark.sql(f"""
        SELECT
            dx.VehicleId, dx.time0, dx.time1,
            (MIN(gps.TS_SEC) - dx.time0) / (60*60*24) AS left_window_to_first_segmented_record,
            (dx.time1 - MAX(gps.TS_SEC)) / (60*60*24) AS last_segmented_record_to_right_window,
            COUNT(DISTINCT(gps.segmentId)) AS n_segments
        FROM gps JOIN dx
            ON gps.VehicleId = dx.VehicleId
            AND gps.TS_SEC >= dx.time0
            AND gps.TS_SEC <= dx.time1
        WHERE gps.segmentId IS NOT NULL
        GROUP BY dx.VehicleId, dx.time0, dx.time1
        ORDER BY dx.VehicleId""")
    s0.createOrReplaceTempView('s0')

    # segment metrics including duration of gap from end of previous segment
    s1 = spark.sql(f"""
        SELECT
            dx.VehicleId, dx.time0, dx.time1, gps.segmentId,
            COUNT(*) AS n_records_segment,
            (MAX(gps.TS_SEC) - MIN(gps.TS_SEC)) / (60*60*24) AS n_days_segment,
            (MIN(gps.TS_SEC) - LAG(MAX(gps.TS_SEC)) OVER(PARTITION BY dx.VehicleId, dx.time0, dx.time1 ORDER BY gps.segmentId)) / (60*60*24) AS n_days_no_segment
        FROM gps JOIN dx
            ON gps.VehicleId = dx.VehicleId
            AND gps.TS_SEC >= dx.time0
            AND gps.TS_SEC <= dx.time1
        WHERE gps.segmentId IS NOT NULL
        GROUP BY dx.VehicleId, dx.time0, dx.time1, gps.segmentId
        ORDER BY dx.VehicleId, gps.segmentId""")
    s1.createOrReplaceTempView('s1')

    # aggregated segment metrics by xid
    s2 = spark.sql(f"""
        SELECT
            VehicleId, time0, time1,
            SUM(n_records_segment) AS n_records_segments,
            SUM(n_days_segment) AS n_days_segments,
            SUM(n_days_no_segment) AS n_days_no_segments
        FROM s1
        GROUP BY VehicleId, time0, time1
        ORDER BY VehicleId, time0, time1""")
    s2.createOrReplaceTempView('s2')

    # merge s0,s1,s2 and create pandas DataFrame
    ds = spark.sql(f"""
        SELECT
            s0.VehicleId, s0.time0, s0.time1,
            s0.left_window_to_first_segmented_record,
            s0.last_segmented_record_to_right_window,
            s0.n_segments,
            s2.n_records_segments,
            s2.n_days_segments,
            s2.n_days_no_segments
        FROM s0 JOIN s2
            ON s0.VehicleId = s2.VehicleId
            AND s0.time0 = s2.time0
            AND s0.time1 = s2.time1
        ORDER BY s0.VehicleId, s0.time0, s0.time1""").toPandas()

    # validate and return
    cols = ['left_window_to_first_segmented_record', 'last_segmented_record_to_right_window', 'n_days_segments', 'n_days_no_segments']
    ds['total_days'] = np.nansum(ds[cols].astype('float'), axis=1)
    assert np.unique(np.round(np.unique(ds['total_days'])).astype('int')).size == 1
    assert np.unique(np.round(np.unique(ds['total_days'])).astype('int'))[0] == days

    return ds

def gps_interval_metrics(spark, datadir, src, dst, service, include_daily_coverage=False):
    """
    distributed gps interval metrics by vehicle-id / segment-id
    """
    assert service in ['EC2', 'EMR']
    geod = Geod(ellps='WGS84')

    def segment_metrics(df):
        """
        metrics applied to individual vehicle-id and segment
        """

        # sort and validate
        df = df.sort_values('TS_SEC')
        lon = df['longitude_gps'].values
        lat = df['latitude_gps'].values
        assert (np.isnan(lon).sum() == 0) and (np.isnan(lat).sum() == 0) and (np.isnan(df['TS_SEC']).sum() == 0)
        # distance interval in meters
        _, _, dx = geod.inv(lons1=lon[1:], lats1=lat[1:], lons2=lon[:-1], lats2=lat[:-1])
        # distance interval in miles
        dx = np.hstack((np.nan, 0.000621371 * dx))
        df['distance_interval_miles'] = dx
        df['cumulative_segment_distance_miles'] = np.hstack((np.nan, np.cumsum(dx[1:])))
        # time interval in sec
        tx = np.hstack((np.nan, np.diff(df['TS_SEC'].values)))
        df['time_interval_sec'] = tx
        df['cumulative_segment_time_days'] = np.hstack((np.nan, (1 / 86400) * np.cumsum(tx[1:])))
        df['mph'] = dx / (tx / 3600)
        return df

    def gps_interval_metrics_vid(pdf):

        # validate pdf, get vehicle-id
        assert pdf.shape[0] == 1
        vid = vids.loc[pdf['id'].values[0], 'VehicleId']

        # load gps data
        df = spark_etl_load_gps(datadir, src, vid, service)

        # segment metrics by segmentId
        if any(~pd.isnull(df['segmentId'])):
            ds = df.groupby('segmentId', group_keys=False).apply(segment_metrics).reset_index(drop=True)
            df = pd.concat((ds, df.loc[pd.isnull(df['segmentId'])]), axis=0).sort_values('TS_SEC').reset_index(drop=True)

        # no segments identified in data
        if all(pd.isnull(df['segmentId'])):
            df = segment_metrics(df)

        # cumulative metrics across all segments
        df['cumulative_distance_miles'] = np.nancumsum(df['distance_interval_miles'])
        df['cumulative_time_days'] = (1 / 86400) * np.nancumsum(df['time_interval_sec'])

        # all time intervals
        df['all_time_interval_sec'] = np.hstack((np.nan, np.diff(df['TS_SEC'])))

        # vehicle state characterization
        df['vehicle_state'] = None
        df.loc[df['mph'] > 0.1, 'vehicle_state'] = 'motion'
        df.loc[(df['mph'] <= 0.1) & (df['time_interval_sec'] < 700), 'vehicle_state'] = 'idle'
        df.loc[(df['mph'] <= 0.1) & (df['time_interval_sec'] >= 700), 'vehicle_state'] = 'off'

        # daily coverage characterization
        if include_daily_coverage:
            x0 = df.assign(index=df.index).resample('D', on='utc')['index'].min().to_frame().rename(columns={'index': 'start index'})
            x1 = df.assign(index=df.index).resample('D', on='utc')['index'].max().to_frame().rename(columns={'index': 'end index'})
            assert all(x0.index == x1.index)
            days = pd.merge(x0, x1, how='inner', left_index=True, right_index=True)
            kws = {'rule': 'D', 'on': 'utc', 'closed': 'right', 'label': 'left'}
            # hours per day covered by time intervals should be 24 hours except first and last days
            hours_per_day = df.resample(**kws)['all_time_interval_sec'].sum() / 3600
            assert all(hours_per_day[1:-1] == 24)
            # hours per day covered by physical vehicle state and null vehicle state
            x0 = df.loc[df['vehicle_state'].isin(['motion', 'idle', 'off'])].resample(**kws)['time_interval_sec'].sum() / 3600
            x1 = df.loc[pd.isnull(df['vehicle_state'])].resample(**kws)['all_time_interval_sec'].sum() / 3600
            coverage = pd.merge(x0, x1, left_index=True, right_index=True, how='outer')
            coverage.columns = ['daily_coverage_hours', 'daily_null_coverage_hours']
            assert all(np.nansum(coverage.values, axis=1)[1:-1] == 24)
            coverage['daily_coverage_frac'] = coverage['daily_coverage_hours'].values / np.nansum(coverage.values, axis=1)
            coverage = coverage.fillna(0).iloc[1:-1]

            # join df and coverage DataFrames
            assert len(set(coverage.index.values).intersection(df['utc'].values)) == coverage.shape[0]
            df['utc-date'] = np.array([x.date() for x in df['utc']]).astype('datetime64[ns]')
            df = pd.merge(left=df, right=coverage, left_on='utc-date', right_index=True, how='left')
            del df['utc-date']

        # save modified gps data and clean up
        spark_etl_save_gps(df, datadir, dst, vid, service)

        return pdf

    # pyspark code to distribute gps_interval_metrics_vid
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps').toPandas()
    vx = spark.range(start=0, end=vids.shape[0], step=1, numPartitions=int(1.5 * vids.shape[0]))
    dx = vx.groupby('id').applyInPandas(gps_interval_metrics_vid, schema=vx.schema).toPandas()
    # debug
    # vx = vx.toPandas()
    # gps_interval_metrics_vid(vx.loc[vx.index == 12])
    # gps_interval_metrics_vid(vx.loc[vids['VehicleId'] == ''])

def gps_enrichment_dbs(rc):
    """
    connection objects for gps enrichment databases
    - rc as 'all' - returns list of sqlalchemy connection objects for all keys in dbs dict
    - rc as int - returns tuple of (sqlalchemy connection object, schema) for specific database
    """
    assert (rc == 'all') or isinstance(rc, int)

    # initialize dbs dict
    dbs = {}

    # domain-a/b/c/d databases - only used for gps enrichment (can be scaled and used as needed)
    for x, xc in zip([0], ['a']):
    # for x, xc in zip([0, 1], ['a', 'b']):
        dbs[x] = {}
        dbs[x]['server'] = f'dev-mapping-domain-{xc}-cluster.cluster-cctoq0yyopdx.us-west-2.rds.amazonaws.com'
        dbs[x]['database'] = 'services'
        dbs[x]['username'] = 'osm_limited'
        dbs[x]['password'] = '27f90d43a35596ca930fef872a5db4a1'
        dbs[x]['schema'] = 'user_osm_limited'

    # return sqlalchemy database connection objects for all databases as a list
    if rc == 'all':
        return [
            sa.create_engine(f"""postgresql://{dbs[x]['username']}:{dbs[x]['password']}@{dbs[x]['server']}/{dbs[x]['database']}""").connect()
            for x in dbs.keys()]

    # return tuple of (sqlalchemy connection object, schema) for specific database
    else:
        assert isinstance(rc, int)
        assert rc in dbs.keys()
        return (
            sa.create_engine(f"""postgresql://{dbs[rc]['username']}:{dbs[rc]['password']}@{dbs[rc]['server']}/{dbs[rc]['database']}""").connect(),
            dbs[rc]['schema'])

def gps_enrich_dc(spark, datadir, src, dst, service):
    """
    distributed gps enrichment via Dennis Cheng SQL function
    """
    assert service in ['EC2', 'EMR']

    # gps enrichment function parameters
    enrichment_function = 'lytxlab_riskcore_enrichgps_trip'
    enrichment_schema = 'osm221107'

    # enrichment function schema from all dbs, validate all are same
    conns = gps_enrichment_dbs(rc='all')
    dbx = [database_function_schema(conn=x, schema=enrichment_schema, function=enrichment_function) for x in conns]
    [x.close() for x in conns]
    assert all([x.equals(dbx[0]) for x in dbx])
    enrichment_function_schema = dbx[0]

    def gps_enrich_dc_vid(pdf):

        # validate pdf, get vehicle-id
        assert pdf.shape[0] == 1
        vid = vids.loc[pdf['id'].values[0], 'VehicleId']

        # load gps data
        df = spark_etl_load_gps(datadir, src, vid, service)

        # clean COMPANY_ID column
        if np.all(pd.isnull(df['COMPANY_ID'])):
            df['COMPANY_ID'] = 0
        else:
            assert np.any(~pd.isnull(df['COMPANY_ID']))
            df.loc[pd.isnull(df['COMPANY_ID']), 'COMPANY_ID'] = df.loc[~pd.isnull(df['COMPANY_ID']), 'COMPANY_ID'].iloc[0]
        df['COMPANY_ID'] = df['COMPANY_ID'].astype('int')

        # connection objects and schema for random gps enrichment database
        conn, schema = gps_enrichment_dbs(rc=int(vids.loc[pdf['id'].values[0], 'db']))

        # initialize enriched DataFrame, scan over segments
        enriched = pd.DataFrame()
        now = datetime.now()
        c0 = ['TS_SEC', 'TS_USEC', 'COMPANY_ID', 'HEADING', 'SERIAL_NUMBER', 'longitude_gps', 'latitude_gps', 'SPEED', 'VehicleId']
        for segment in tqdm(np.sort(pd.unique(df['segmentId'])), desc='scanning segments', disable=False):

            # data input to enrichment function
            if not np.isnan(segment):
                dx = df.loc[df['segmentId'] == segment, c0]
            else:
                dx = df.loc[pd.isnull(df['segmentId']), c0]
            for col in ['TS_SEC', 'TS_USEC', 'HEADING', 'SPEED', 'SERIAL_NUMBER']:
                dx[col.lower()] = dx.pop(col)
            dx['vehicle_id'] = [x.lower() for x in dx.pop('VehicleId')]
            dx['longitude'] = dx.pop('longitude_gps')
            dx['latitude'] = dx.pop('latitude_gps')
            dx['company_id'] = dx.pop('COMPANY_ID').astype('int')
            dx['timestamp'] = [datetime.utcfromtimestamp(x) for x in dx['ts_sec']]
            name = f"""deleteme{vid.replace('-','_').lower()}"""
            dx.to_sql(name=name, con=conn.engine, schema=schema, if_exists='replace', index=False)

            # run enrichment function
            sql = f"""SELECT * FROM {enrichment_schema}.{enrichment_function}('{schema}.{name}', null, '{vid.lower()}')"""
            try:
                de = pd.read_sql_query(con=conn, sql=sa.text(sql)).sort_values('ts_sec').reset_index(drop=True)
            except:
                raise ValueError(f'{gethostname()}, {vid}')

            # drop data input to enrichment function temp table
            conn.execute(sa.text(f'DROP TABLE {schema}.{name}'))
            conn.commit()

            # validate consistency with dx
            assert de.shape[0] == dx.shape[0]
            de.index = dx.index
            assert sorted(np.array(list(set(list(dx.columns)).intersection(de.columns)))) == ['company_id', 'heading', 'latitude', 'longitude', 'serial_number', 'timestamp', 'ts_sec', 'ts_usec', 'vehicle_id']
            assert [x.lower() for x in dx['vehicle_id']] == [x.lower() for x in de['vehicle_id']]
            assert np.all(dx['ts_sec'].values == de['ts_sec'].values)
            assert np.all(dx['serial_number'].values == de['serial_number'].values)
            assert np.all(dx['timestamp'].values == de['timestamp'].values)
            assert np.all(np.isclose(dx['latitude'].values, de['latitude'].values, equal_nan=True))
            assert np.all(np.isclose(dx['longitude'].values, de['longitude'].values, equal_nan=True))
            de = de[list(set(list(de.columns)).difference(dx.columns))]

            # convert datatypes
            assert de.shape[1] == len(set(list(de.columns)).intersection(enrichment_function_schema['column'].values))
            for col in de.columns:
                t0 = enrichment_function_schema.loc[enrichment_function_schema['column'] == col, 'datatype'].iloc[0]
                if t0 == 'text':
                    de[col] = de[col].astype('object')
                elif t0 in ['float8', 'numeric', 'int8', 'int4', 'bool']:
                    de[col] = de[col].astype('float')
                elif t0 == 'timestamp':
                    de[col] = de[col].astype('datetime64[ns]')
                else:
                    raise TypeError()

            # rename columns
            de.columns = [x.replace('gps', 'gpse') if x[:3] == 'gps' else 'gpse_' + x for x in de.columns]
            de.columns = [x.replace('__', '_') for x in de.columns]

            # concat to enriched DataFrame
            enriched = pd.concat((enriched, de))

        # clean and validate enriched data
        conn.close()
        total_sec = (datetime.now() - now).total_seconds()
        enriched = enriched.sort_index()
        assert (df.shape[0] == enriched.shape[0])
        assert (len(set(df.columns).intersection(enriched.columns)) == 0)
        df = pd.concat((df, enriched), axis=1).copy()
        assert np.all(np.sort(df['TS_SEC'].values) == df['TS_SEC'].values)
        df['enrichment_minutes'] = total_sec / 60

        # save modified gps data and clean up
        spark_etl_save_gps(df, datadir, dst, vid, service)

        return pdf

    # pyspark code to distribute gps_enrich_dc_vid
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps').toPandas()
    vids['db'] = np.random.choice(np.arange(len(conns)), size=vids.shape[0], replace=True)
    vx = spark.range(start=0, end=vids.shape[0], step=1, numPartitions=int(1.5 * vids.shape[0]))
    dx = vx.groupby('id').applyInPandas(gps_enrich_dc_vid, schema=vx.schema).toPandas()
    # debug
    # vx = vx.toPandas()
    # gps_enrich_dc_vid(vx.loc[vx.index == 47])
    # gps_enrich_dc_vid(vx.loc[vids['VehicleId'] == ''])

def gps_enrich_dc_normalized(spark, datadir, src, service):
    """
    distributed distance-normalized gps enrichment via Dennis Cheng SQL function
    """
    assert service in ['EC2', 'EMR']
    transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform

    # database connection objects
    conns = gps_enrichment_dbs(rc='all')

    # validate consistent schema for osm221107.lytxlab_riskcore_enrichgps_trip_core on all dbs
    dbx = [database_function_schema(conn=x, schema='osm221107', function='lytxlab_riskcore_enrichgps_trip_core') for x in conns]
    assert all([x.equals(dbx[0]) for x in dbx])
    core_enrichment_function_schema = dbx[0]

    # validate consistent schema for osm221107.lytxlab_riskcore_normalize_and_enrich_gps_segments on all dbs
    dbx = [database_function_schema(conn=x, schema='osm221107', function='lytxlab_riskcore_normalize_and_enrich_gps_segments') for x in conns]
    assert all([x.equals(dbx[0]) for x in dbx])
    normalized_enrichment_function_schema = dbx[0]

    # close database connection objects
    [x.close() for x in conns]

    # segment columns to extract from normalized enrichment function
    cols = np.sort(normalized_enrichment_function_schema['column'])
    c0 = np.array(['segment__id', 'segment__geomsegment4326', 'segment__length_meters', 'segment__enrichgps_trip_gps_segment_id_range', 'segment__timestamp_range'])
    cols = np.sort(list(set(cols).difference(c0)))
    c1 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('hotspot' in x.lower()) and (x[:9] == 'segment__') and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c1)))
    c2 = np.array([x for x in cols if 'segment__curr' in x.lower()])
    cols = np.sort(list(set(cols).difference(c2)))
    c3 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('segment__intersection' in x.lower()) and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c3)))
    c4 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('segment__is_intersection' in x.lower()) and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    c4 = np.hstack((c4, np.array(['segment__is_publicroadintersection_count'])))
    cols = np.sort(list(set(cols).difference(c4)))
    c5 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('segment__private_roads' in x.lower()) and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c5)))
    c6 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('segment__public_roads' in x.lower()) and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c6)))
    keep_segments = np.hstack((c0, c1, c2, c3, c4, c5, c6))
    cols = np.array([x for x in cols if x[:9] != 'segment__'])

    # segment group colums to extract from normalized enrichment function
    c0 = np.array(['segmentgroup__geomsegment4326', 'segmentgroup__segment__id_range', 'segmentgroup__maneuver', 'segmentgroup__length_meters', 'segmentgroup__segment__timestamp_range', 'segmentgroup__is_start_segment', 'segmentgroup__is_end_segment', 'segmentgroup__segment_count'])
    cols = np.sort(list(set(cols).difference(c0)))
    c1 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('hotspot' in x.lower()) and (x[:14] == 'segmentgroup__') and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c1)))
    c2 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('segmentgroup__intersections' in x.lower()) and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c2)))
    c3 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('segmentgroup__private_roads' in x.lower()) and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c3)))
    c4 = np.array([x for _, (x, dx) in normalized_enrichment_function_schema.iterrows() if ('segmentgroup__public_roads' in x.lower()) and (dx not in ['_int8', '_float8', '_numeric', 'geometry'])])
    cols = np.sort(list(set(cols).difference(c4)))
    keep_segmentgroups = np.hstack((c0, c1, c2, c3, c4))

    def gps_enrich_dc_vid(pdf):

        # validate pdf, get vehicle-id
        assert pdf.shape[0] == 1
        vid = vids.loc[pdf['id'].values[0], 'VehicleId']

        # load gps data
        df = spark_etl_load_gps(datadir, src, vid, service)

        # clean COMPANY_ID column
        if np.all(pd.isnull(df['COMPANY_ID'])):
            df['COMPANY_ID'] = 0
        else:
            assert np.any(~pd.isnull(df['COMPANY_ID']))
            df.loc[pd.isnull(df['COMPANY_ID']), 'COMPANY_ID'] = df.loc[~pd.isnull(df['COMPANY_ID']), 'COMPANY_ID'].iloc[0]
        df['COMPANY_ID'] = df['COMPANY_ID'].astype('int')

        # connection objects and schema for random gps enrichment database
        conn, schema = gps_enrichment_dbs(rc=int(vids.loc[pdf['id'].values[0], 'db']))

        # initialize enriched DataFrame, scan over segments
        segment_enriched = pd.DataFrame()
        segmentgroup_enriched = pd.DataFrame()
        now = datetime.now()
        dc = ['TS_SEC', 'TS_USEC', 'COMPANY_ID', 'HEADING', 'SERIAL_NUMBER', 'longitude_gps', 'latitude_gps', 'SPEED', 'VehicleId']
        segments = np.sort(pd.unique(df.loc[~pd.isnull(df['segmentId']), 'segmentId']))
        for xs, segment in enumerate(tqdm(segments, desc='scanning segments', disable=False)):
            # segment = 0.0

            # data input to core enrichment function
            dx = df.loc[df['segmentId'] == segment, dc]
            for col in ['TS_SEC', 'TS_USEC', 'HEADING', 'SPEED', 'SERIAL_NUMBER']:
                dx[col.lower()] = dx.pop(col)
            dx['vehicle_id'] = [x.lower() for x in dx.pop('VehicleId')]
            dx['longitude'] = dx.pop('longitude_gps')
            dx['latitude'] = dx.pop('latitude_gps')
            dx['company_id'] = dx.pop('COMPANY_ID').astype('int')
            dx['timestamp'] = [datetime.utcfromtimestamp(x) for x in dx['ts_sec']]
            assert np.all(np.sort(dx['ts_sec'].values) == dx['ts_sec'].values)
            name = f"""deleteme{vid.replace('-','_').lower()}"""
            dx.to_sql(name=name, con=conn.engine, schema=schema, if_exists='replace', index=False)

            # run core enrichment function
            sql = f"""
                SELECT *
                FROM osm221107.lytxlab_riskcore_enrichgps_trip_core('{schema}.{name}', null, '{vid.lower()}')"""
            try:
                de = pd.read_sql_query(con=conn, sql=sa.text(sql)).sort_values('ts_sec').reset_index(drop=True)
            except:
                raise ValueError(f'{gethostname()}, {vid}, segment{segment:.0f}')

            # drop data input to core enrichment function
            conn.execute(sa.text(f'DROP TABLE {schema}.{name}'))
            conn.commit()

            # validate consistency with dx and convert datatypes
            assert de.shape[0] == dx.shape[0]
            de.index = dx.index
            assert pd.unique(de['id']).size == de.shape[0]
            assert sorted(np.array(list(set(list(dx.columns)).intersection(de.columns)))) == ['company_id', 'heading', 'latitude', 'longitude', 'serial_number', 'timestamp', 'ts_sec', 'ts_usec', 'vehicle_id']
            assert [x.lower() for x in dx['vehicle_id']] == [x.lower() for x in de['vehicle_id']]
            assert np.all(dx['ts_sec'].values == de['ts_sec'].values)
            assert np.all(dx['serial_number'].values == de['serial_number'].values)
            assert np.all(dx['timestamp'].values == de['timestamp'].values)
            assert np.all(np.isclose(dx['latitude'].values, de['latitude'].values, equal_nan=True))
            assert np.all(np.isclose(dx['longitude'].values, de['longitude'].values, equal_nan=True))
            de = align_dataframe_datatypes_sql(de, core_enrichment_function_schema)

            # write core table to database, reduce de, create conversion dictionaries
            name = name + '_core'
            de.to_sql(name=name, con=conn.engine, schema=schema, if_exists='replace', index=False)
            de = de[['id']].copy()

            # run normalized enrichment function
            sql = f"""
                -- SELECT *
                SELECT {','.join([','.join(keep_segments), ','.join(keep_segmentgroups)])}
                FROM osm221107.lytxlab_riskcore_normalize_and_enrich_gps_segments_core(
                    '{schema}.{name}', null, '{vid.lower()}', 30.0)
                ORDER BY segment__id"""
            try:
                dn = pd.read_sql_query(con=conn, sql=sa.text(sql))
            except:
                raise ValueError(f'{gethostname()}, {vid}, segment{segment:.0f}')
            dn = align_dataframe_datatypes_sql(df=dn, ds=normalized_enrichment_function_schema.loc[normalized_enrichment_function_schema['column'].isin(np.hstack((keep_segments, keep_segmentgroups)))])
            # dn = align_dataframe_datatypes_sql(df=dn, ds=normalized_enrichment_function_schema)

            # drop data input to normalized enrichment function
            conn.execute(sa.text(f'DROP TABLE {schema}.{name}'))
            conn.commit()
            # name = name + '_enriched'
            # dn.to_sql(name=name, con=conn.engine, schema=schema, if_exists='replace', index=False)

            # null case
            if dn.size == 0:
                continue

            # segment vehicle-id, segment-id, segment-distance, lat/lon
            dn['VehicleId'] = vid
            dn['segmentId'] = np.full(dn.shape[0], segment).astype('float')
            dn['distance_meters'] = dn.pop('segment__length_meters')
            geom_str = dn['segment__geomsegment4326'].values
            geom = from_wkb(geom_str)
            dn['longitude_gps'], dn['latitude_gps'] = np.nan, np.nan
            for x, gx in enumerate(geom):
                assert gx.geom_type in ['LineString', 'MultiLineString']
                dn.loc[x, 'longitude_gps'] = np.array(gx.xy[0])[0] if gx.geom_type == 'LineString' else np.array(gx.geoms[0].xy[0])[0] if gx.geom_type == 'MultiLineString' else None
                dn.loc[x, 'latitude_gps'] = np.array(gx.xy[1])[0] if gx.geom_type == 'LineString' else np.array(gx.geoms[0].xy[1])[0] if gx.geom_type == 'MultiLineString' else None
            dn['longitude'], dn['latitude'] = transform(xx=dn['longitude_gps'].values, yy=dn['latitude_gps'].values)

            # segment indices, timestamps, mph
            xr = dn.pop('segment__enrichgps_trip_gps_segment_id_range')
            x0 = np.array([x.lower for x in xr])
            x1 = np.array([x.upper - 1 for x in xr])
            assert all(np.in1d(x0, de['id'].values))
            assert all(np.in1d(x1, de['id'].values))
            assert all(x1 > x0)
            tr = dn.pop('segment__timestamp_range')
            ts_min = np.array([x.lower for x in tr])
            ts_max = np.array([x.upper for x in tr])
            assert all(ts_max > ts_min)
            assert all(ts_max[:-1] == ts_min[1:])
            assert np.unique(ts_max).size == ts_max.size
            dn['time_interval_sec'] = [x.total_seconds() for x in ts_max - ts_min]
            dn['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in ts_max]
            dn['utc'] = ts_max
            assert np.all(np.sort(dn['TS_SEC'].values) == dn['TS_SEC'].values)
            assert np.all(np.sort(dn['utc'].values) == dn['utc'].values)
            dn['mph'] = (0.000621371 * dn['distance_meters'] / (dn['time_interval_sec'] / 3600))

            # validate unique and sorted segment__id, and increasing segment__id wrt previous segment
            assert (np.unique(dn['segment__id']).size == dn.shape[0])
            assert np.all(np.sort(dn['segment__id'].values) == dn['segment__id'].values)
            if xs > 0:
                assert dn.loc[0, 'segment__id'] > segment_enriched.iloc[-1]['segment__id']

            # validate consistent null-values in segmentgroup columns
            sg_start = dn.pop('segmentgroup__is_start_segment').astype('bool')
            sg_end = dn.pop('segmentgroup__is_end_segment').astype('bool')
            ok = pd.isnull(dn['segmentgroup__segment__id_range'])
            cx = np.array([x for x in dn.columns if x[:13] == 'segmentgroup_'])
            assert all([np.all(pd.isnull(dn.loc[ok, x])) for x in cx])

            # extract segmentgroup data, reduce dn
            cn = np.array([x for x in dn.columns if x not in cx])
            assert cx.size + cn.size == dn.shape[1]
            ds = dn.loc[~ok, cx].copy().reset_index(drop=True)
            dn = dn[cn].copy()
            assert pd.isnull(ds['segmentgroup__maneuver']).sum() == 0

            # segmentgroup indices and timestamps
            assert pd.isnull(ds['segmentgroup__segment__id_range']).sum() == 0
            xr = ds.pop('segmentgroup__segment__id_range')
            x0 = np.array([x.lower for x in xr])
            x1 = np.array([x.upper - 1 for x in xr])
            assert all(np.in1d(x0, de['id'].values))
            assert all(np.in1d(x1, de['id'].values))
            assert all(x1 > x0)
            assert pd.isnull(ds['segmentgroup__segment__timestamp_range']).sum() == 0
            tr = ds.pop('segmentgroup__segment__timestamp_range')
            ts_min = np.array([x.lower for x in tr])
            ts_max = np.array([x.upper for x in tr])
            assert all(ts_max > ts_min)
            assert all(ts_max[:-1] == ts_min[1:])
            assert np.unique(ts_max).size == ts_max.size

            # segmentgroup vehicle-id, segment-id, segment-distance, lat/lon
            ds['VehicleId'] = vid
            ds['segmentId'] = np.full(ds.shape[0], segment).astype('float')
            ds['distance_meters'] = ds.pop('segmentgroup__length_meters')
            geom_str = ds['segmentgroup__geomsegment4326'].values
            geom = from_wkb(geom_str)
            ds['longitude_gps'], ds['latitude_gps'] = np.nan, np.nan
            for x, gx in enumerate(geom):
                assert gx.geom_type in ['LineString', 'MultiLineString']
                ds.loc[x, 'longitude_gps'] = np.array(gx.xy[0])[0] if gx.geom_type == 'LineString' else np.array(gx.geoms[0].xy[0])[0] if gx.geom_type == 'MultiLineString' else None
                ds.loc[x, 'latitude_gps'] = np.array(gx.xy[1])[0] if gx.geom_type == 'LineString' else np.array(gx.geoms[0].xy[1])[0] if gx.geom_type == 'MultiLineString' else None
            ds['longitude'], ds['latitude'] = transform(xx=ds['longitude_gps'].values, yy=ds['latitude_gps'].values)

            # segmentgroup start and end
            assert sg_start.sum() == sg_end.sum() == ds.shape[0]
            sg_start = np.where(sg_start)[0]
            sg_end = np.where(sg_end)[0]
            assert all(sg_end >= sg_start)
            assert sg_end.max() <= dn.shape[0]
            assert all(sg_start[1:] - sg_end[:-1] == 1)
            ds['sg_start'] = sg_start + segment_enriched.shape[0]
            ds['sg_end'] = sg_end + segment_enriched.shape[0]

            # segmentgroup time data
            ds['time_interval_sec'] = [x.total_seconds() for x in ts_max - ts_min]
            ds['TS_SEC'] = [(x - datetime(1970, 1, 1)).total_seconds() for x in ts_max]
            ds['utc'] = ts_max
            assert np.all(np.sort(ds['TS_SEC'].values) == ds['TS_SEC'].values)
            assert np.all(np.sort(ds['utc'].values) == ds['utc'].values)

            # concat to enriched DataFrames and clean up
            segment_enriched = pd.concat((segment_enriched, dn))
            segmentgroup_enriched = pd.concat((segmentgroup_enriched, ds))

        # clean and validate enriched data
        conn.close()
        total_sec = (datetime.now() - now).total_seconds()
        for dx in [segment_enriched, segmentgroup_enriched]:
            assert pd.isnull(dx['segmentId']).sum() == 0
            assert np.all(np.sort(dx['TS_SEC'].values) == dx['TS_SEC'].values)
        segment_enriched = segment_enriched.reset_index(drop=True)
        segment_enriched['cumulative_distance_miles'] = 0.000621371 * np.cumsum(segment_enriched['distance_meters'])
        segment_enriched['enrichment_minutes'] = total_sec / 60
        assert all(np.sort(segment_enriched['segment__id'].values) == segment_enriched['segment__id'].values)
        segmentgroup_enriched = segmentgroup_enriched.reset_index(drop=True)
        segmentgroup_enriched['cumulative_distance_miles'] = 0.000621371 * np.cumsum(segmentgroup_enriched['distance_meters'])
        segmentgroup_enriched['enrichment_minutes'] = total_sec / 60
        assert all(np.sort(segmentgroup_enriched['sg_start'].values) == segmentgroup_enriched['sg_start'].values)

        # save modified gps data
        spark_etl_save_gps(segment_enriched, datadir, 'gpse.parquet', vid, service)
        spark_etl_save_gps(segmentgroup_enriched, datadir, 'gpsm.parquet', vid, service)

        return pdf

    # pyspark code to distribute gps_enrich_dc_vid
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps').toPandas()
    vids['db'] = np.random.choice(np.arange(len(conns)), size=vids.shape[0], replace=True)
    vx = spark.range(start=0, end=vids.shape[0], step=1, numPartitions=int(1.5 * vids.shape[0]))
    dx = vx.groupby('id').applyInPandas(gps_enrich_dc_vid, schema=vx.schema).toPandas()
    # debug
    # vx = vx.toPandas()
    # gps_enrich_dc_vid(vx.loc[vx.index == 0])
    # segment 1, 9100FFFF-48A9-CB63-7941-A8A3E0CF0000
    # segment 4, 9100FFFF-48A9-E563-74FB-60A3E15B0000
    # segment 0, 9100FFFF-48A9-CB63-A364-A8A3E03F0000
    # gps_enrich_dc_vid(vx.loc[vids['VehicleId'] == ''])

def interpolate_gps(spark, loc):
    """
    distributed interpolation of gps data by vehicle-id
    - needs review before using again
    """

    def interpolate_vehicle_gps(pdf):
        """
        interpolate gps data for individual vehicle-id
        """

        # load gps data from parquet file
        assert pdf.shape[0] == 1
        vid = pdf['VehicleId'].values[0]
        fn = glob(os.path.join(loc, f'VehicleId={vid}', '*.parquet'))
        assert len(fn) == 1
        fn = fn[0]
        df = pq.ParquetFile(fn).read().to_pandas().sort_values('TS_SEC').reset_index(drop=True)
        assert all(np.sort(df['TS_SEC']) == df['TS_SEC'].values)
        assert all(np.sort(df['utc']) == df['utc'].values)
        df['VehicleId'] = vid

        # identify day indices
        x0 = df.assign(index=df.index).resample('D', on='utc')['index'].min().to_frame().rename(columns={'index': 'start index'})
        x1 = df.assign(index=df.index).resample('D', on='utc')['index'].max().to_frame().rename(columns={'index': 'end index'})
        assert all(x0.index == x1.index)
        dx = pd.merge(x0, x1, how='inner', left_index=True, right_index=True)

        # initialize interpolate records, scan over day indices after first day
        records = defaultdict(list)
        already_present_counter = 0
        for x, (utc, (x0, x1)) in enumerate(dx.iloc[1:].iterrows()):

            # previous record and subsequent record for interpolation
            xp = int(np.nanmax(dx.iloc[:x + 1].values))
            rp = df.loc[xp]
            xs = int(np.nanmin(dx.iloc[x + 1:].values))
            rs = df.loc[xs]

            # timestamp of the interpolated record
            t0 = utc.normalize()
            assert rp['utc'] < t0 <= rs['utc']

            # identify if any data for utc
            assert (~np.isnan(x0) and ~np.isnan(x1)) or (np.isnan(x0) and np.isnan(x1))
            day = ~np.isnan(x0) and ~np.isnan(x1)

            # no data for current day
            if not day:
                assert pd.Timestamp(rs['utc'].date()) > t0

            # data for current day, continue if t0 already a record
            if day:
                if t0 == df.loc[x0, 'utc']:
                    already_present_counter += 1
                    continue
                assert (rs['utc'] > t0) and (pd.Timestamp(rs['utc'].date()) == t0)

            # interpolation parameters
            d0 = (t0 - rp['utc']).total_seconds()
            d1 = (rs['utc'] - rp['utc']).total_seconds()
            assert d0 < d1
            frac = d0 / d1

            # interpolated record common items
            records['TS_SEC'].append(int(rp['TS_SEC'] + frac * (rs['TS_SEC'] - rp['TS_SEC'])))
            assert datetime.utcfromtimestamp(records['TS_SEC'][-1]) == t0
            records['utc'].append(t0)
            for field in ['longitude', 'latitude', 'longitude_gps', 'latitude_gps']:
                records[field].append(rp[field] + frac * (rs[field] - rp[field]))

            # interpolated segmentId
            if rp['segmentId'] == rs['segmentId']:
                records['segmentId'].append(rp['segmentId'])

            # atypical case, prev or subsequent record is not segmented
            else:
                assert np.isnan(rp['segmentId']) or np.isnan(rs['segmentId'])
                records['segmentId'].append(np.nan)

        # validate
        records = pd.DataFrame(records)
        records['VehicleId'] = vid
        assert records.shape[0] + already_present_counter == dx.shape[0] - 1

        # identify interpolated records, merge with df
        df['interpolated'] = False
        if records.size > 0:
            records['interpolated'] = True
            assert sorted(records.columns) == sorted(df.columns)
            df = pd.concat((records, df), axis=0).sort_values('TS_SEC').reset_index(drop=True)
            assert all(np.sort(df['utc']) == df['utc'].values)

        # rewrite data and remove original data, return pdf (Spark boilerplate)
        df.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])
        os.remove(fn)
        return pdf

    # distributed gps enrichment by vehicle-id
    vids = spark.sql(f'SELECT DISTINCT VehicleId FROM gps')
    dx = vids.groupby('VehicleId').applyInPandas(interpolate_vehicle_gps, schema=vids.schema).toPandas()
    # debug
    # vids = vids.toPandas()
    # interpolate_vehicle_gps(vids.loc[vids.index == 7])

def distributed_geocode(pdf, loc):
    """
    update gps dataset with following columns
    - timezone, eg America/New_York
    - localtime, datetime object
    - day_of_week, eg Sun
    - weekday, True/False
    - state, eg South Carolina
    - county, eg Laurens County
    - country, eg US

    needs review before using again, was used as:
    xs = spark.sql(f'SELECT DISTINCT VehicleId FROM gps')
    xs.groupby('VehicleId').applyInPandas(partial(lytx.distributed_geocode, loc=loc), schema=xs.schema).toPandas()
    gps = spark.read.parquet(loc)
    gps.createOrReplaceTempView('gps')
    assert lytx.count_all(spark, 'gps') == r0
    """
    import reverse_geocoder
    from timezonefinder import TimezoneFinder

    # geospatial processing objects
    tzf = TimezoneFinder()
    rg = reverse_geocoder.RGeocoder()

    # inner function (Spark boilerplate)
    def inner(pdf):

        # load data from parquet file
        assert pdf.shape[0] == 1
        vid = pdf['VehicleId'].values[0]
        fn = glob(os.path.join(loc, f'VehicleId={vid}', '*.parquet'))
        assert len(fn) == 1
        fn = fn[0]
        df = pq.ParquetFile(fn).read()
        assert all([x in df.column_names for x in ['latitude_gps', 'longitude_gps', 'TS_SEC']])
        df = df.to_pandas().sort_values('TS_SEC').reset_index(drop=True)
        df['VehicleId'] = vid

        # timestamp objects and valid lat/lon
        dt = np.array([pd.Timestamp(datetime.utcfromtimestamp(x)) for x in df['TS_SEC']])
        lat = df['latitude_gps'].values
        lon = df['longitude_gps'].values
        assert (lat.size > 0) and (lon.size > 0) and all(~np.isnan(lon)) and all(~np.isnan(lat))

        # timezone, localtime, day-of-week, weekday
        timezone = np.array([tzf.timezone_at(lng=a, lat=b) for a, b in zip(lon, lat)])
        localtime = np.array([a.tz_localize('UTC').astimezone(b).tz_localize(None) for a, b in zip(dt, timezone)])
        dow = np.array([x.strftime('%a') for x in localtime])
        weekday = np.array([False if x in ['Sat', 'Sun'] else True for x in dow])
        df['timezone'] = timezone
        df['localtime'] = localtime
        df['day_of_week'] = dow
        df['weekday'] = weekday

        # state, county, country
        locations = rg.query([(a, b) for a, b in zip(lat, lon)])
        df['state'] = np.array([x['admin1'] for x in locations])
        df['county'] = np.array([x['admin2'] for x in locations])
        df['country'] = np.array([x['cc'] for x in locations])

        # rewrite data and remove original data, return pdf (Spark boilerplate)
        df.to_parquet(path=loc, engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])
        os.remove(fn)
        return pdf

    # Spark boilerplate
    return inner(pdf)

# misc aws
def extract_and_save_video(record, fn, assert_exists=False, keep_dce=False):
    """
    extract video from s3 repository based on record as pandas Series, save to fn
    """

    # validate
    assert isinstance(record, pd.Series) and ('EventFilePath' in record) and ('EventFileName' in record)
    assert fn[-4:] == '.mkv'

    # build s3 key
    path = os.sep.join(record['EventFilePath'][15:].split('\\'))
    key = f"""dce-files/{path}/{record['EventFileName']}"""

    # download s3 url
    fx = fn[:-4] + '.dce'
    response = boto3.client('s3').list_objects_v2(Bucket='lytx-amlnas-us-west-2', Prefix=key)
    if assert_exists:
        assert ('Contents' in response.keys()) and (len(response['Contents']) == 1)
    if ('Contents' not in response.keys()) or (len(response['Contents']) != 1):
        return
    boto3.client('s3').download_file(Bucket='lytx-amlnas-us-west-2', Key=key, Filename=fx)

    # process video and clean up
    cmd = f'conda run -n ffmpeg3 --cwd {os.path.split(fx)[0]} '
    cmd += 'python /mnt/home/russell.burdt/miniconda3/envs/ffmpeg3/lib/python3.9/site-packages/dceutils/dce2mkv.py '
    cmd += fx
    os.system(cmd)
    if not keep_dce:
        os.remove(fx)
    f0 = os.path.join(os.path.split(fx)[0], f'{os.path.split(fx)[1][:-4]}_discrete.mkv')
    f1 = os.path.join(os.path.split(fx)[0], f'{os.path.split(fx)[1][:-4]}_merged.mkv')
    assert os.path.isfile(f0) and os.path.isfile(f1)
    os.remove(f0)
    os.rename(src=f1, dst=fn)

def process_aws_cost_data(df, timezone=None, convert_dates_only=False):
    """
    return clean aws cost data as Series and DataFrames:
        df0 - copy of input DataFrame
        df - input DataFrame with all null columns removed and line_item_usage_start/end_date converted
        sv - single value data as a Series
        mv - multi-value data as a DataFrame, columns with all duplicate data filtered to first sorted column name
        ds - multi-value string data as a DataFrame
        ds - multi-value numeric data as a DataFrame
    example usage,
    df0, df, sv, mv, ds, dn = process_aws_cost_data(df, timezone='America/Los_Angeles')
    -
    """

    # validate, copy initial data
    assert isinstance(df, pd.DataFrame)
    if timezone is not None:
        assert timezone in pytz.all_timezones
    df0 = df.copy()

    # remove all null columns and validate
    ok = ~np.all(pd.isnull(df), axis=0)
    ok = ok[ok.values].index.to_numpy()
    df = df.loc[:, ok].copy()
    assert np.all(np.any(~pd.isnull(df), axis=0).values)

    # validate hour resolution
    df['line_item_usage_end_date'] = [pd.Timestamp(x) for x in df.pop('line_item_usage_end_date')]
    df['line_item_usage_start_date'] = [pd.Timestamp(x) for x in df.pop('line_item_usage_start_date')]
    diff = df['line_item_usage_end_date'] - df['line_item_usage_start_date']
    assert all([x.total_seconds() == 3600 for x in diff])
    if timezone is not None:
        def convert(x):
            return x.tz_localize('UTC').astimezone(timezone).tz_localize(None)
        df['line_item_usage_end_date'] = [convert(x) for x in df.pop('line_item_usage_end_date')]
        df['line_item_usage_start_date'] = [convert(x) for x in df.pop('line_item_usage_start_date')]
    df['line_item_usage_end_day'] = [pd.Timestamp(x.date()) for x in df['line_item_usage_end_date']]
    df['line_item_usage_start_day'] = [pd.Timestamp(x.date()) for x in df['line_item_usage_start_date']]

    # stop here if requested
    if convert_dates_only:
        return df0, df, None, None, None, None

    # single-value data as a Series, multi-value data as a DataFrame
    sv = df.nunique()
    sv = sv[sv.values == 1].index
    mv = np.array(list(set(df.columns).difference(sv)))
    assert sorted(np.hstack((sv, mv))) == sorted(df.columns)
    sv = df.loc[0, sorted(sv)]
    sv.name = None
    mv = df.loc[:, sorted(mv)]

    # remove columns with all duplicate data
    ok = ~mv.T.duplicated(keep='first')
    cols = ok.index[ok.values].to_numpy()
    mv = df.loc[:, cols]

    # multi-value str and numeric data as DataFrames
    co = mv.dtypes.loc[mv.dtypes == object].index.to_numpy()
    cn = np.array(list(set(mv.columns).difference(co)))
    assert sorted(np.hstack((co, cn))) == sorted(mv.columns)
    ds = mv.loc[:, sorted(co)]
    dn = mv.loc[:, sorted(cn)]

    return df0, df, sv, mv, ds, dn
