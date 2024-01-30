
"""
extract data for validation of new data platform
- based on raw trackpoints data for individual vehicles on a single day
"""

import pickle
import pandas as pd
import numpy as np
from time import sleep
from lytx import get_conn
from socket import gethostname
from datetime import datetime, timedelta
from pyproj import Transformer
from tqdm import tqdm
from ipdb import set_trace


# get connection and results filename based on host
if gethostname() == 'rburdt-840g4':
    host = 'local'
    conn = get_conn('snowflake')
    fn = r'c:/Users/russell.burdt/Data/data-platform/results-new-data-platform.p'
elif gethostname() == 'russell-ubuntu-1-8brrp':
    host = 'pod'
    conn = get_conn('hadoop')
    fn = r'/mnt/AML_Dev/sandbox/russell.burdt/data/results-hadoop.p'

def convert_time(time0, time1):
    """
    convert time bounds in %Y-%m-%d format to epoch time and ymd strings representing hadoop partitions
    """
    time_to_ymd = lambda x, d: (datetime.strptime(x, '%Y-%m-%d') + timedelta(days=d)).strftime('%Y%m%d')
    time_to_epoch = lambda x: int((datetime.strptime(x, '%Y-%m-%d') - datetime(1970, 1, 1)).total_seconds())
    ymd0, ymd1 = time_to_ymd(time0, -1), time_to_ymd(time1, 3)
    epoch0, epoch1 = time_to_epoch(time0), time_to_epoch(time1)
    return ymd0, ymd1, epoch0, epoch1

# cases from Dennis Cheng email Friday, July 30, 2021 6:33 PM
cases = pd.DataFrame(data={
    'vehicleid': [
        '9100ffff-48a9-cb63-732f-a8a3e0cf0000',
        '9100ffff-48a9-cb63-732f-a8a3e0cf0000',
        '9100ffff-48a9-cb63-81b2-a8a3e03f0000',
        '9100ffff-48a9-cb63-77da-a8a3e0cf0000',
        '9100ffff-48a9-cb63-77d5-a8a3e0cf0000',
        '9100ffff-48a9-cb63-bb11-a8a3e03f0000',
        '9100ffff-48a9-cb63-a32f-a8a3e03f0000',
        '9100ffff-48a9-cc63-31cc-a8a3e03f0000',
        '9100ffff-48a9-cb63-a312-a8a3e03f0000'],
    'timestamp': [datetime.strptime(x, '%m/%d/%Y %H:%M') for x in [
        '7/20/2021 19:53',
        '7/14/2021 21:11',
        '7/14/2021 19:50',
        '7/14/2021 19:47',
        '7/14/2021 20:04',
        '7/14/2021 19:42',
        '7/14/2021 19:50',
        '7/14/2021 20:42',
        '7/17/2021 08:17']]})

# initialize results, geo-transform, scan over cases
results = {}
transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
for x, row in tqdm(cases.iterrows(), desc='scanning cases', total=cases.shape[0]):

    # get time parameters for row
    time0 = row['timestamp'].strftime('%Y-%m-%d')
    time1 = (row['timestamp'] + timedelta(days=1)).strftime('%Y-%m-%d')
    ymd0, ymd1, epoch0, epoch1 = convert_time(time0, time1)

    # snowflake query
    if host == 'local':
        query = """
            SELECT TSSEC, LATITUDE, LONGITUDE, SPEED
            FROM GPS.GPS_ENRICHED_TBL
            WHERE TSSEC BETWEEN {epoch0} AND {epoch1}
            AND VEHICLEID = '{vid}'""".format(
                epoch0=epoch0, epoch1=epoch1, vid=row['vehicleid'])

    # hadoop query
    elif host == 'pod':
        query = """
            SELECT gpsdatetime, latitude, longitude, speed
            FROM geo_master.trackpoints
            WHERE yearmonthday BETWEEN {ymd0} AND {ymd1}
            AND gpsdatetime BETWEEN '{time0}' AND '{time1}'
            AND vehicleid = '{vid}'""".format(
                time0=time0, time1=time1, ymd0=ymd0, ymd1=ymd1, vid=row['vehicleid'].upper())

    # run and time query (use retry logic for hadoop source)
    if host == 'local':
        df = pd.read_sql_query(query, conn)
        df['gpsdatetime'] = pd.to_datetime(df['TSSEC'], unit='s')
        df.columns = [x.lower() for x in df.columns]
    elif host == 'pod':
        counter = 0
        df = None
        while df is None:
            try:
                df = pd.read_sql_query(query, conn)
            except:
                counter += 1
                sleep(10)
                conn = get_conn('hadoop')
                print('hadoop query failed, retry number {}'.format(counter))

    # convert units
    webm = np.full((df.shape[0], 2), np.nan)
    for xi, (lon, lat) in df[['longitude', 'latitude']].iterrows():
        webm[xi, :] = transform(lon, lat)
    df[['longitude', 'latitude']] = webm

    # update results
    results[x] = {}
    results[x]['query'] = query
    results[x]['vid'] = row['vehicleid']
    results[x]['time0'] = time0
    results[x]['time1'] = time1
    results[x]['data'] = df
with open(fn, 'wb') as fid:
    pickle.dump(results, fid)
