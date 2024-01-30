
"""
extract data for validation of new data platform
- based on num of trackpoints per vehicle id per day
"""

import pandas as pd
import pickle
from time import sleep
from lytx import get_conn
from socket import gethostname
from datetime import datetime, timedelta
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

def get_num_trackpoints_by_vid(host, conn, time0, time1, lat0=32.5, lat1=33.1, lon0=-117.5, lon1=-116.7):
    """
    build and run query for num of trackpoints by vehicleid based on query parameters
    """

    # convert time parameters
    ymd0, ymd1, epoch0, epoch1 = convert_time(time0, time1)

    # snowflake source
    if host == 'local':
        query = """
            SELECT
                VEHICLEID,
                COUNT(1) AS ntps,
                DATEADD(second, MIN(TSSEC), '1970-01-01') AS tmin,
                DATEADD(second, MAX(TSSEC), '1970-01-01') AS tmax
            FROM GPS.GPS_ENRICHED_TBL
            WHERE TSSEC BETWEEN {epoch0} AND {epoch1}
            AND latitude BETWEEN {lat0} AND {lat1}
            AND longitude BETWEEN {lon0} AND {lon1}
            AND VEHICLEID <> '00000000-0000-0000-0000-000000000000'
            GROUP BY VEHICLEID
            ORDER BY VEHICLEID""".format(
                epoch0=epoch0, epoch1=epoch1, lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1)

    # hadoop source
    elif host == 'pod':
        query = """
            SELECT
                vehicleid,
                COUNT(1) AS ntps,
                MIN(gpsdatetime) AS tmin,
                MAX(gpsdatetime) AS tmax
            FROM geo_master.trackpoints
            WHERE yearmonthday BETWEEN {ymd0} AND {ymd1}
            AND gpsdatetime BETWEEN '{time0}' AND '{time1}'
            AND latitude BETWEEN {lat0} AND {lat1}
            AND longitude BETWEEN {lon0} AND {lon1}
            AND vehicleid <> '00000000-0000-0000-0000-000000000000'
            GROUP BY vehicleid
            ORDER BY vehicleid""".format(
                time0=time0, time1=time1, ymd0=ymd0, ymd1=ymd1, lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1)

    # run and time query (use retry logic for hadoop source)
    now = datetime.now()
    if host == 'local':
        df = pd.read_sql_query(query, conn)
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

    df.columns = [x.lower() for x in df.columns]
    df['vehicleid'] = [x.upper() for x in df['vehicleid']]
    sec = (datetime.now() - now).total_seconds()

    return df, sec, query

# get num trackpoints by vid for indiviudal days and save results in a dictionary
results = {}
for day in tqdm(range(1, 29), desc='scanning days'):
    time0 = '2021-07-{:02d}'.format(day)
    time1 = '2021-07-{:02d}'.format(day + 1)
    df, sec, query = get_num_trackpoints_by_vid(host, conn, time0, time1)
    results[day] = {}
    results[day]['query'] = query
    results[day]['sec'] = sec
    results[day]['data'] = df
with open(fn, 'wb') as fid:
    pickle.dump(results, fid)
