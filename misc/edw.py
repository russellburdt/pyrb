"""
data analysis from EDW
"""

import os
import pyodbc
import numpy as np
import pandas as pd
from shutil import copyfile
from socket import gethostname
print('host: {}'.format(gethostname()))


def get_columns(conn, table):
    """
    return columns in table from connection object
    """
    cursor = conn.cursor()
    schema, table = table.split('.')
    cursor.columns(table=table, schema=schema)
    data = np.array(cursor.fetchall())
    return pd.DataFrame({
        'column': [x[3] for x in data],
        'SQL type': [x[5] for x in data]}).sort_values('column').reset_index(drop=True)

# get connection object, get columns in specific tables
conn = pyodbc.connect('DSN=EDW')
cols = {}
for table in ['flat.Events', 'gps.Trips', 'hs.EventTriggerTypes_i18n', 'hs.Behaviors_i18n']:
    cols[table] = get_columns(conn, table)

# get flat.Event records in a time window
time0, time1 = '2021-05-01', '2021-05-02'
query = """
    SELECT *
    FROM flat.Events
    WHERE RecordDate between '{}' and '{}'
    AND Deleted = 0
    AND EventFileName IS NOT NULL""".format(time0, time1)
x1 = pd.read_sql_query(query, conn, parse_dates=['RecordDate'])

# get metadata for flat.Events in a time window
time0, time1 = '2021-04-01', '2021-05-01'
query = """
    SELECT
        COUNT(*) AS n_events,
        COUNT(DISTINCT(SerialNumber)) AS n_devices,
        COUNT(DISTINCT(CompanyId)) AS n_companies,
        MIN(RecordDate) AS tmin,
        MAX(RecordDate) AS tmax
    FROM flat.Events
    WHERE RecordDate BETWEEN '{}' and '{}'""".format(time0, time1)
x2 = pd.read_sql_query(query, conn, parse_dates=['tmin', 'tmax']).squeeze()

# get trigger and behavior metadata, identify common names
x3 = pd.read_sql_query("""SELECT * FROM hs.EventTriggerTypes_i18n""", conn)
x4 = pd.read_sql_query("""SELECT * FROM hs.Behaviors_i18n""", conn)
x4b = pd.read_sql_query("""SELECT * FROM hs.Behaviors_i18n WHERE Name LIKE '%collision%'""", conn)

# get common behavior and trigger occurrences
query = """
    SELECT T.Id AS TriggerId, B.Id AS BehaviorId, T.Name AS Name
    FROM hs.EventTriggerTypes_i18n AS T
        INNER JOIN hs.Behaviors_i18n AS B
        ON T.Name = B.Name"""
x5 = pd.read_sql_query(query, conn)

# validate that all non-null 'BehaviourStringIds' values are lists of numbers as a string, e.g. '5,119' and '8,43,95'
time0, time1 = '2021-04-01', '2021-05-01'
query = """
    SELECT DISTINCT BehaviourStringIds
    FROM flat.Events
    WHERE BehaviourStringIds IS NOT NULL
    AND RecordDate between '{}' and '{}'
    AND Deleted = 0
    AND EventFileName IS NOT NULL""".format(time0, time1)
x6 = pd.read_sql_query(query, conn)
valid = [str(x) for x in range(10)] + [',']
for value in x6['BehaviourStringIds'].values:
    assert all([xi in valid for xi in value])

# basic trigger stats in a time window
time0, time1 = '2021-04-01', '2021-05-01'
query = """
    SELECT
        T.Name,
        E.EventTriggerTypeId,
        COUNT(*) AS n_events,
        STR(100 * CAST(SUM(CASE WHEN EventScore > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 5, 2) AS percent_scored,
        STR(100 * CAST(SUM(CASE WHEN IsCoachable = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 5, 2) AS percent_coachable,
        STR(100 * CAST(SUM(CASE WHEN BehaviourStringIds IS NOT NULL THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 5, 2) AS percent_with_any_behaviours
    FROM flat.Events AS E
        INNER JOIN hs.EventTriggerTypes_i18n AS T
        ON T.Id = E.EventTriggerTypeId
    WHERE E.RecordDate BETWEEN '{}' and '{}'
    GROUP BY E.EventTriggerTypeId, T.Name
    ORDER BY n_events DESC""".format(time0, time1)
x7 = pd.read_sql_query(query, conn, parse_dates=['RecordDate'])

# basic behavior stats in a time window
time0, time1 = '2021-04-01', '2021-04-02'
query = """
    SELECT EventId, BehaviourStringIds
    FROM flat.Events
    WHERE deleted = 0
    AND RecordDate BETWEEN '{}' AND '{}'
    """.format(time0, time1)
x8a = pd.read_sql_query(query, conn, parse_dates=['RecordDate'])
query = """
    SELECT EventId, BehaviourStringIds, value AS BehaviorId
    FROM flat.Events AS E
        CROSS APPLY STRING_SPLIT(COALESCE(BehaviourStringIds, '-1'), ',')
    WHERE deleted = 0
    AND RecordDate BETWEEN '{}' AND '{}'""".format(time0, time1)
x8b = pd.read_sql_query(query, conn, parse_dates=['RecordDate'])
query = """
    SELECT
        value AS BehaviorId,
        COUNT(DISTINCT(EventId)) AS n_events_with_behavior
    FROM flat.Events AS E
        CROSS APPLY STRING_SPLIT(COALESCE(BehaviourStringIds, '-1'), ',')
    WHERE deleted = 0
    AND RecordDate BETWEEN '{}' AND '{}'
    GROUP BY value""".format(time0, time1)
x8c = pd.read_sql_query(query, conn, parse_dates=['RecordDate'])
query = """
    WITH A AS (
        SELECT
            value AS BehaviorId,
            COUNT(DISTINCT(EventId)) AS n_events_with_behavior
        FROM flat.Events AS E
            CROSS APPLY STRING_SPLIT(COALESCE(BehaviourStringIds, '-1'), ',')
        WHERE deleted = 0
        AND RecordDate BETWEEN '{}' AND '{}'
        GROUP BY value)
    SELECT B.Name, A.BehaviorId, A.n_events_with_behavior
    FROM A
        LEFT JOIN hs.Behaviors_i18n AS B
        ON A.BehaviorId = B.Id
    ORDER BY A.n_events_with_behavior DESC""".format(time0, time1)
x8 = pd.read_sql_query(query, conn)

# get valid dce filenames in a time window
time0, time1 = '2021-04-01', '2021-04-02'
query = """
    SELECT
        EventId,
        REPLACE(REPLACE(CONCAT(
            EventFilePath, '/',
            CAST(CONCAT('<t>', REPLACE(EventFileName, '_', '</t><t>'), '</t>') AS XML).value('/t[2]', 'varchar(100)'), '/', EventFileName), '\\\\drivecam.net', '/mnt'), '\\', '/') AS fn
    FROM flat.Events
    WHERE deleted = 0
    AND RecordDate BETWEEN '{}' AND '{}'
    """.format(time0, time1)
x9 = pd.read_sql_query(query, conn, parse_dates=['RecordDate'])

# full metadata for collisions in a time window, copy single event .dce file to X-data
time0, time1 = '2021-05-01', '2021-05-07'
query = """
    SELECT
        E.EventId,
        E.RecordDate,
        T.Name AS EventName,
        E.CompanyId,
        C.CompanyName,
        C.IndustryDesc,
        E.DriverId,
        E.GroupId,
        E.VehicleId,
        E.VehicleTypeId,
        E.Latitude,
        E.Longitude,
        E.EventTriggerTypeId,
        E.BehaviourStringIds,
        REPLACE(REPLACE(CONCAT(
            E.EventFilePath, '/',
            CAST(CONCAT('<t>', REPLACE(E.EventFileName, '_', '</t><t>'), '</t>') AS XML).value('/t[2]', 'varchar(100)'), '/', E.EventFileName), '\\\\drivecam.net', '/mnt'), '\\', '/') AS fn
    FROM flat.Events AS E
        LEFT JOIN hs.EventTriggerTypes_i18n AS T
        ON T.Id = E.EventTriggerTypeId
        LEFT JOIN flat.Companies AS C
        ON C.CompanyId = E.CompanyId
    WHERE deleted = 0
    AND RecordDate BETWEEN '{}' AND '{}'
    AND BehaviourStringIds LIKE '%47%'
    AND EventFileName LIKE 'event%'
    """.format(time0, time1)
x10 = pd.read_sql_query(query, conn, parse_dates=['RecordDate'])
# x=51; copyfile(x10.loc[x, 'fn'], os.path.join(r'/mnt/AML_Dev/sandbox/russell.burdt/data', os.path.split(x10.loc[x, 'fn'])[1]))
# print(x10.loc[x].squeeze())

# get trips and number of events in a time and geo window
time0, time1 = '2021-05-01', '2021-05-02'
lat0, lat1 = 33.5, 34.3
lon0, lon1 = -116.8, -115.6
query = """
    SELECT
        T.Id AS TripId,
        COUNT(1) AS n_events
        -- MIN(T.EventRecorderId) AS EventRecorderId,
        -- MIN(T.VehicleId) AS VehicleId,
        -- MIN(T.GroupId) AS GroupId,
        -- MIN(T.Distance) AS Distance,
        -- MIN(T.StartTimeUTC) AS StartTimeUTC,
        -- MIN(T.EndTimeUTC) AS EndTimeUTC,
        -- MIN(DATEDIFF(MINUTE, T.StartTimeUTC, T.EndTimeUTC)) AS Duration
        -- T.StartPosition.Lat AS Lat0,
        -- T.StartPosition.Long AS Lon0,
        -- T.EndPosition.Lat AS Lat1,
        -- T.EndPosition.Long AS Lon1,
        -- V.RecordDate,
        -- V.EventTriggerTypeId
    FROM gps.Trips AS T
        LEFT JOIN flat.EventRecorderConfigs AS E
        ON E.EventRecorderId = T.EventRecorderId
        INNER JOIN flat.Events AS V
        ON V.VehicleId = T.VehicleId AND V.RecordDate BETWEEN T.StartTimeUTC AND T.EndTimeUTC
    WHERE T.StartTimeUTC BETWEEN '{time0}' AND '{time1}'
    AND T.EndTimeUTC BETWEEN '{time0}' AND '{time1}'
    AND T.EndTimeUTC > T.StartTimeUTC
    AND DATEDIFF(MINUTE, T.StartTimeUTC, T.EndTimeUTC) BETWEEN 60 AND 720
    AND T.StartPosition.Lat BETWEEN {lat0} AND {lat1}
    AND T.EndPosition.Lat BETWEEN {lat0} AND {lat1}
    AND T.StartPosition.Long BETWEEN {lon0} AND {lon1}
    AND T.EndPosition.Long BETWEEN {lon0} AND {lon1}
    AND T.VehicleId <> '00000000-0000-0000-0000-000000000000'
    AND E.GpsTrailEnable = 1
    GROUP BY T.Id
    ORDER BY T.Id
    """.format(time0=time0, time1=time1, lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1)
x11 = pd.read_sql_query(query, conn, parse_dates=['StartTimeUTC', 'EndTimeUTC'])