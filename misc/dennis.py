
"""
event-videos from own device to s3 for Dennis
"""

import os
import lytx
import boto3
import numpy as np
import pandas as pd
import sqlalchemy as sa
from glob import glob
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


# event and behavior decoders
edw = lytx.get_conn('edw')
events = pd.read_sql_query(con=edw, sql=sa.text(f'SELECT Id, Name FROM hs.EventTriggerTypes_i18n'))
events_sub = pd.read_sql_query(con=edw, sql=sa.text(f'SELECT EventTriggerSubTypeId, UILabel FROM hs.EventTriggerSubTypes_i18n'))
behaviors = pd.read_sql_query(con=edw, sql=sa.text(f'SELECT Id, Name FROM hs.Behaviors_i18n'))

# own device metadata
dm = pd.read_sql_query(con=edw, sql=sa.text(f"""
    SELECT ERA.VehicleId, D.SerialNumber, ERA.CreationDate, ERA.DeletedDate
    FROM hs.EventRecorderAssociations AS ERA
    LEFT JOIN flat.Devices AS D ON ERA.EventRecorderId = D.DeviceId
    WHERE D.SerialNumber='MV01000865'
    AND ERA.VehicleId <> '00000000-0000-0000-0000-000000000000'"""))
assert dm.shape[0] == 1
vid = dm['VehicleId'].iloc[0]

# own device events in prev year
t1 = pd.Timestamp(datetime.now())
t0 = t1 - pd.Timedelta(days=365)
de = pd.read_sql_query(con=edw, sql=sa.text(f"""
    SELECT VehicleId, RecordDate, Latitude, Longitude, EventId, EventRecorderId, EventRecorderFileId, EventFilePath, EventFileName,
        BehaviourStringIds, CustomerEventIdString, SpeedAtTrigger, EventTriggerTypeId, EventTriggerSubTypeId
    FROM flat.Events
    WHERE Deleted=0 AND RecordDate BETWEEN '{datetime.isoformat(t0)}' AND '{datetime.isoformat(t1)}' AND VehicleId = '{vid}'"""))

# save videos and metadata for own device
df = defaultdict(list)
for _, xe in tqdm(de.iterrows(), desc='own device videos', total=de.shape[0]):
    fn = os.path.join(r'/mnt/home/russell.burdt/data/misc', xe['EventId'] + '.mkv')
    lytx.extract_and_save_video(record=xe, fn=fn, assert_exists=False, keep_dce=False)
    if os.path.isfile(fn):
        for col in ['VehicleId', 'EventId', 'EventRecorderId', 'Latitude', 'Longitude', 'CustomerEventIdString', 'SpeedAtTrigger']:
            df[col].append(xe[col])
        df['RecordDate'].append(datetime.isoformat(xe['RecordDate']))
        if xe['BehaviourStringIds'] is None:
            df['BehaviourStringIds'].append('')
            df['BehaviourNames'].append('')
        else:
            assert isinstance(xe['BehaviourStringIds'], str)
            df['BehaviourStringIds'].append(xe['BehaviourStringIds'])
            df['BehaviourNames'].append(','.join(behaviors.loc[behaviors['Id'].isin(np.array(xe['BehaviourStringIds'].split(',')).astype('int')), 'Name'].values))
        df['EventTriggerTypeId'].append(xe['EventTriggerTypeId'])
        df['EventName'].append(events.loc[events['Id'] == xe['EventTriggerTypeId'], 'Name'].iloc[0])
df = pd.DataFrame(df)
df.to_csv(os.path.join(r'/mnt/home/russell.burdt/data/misc', 'records.csv'), index=False)

# upload data to s3
client = boto3.client('s3')
fns = glob(os.path.join(r'/mnt/home/russell.burdt/data/misc', '*'))
for fn in tqdm(fns, desc='upload to s3'):
    client.upload_file(Filename=fn, Bucket='russell-lab-s3', Key=f'for-dennis/{os.path.split(fn)[1]}')
