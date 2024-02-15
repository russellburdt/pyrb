
"""
metrics based on JD event-review dashboard
"""

import os
import lytx
import numpy as np
import pandas as pd
from functools import partial
from pyspark.sql.types import StructType, StructField, StringType, BooleanType
from datetime import datetime
from tqdm import tqdm
from ipdb import set_trace


# datadir and spark session
datadir = r'/mnt/home/russell.burdt/data/driver-safety/event-review/dft'
assert os.path.isdir(datadir)
spark = lytx.spark_session(memory='8g', cores='*')

# metadata and parquet datasets
dx = pd.read_pickle(os.path.join(datadir, 'event-recorder-associations.p'))
dt = pd.read_pickle(os.path.join(datadir, 'intervals.p'))
dp = pd.read_pickle(os.path.join(datadir, 'population-dataframe.p'))
dbv = pd.read_pickle(os.path.join(datadir, 'time-bounds-vehicle.p'))
events = spark.read.parquet(os.path.join(datadir, 'events.parquet'))
events.createOrReplaceTempView('events')
behaviors = spark.read.parquet(os.path.join(datadir, 'behaviors.parquet'))
behaviors.createOrReplaceTempView('behaviors')
dce = spark.read.parquet(os.path.join(datadir, 'dce_scores.parquet'))
dce.createOrReplaceTempView('dce')
dcex = lytx.coverage_dce_scores_events(spark)

# threshold dictionary
thresh = {'collision': np.linspace(2e-5, 1e-4, 9)}
thresh['braking'] = np.tile(0.38, thresh['collision'].size)
thresh['cornering'] = np.tile(0.38, thresh['collision'].size)

# scan over thresh
dm = []
for collision, braking, cornering in zip(thresh['collision'], thresh['braking'], thresh['cornering']):

    # status
    print(f'collision - {collision}, braking - {braking}, cornering - {cornering}')

    # event-review metrics for thresh
    dmt = pd.Series(dtype='float')
    dmt['collision threshold'] = collision
    dmt['braking threshold'] = braking
    dmt['cornering threshold'] = cornering
    dmt['num accel events'] = spark.sql(f"""
        SELECT COUNT(EventId) AS nc
        FROM events
        WHERE EventTriggerTypeId=30""").toPandas().values.flatten()[0]
    dmt['num accel events, any score'] = spark.sql(f"""
        SELECT COUNT(DISTINCT(events.EventId))
        FROM events
        INNER JOIN dce on events.EventId = dce.EventId""").toPandas().values.flatten()[0]
    dxx = spark.sql(f"""
        WITH nx AS (
            SELECT dce.ModelKey, events.EventId,
                (CASE WHEN (events.BehaviourStringIds IS NOT NULL) THEN 1 ELSE 0 END) AS behaviors,
                (CASE WHEN (events.ReviewerId <> '00000996-0000-0000-0000-000000000000') THEN 1 ELSE 0 END) AS reviewed,
                (CASE WHEN (MAX(ABS(dce.ModelValue)) >= {collision}) AND (dce.ModelKey = 'collision') THEN 1 ELSE 0 END) AS collision,
                (CASE WHEN (MAX(ABS(dce.ModelValue)) >= {braking}) AND (dce.ModelKey = 'braking') THEN 1 ELSE 0 END) AS braking,
                (CASE WHEN (MAX(ABS(dce.ModelValue)) >= {cornering}) AND (dce.ModelKey = 'cornering') THEN 1 ELSE 0 END) AS cornering
            FROM events
            INNER JOIN dce on events.EventId = dce.EventId
            GROUP BY events.EventId, events.ReviewerId, events.BehaviourStringIds, dce.ModelKey),
        thresh AS (
            SELECT EventId, MAX(reviewed) AS reviewed,
                MAX(behaviors) AS behaviors, MAX(collision) AS collision, MAX(braking) AS braking, MAX(cornering) AS cornering
            FROM nx
            GROUP BY EventId)
        -- row 0, scores over thresh
        SELECT SUM(thresh.collision) AS a, SUM(thresh.braking) AS b, SUM(thresh.cornering) AS c
        FROM thresh
            UNION ALL
        -- row 1, any score over thresh
        SELECT COUNT(*) AS a, 0 AS b, 0 AS c
        FROM thresh
        WHERE CAST(collision AS BOOLEAN) or CAST(braking AS BOOLEAN) or CAST(cornering AS BOOLEAN)
            UNION ALL
        -- row 2, all scores under thresh
        SELECT COUNT(*) AS a, 0 AS b, 0 AS c
        FROM thresh
        WHERE NOT(CAST(collision AS BOOLEAN) or CAST(braking AS BOOLEAN) or CAST(cornering AS BOOLEAN))
            UNION ALL
        -- row 3, any score over thresh, reviewed
        SELECT COUNT(*) AS a, 0 AS b, 0 AS c
        FROM thresh
        WHERE CAST(reviewed AS BOOLEAN)
        AND (CAST(collision AS BOOLEAN) or CAST(braking AS BOOLEAN) or CAST(cornering AS BOOLEAN))
            UNION ALL
        -- row 4, all scores under thresh, reviewed
        SELECT COUNT(*) AS a, 0 AS b, 0 AS c
        FROM thresh
        WHERE CAST(reviewed AS BOOLEAN)
        AND NOT(CAST(collision AS BOOLEAN) or CAST(braking AS BOOLEAN) or CAST(cornering AS BOOLEAN))
            UNION ALL
        -- row 5, any score over thresh, reviewed, with behaviors
        SELECT COUNT(*) AS a, 0 AS b, 0 AS c
        FROM thresh
        WHERE CAST(reviewed AS BOOLEAN)
        AND CAST(behaviors AS BOOLEAN)
        AND (CAST(collision AS BOOLEAN) or CAST(braking AS BOOLEAN) or CAST(cornering AS BOOLEAN))
            UNION ALL
        -- row 6, any score over thresh, reviewed, no behaviors
        SELECT COUNT(*) AS a, 0 AS b, 0 AS c
        FROM thresh
        WHERE CAST(reviewed AS BOOLEAN)
        AND NOT CAST(behaviors AS BOOLEAN)
        AND (CAST(collision AS BOOLEAN) or CAST(braking AS BOOLEAN) or CAST(cornering AS BOOLEAN))""").toPandas()
    dmt['collision score over thresh'] = dxx.loc[0, 'a']
    dmt['braking score over thresh'] = dxx.loc[0, 'b']
    dmt['cornering score over thresh'] = dxx.loc[0, 'c']
    dmt['any score over thresh'] = dxx.loc[1, 'a']
    dmt['all scores under thresh'] = dxx.loc[2, 'a']
    dmt['any score over thresh, reviewed'] = dxx.loc[3, 'a']
    dmt['all scores under thresh, reviewed'] = dxx.loc[4, 'a']
    dmt['any score over thresh, reviewed, with behaviors'] = dxx.loc[5, 'a']
    dmt['any score over thresh, reviewed, no behaviors'] = dxx.loc[6, 'a']
    dm.append(dmt)
dm = pd.DataFrame(dm)
for x in [x for x in dm.columns if x not in ['collision threshold', 'braking threshold', 'cornering threshold']]:
    dm[x] = dm[x].astype('int')
dm.to_pickle(os.path.join(datadir, 'dashboard.p'))
