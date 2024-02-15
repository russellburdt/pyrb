
"""
misc EventRecorder queries
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from lytx import get_conn, get_columns

# columns
edw = get_conn('edw')
events = get_columns(edw, 'flat.Events')
er = get_columns(edw, 'hs.EventRecorders')
era = get_columns(edw, 'hs.EventRecorderAssociations')
erf = get_columns(edw, 'hs.EventRecorderFiles')
erft = get_columns(edw, 'hs.EventRecorderFileTriggers')
erfto = get_columns(edw, 'hs.EventRecorderFileTriggerOutputs')

# ER data for multiple ERs
dp = pd.read_pickle(r'/mnt/home/russell.burdt/data/collision-model/munich-re/dp.p')
x = [55]
ers = ','.join([f"""'{x}'""" for x in dp.loc[x, 'ER-Id'].values])
vids = ','.join([f"""'{x}'""" for x in dp.loc[x,'VehicleId'].values])
tmin = dp.loc[x, 'time0'].min().strftime('%m-%d-%Y %H:%M:%S')
tmax = dp.loc[x, 'time1'].max().strftime('%m-%d-%Y %H:%M:%S')
now = datetime.now()
query0 = f"""
    SELECT
        E.EventId,
        E.RecordDate,
        E.Latitude,
        E.Longitude,
        E.EventTriggerTypeId,
        E.EventRecorderId,
        E.EventRecorderFileId,
        E.EventFileName
    FROM flat.Events AS E
    WHERE E.Deleted = 0
    AND E.VehicleId IN ({vids})
    AND RecordDate BETWEEN '{tmin}' AND '{tmax}'"""
dx0 = pd.read_sql_query(query0, edw)
print(f'query flat.Events, {(datetime.now() - now).total_seconds():.1f}sec')
now = datetime.now()
query1 = f"""
    SELECT
        ERF.EventRecorderId,
        ERF.EventRecorderFileId,
        ERF.CreationDate,
        ERF.FileName,
        ERF.EventTriggerTypeId
    FROM hs.EventRecorderFiles AS ERF
    WHERE ERF.EventRecorderId IN ({ers})
    AND ERF.CreationDate BETWEEN '{tmin}' AND '{tmax}'"""
dx1 = pd.read_sql_query(query1, edw)
print(f'query hs.EventRecorderFiles, {(datetime.now() - now).total_seconds():.1f}sec')
now = datetime.now()
query2 = f"""
    SELECT
        ERF.EventRecorderId,
        ERF.EventRecorderFileId,
        ERF.CreationDate,
        ERF.FileName,
        ERF.EventTriggerTypeId,
        ERFT.TriggerTime,
        ERFT.Position.Lat AS lat,
        ERFT.Position.Long as lon,
        ERFT.ForwardExtremeAcceleration,
        ERFT.SpeedAtTrigger,
        ERFT.PostedSpeedLimit
    FROM hs.EventRecorderFiles AS ERF
        LEFT JOIN hs.EventRecorderFileTriggers AS ERFT
        ON ERFT.EventRecorderFileId = ERF.EventRecorderFileId
    WHERE ERF.EventRecorderId IN ({ers})
    AND ERF.CreationDate BETWEEN '{tmin}' AND '{tmax}'"""
dx2 = pd.read_sql_query(query2, edw)
print(f'query hs.EventRecorderFiles/Triggers, {(datetime.now() - now).total_seconds():.1f}sec')

dx3 = pd.merge(left=dx0, right=dx2, on='EventRecorderFileId', how='inner')
assert dx3.shape[0] == dx0.shape[0]
# assert all(dx2['CreationDate'] > dx2['RecordDate'])



"""
BCP "SELECT top(2)  FROM [EDW].[hs].[EventRecorderFileTriggers]" queryout "./edw_hs_EventRecorderFileTriggers.csv" -c -S "PHV0V-DWHSQL01" -T

SELECT *
FROM [EDW].[hs].[EventRecorderFiles]
ORDER BY EventRecorderFileId
OFFSET 1000000 ROWS FETCH NEXT 3000 ROWS ONLY;
"""


# ER association and model for individual vehicle
# vid = '9100FFFF-48A9-CB63-A300-A8A3E03F0000'
# conn = get_conn('edw')
# query2 = f"""
#     SELECT
#         ERA.VehicleId,
#         ERA.EventRecorderId,
#         ERA.CreationDate,
#         ERA.DeletedDate,
#         ER.SerialNumber,
#         ER.Model
#     FROM hs.EventRecorderAssociations AS ERA
#         LEFT JOIN hs.EventRecorders AS ER
#         ON ER.Id = ERA.EventRecorderId
#     WHERE ERA.VehicleId = '{vid}'
#     ORDER BY ERA.CreationDate ASC"""
# dx2 = pd.read_sql_query(query2, conn)


"""
select top (10) *
from [hs].[EventRecorderFiles] ERF
left outer join [hs].[EventRecorderFileTriggers] ERFT on ERFT.EventRecorderFileId = ERF.EventRecorderFileId
left outer join [hs].[EventRecorderFileTriggerOutputs] ERFTO on ERFTO.EventRecorderFileTriggerId = ERFT.EventRecorderFileTriggerId
where ERF.CreationDate > dateadd(day, -7, sysdatetime())
"""

"""
select *
from edw.edw_hs_eventrecorderfiles_01_2020_to_11_22_2021 ERF
left outer join edw.edw_hs_eventrecorderfiletriggers_01_2020_to_11_22_2021 ERFT on ERFT.hs_eventrecorderfiletriggers_eventrecorderfileid = ERF.hs_eventrecorderfiles_eventrecorderfileid
left outer join edw.edw_hs_eventrecorderfiletriggeroutput_01_2020_to_11_22_2021 ERFTO on ERFTO.hs_eventrecorderfiletriggeroutputs_eventrecorderfiletriggerid = ERFT.hs_eventrecorderfiletriggers_eventrecorderfileid
where ERF.hs_eventrecorderfiles_creationdate > now()-'7 days'::interval
limit 10
"""

"""
select ERF.EventRecorderId
, ERF.EventRecorderFileId
, ERF.EventTriggerTypeId
, ERF.IsDownloaded -- was event transferred from device to cloud
, ERF.AudioEnabled
, ERF.InsideVideoEnabled
, ERF.OutsideVideoEnabled
, ERFT.TriggerTime
, ERFT.[Position]
, ERFT.PostedSpeedLimit --from device map
, ERFT.SpeedAtTrigger
, ERFT.Heading
, ERFT.TriggerVersion
, ERFT.ForwardErraticThreshold
, ERFT.LateralErraticThreshold
, ERFT.VehicleProfileId
, ERFT.ForwardExtremeAcceleration
, ERFT.LateralExtremeAcceleration
, ERFT.VerticalExtremeAcceleration
, ERFT.MeanForwardAcceleration
, ERFT.MeanLateralAcceleration
, case when ProbabilityAccelerating > ProbabilityBraking
                and ProbabilityAccelerating > ProbabilityLeftCornering
                and ProbabilityAccelerating > ProbabilityRightCornering
                and ProbabilityAccelerating > ProbabilityOther
                and ProbabilityAccelerating > ProbabilityRoughRoad
                and ERF.EventTriggerTypeId in (2,3,30,31,32) then 1 else 0 end as isAccelerating
, case when ProbabilityBraking > ProbabilityAccelerating
                and ProbabilityBraking > ProbabilityLeftCornering
                and ProbabilityBraking > ProbabilityRightCornering
                and ProbabilityBraking > ProbabilityOther
                and ProbabilityBraking > ProbabilityRoughRoad
                and ERF.EventTriggerTypeId in (2,3,30,31,32) then 1 else 0 end as isBraking
, case when (ProbabilityLeftCornering > ProbabilityAccelerating or ProbabilityRightCornering > ProbabilityAccelerating)
                and (ProbabilityLeftCornering > ProbabilityBraking or ProbabilityRightCornering > ProbabilityBraking)
                and (ProbabilityLeftCornering > ProbabilityOther or ProbabilityRightCornering > ProbabilityOther)
                and (ProbabilityLeftCornering > ProbabilityRoughRoad or ProbabilityRightCornering > ProbabilityRoughRoad)
                and ERF.EventTriggerTypeId in (2,3,30,31,32) then 1 else 0 end as isCornering
, case when ProbabilityOther > ProbabilityBraking
                and ProbabilityOther > ProbabilityLeftCornering
                and ProbabilityOther > ProbabilityRightCornering
                and ProbabilityOther > ProbabilityAccelerating
                and ProbabilityOther > ProbabilityRoughRoad
                and ERF.EventTriggerTypeId in (2,3,30,31,32) then 1 else 0 end as isOthner
, case when ProbabilityRoughRoad > ProbabilityBraking
                and ProbabilityRoughRoad > ProbabilityLeftCornering
                and ProbabilityRoughRoad > ProbabilityRightCornering
                and ProbabilityRoughRoad > ProbabilityAccelerating
                and ProbabilityRoughRoad > ProbabilityOther
                and ERF.EventTriggerTypeId in (2,3,30,31,32) then 1 else 0 end as isOthner
, ERFTO.SelectionRate -- BERP seletion rate to achieve target rate
from [hs].[EventRecorderFiles] ERF
left outer join [hs].[EventRecorderFileTriggers] ERFT on ERFT.EventRecorderFileId = ERF.EventRecorderFileId
left outer join [hs].[EventRecorderFileTriggerOutputs] ERFTO on ERFTO.EventRecorderFileTriggerId = ERFT.EventRecorderFileTriggerId
"""

"""
select *
from edw.edw_hs_eventrecorderfiletriggers_01_2020_to_11_22_2021 as tERFT
    LEFT JOIN edw.edw_hs_eventrecorderfiles_01_2020_to_11_22_2021 tERF ON tERF.hs_eventrecorderfiles_eventrecorderfileid=tERFT.hs_eventrecorderfiletriggers_eventrecorderfiletriggerid
    LEFT JOIN edw.edw_hs_eventrecorderfiletriggeroutput_01_2020_to_11_22_2021 as tERFTO ON tERFTO.hs_eventrecorderfiletriggeroutputs_eventrecorderfiletriggerid=tERFT.hs_eventrecorderfiletriggers_eventrecorderfiletriggerid
    LEFT JOIN edw.edw_flat_devices_01_2020_to_11_22_2021 as tFD on tFD.flat_devices__deviceid=tERF.hs_eventrecorderfiles_eventrecorderid
    LEFT JOIN edw.edw_flat_companies_01_2020_to_11_22_2021 as tFC on tFC.flat_companies_companyid=tFD.flat_devices__companyid

    --TODO: Need to understand how to join this table
--  LEFT JOIN edw.edw_flat_vehicletypes_i18n_01_2020_to_11_22_2021 as tFVT on tFVT.hs_vehicletypes__id=tFD.flat_devices__companyid

    FULL OUTER JOIN edw.edw_gps_trips_01_2020_to_11_24_2021 tGT on (
        -- Is this join correct and will return 1 trip per tERFT?
        -- The full out join ensures all EventRecorderFileTriggers and gps_trips regardless if they match
        -- and null for each respectively if they do not match
        tGT.gps_trips__vehicleid=tFD.flat_devices__vehicleid
        AND tERFT.hs_eventrecorderfiletriggers_triggertime BETWEEN tGT.gps_trips__starttimeutc AND tGT.gps_trips__endtimeutc
    )
limit 10
"""

"""
select
    hs_eventrecorderfiletriggeroutputs_eventrecorderfiletriggeroutp,
    hs_eventrecorderfiletriggeroutputs_eventrecorderfiletriggerid,
    hs_eventrecorderfiletriggeroutputs_manifestbatchid,
    hs_eventrecorderfiletriggeroutputs_algorithmversion,
    hs_eventrecorderfiletriggeroutputs_selectionrate,
    hs_eventrecorderfiletriggeroutputs_triggersortedrankorder,
    hs_eventrecorderfiletriggeroutputs_triggerscore,
    hs_eventrecorderfiletriggeroutputs_overallsortedrankorder,
    hs_eventrecorderfiletriggeroutputs_overallscore,
    hs_eventrecorderfiletriggeroutputs_reviewselectionflag,
    hs_eventrecorderfiletriggeroutputs_excludedflag,
    hs_eventrecorderfiletriggeroutputs_isresearchflag,
    hs_eventrecorderfiletriggeroutputs_reviewselectionmethod,
    hs_eventrecorderfiletriggeroutputs_creationdate,
    hs_eventrecorderfiletriggeroutputs_arevision,
    hs_eventrecorderfiletriggeroutputs_arevisiondate,
    hs_eventrecorderfiletriggeroutputs_iscritical,
    hs_eventrecorderfiletriggeroutputs_excludedcode,
    hs_eventrecorderfiletriggeroutputs_weathercode,
    hs_eventrecorderfiletriggeroutputs_roadintercode,
    hs_eventrecorderfiletriggeroutputs_trafficcode,
    hs_eventrecorderfiletriggeroutputs_stackid,
    hs_eventrecorderfiletriggeroutputs_edwcreationdate,
    hs_eventrecorderfiletriggeroutputs_edwupdatedate,
    count(*)
from edw.edw_hs_eventrecorderfiletriggeroutput_01_2020_to_11_22_2021
group by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
having count(*) > 1
order by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
limit 100
"""

"""
select *
from edw.edw_hs_eventrecorderfiletriggers_01_2020_to_11_22_2021 as tERFT
    LEFT JOIN edw.edw_hs_eventrecorderfiles_01_2020_to_11_22_2021 tERF ON tERF.hs_eventrecorderfiles_eventrecorderfileid=tERFT.hs_eventrecorderfiletriggers_eventrecorderfiletriggerid
    -- remove ERFTO since it causes 1 to many
    -- LEFT JOIN edw.edw_hs_eventrecorderfiletriggeroutput_01_2020_to_11_22_2021 as tERFTO ON tERFTO.hs_eventrecorderfiletriggeroutputs_eventrecorderfiletriggerid=tERFT.hs_eventrecorderfiletriggers_eventrecorderfiletriggerid
    LEFT JOIN edw.edw_hs_eventrecorderassociations_up_to_12_06_2021 as tERA on (
        tERA.hs_eventrecorderassociations__eventrecorderid=tERF.hs_eventrecorderfiles_eventrecorderid
        AND tERFT.hs_eventrecorderfiletriggers_triggertime >= tERA.hs_eventrecorderassociations__CreationDate
        AND tERFT.hs_eventrecorderfiletriggers_triggertime <= tERA.hs_eventrecorderassociations__deleteddate
    )
    LEFT JOIN edw.edw_flat_devices_01_2020_to_11_22_2021 as tFD_ERFTO on tFD_ERFTO.flat_devices__deviceid=tERA.hs_eventrecorderassociations__eventrecorderid

limit 100
"""

"""
SELECT *
FROM
edw.edw_hs_eventrecorderfiletriggers_01_2020_to_11_22_2021 as tERFT
        LEFT JOIN edw.edw_hs_eventrecorderfiles_01_2020_to_11_22_2021 tERF ON tERF.hs_eventrecorderfiles_eventrecorderfileid=tERFT.hs_eventrecorderfiletriggers_eventrecorderfiletriggerid
        LEFT JOIN edw.edw_hs_eventrecorderassociations_01_2020_to_11_22_2021 as tERA on (
            tERA.hs_eventrecorderassociations_eventrecorderid=tERF.hs_eventrecorderfiles_eventrecorderid
            AND tERFT.hs_eventrecorderfiletriggers_triggertime > tERA.InitialDockDate
            AND tERFT.hs_eventrecorderfiletriggers_triggertime < tERA.DeletedDate
        )
        LEFT JOIN edw.edw_flat_devices_01_2020_to_11_22_2021 as tFD_ERFTO on tFD_ERFTO.flat_devices__deviceid=tERA.hs_eventrecorderassociations_eventrecorderid
        LEFT JOIN edw.edw_flat_companies_01_2020_to_11_22_2021 as tFC on (
            tFC.flat_companies_companyid=tFD_ERFTO.flat_devices__companyid
        )
"""

"""
-- TODO return gps_trips for each ERFT if available
--      FULL OUTER JOIN edw.edw_gps_trips_01_2020_to_11_24_2021 as tGT on (
--          -- Is this join correct and will return 1 trip per tERFT?
--          -- The full out join ensures all EventRecorderFileTriggers and gps_trips regardless if they match
--          -- and null for each respectively if they do not match
--          tGT.gps_trips__vehicleid=tFD_ERFTO.flat_devices__vehicleid
--          AND tERFT.hs_eventrecorderfiletriggers_triggertime BETWEEN tGT.gps_trips__starttimeutc AND tGT.gps_trips__endtimeutc
--      )
tGT.gps_trips__vehicleid=tFD_ERFTO.flat_devices__vehicleid
changed to
tGT.gps_trips__EventRecorderID=tFD_ERFTO.flat_devices__deviceid
"""

"""
select t1.vehicle_id_record_count as Vehicle_Event_Count, t2.*
from edw.edw_hs_eventrecorderfiletriggers_01_2020_to_11_enriched_status t1
    left join edw.edw_flat_companies_01_2020_to_11_22_2021 t2 on t1.company_id=flat_companies_companyid
order by vehicle_id_record_count desc
limit 100
"""

"""
select
    vehicle_id,
    hs_eventrecorderfiles_eventtriggertypeid,
    hs_eventrecorderfiletriggers_atgtrigsubtype,
    count(*)
from edw.edw_hs_eventrecorderfiletriggers_01_2020_to_11_22_2021_joined
where vehicle_id='9100ffff-48a9-d463-9a0b-3a63f3ff0000'
group by 1, 2, 3
order by 4 desc
limit 100
"""