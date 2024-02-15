
"""
extract GPS enriched metrics from Parquet data source, compare to same metrics from DC
"""

import os
import utils
import numpy as np
import pandas as pd
from lytx import get_conn
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

# datadir and dcm
datadir = r'/mnt/home/russell.burdt/data/collision-model/v1/dft1b'
dcm = pd.read_pickle(os.path.join(datadir, 'dcm.p'))

# spark session
conf = SparkConf()
conf.set('spark.driver.memory', '32g')
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.session.timeZone', 'UTC')
conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
conf.set('spark.sql.shuffle.partitions', 20000)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# enriched gps dataset
gps = spark.read.parquet(os.path.join(datadir, 'gps.parquet'))
gps.createOrReplaceTempView('gps')

# vehicle evaluation window by VehicleId
dcm.index.name = 'rid'
dcm = dcm.reset_index(drop=False)
dfv = dcm.loc[:, ['rid', 'VehicleId', 'time0', 'time1']].copy()
dfv['time0'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time0']]
dfv['time1'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time1']]
dfv = broadcast(spark.createDataFrame(dfv))
dfv.cache()
dfv.createOrReplaceTempView('dfv')

# enriched gps metrics
de = utils.gpse_metrics(spark, prefix='gpse')

# merge collision model population and metrics DataFrames
df = pd.merge(left=dcm, right=de, on='rid', how='left')
assert df.shape[0] == dcm.shape[0]

# DC enriched gps metrics
lab = get_conn('lytx-lab')
fn = r'/mnt/home/russell.burdt/data/collision-model/dc/tripreport20220401.p'
if os.path.isfile(fn):
    dc0 = pd.read_pickle(fn)
    dc0['gps_seg__manuever_straight__all_stop_complexity__avg'] = 0
    dc0['gps_seg__manuever_straight__stopsign_complexity__avg'] = 0
    dc = dc0.copy()
else:
    dc = pd.read_sql_query(f'SELECT * FROM insurance_model.distfreighttruckingcrash_08012021_11302021_tripreport20220401', lab)
    dc.to_pickle(fn)

# initialize dict to align enriched GPS metrics with DC enriched GPS metrics
align = {}

# trip duration metrics
align = {**align, **{
    'gpse_travel_duration_hours_sum': 'travel_duration_hours__sum',
    'gpse_travel_duration_moving_hours_sum': 'travel_duration_moving_hours__sum',
    'gpse_travel_duration_idle_hours_sum': 'travel_duration_idle_hours__sum'}}

# hotspot metrics
align = {**align, **{
    'gpse_all_hotspots_entered_sum': 'crash_all_hotspots_entered__sum',
    'gpse_hotspots_entered_intersection_sum': 'crash__hotspots_entered_intersection__sum',
    'gpse_hotspots_incidentrate_intersection_avg': 'crash__hotspots_incidentrate_intersection__avg',
    'gpse_hotspots_entered_intersection_complexity_avg': 'crash__hotspots_entered_intersection_complexity__avg',
    'gpse_hotspots_severity_index_avg': 'crash_hotspots_rankgroup_severity_index_avg',
    'gpse_hotspots_incidents_sum': 'crash_hotspots_incidents_sum'}}
for hs in [
        'animal',
        'bicyclist',
        'lowclearance',
        'intersection',
        'pedestrian',
        'slowing_traffic',
        'train',
        'turn_curve']:
    align = {**align, **{f'gpse_{hs}_hotspots_entered_sum': f'crash_{hs}_hotspots_entered__sum'}}
for hs in [
        'injury_incidents',
        'fatal_incidents',
        'pedestriansinvolved',
        'pedestriansinvolvedunder18',
        'cyclistsinvolved',
        'cyclistsinvolvedunder18']:
    align = {**align, **{f'gpse_hotspots_{hs}_sum': f'crash_hotspots_{hs}_sum'}}

# urban density metrics
align = {**align, **{
    'gpse_urban_density_km_of_road_per_sq_km_avg': 'urban_density_km_of_road_per_sq_km__avg',
    'gpse_urban_density_30_plus_km_sq_km_idle_hours_sum': 'urban_density_30_plus_km_sqkm_idle__hours_sum',
    'gpse_urban_density_30_plus_km_sq_km_moving_hours_sum': 'urban_density_30_plus_km_sqkm_moving__hours_sum'}}
for r0, r1 in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)):
    align = {**align, **{
        f'gpse_urban_density_{r0}_to_{r1}_km_sq_km_moving_hours_sum': f'urban_density_{r0}_to_{r1}_km_sqkm_moving__hours_sum',
        f'gpse_urban_density_{r0}_to_{r1}_km_sq_km_idle_hours_sum': f'urban_density_{r0}_to_{r1}_km_sqkm_idle__hours_sum'}}

# speed metrics
align = {**align, **{
    'gpse_delta_speed_mph_vs_speed_limit_avg': 'speed_mph_delta__gps_speed_vs_road_speed_limit__avg',
    'gpse_delta_speed_mph_vs_speed_limit_under_count': 'speed_mph_delta__gps_speed_vs_road_speed_limit__under__count',
    'gpse_delta_speed_mph_vs_speed_limit_under_avg': 'speed_mph_delta__gps_speed_vs_road_speed_limit__under__avg',
    'gpse_delta_speed_mph_vs_speed_limit_under_hours': 'speed_mph_delta__gps_speed_vs_road_speed_limit__under__hours',
    'gpse_delta_speed_mph_vs_speed_limit_over_count': 'speed_mph_delta__gps_speed_vs_road_speed_limit__over__count',
    'gpse_delta_speed_mph_vs_speed_limit_over_avg': 'speed_mph_delta__gps_speed_vs_road_speed_limit__over__avg',
    'gpse_delta_speed_mph_vs_speed_limit_over_hours': 'speed_mph_delta__gps_speed_vs_road_speed_limit__over__hours',
    'gpse_speed_vs_road_speed_limit_equal_count': 'speed_mph_delta__gps_speed_vs_road_speed_limit__equal__count',
    'gpse_speed_vs_road_speed_limit_equal_hours': 'speed_mph_delta__gps_speed_vs_road_speed_limit__equal__hours'}}

# distance and duration by road type
align = {**align, **{
    'gpse_travel_distance_meters_sum': 'travel_distance_meters__sum',

    'gpse_travel_distance_meters_roadcode_null_sum': 'distance_traveled_meters__road_class_null_off_road__sum',
    'gpse_moving_duration_hours_roadcode_null_sum': 'moving_duration_hours__road_class_null_off_road__sum',
    'gpse_idle_duration_hours_roadcode_null_sum': 'idle_duration_hours__road_class_null_off_road__sum',

    'gpse_travel_distance_meters_roadclass_localroad_sum': 'distance_traveled_meters__road_class_local__sum',
    'gpse_travel_distance_meters_roadclass_highwayramp_sum': 'distance_traveled_meters__road_class_highway_ramp__sum',
    'gpse_travel_distance_meters_roadclass_highwayroad_sum': 'distance_traveled_meters__road_class_highway_road__sum',

    'gpse_moving_duration_hours_roadclass_localroad_sum': 'moving_duration_hours__road_class_local__sum',
    'gpse_moving_duration_hours_roadclass_highwayramp_sum': 'moving_duration_hours__road_class_highway_ramp__sum',
    'gpse_moving_duration_hours_roadclass_highwayroad_sum': 'moving_duration_hours__road_class_highway_road__sum',

    'gpse_idle_duration_hours_roadclass_localroad_sum': 'idle_duration_hours__road_class_local__sum',
    'gpse_idle_duration_hours_roadclass_highwayramp_sum': 'idle_duration_hours__road_class_highway_ramp__sum',
    'gpse_idle_duration_hours_roadclass_highwayroad_sum': 'idle_duration_hours__road_class_highway_road__sum',

    'gpse_travel_distance_meters_roadcode_5123_living_street_sum': 'distance_traveled_meters__road_class_5123_livings_treet__sum',
    'gpse_moving_duration_hours_roadcode_5123_living_street_sum': 'moving_duration_hours__road_class_5123_livings_treet__sum',
    'gpse_idle_duration_hours_roadcode_5123_living_street_sum': 'idle_duration_hours__road_class_5123_livings_treet__sum'}}
for rc, desc in (
        (5111, 'motorway'),
        (5112, 'trunk'),
        (5113, 'primary'),
        (5114, 'secondary'),
        (5115, 'tertiary'),
        (5121, 'unclassified'),
        (5122, 'residential'),
        # (5123, 'living_street'),
        (5131, 'motorway_link'),
        (5132, 'trunk_link'),
        (5133, 'primary_link'),
        (5134, 'secondary_link'),
        (5135, 'tertiary_link')):
    align = {**align, **{
        f'gpse_travel_distance_meters_roadcode_{rc}_{desc}_sum': f'distance_traveled_meters__road_class_{rc}_{desc}__sum',
        f'gpse_moving_duration_hours_roadcode_{rc}_{desc}_sum': f'moving_duration_hours__road_class_{rc}_{desc}__sum',
        f'gpse_idle_duration_hours_roadcode_{rc}_{desc}_sum': f'idle_duration_hours__road_class_{rc}_{desc}__sum'}}

# road properties
align = {**align, **{
    'gpse_road_speed_limit_estimate_avg': 'road_speed_limit_estimate__avg',
    'gpse_road_angle_degrees_avg': 'road_angle_degrees__avg',
    'gpse_roadclass_localroad_sum': 'road_class_local__sum',
    'gpse_roadclass_highwayramp_sum': 'road_class_highway_ramp__sum',
    'gpse_roadclass_highwayroad_sum': 'road_class_highway_road__sum',
    'gpse_private_road_sum': 'road_class_0_private_road__sum',
    'gpse_road_code_null_off_road_sum': 'road_class_null_off_road__sum',
    'gpse_roadcode_5123_living_street_sum': 'road_class_5123_livings_treet__sum'}}
for rc, desc in (
        (5111, 'motorway'),
        (5112, 'trunk'),
        (5113, 'primary'),
        (5114, 'secondary'),
        (5115, 'tertiary'),
        (5121, 'unclassified'),
        (5122, 'residential'),
        # (5123, 'living_street'),
        (5131, 'motorway_link'),
        (5132, 'trunk_link'),
        (5133, 'primary_link'),
        (5134, 'secondary_link'),
        (5135, 'tertiary_link')):
    align = {**align, **{
        f'gpse_roadcode_{rc}_{desc}_sum': f'road_class_{rc}_{desc}__sum'}}

# average annual daily traffic by road type
align = {**align, **{
    'gpse_roadclass_localroad_aadt_avg': 'road_class_local_aadt__avg',
    'gpse_roadclass_highwayramp_aadt_avg': 'road_class_highway_ramp_aadt__avg',
    'gpse_roadclass_highwayroad_aadt_avg': 'road_class_highway_road_aadt__avg',
    'gpse_roadcode_5123_living_street_aadt_avg': 'road_class_5123_livings_street_aadt__avg'}}
for rc, desc in (
        (5111, 'motorway'),
        (5112, 'trunk'),
        (5113, 'primary'),
        (5114, 'secondary'),
        (5115, 'tertiary'),
        (5121, 'unclassified'),
        (5122, 'residential'),
        # (5123, 'living_street'),
        (5131, 'motorway_link'),
        (5132, 'trunk_link'),
        (5133, 'primary_link'),
        (5134, 'secondary_link'),
        (5135, 'tertiary_link')):
    align = {**align, **{
        f'gpse_roadcode_{rc}_{desc}_aadt_avg': f'road_class_{rc}_{desc}_aadt__avg'}}

# trip duration based on average annual daily traffic by road type
align = {**align, **{
    'gpse_roadcode_5123_living_street_aadt_0_1000_hours_sum': 'road_class_5123_livings_street_aadt_0_1000__hours_sum',
    'gpse_roadcode_5123_living_street_aadt_1k_10k_hours_sum': 'road_class_5123_livings_street_aadt_1k_to_10k__hours_sum',
    'gpse_roadcode_5123_living_street_aadt_10k_100k_hours_sum': 'road_class_5123_livings_street_aadt_10k_to_100k__hours_sum',
    'gpse_roadcode_5123_living_street_aadt_100k_300k_hours_sum': 'road_class_5123_livings_street_aadt_100k_to_300k__hours_sum',
    'gpse_roadcode_5123_living_street_aadt_300k_plus_hours_sum': 'road_class_5123_livings_street_aadt_300k_plus__hours_sum'}}
for r0, r1 in ((0, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 3e5), (3e5, 'plus')):
    assert isinstance(r0, (int, float)) and (isinstance(r1, (int, float)) or (r1 == 'plus'))
    if r1 != 'plus':
        if r1 <= 1e3:
            wa = wb = f'{int(r0)}_{int(r1)}'
        else:
            assert r0 >= 1e3
            wa = f'{r0 / 1000:.0f}k_{r1 / 1000:.0f}k'
            wb = f'{r0 / 1000:.0f}k_to_{r1 / 1000:.0f}k'
    else:
        assert r0 >= 1e3
        wa = wb = f'{r0 / 1000:.0f}k_plus'
    align = {**align, **{
        f'gpse_roadclass_localroad_aadt_{wa}_hours_sum': f'road_class_local_aadt_{wb}__hours_sum',
        f'gpse_roadclass_highwayramp_aadt_{wa}_hours_sum': f'road_class_highway_ramp_aadt_{wb}__hours_sum',
        f'gpse_roadclass_highwayroad_aadt_{wa}_hours_sum': f'road_class_highway_road_aadt_{wb}__hours_sum'}}
    for rc, desc in (
                (5111, 'motorway'),
                (5112, 'trunk'),
                (5113, 'primary'),
                (5114, 'secondary'),
                (5115, 'tertiary'),
                (5121, 'unclassified'),
                (5122, 'residential'),
                # (5123, 'living_street'),
                (5131, 'motorway_link'),
                (5132, 'trunk_link'),
                (5133, 'primary_link'),
                (5134, 'secondary_link'),
                (5135, 'tertiary_link')):
        align = {**align, **{
            f'gpse_roadcode_{rc}_{desc}_aadt_{wa}_hours_sum': f'road_class_{rc}_{desc}_aadt_{wb}__hours_sum'}}

# road feature metrics
for road_type in ['public_road_intersection', 'highway_ramp_junction', 'private_road_intersection']:
    for distance in [0, 10, 20, 30, 60]:
        align = {**align, **{
            f'gpse_{road_type}_within_{distance}_meters_avg':
                f'{road_type}_within_{distance}_meters',
            f'gpse_{road_type}_complexity_within_{distance}_meters_avg':
                f'{road_type}_complexity_within_{distance}_meters'}}

# road corridor metrics
align = {**align, **{
    'gpse_private_roads_length_in_corridor_meters_sum': 'private_roads_length_in_corridor_meters',
    'gpse_local_road_length_in_corridor_meters_sum': 'local_road_length_in_corridor_meters',
    'gpse_local_road_width_avg_in_corridor_meters': 'local_road_width_avg_in_corridor_meters',
    'gpse_local_road_maxspeedestimate_kmh_avg_in_corridor_meters': 'local_road_maxspeedestimate_kmh_avg_in_corridor_meters',
    'gpse_local_road_aadt_avg_in_corridor': 'local_road__aadt_avg_in_corridors'}}
for cx in ['public_road_intersection', 'highway_ramp_junction', 'private_road_intersection']:
    align = {**align, **{
        f'gpse_{cx}_in_corridor_sum': f'{cx}_in_corridor',
        f'gpse_{cx}_complexity_in_corridor_sum': f'{cx}_complexity_in_corridor'}}
for cx in ['public_roads', 'highway_road', 'highway_ramp']:
    align = {**align, **{
        f'gpse_{cx}_length_in_corridor_meters_sum': f'{cx}_length_in_corridor_meters',
        f'gpse_{cx}_width_avg_in_corridor_meters': f'{cx}_width_avg_in_corridor_meters',
        f'gpse_{cx}_maxspeedestimate_kmh_avg_in_corridor_meters': f'{cx}_maxspeedestimate_kmh_avg_in_corridor_meters',
        f'gpse_{cx}_aadt_avg_in_corridor': f'{cx}__aadt_avg_in_corridor'}}

# road features traversed on route
align = {**align, **{
    'gpse_all_intersections_traversed_sum': 'gps_seg__all_intersections_traversed__sum',
    'gpse_all_intersections_complexity_avg': 'gps_seg__all_intersections_traversed_complexity__avg',
    'gpse_all_intersections_complexity_min': 'gps_seg__all_intersections_traversed_complexity_min__avg',
    'gpse_all_intersections_complexity_max': 'gps_seg__all_intersections_traversed_complexity_max__avg',
    'gpse_all_intersections_traversed_duration_hours_sum': 'gps_seg__all_intersections_traversed_duration_hours__sum',
    'gpse_all_intersections_traversed_moving_hours_sum': 'gps_seg__all_intersections_traversed_moving_hours__sum',
    'gpse_all_intersections_traversed_idle_hours_sum': 'gps_seg__all_intersections_traversed_idle_hours__sum',
    'gpse_traffic_signals_traversed_sum': 'gps_seg__traffic_signal_intersections_traversed__sum',
    'gpse_traffic_signals_complexity_avg': 'gps_seg__intersection_signal_traversed_complexity__avg',
    'gpse_traffic_signals_complexity_min': 'gps_seg__intersection_signal_traversed_complexity_min__avg',
    'gpse_traffic_signals_complexity_max': 'gps_seg__intersection_signal_traversed_complexity_max__avg',
    'gpse_traffic_signals_traversed_duration_hours_sum': 'gps_seg__intersection_signal_traversed_duration_hours__sum',
    'gpse_traffic_signals_traversed_moving_hours_sum': 'gps_seg__public_intersection_signal_traversed_moving_hours__sum',
    'gpse_traffic_signals_traversed_idle_hours_sum': 'gps_seg__public_intersection_signal_traversed_idle_hours__sum'}}
for ca, cb in (
        ('ramp_junctions', 'ramp_junctions'),
        ('crosswalks', 'crosswalk_intersections'),
        ('railway_crossings', 'railway_intersections')):
    align = {**align, **{
        f'gpse_{ca}_traversed_sum': f'gps_seg__{cb}_traversed__sum',
        f'gpse_{ca}_complexity_avg': f'gps_seg__{cb}_traversed_complexity__avg',
        f'gpse_{ca}_complexity_min': f'gps_seg__{cb}_traversed_complexity_min__avg',
        f'gpse_{ca}_complexity_max': f'gps_seg__{cb}_traversed_complexity_max__avg',
        f'gpse_{ca}_traversed_duration_hours_sum': f'gps_seg__{cb}_traversed_duration_hours__sum',
        f'gpse_{ca}_traversed_moving_hours_sum': f'gps_seg__{cb}_traversed_moving_hours__sum',
        f'gpse_{ca}_traversed_idle_hours_sum': f'gps_seg__{cb}_traversed_idle_hours__sum'}}
for ca, cb in (
        ('public_intersections', 'public_intersection'),
        ('private_intersections', 'private_intersection')):
    align = {**align, **{
        f'gpse_{ca}_traversed_sum': f'gps_seg__{cb}_traversed__sum',
        f'gpse_{ca}_complexity_avg': f'gps_seg__{cb}_traversed_complexity__avg',
        f'gpse_{ca}_complexity_min': f'gps_seg__{cb}_traversed_complexity_min__avg',
        f'gpse_{ca}_complexity_max': f'gps_seg__{cb}_traversed_complexity_max__avg',
        f'gpse_{ca}_traversed_duration_hours_sum': f'gps_seg__{cb}_traversed_duration_hours__sum',
        f'gpse_{ca}_traversed_moving_hours_sum': f'gps_seg__{ca}_traversed_moving_hours__sum',
        f'gpse_{ca}_traversed_idle_hours_sum': f'gps_seg__{ca}_traversed_idle_hours__sum'}}
for ca, cb, cc in (
        ('allstop', 'allstop_intersections', 'allstop_intersection'),
        ('stopsign', 'stopsign_intersections', 'stopsign_intersection'),
        ('yieldsign', 'yieldsign_intersections', 'yieldsign_intersection')):
    align = {**align, **{
        f'gpse_{ca}_traversed_sum': f'gps_seg__{cb}_traversed__sum',
        f'gpse_{ca}_complexity_avg': f'gps_seg__{cb}_traversed_complexity__avg',
        f'gpse_{ca}_complexity_min': f'gps_seg__{cb}_traversed_complexity_min__avg',
        f'gpse_{ca}_complexity_max': f'gps_seg__{cb}_traversed_complexity_max__avg',
        f'gpse_{ca}_traversed_duration_hours_sum': f'gps_seg__{cc}_traversed_duration_hours__sum',
        f'gpse_{ca}_traversed_moving_hours_sum': f'gps_seg__{cb}_traversed_moving_hours__sum',
        f'gpse_{ca}_traversed_idle_hours_sum': f'gps_seg__{cb}_traversed_idle_hours__sum'}}

# maneuver count by road type
for m1, m2 in (
        ('straight', 'straight'),
        ('left', 'left'),
        ('leftextreme', 'left_extreme'),
        ('leftshallow', 'left_shallow'),
        ('right', 'right'),
        ('rightextreme', 'right_extreme'),
        ('rightshallow', 'right_shallow')):
    for c1, c2 in (
            ('public_roads', ''),
            ('private_roads', 'off_road_')):
        align = {**align, **{
            f'gpse_{c1}_maneuver_{m1}_sum': f'gps_seg__{c2}manuever_{m2}__sum'}}

# intersection count by maneuver and intersection type
for m1, m2 in (
        ('straight', 'straight'),
        ('left', 'left'),
        ('leftextreme', 'left_extreme'),
        ('leftshallow', 'left_shallow'),
        ('right', 'right'),
        ('rightextreme', 'right_extreme'),
        ('rightshallow', 'right_shallow')):
    for c1, c2 in (
            ('intersection', 'public_intersection'),
            ('serviceroadintersection', 'private_intersection'),
            ('rampjunction', 'ramp_junction'),
            ('trafficsignal', 'intersection_signal'),
            ('crosswalk', 'intersection_crosswalk'),
            ('railwaylevelcrossing', 'railway_crossing'),
            ('yieldsign', 'yieldsign'),
            ('stopsign', 'stopsign'),
            ('sign_allwaystop', 'allstopsign')):
        align = {**align, **{
            f'gpse_intersection_count_{c1}_maneuver_{m1}_sum': f'gps_seg__{c2}__manuever_{m2}__sum'}}

# average intersection complexity metrics by maneuver and intersection type
for m1, m2 in (
        ('straight', 'straight'),
        ('left', 'left'),
        ('leftextreme', 'left_extreme'),
        ('leftshallow', 'left_shallow'),
        ('right', 'right'),
        ('rightextreme', 'right_extreme'),
        ('rightshallow', 'right_shallow')):
    for c1, c2 in (
            ('intersection', 'pub_inters'),
            ('serviceroadintersection', 'priv_int'),
            ('rampjunction', 'ramp_junct'),
            ('trafficsignal', 'signaled'),
            ('stopsign', 'stopsign'),
            ('sign_allwaystop', 'all_stop')):
        align = {**align, **{
            f'gpse_intersection_complexity_avg_{c1}_maneuver_{m1}_avg':
                f'gps_seg__manuever_{m2}__{c2}_complexity__avg',
            f'gpse_intersection_complexity_min_{c1}_maneuver_{m1}_avg':
                f'gps_seg__manuever_{m2}__{c2}_complexity_min__avg',
            f'gpse_intersection_complexity_max_{c1}_maneuver_{m1}_avg':
                f'gps_seg__manuever_{m2}__{c2}_complexity_max__avg'}}

# merge enriched metrics
df = df[['VehicleId', 'time0', 'time1'] + list(align.keys())].copy()
dc['VehicleId'] = dc.pop('vehicle_id')
cols = sorted(align.values())
dc = dc[['VehicleId', 'time0', 'time1'] + cols].copy()
align2 = dict([(value, key) for key, value in align.items()])
cols2 = [align2[x].replace('gpse', 'DC') for x in cols]
dc.columns = ['VehicleId', 'time0', 'time1'] + cols2
dxx = pd.merge(df, dc, on=['VehicleId', 'time0', 'time1'], how='inner')
dxx.to_pickle(r'/mnt/home/russell.burdt/data/dxx.p')
