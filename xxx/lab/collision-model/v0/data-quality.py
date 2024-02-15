
"""
data quality analysis from collision prediction model metadata and coverage artifacts
"""

import os
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from ipdb import set_trace
from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs, metric_distribution

# mpl
plt.style.use('bmh')
size = 18

# load data
dcm = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/dcm.p')
ds = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/gps_segmentation_metrics.p')
de = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/metadata/event_recorder_associations.p')
dem = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/metadata/event_recorder_association_metrics.p')
dp = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/metadata/positive_instances.p')
bounds = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/coverage/bounds.p')
records = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/coverage/records.p')
dce = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/coverage/dce_scores_events_coverage.p')
dte = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/coverage/triggers_events_coverage.p')

def event_recorder_association_metrics():

    # duration of ER associations in days
    x = (1 / (60 * 60 * 24)) * np.concatenate(dem['duration of ER associations in sec'].values)
    title = 'distribution of duration of Event Recorder associations in days'
    metric_distribution(
        x=x,
        bins=np.arange(0, 400, 1),
        title=title,
        xlabel='duration in days',
        figsize=(12, 4),
        size=16,
        logscale=True)

    # seconds between consecutive ER associations
    x = np.concatenate(dem['sec between consecutive ER associations without metadata change'].values)
    title = 'distribution of seconds between consecutive ER associations without metadata change for same vehicle'
    title += f'\n{100 * (x < 1).sum() / x.size:.3f} percent less than 1 second'
    metric_distribution(
        x=x,
        bins=np.arange(0, 90, 1),
        title=title,
        xlabel='seconds between consecutive ER associations',
        figsize=(14, 6),
        size=16,
        logscale=True)
    x = np.concatenate(dem['sec between consecutive ER associations with metadata change'].values)
    title = 'distribution of seconds between consecutive ER associations with metadata change for same vehicle'
    title += f'\n{100 * (x < 1).sum() / x.size:.3f} percent less than 1 second'
    metric_distribution(
        x=x,
        bins=np.arange(0, 90, 1),
        title=title,
        xlabel='seconds between consecutive ER associations',
        figsize=(14, 6),
        size=16,
        logscale=True)

def collision_count_metrics():

    # collision-count vs time of day, day of week, month of year
    crt = pd.DataFrame({
        'hour': np.array([int(x['localtime'].strftime('%H')) for _, x in dp.iterrows()]),
        'day': np.array([int(x['localtime'].strftime('%w')) for _, x in dp.iterrows()]),
        'month': np.array([int(x['localtime'].strftime('%m')) for _, x in dp.iterrows()])})
    fig, ax = open_figure('collision count vs time', 3, 1, figsize=(12, 8))
    ax[0].plot(crt['hour'].value_counts().sort_index(), 'x-', ms=12, lw=3)
    ax[0].set_xticks(range(24))
    ax[0].set_xlim(-0.1, 23.1)
    format_axes('hour of day', 'collision count', 'collision count vs hour of day', ax[0])
    ax[1].plot(crt['day'].value_counts().sort_index(), 'x-', ms=12, lw=3)
    ax[1].set_xticks(range(7))
    ax[1].set_xlim(-0.1, 6.1)
    format_axes('day of week, 0=Sunday', 'collision count', 'collision count vs day of week', ax[1])
    ax[2].plot(crt['month'].value_counts().sort_index(), 'x-', ms=12, lw=3)
    ax[2].set_xticks(range(1, 13))
    ax[2].set_xlim(0.9, 12.1)
    format_axes('month of year', 'collision count', 'collision count vs month of year', ax[2])
    largefonts(16)
    fig.tight_layout()

def records_per_day_per_vehicle():

    for src in records.keys():
        dx = records[src]
        fig, ax = open_figure(f'records per vehicle vs day, {src}', 3, 1, figsize=(18, 8), sharex=True)
        ax[0].plot(dx['day'], dx['nx'], 'x-', ms=8, lw=3)
        format_axes('', '', 'number of vehicles vs day', ax[0], apply_concise_date_formatter=True)
        ax[1].plot(dx['day'], dx['nr'], 'x-', ms=8, lw=3)
        format_axes('', '', 'number of vehicle-records vs day', ax[1], apply_concise_date_formatter=True)
        ax[2].plot(dx['day'], dx['nrx'], 'x-', ms=8, lw=3)
        format_axes('', '', 'number of records per vehicle vs day', ax[2], apply_concise_date_formatter=True)
        largefonts(18)
        fig.tight_layout()

def gps_segment_coverage():

    # gps segmentation metrics for all vehicle evaluations
    dx = dcm[['VehicleId', 'time0', 'time1']].copy()
    dx['time0'] = [(pd.Timestamp(x) - pd.Timestamp(1970, 1, 1)).total_seconds() for x in dx['time0'].values]
    dx['time1'] = [(pd.Timestamp(x) - pd.Timestamp(1970, 1, 1)).total_seconds() for x in dx['time1'].values]
    df = pd.merge(left=dx, right=ds, on=['VehicleId', 'time0', 'time1'], how='left')
    df = df.fillna(0).reset_index(drop=True)

    # cumulative distribution functions
    days = np.unique(np.round(np.unique(ds['total_days'])).astype('int'))[0]
    bins = np.linspace(0, days, 100)
    metric_distribution(x=df['n_days_segments'].values, bins=bins, title=f'distribution of n_days_segments', xlabel='days', size=size)
    metric_distribution(x=df['n_days_no_segments'].values, bins=bins, title=f'distribution of n_days_no_segments', xlabel='days', size=size)
    bins = np.linspace(0, 1.01 * ds['n_segments'].max(), 100)
    metric_distribution(x=df['n_segments'].values, bins=bins, title=f'distribution of n_segments', xlabel='count', size=size, logscale=True)
    bins = np.linspace(0, 1.01 * ds['n_records_segments'].max(), 100)
    metric_distribution(x=df['n_records_segments'].values, bins=bins, title=f'distribution of n_records_segments', xlabel='count', size=size, logscale=True)

def dce_scores_events_coverage():

    fig, ax = open_figure('dce-score vs events coverage', figsize=(12, 6))
    ax.plot(dce['time1'], dce['coverage'], '.-', lw=3, ms=10)
    title = 'Coverage of dce-model to recorded (27,30,31) events in a 30-day window'
    title += '\nfor all vehicles in collision prediction model population'
    format_axes('', '%', title, ax, apply_concise_date_formatter=True)
    ax.set_ylim(0, 101)
    ax.set_yticks(np.arange(0, 101, 10))
    largefonts(size)
    fig.tight_layout()

def triggers_events_coverage():

    fig, ax = open_figure('num of triggered vs recorded events', 2, 1, figsize=(16, 6))
    ax[0].plot(dte['ne'].values, '.-', ms=12, lw=3, label='num recorded events')
    ax[0].plot(dte['nte'].values, '.-', ms=12, lw=3, label='num triggered events\nwith same EventRecorderFileId')
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, handlelength=4)
    title = 'num of triggered and recorded events, all vehicle evaluations'
    format_axes('EventTriggerTypeId', 'num', title, ax[0])
    ax[1].plot(100 * dte['frac'].values, '.-', ms=12, lw=3)
    title = 'ratio of num of triggered to recorded events'
    format_axes('EventTriggerTypeId', '%', title, ax[1])
    for x in ax:
        x.set_xticks(dte.index.values)
        x.set_xticklabels(dte['NameId'].values)
    ax[0].set_yscale('log')
    ax[1].set_ylim(0, 100)
    largefonts(size)
    fig.tight_layout()

# data quality charts
event_recorder_association_metrics()
collision_count_metrics()
records_per_day_per_vehicle()
gps_segment_coverage()
dce_scores_events_coverage()
triggers_events_coverage()

cdir = r'c:/Users/russell.burdt/Downloads/charts'
if not os.path.isdir(cdir):
    os.mkdir(cdir)
save_pngs(cdir)
plt.show()
