
"""
validation of new data platform based on extracted data
- based on num of trackpoints per vehicle id per day
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pyrb import open_figure, format_axes, largefonts, save_pngs
from collections import defaultdict
from tqdm import tqdm
plt.style.use('bmh')


# read results based on hadoop and new data platform
fn0 = r'c:/Users/russell.burdt/Data/data-platform/7-1-2021 to 7-28-2021/results-hadoop.p'
fn1 = r'c:/Users/russell.burdt/Data/data-platform/7-1-2021 to 7-28-2021/results-new-data-platform.p'
with open(fn0, 'rb') as fid:
    r0 = pickle.load(fid)
with open(fn1, 'rb') as fid:
    r1 = pickle.load(fid)

# create metrics vs day
days = list(r0.keys())
assert list(r1.keys()) == days
results = defaultdict(list)
for day in days:
    results['day'].append(day)
    results['query time sec, hadoop'].append(r0[day]['sec'])
    results['query time sec, AWS'].append(r1[day]['sec'])
    results['query time ratio, AWS vs hadoop'].append(r0[day]['sec'] / r1[day]['sec'])
    results['num vehicle ids, hadoop'].append(r0[day]['data'].shape[0])
    results['num vehicle ids, AWS'].append(r1[day]['data'].shape[0])
    results['percentage of same vehicleids in AWS wrt hadoop'].append(
        100 * len(set(r0[day]['data']['vehicleid']).intersection(r1[day]['data']['vehicleid'])) / r0[day]['data'].shape[0])
    df = pd.merge(left=r0[day]['data'][['vehicleid', 'ntps']], right=r1[day]['data'][['vehicleid', 'ntps']], on='vehicleid', how='inner')
    tps_diff = (df['ntps_x'] - df['ntps_y']).values
    results['mean difference in num of trackpoints by vehicleid in AWS wrt hadoop'].append(tps_diff.mean())
    results['max difference in num of trackpoints by vehicleid in AWS wrt hadoop'].append(tps_diff.max())
    results['min difference in num of trackpoints by vehicleid in AWS wrt hadoop'].append(tps_diff.min())
results = pd.DataFrame(results)

# query time metrics
fig, ax = open_figure('Query Times', 3, 1, figsize=(8, 4), sharex=True)
for x, (signal, ylabel, ylim) in enumerate(zip(
        ['query time sec, hadoop', 'query time sec, AWS', 'query time ratio, AWS vs hadoop'],
        ['sec', 'sec', ''],
        [(50, 150), (0, 10), (0, 50)])):

    ax[x].plot(results['day'], results[signal], '.-')
    format_axes('day in July', ylabel, signal, ax[x])
    ax[x].set_ylim(ylim)
    ax[x].set_xlim(0.5, day + 0.5)
    ax[x].set_xticks(range(1, day + 1))
largefonts(8)
fig.tight_layout()

# vehicle id and trackpoints comparison
fig, ax = open_figure('Vehicle ID and Num of Trackpoints Comparison', 4, 1, figsize=(8, 6), sharex=True)
for x, (signal, ylabel, ylim) in enumerate(zip(
        ['num vehicle ids, hadoop', 'num vehicle ids, AWS', 'percentage of same vehicleids in AWS wrt hadoop', 'mean difference in num of trackpoints by vehicleid in AWS wrt hadoop'],
        ['', '', '%', ''],
        [(0, 3000), (0, 3000), (0, 100), (0, 100)])):

    ax[x].plot(results['day'], results[signal], '.-')
    format_axes('day in July', ylabel, signal, ax[x])
    ax[x].set_ylim(ylim)
    ax[x].set_xlim(0.5, day + 0.5)
    ax[x].set_xticks(range(1, day + 1))
largefonts(8)
fig.tight_layout()

plt.show()
