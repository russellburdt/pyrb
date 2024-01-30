
"""
coverage vs road-conditions
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pyrb.mpl import open_figure, save_pngs, format_axes, largefonts
from collections import defaultdict
from ipdb import set_trace
plt.style.use('bmh')


# load data
# datadir = r'/mnt/home/russell.burdt/data/collision-reconstruction/app'
datadir = r'c:/Users/russell.burdt/Downloads/cr'
assert os.path.isdir(datadir)
dm = pd.read_pickle(os.path.join(datadir, 'population.p'))
days, vehicles = (dm.loc[0, 't1'] - dm.loc[0, 't0']).days, dm.shape[0]
dx = pd.read_pickle(os.path.join(datadir, 'collisions.p'))

# artificially reduce fleet size
vids = np.random.choice(dm['VehicleId'], size=100000, replace=False)
dm = dm.loc[dm['VehicleId'].isin(vids)].reset_index(drop=True)
dx = dx.loc[dx['VehicleId'].isin(vids)].reset_index(drop=True)
days, vehicles = (dm.loc[0, 't1'] - dm.loc[0, 't0']).days, dm.shape[0]

# coverage config
config = (
    (np.ones(dx.shape[0]).astype('bool'),
        f'{dm.shape[0]} vehicle fleet size'),)
#     (dx['SpeedAtTrigger'].values >= 20,
#         'roads-all, hours-all, speeds>20'),
#     (dx['SpeedAtTrigger'].values < 20,
#         'roads-all, hours-all, speeds<20'),
#     (dx['localhour'].isin([8,9,10,11,12,13,14,15,16,17]).values,
#         'roads-all, hours-8a-6p, speeds-all'),
#     (~dx['localhour'].isin([8,9,10,11,12,13,14,15,16,17]).values,
#         'roads-all, hours-not-8a-6p, speeds-all'),
#     (~pd.isnull(dx['publicroad1']).values,
#         'roads-public, hours-all, speeds-all'),
#     (pd.isnull(dx['publicroad1']).values,
#         'roads-not-public, hours-all, speeds-all'))

# coverage DataFrame
fns = glob(os.path.join(datadir, 'nearby_vehicles*.p'))
dc = defaultdict(list)
same_company = False
for fn in fns:
    df = pd.read_pickle(fn)
    df = df.loc[df['VehicleId'].isin(vids)].reset_index(drop=True)
    for ok, label in config:
        assert ok.size == dx.shape[0]
        dc['td'].append(df.loc[0, 'td'])
        dc['xd'].append(df.loc[0, 'xd'])
        dc['label'].append(label)
        dc['num collisions'].append(ok.sum())
        if same_company:
            coverage = 0
            for sx in pd.unique(df['id']):
                if np.any(dx.loc[sx, 'CompanyName'] == df.loc[df['id'] == sx, 'CompanyName'].values):
                    coverage += 1
            coverage = 100 * coverage / ok.sum()
        else:
            coverage = 100 * len(set(pd.unique(df['id'])).intersection(dx.loc[ok].index)) / ok.sum()
        dc['coverage'].append(coverage)
dc = pd.DataFrame(dc).sort_values(['td', 'coverage']).reset_index(drop=True)

# plot results
title = f'coverage of collisions by nearby vehicles within 40m over {days} days'
fig, ax = open_figure(title, figsize=(12, 6))
groups = dc.groupby('label').groups
keys = [x[1] for x in config]
groups = {a: b for a, b in zip(keys, [groups[x] for x in keys])}
for label, xs in groups.items():
    ax.plot(dc.loc[xs, 'td'], dc.loc[xs, 'coverage'], 'x-', ms=12, lw=4, label=f"""{label}, {dc.loc[xs].iloc[0]['num collisions']} collisions""")
format_axes('min time between gps record for nearby vehicle and collision timestamp', '% of collisions with any nearby vehicle', title, ax)
ax.legend(loc='upper left', numpoints=3, handlelength=5)
ax.set_ylim(0, 50)
ax.set_yticks(np.arange(0, 51, 5))
ax.set_xscale('log')
ax.set_xticks([10, 30, 60, 300])
ax.set_xticklabels(['10sec', '30sec', '60sec', '5min'])
largefonts(18)
fig.tight_layout()
# save_pngs(r'/mnt/home/russell.burdt/data')
plt.show()
