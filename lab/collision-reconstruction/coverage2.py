
"""
coverage vs fleet size
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pyrb.mpl import open_figure, save_pngs, format_axes, largefonts
from collections import defaultdict
from scipy.optimize import curve_fit
from ipdb import set_trace
plt.style.use('bmh')


# load data
datadir = r'c:/Users/russell.burdt/Downloads/cr'
assert os.path.isdir(datadir)
dm = pd.read_pickle(os.path.join(datadir, 'population.p'))
days, vehicles = (dm.loc[0, 't1'] - dm.loc[0, 't0']).days, dm.shape[0]
dx = pd.read_pickle(os.path.join(datadir, 'collisions.p'))
fns = glob(os.path.join(datadir, 'nearby_vehicles*.p'))

# scan over fleet sizes
dc = defaultdict(list)
fss = np.arange(100, 701, 20)
for fs in fss:

    # subset of collisions for fs
    vids = np.random.choice(dm['VehicleId'], size=int(1000 * fs), replace=False)
    dxx = dx.loc[dx['VehicleId'].isin(vids)]

    # scan over nearby vehicle definitions
    for fn in fns:

        # subset of nearby vehicles for fs
        df = pd.read_pickle(fn)
        dfx = df.loc[df['VehicleId'].isin(vids)].reset_index(drop=True)

        # coverage for nearby vehicle definition and fs, save results
        dc['td'].append(df.loc[0, 'td'])
        dc['xd'].append(df.loc[0, 'xd'])
        dc['coverage'].append(100 * len(set(pd.unique(dfx['id'])).intersection(dxx.index)) / dxx.shape[0])
        dc['fleet size'].append(fs)
        dc['num collisions'].append(dxx.shape[0])
dc = pd.DataFrame(dc).sort_values(['fleet size', 'td']).reset_index(drop=True)

# function to curve-fit
def func(x, a, b):
    return a * x + b

# coverage vs fleet size
xc = np.arange(100, 2001, 20)
fig, ax = open_figure('coverage vs fleet size', figsize=(12, 6))
for td in np.sort(pd.unique(dc['td'])):
    x = dc.loc[dc['td'] == td, 'fleet size'].values
    y = dc.loc[dc['td'] == td, 'coverage'].values
    p = ax.plot(x, y, 'x-', ms=12, lw=4, label=f'time-window, {td}sec')[0]
    (a, b), _ = curve_fit(func, x, y)
    ax.plot(xc, func(xc, a, b), '--', lw=3, color=p.get_color())
format_axes('fleet size, thousand vehicles', '% of collisions with any nearby vehicle', 'extrapoloated coverage vs fleet size', ax)
ax.legend(loc='upper left', numpoints=3, handlelength=5)
ax.set_ylim(-1, 101)
ax.set_yticks(np.arange(0, 101, 10))
ax.set_xlim(-1, 2001)
ax.set_xticks(np.arange(0, 2001, 200))
largefonts(18)
fig.tight_layout()
# # save_pngs(r'/mnt/home/russell.burdt/data')
plt.show()
