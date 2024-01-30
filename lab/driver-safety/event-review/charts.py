
"""
charts based on JD event-review dashboard
"""

import os
import lytx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrb.mpl import open_figure, largefonts, format_axes, save_pngs
from ipdb import set_trace
plt.style.use('bmh')


# load data
df = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/dashboard.p')

# initialize event-review chart
x = df['collision threshold'].values * (1e5)
kws = {'ms': 10, 'lw': 3}
fig, ax = open_figure('num events vs collision threshold', 2, 1, sharex=True, figsize=(18, 8))

# event counts - all data
p = ax[0].plot(x, df['num accel events'].values, 'x-', label='num accel events', **kws)[0]
ax[0].plot(x, df['num accel events, any score'].values, 'x--', label='num accel events, any score', color=p.get_color(), **kws)
p = ax[0].plot(x, df['collision score over thresh'].values, 'x-', label='collision score over thresh', **kws)[0]
ax[0].plot(x, df['braking score over thresh'].values, linestyle='-', marker='$b$', label='braking score over thresh', color=p.get_color(), alpha=0.6, **kws)
ax[0].plot(x, df['cornering score over thresh'].values, linestyle='-', marker='$c$', label='cornering score over thresh', color=p.get_color(), alpha=0.6, **kws)
p = ax[0].plot(x, df['any score over thresh'].values, 'x-', label='any score over thresh', **kws)[0]
ax[0].plot(x, df['any score over thresh, reviewed'].values, 'x--', label='any score over thresh, reviewed', color=p.get_color(), **kws)
p = ax[0].plot(x, df['any score over thresh, reviewed, with behaviors'].values, 'x-', label='any score over thresh, reviewed, with behaviors', **kws)[0]
ax[0].plot(x, df['any score over thresh, reviewed, no behaviors'].values, 'x--', label='any score over thresh, reviewed, no behaviors', color=p.get_color(), **kws)
format_axes('collision threshold (1e-5)', 'num events', 'num events over thresh vs collision threshold', ax[0])

# percentages - all data
for signal in [
        'any score over thresh',
        'any score over thresh, reviewed, with behaviors',
        'any score over thresh, reviewed, no behaviors']:
    y = 100 * (df.loc[0, signal] - df[signal].values) / df.loc[0, signal]
    ax[1].plot(x, y, 'x-', label=signal, **kws)
format_axes('collision threshold (1e-5)', 'percentage', 'percent reduction wrt 2e-5 collision threshold', ax[1])

# clean up
for x in ax:
    x.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, handlelength=5, fancybox=True, shadow=True)
    x.xaxis.set_tick_params(labelbottom=True)
largefonts(18)
fig.tight_layout()
plt.show()


