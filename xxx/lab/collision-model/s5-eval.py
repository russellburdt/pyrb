
"""
collision prediction model eval based on prediction probabilities dataframe
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
plt.style.use('bmh')


# model artifacts (either from cross-val or train-test)
dx = pd.read_pickle(r'c:/Users/russell.burdt/Downloads/results.p')

# data for pdf and cdf
neg, pos = dx.loc[~dx['outcome'], 'pred'].values, dx.loc[dx['outcome'], 'pred'].values
bins = np.linspace(0, 1, 100)
centers = (bins[1:] + bins[:-1]) / 2
posx = np.digitize(pos, bins)
negx = np.digitize(neg, bins)
posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
assert (posx[-1] == 0) and (negx[-1] == 0)
posx, negx = posx[:-1], negx[:-1]
posx = posx / posx.sum()
negx = negx / negx.sum()

# pdf
fig, ax = open_figure('pdf and cdf', 2, 1, figsize=(18, 8))
ax[0].plot(centers, posx, '.-', ms=8, lw=3, label=f'{pos.size} collision probabilities')
ax[0].plot(centers, negx, '.-', ms=8, lw=3, label=f'{neg.size} non-collision probabilities')
xlabel = 'collision-prediction model probability'
title = 'pdf of collision-prediction model probabilities'
format_axes(xlabel, 'PDF', title, ax[0])
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, handlelength=4)

# cdf
posx = np.cumsum(posx)
negx = np.cumsum(negx)
ks = ks_2samp(pos, neg)
ax[1].plot(centers, posx, '.-', ms=8, lw=3, label=f'{pos.size} collision probabilities')
ax[1].plot(centers, negx, '.-', ms=8, lw=3, label=f'{neg.size} non-collision probabilities')
title = 'cdf of collision-prediction model probabilities\n'
title += f'KS statistic {ks.statistic:.2f}, KS p-value {ks.pvalue:.2f}'
format_axes(xlabel, 'CDF', title, ax[1])
ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, handlelength=4)
largefonts(20)
fig.tight_layout()

# vehicle eval segmentation curve
fig, ax = open_figure('vehicle eval segmentation curve', figsize=(16, 6))
x = 100 * np.arange(1, dx.shape[0] + 1) / dx.shape[0]
y = 100 * dx.sort_values('pred', ascending=False)['outcome'].values.cumsum() / dx['outcome'].sum()
ax.plot(x, y, '.-', lw=4)
title = 'cumulative percentage of collisions\n'
title += 'vehicle evals sorted by descending prediction probability'
format_axes('percentage of vehicle evals', 'cumulative percentage of collisions', title, ax)
largefonts(20)
fig.tight_layout()

# save figures
save_pngs(r'c:/Users/russell.burdt/Downloads')
