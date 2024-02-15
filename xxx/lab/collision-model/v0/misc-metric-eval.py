
"""
distributions of individual vehicle evaluation metrics by collision / non-collision populations
"""

import os
import config
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyrb.mpl import open_figure, largefonts, format_axes, save_pngs
from tqdm import tqdm
from ipdb import set_trace


# collision model data
adir = r'c:\Users\russell.burdt\Downloads\artifacts-2022-04-15-11-10-15'
dc = pd.read_pickle(os.path.join(adir, 'configuration.p'))
dfm = pd.read_pickle(os.path.join(adir, r'feature-importance.p'))

# model feature importance vs permutation feature importance
title = f'model feature importance vs permutation feature importance, {os.path.split(adir)[1]}'
fig, ax = open_figure(title, 1, 2, figsize=(14, 8))
ax[0].plot(dfm['model feature importance'], dfm['roc_auc']['importances_mean'], 'x', ms=12, mew=3)
format_axes('model feature importance', 'permutation feature importance', 'roc_auc permutation\nvs model feature importance', ax[0])
ax[1].plot(dfm['model feature importance'], dfm['average_precision']['importances_mean'], 'x', ms=12, mew=3)
format_axes('model feature importance', 'permutation feature importance', 'average_precision permutation\nvs model feature importance', ax[1])
largefonts(18)
fig.tight_layout()

# model feature importance vs mean-differences feature importance
features = dfm['features']
dist_importance = dfm['distribution feature importance']
ok = ~np.isnan(dist_importance)
dist_importance, features = dist_importance[ok], features[ok]
ok = np.argsort(dist_importance)[::-1]
dist_importance, features = dist_importance[ok], features[ok]
title = f'model feature importance vs mean-differences feature importance, {os.path.split(adir)[1]}'
fig, ax = open_figure(title, figsize=(10, 6))
ax.plot(dfm['model feature importance'], dfm['distribution feature importance'], 'x', ms=12, mew=3)
format_axes('model feature importance', 'KS test statistic', 'distribution vs model feature importance', ax)
largefonts(18)
fig.tight_layout()

# data for metric distributions
df = pd.read_pickle(os.path.join(adir, 'ml-data.p'))
dcm = pd.read_pickle(os.path.join(adir, 'population-data.p'))

# metric distributions
metric = 'gps_days'
x = df[metric].values
xp = x[dcm['collision'].values]
xn = x[~dcm['collision'].values]
xp = xp[~np.isnan(xp)]
xn = xn[~np.isnan(xn)]
assert (xp.size > 0) and (xn.size > 0)
label_positive = f'{xp.size} vehicle evaluations\nleading to a collision'
label_negative = f'{xn.size} vehicle evaluations\nNOT leading to a collision'
xr = min(xp.min(), xn.min()), max(xp.max(), xn.max())
xs = np.diff(xr) / 1000
xs = 1
dist = 'both'
if dist == 'collision':
    fc, ac = utils.metric_distribution(
        x=xp, xr=xr, xs=xs, title=f'distribution of {metric}, collision, {os.path.split(adir)[1]}', ax_title=f'distribution of {metric}',
        xlabel=metric, legend=label_positive + f'\nmean={xp.mean():.4f}', figsize=(10, 6), size=18, logscale=True, pdf=False, alpha=0.4,
        loc='upper right', bbox_to_anchor=None)
elif dist == 'non-collision':
    fn, an = utils.metric_distribution(
        x=xn, xr=xr, xs=xs, title=f'distribution of {metric}, non-collision, {os.path.split(adir)[1]}', ax_title=f'distribution of {metric}',
        xlabel=metric, legend=label_negative + f'\nmean={xn.mean():.4f}', figsize=(10, 6), size=18, logscale=True, pdf=False, alpha=0.4,
        loc='upper right', bbox_to_anchor=None)
elif dist == 'both':
    utils.metric_distribution(
        x=xp, xr=xr, xs=xs, title=f'distribution of {metric}, {os.path.split(adir)[1]}', ax_title=f'distribution of {metric}',
        xlabel=metric, legend=label_positive + f'\nmean={xp.mean():.4f}', figsize=(10, 6), size=18, logscale=True, pdf=True, alpha=0.4,
        loc='upper left', bbox_to_anchor=(1, 1))
    utils.metric_distribution(
        x=xn, xr=xr, xs=xs, title=f'distribution of {metric}, {os.path.split(adir)[1]}', ax_title=f'distribution of {metric}',
        xlabel=metric, legend=label_negative + f'\nmean={xn.mean():.4f}', figsize=(10, 6), size=18, logscale=True, pdf=True, alpha=0.4,
        loc='upper left', bbox_to_anchor=(1, 1))

plt.show()
