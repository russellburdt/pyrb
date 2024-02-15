
"""
behavior analysis charts, single metrics object
"""

import os
import lytx
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrb.mpl import expanding_bar_chart, save_pngs


# behavior metrics data
with open(r'c:/Users/russell.burdt/Downloads/metrics.p', 'rb') as fid:
    dm = pickle.load(fid)
x1, x2, x12 = dm['x1'], dm['x2'], dm['x12']
x3, x4, x34 = dm['x3'], dm['x4'], dm['x34']
rfc, lr = dm['rfc-model'], dm['lr-model']
rx = dm['relative occurrence']

# modify labels with relative occurrence
def convert(x):
    rv = []
    for label in labels:
        rv.append(f"""{label} ({rx.loc[rx['behavior'] == label, 'value'].iloc[0]:.1f}%)""")
    return np.array(rv)

# chart of x1 and x2
n = 14
title = 'percentage of vehicle evals with any individual behavior'
x, labels = 100 * x1['value'].values, x1['behavior'].values
x, labels = x[::-1], labels[::-1]
fig, ax = expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='percentage', legend=f"""{x1['num evals'].iloc[0]} collision evals""")
x, labels = 100 * x2['value'].values, x2['behavior'].values
x, labels = x[::-1], labels[::-1]
fig, ax = expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='percentage', legend=f"""{x2['num evals'].iloc[0]} non-collision evals""")
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax - n - 5, ymax)

# x12 discrepancy chart
title = 'difference in percentage of collision vs non-collision evals with any individual behavior'
x, labels = 100 * x12['abs-diff'].values, x12['behavior'].values
x, labels = x[:n][::-1], labels[:n][::-1]
expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='percentage')

# chart of x3 and x4
n = 14
title = 'average count of individual behaviors for vehicle evals'
x, labels = x3['value'].values, x3['behavior'].values
x, labels = x[::-1], labels[::-1]
fig, ax = expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='average count', legend=f"""{x3['num evals'].iloc[0]} collision evals""")
x, labels = x4['value'].values, x4['behavior'].values
x, labels = x[::-1], labels[::-1]
fig, ax = expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='average count', legend=f"""{x4['num evals'].iloc[0]} non-collision evals""")
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax - n - 5, ymax)

# x34 discrepancy chart
title = 'difference in average count of individual behaviors for collision vs non-collision evals'
x, labels = x34['abs-diff'].values, x34['behavior'].values
x, labels = x[:n][::-1], labels[:n][::-1]
expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='average count')

# rfc vs lr model feature importance
n = 20
title = 'random forest vs logistic regression model feature importance'
x, labels = rfc['value'].values, rfc['behavior'].values
x, labels = x[::-1], labels[::-1]
x = x / x.max()
fig, ax = expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='normalized feature importance', legend='Random Forest')
x, labels = lr['value'].values, lr['behavior'].values
x, labels = x[::-1], labels[::-1]
x = x / x.max()
fig, ax = expanding_bar_chart(x=x, labels=labels, figsize=(18, 8), size=18, title=title, xlabel='normalized feature importance', legend='Logistic Regression')
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax - n - 5, ymax)

# composite score and chart
cs = pd.merge(on='behavior', how='inner',
    left=x12[['behavior', 'abs-diff']].rename(columns={'abs-diff': 'xa'}),
    right=x34[['behavior', 'abs-diff']].rename(columns={'abs-diff': 'xb'}))
cs = pd.merge(on='behavior', how='inner', left=cs, right=rfc.rename(columns={'value': 'xc'}))
cs = pd.merge(on='behavior', how='inner', left=cs, right=lr.rename(columns={'value': 'xd'}))
for col in ['xa', 'xb', 'xc', 'xd']:
    cs[col] = cs[col].values / cs[col].max()
cs['composite'] = (cs['xa'] + cs['xb'] + cs['xc'] + cs['xd']) / 4
cs = cs.sort_values('composite', ascending=False).reset_index(drop=True)
title = 'composite score (average of normalized metric values) for individual behaviors'
x, labels = cs['composite'].values, cs['behavior'].values
x, labels = x[:n][::-1], labels[:n][::-1]
expanding_bar_chart(x=x, labels=convert(labels), figsize=(18, 8), size=18, title=title, xlabel='composite score (0 to 1)')

# clean up
# plt.show()
save_pngs(r'c:/Users/russell.burdt/Downloads')
