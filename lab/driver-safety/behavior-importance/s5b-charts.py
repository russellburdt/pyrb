
"""
behavior analysis composite chart, more than one metrics object
"""

import os
import lytx
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrb.mpl import expanding_bar_chart, save_pngs
from ipdb import set_trace


# metrics objects
n = 100
for fn, label in (
    ('metrics-dist.p', 'Distribution'),
    ('metrics-ft.p', 'Freight/Trucking')):

    # behavior metrics data
    with open(os.path.join(r'c:/Users/russell.burdt/Downloads', fn), 'rb') as fid:
        dm = pickle.load(fid)
    x1, x2, x12 = dm['x1'], dm['x2'], dm['x12']
    x3, x4, x34 = dm['x3'], dm['x4'], dm['x34']
    rfc, lr = dm['rfc-model'], dm['lr-model']

    # composite metric
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
    expanding_bar_chart(x=x, labels=labels, legend=label, figsize=(18, 8), size=18, title=title, xlabel='composite score (0 to 1)')

# clean up
plt.show()
# save_pngs(r'c:/Users/russell.burdt/Downloads')
