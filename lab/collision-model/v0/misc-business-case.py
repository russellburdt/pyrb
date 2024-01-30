
"""
collision-prediction-model business case calculations
"""

import os
import utils
import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import ks_2samp
from pyrb.processing import int_to_ordinal
from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
from ipdb import set_trace
from tqdm import tqdm
plt.style.use('bmh')


# collision-prediction model results
fn, log_scale = r'c:/Users/russell.burdt/Downloads/artifacts-2022-04-19-17-33-34.p', False
fn, log_scale = r'c:/Users/russell.burdt/Downloads/artifacts-2022-04-19-20-34-03.p', True
df = pd.read_pickle(fn)

# text summary, num vehicle evaluations by industry
summary = f"""num vehicle evals, {df.shape[0]:.0f}"""
for ind, count in df['IndustryDesc'].value_counts().iteritems():
    summary += f'\nnum evals in {ind}, {count}'

# text summary, num collisions
n0 = df.shape[0]
nc = df['collision'].sum()
summary += f"""\nnum evals with any collision, {nc} ({100 * nc / n0:.3f}% of {n0})"""

# text summary, num severity records
na = np.logical_or.reduce((
    (df['num severity 1'] > 0).values,
    (df['num severity 2'] > 0).values,
    (df['num severity 3'] > 0).values,
    (df['num severity 4'] > 0).values)).sum()
summary += f"""\nnum evals with any severity record, {na} ({100 * na / n0:.3f}% of {n0})"""
ns = df['num severity 1'] + df['num severity 2'] + df['num severity 3'] + df['num severity 4']
for x in range(1, int(ns.max() + 1)):
    if x == 1:
        summary += f"""\nnum evals with exactly {x} severity record, {(ns == x).sum()}"""
    else:
        summary += f"""\nnum evals with exactly {x} severity records, {(ns == x).sum()}"""
total = 0
for sev in [1, 2, 3, 4]:
    summary += f"""\nnum severity {sev} records, {df[f'num severity {sev}'].sum():.0f}"""
    total += df[f'num severity {sev}'].sum()
summary += f"""\ntotal num severity records, {total:.0f}"""

# text summary, vehicle evaluation windows
dts = df[['time0', 'time1', 'time2']]
dts = dts.loc[~dts.duplicated()].sort_values('time1').reset_index(drop=True)
dts.index = range(1, dts.shape[0] + 1)
tc = lambda x: pd.Timestamp(x).strftime('%d %b %Y')
predictor_days = pd.unique(dts['time1'] - dts['time0'])
assert predictor_days.size == 1
predictor_days = int(predictor_days[0].astype('float') * (1e-9) / (60 * 60 * 24))
collision_days = pd.unique(dts['time2'] - dts['time1'])
pxs = np.array([f'{tc(a)} to {tc(b)}' for x, (a, b, c) in dts.iterrows()])
cxs = np.array([f'{tc(b)} to {tc(c)}' for x, (a, b, c) in dts.iterrows()])
assert pxs.size == cxs.size
summary += f'\npredictor intervals, all {predictor_days} days\n'
summary += '\n'.join(pxs)
summary += f'\ncollision intervals\n'
summary += '\n'.join(cxs)

# consistent chart colors by industry
industries = np.sort(pd.unique(df['IndustryDesc']))
colors = plt.cm.tab10([x / industries.size for x in range(industries.size)])

# visualization of collision-prediction model definition
fig, ax = open_figure(f'collision-prediction model definition, {os.path.split(fn)[1][:-2]}', 4, 1, figsize=(16, 9))
# predictor and collision intervals
for x, (time0, time1, time2) in dts.iterrows():
    p0 = ax[0].fill_between(x=np.array([time0, time1]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='darkblue', alpha=0.8)
    p1 = ax[0].fill_between(x=np.array([time1, time2]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='orange', alpha=0.2)
    if x == 1:
        p0.set_label(f'predictor interval, {predictor_days} days')
        p1.set_label(f'collision interval')
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
format_axes('', '', 'predictor and collision intervals in collision-prediction model', ax[0], apply_concise_date_formatter=True)
ax[0].set_yticks(dts.index)
ax[0].set_yticklabels([f'{int_to_ordinal(x)} interval' for x in dts.index])
# vehicle evaluations by industry
for industry, color in zip(industries, colors):
    ts = df.loc[df['IndustryDesc'] == industry, 'time1'].value_counts().sort_index()
    ax[1].plot(ts.index, ts.values, '.-', lw=3, ms=10, color=color, label=f'{industry}, {ts.values.sum()} total')
if log_scale:
    ax[1].set_yscale('log')
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
format_axes('', 'count', f'num vehicle evaluations by industry, {df.shape[0]} total', ax[1], apply_concise_date_formatter=True)
# collision counts by industry
total = 0
for industry, color in zip(industries, colors):
    ts = df.loc[(df['IndustryDesc'] == industry) & (df[f'collision']), 'collision_ts'].values
    if ts.size == 0:
        continue
    ts = np.sort(np.concatenate(ts))
    count = np.cumsum(np.ones(ts.size))
    ax[2].plot(ts, count, '.-', lw=3, ms=10, color=color, label=f'{industry}, {ts.size} total')
    total += ts.size
if log_scale:
    ax[2].set_yscale('log')
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
format_axes('', 'count', f'cumulative num collisions by industry, {total} total', ax[2], apply_concise_date_formatter=True)
# collision percentages by industry
te, tc = 0, 0
for industry, color in zip(industries, colors):
    evals = df.loc[df['IndustryDesc'] == industry].groupby('time1')['collision'].count()
    collisions = df.loc[df['IndustryDesc'] == industry].groupby('time1')['collision'].sum()
    rate = 100 * collisions / evals
    te += evals.sum()
    tc += collisions.sum()
    ax[3].plot(rate.index, rate.values, '.-', lw=3, ms=10, color=color, label=f'{industry}, {100 * collisions.sum() / evals.sum():.2f}% total')
ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
format_axes('', '%', f'percentage of vehicle evaluations with any collisions in collision interval, {100 * tc / te:.2f}% total', ax[3], apply_concise_date_formatter=True)
# clean up
for x in ax:
    x.set_xlim(dts.iloc[0]['time0'] - pd.Timedelta(days=3), dts.iloc[-1]['time2'] + pd.Timedelta(days=3))
largefonts(14)
fig.tight_layout()
fig.subplots_adjust(hspace=0.4)

# visualization of collision severity counts
fig, ax = open_figure(f'collision-prediction model severity counts, {os.path.split(fn)[1][:-2]}', 5, 1, figsize=(16, 9), sharex=True)
# predictor and collision intervals
for x, (time0, time1, time2) in dts.iterrows():
    p0 = ax[0].fill_between(x=np.array([time0, time1]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='darkblue', alpha=0.8)
    p1 = ax[0].fill_between(x=np.array([time1, time2]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='orange', alpha=0.2)
    if x == 1:
        p0.set_label(f'predictor interval, {predictor_days} days')
        p1.set_label(f'collision interval')
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
format_axes('', '', 'predictor and collision intervals in collision-prediction model', ax[0], apply_concise_date_formatter=True)
ax[0].set_yticks(dts.index)
ax[0].set_yticklabels([f'{int_to_ordinal(x)} interval' for x in dts.index])
# scan over collision severities
for sev in [1, 2, 3, 4]:
    c0 = df[f'num severity {sev}'] > 0
    total = 0
    for industry, color in zip(industries, colors):
        c1 = df['IndustryDesc'] == industry
        ts = np.array([], dtype=np.datetime64)
        for _, row in df[c0 & c1].iterrows():
            ok = row['severity order'] == sev
            assert np.any(ok)
            ts = np.hstack((ts, row['collision_ts'][ok]))
        ts = np.sort(ts)
        count = np.cumsum(np.ones(ts.size))
        if ts.size == 0:
            continue
        total += ts.size

        # plot collision timestamps for severity and industry
        ax[sev].plot(ts, count, '.-', lw=3, ms=10, label=f'{industry}, {ts.size} total', color=color)

    # clean up
    format_axes('', 'count', f'cumulative num severity {sev} collisions by industry, {total} total', ax[sev], apply_concise_date_formatter=True)
    ax[sev].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
    if log_scale:
        ax[sev].set_yscale('log')
# clean up
largefonts(14)
fig.tight_layout()

# visualization of core model prediction probabilities
fig, ax0 = open_figure(f'collision-prediction model probability distributions, {os.path.split(fn)[1][:-2]}', figsize=(10, 6))
ax1 = ax0.twinx()
pos = df.loc[df['collision'], 'prediction probability'].values
neg = df.loc[~df['collision'], 'prediction probability'].values
bins = np.linspace(0, 1, 50)
width = np.diff(bins)[0]
centers = (bins[1:] + bins[:-1]) / 2
posx = np.digitize(pos, bins)
negx = np.digitize(neg, bins)
posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
assert (posx[-1] == 0) and (negx[-1] == 0)
posx, negx = posx[:-1], negx[:-1]
l0 = ax0.plot(centers, negx, '-', color='orange', lw=3, label=f"""{(~df['collision']).sum()} non-collision probabilities""")[0]
l1 = ax1.plot(centers, posx, '--', color='darkblue', lw=3, label=f"""{df['collision'].sum()} collision probabilities""")[0]
ax0.set_xlabel('collision-prediction model probability')
ax0.set_ylabel('count of non-collision probabilities')
ax1.set_ylabel('count of collision probabilities')
ax0.set_title('distribution of collision-prediction model probabilities')
labels = [x.get_label() for x in [l0, l1]]
ax0.legend([l0, l1], labels, loc='upper right', numpoints=3, shadow=True, fancybox=True, handlelength=4)
for ax in [ax0, ax1]:
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.grid(False)
largefonts(14)
fig.tight_layout()

# visualization of core model cumulative distribution functions and KS test statistic
fig, ax = open_figure(f'collision-prediction model cumulative distribution functions, {os.path.split(fn)[1][:-2]}', figsize=(18, 6))
pos = df.loc[df['collision'], 'prediction probability'].values
neg = df.loc[~df['collision'], 'prediction probability'].values
bins = np.linspace(0, 1, 50)
width = np.diff(bins)[0]
centers = (bins[1:] + bins[:-1]) / 2
posx = np.digitize(pos, bins)
negx = np.digitize(neg, bins)
posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
assert (posx[-1] == 0) and (negx[-1] == 0)
posx, negx = posx[:-1], negx[:-1]
posx = np.cumsum(posx / posx.sum())
negx = np.cumsum(negx / negx.sum())
ax.plot(centers, negx, '-', color='green', lw=3, label=f"""{(~df['collision']).sum()} non-collision probabilities""")[0]
ax.plot(centers, posx, '--', color='purple', lw=3, label=f"""{df['collision'].sum()} collision probabilities""")[0]
ks = ks_2samp(pos, neg)
x = np.argmax(np.abs(posx - negx))
ax.plot(np.tile(centers[x], 2), np.array([negx[x], posx[x]]), '-', color='darkorange', lw=3, label='max distance')
title = 'cumulative distribution functions of collision-prediction model probabilities'
title += f'\nKS statistic {ks.statistic:.2f}, KS p-value {100 * ks.pvalue:.3f}%'
format_axes('collision-prediction model probability', 'cdf', title, ax)
ax.legend(loc='upper left', numpoints=3, shadow=True, fancybox=True, handlelength=4)
ax.set_xlim(-0.01, 1.01)
ax.set_xticks(np.arange(0, 1.01, 0.1))
largefonts(14)
fig.tight_layout()

# create DataFrame of KS test results by severity and industry
ksr = defaultdict(list)
for industry in industries:

    # all negative vehicle evaluations
    dx0 = df.loc[(df['IndustryDesc'] == industry) & (~df['collision']), 'prediction probability'].values

    # all positive vehicle evaluations
    dx1 = df.loc[(df['IndustryDesc'] == industry) & (df['collision']), 'prediction probability'].values
    if (dx0.size == 0) or (dx1.size == 0):
        continue

    # KS test and save results to DataFrame
    ks = ks_2samp(dx0, dx1)
    ksr['industry'].append(industry)
    ksr['population 1'].append(f'{dx0.size} negative vehicle evaluations')
    ksr['population 2'].append(f'{dx1.size} positive vehicle evaluations')
    ksr['KS statistic'].append(f'{ks.statistic:.2f}')
    ksr['KS pvalue'].append(f'{100 * ks.pvalue:.3f}%')

    # run KS test by collision severity
    for sev in [1, 2, 3, 4]:
        dx1 = df.loc[(df['IndustryDesc'] == industry) & (df[f'num severity {sev}'] > 0), 'prediction probability'].values
        if dx1.size == 0:
            continue

        # KS test and save results to DataFrame
        ks = ks_2samp(dx0, dx1)
        ksr['industry'].append(industry)
        ksr['population 1'].append(f'{dx0.size} negative vehicle evaluations')
        ksr['population 2'].append(f'{dx1.size} severity {sev} vehicle evaluations')
        ksr['KS statistic'].append(f'{ks.statistic:.2f}')
        ksr['KS pvalue'].append(f'{100 * ks.pvalue:.3f}%')

    # final KS test comparing severity 1 vs 2-3-4
    dx0 = df.loc[(df['IndustryDesc'] == industry) & (df[f'num severity 1'] > 0), 'prediction probability'].values
    ok = np.logical_or.reduce((
        (df['num severity 2'] > 0).values,
        (df['num severity 3'] > 0).values,
        (df['num severity 4'] > 0).values))
    dx1 = df.loc[np.logical_and((df['IndustryDesc'] == industry).values, ok), 'prediction probability'].values
    if (dx0.size == 0) or (dx1.size == 0):
        continue
    ks = ks_2samp(dx0, dx1)
    ksr['industry'].append(industry)
    ksr['population 1'].append(f'{dx0.size} severity 1 vehicle evaluations')
    ksr['population 2'].append(f'{dx1.size} severity 2-3-4 vehicle evaluations')
    ksr['KS statistic'].append(f'{ks.statistic:.2f}')
    ksr['KS pvalue'].append(f'{100 * ks.pvalue:.3f}%')
ksr = pd.DataFrame(ksr)

# visualization of severity prediction probabilities
fig, ax = open_figure(f'collision-prediction model severity distributions, {os.path.split(fn)[1][:-2]}', 5, 1, figsize=(14, 9), sharex=True)
bins = np.linspace(0, 1, 60)
width = np.diff(bins)[0]
centers = (bins[1:] + bins[:-1]) / 2
centers = np.hstack((centers, centers[-1] + width))
alpha = 0.7
# with any collision by industry
for industry, color in zip(industries, colors):
    px = df.loc[(df['collision']) & (df['IndustryDesc'] == industry), 'prediction probability'].values
    if px.size == 0:
        continue
    x = np.digitize(px, bins)
    assert ((x == 0).sum() == 0) and ((x == bins.size).sum() == 0)
    height = np.array([(x == xi).sum() for xi in range(1, bins.size + 1)])
    # height = height / height.sum()
    ax[0].plot(centers, height, '-', lw=1, color=color, alpha=alpha)
    ax[0].bar(x=bins, height=height, align='edge', width=width, color=color, alpha=alpha, label=f'{industry}, {px.size} total, {px.mean():.2f} mean')
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
format_axes('prediction probability', 'bin count', 'dist of prediction probabilities for rows with any collision', ax[0])
# by collision severity and industry
for sev in [1, 2, 3, 4]:
    for industry, color in zip(industries, colors):
        px = df.loc[(df[f'num severity {sev}'] > 0) & (df['IndustryDesc'] == industry), 'prediction probability'].values
        if px.size == 0:
            continue
        x = np.digitize(px, bins)
        assert ((x == 0).sum() == 0) and ((x == bins.size).sum() == 0)
        height = np.array([(x == xi).sum() for xi in range(1, bins.size + 1)])
        # height = height / height.sum()
        ax[sev].plot(centers, height, '-', lw=1, color=color, alpha=alpha)
        ax[sev].bar(x=bins, height=height, align='edge', width=width, color=color, alpha=alpha, label=f'{industry}, {px.size} total, {px.mean():.2f} mean')
        # cdf = np.cumsum(pdf)
        # ax[sev].plot(centers, cdf, '-', lw=4, color=color, label=f'{industry}, {px.size} total')
    ax[sev].legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=3, shadow=True, fancybox=True)
    format_axes('prediction probability', 'bin count', f'dist of prediction probabilities for rows with any collision of severity {sev}', ax[sev])
# clean up
for x in ax:
    x.set_xlim(-0.01, 1.01)
    x.set_xticks(np.arange(0, 1.01, 0.1))
    if log_scale:
        x.set_yscale('log')
largefonts(14)
fig.tight_layout()

# nominal insurance loss ratio
nom_rev_in, pay_out = 0, 0
c1 = (df['num severity 1'] > 0).values
c234 = np.logical_or.reduce((
    (df['num severity 2'] > 0).values,
    (df['num severity 3'] > 0).values,
    (df['num severity 4'] > 0).values))
for industry in industries:
    c0 = (df['IndustryDesc'] == industry).values
    # revenue in
    nom_rev_in += c0.sum() * config.BC['nominal monthly premium'][industry]
    # pay out due to severity 1 collisions
    pay_out += (c0 & c1).sum() * config.BC['collision payout'][f'1-{industry}']
    # pay out due to severity 2-3-4 collisions
    pay_out += (c0 & c234).sum() * config.BC['collision payout'][f'234-{industry}']
nom_ins_loss_ratio = pay_out / nom_rev_in

# adjusted insurance loss ratio
adj_rev_in = 0
c0 = df['IndustryDesc'] == 'Distribution'
c1 = df['IndustryDesc'] == 'Freight/Trucking'
c2 = df['prediction probability'] < 0.30
c3 = (df['prediction probability'] >= 0.30) & (df['prediction probability'] < 0.45)
c4 = df['prediction probability'] >= 0.45
assert (c0 & c2).sum() + (c0 & c3).sum() + (c0 & c4).sum() + (c1 & c2).sum() + (c1 & c3).sum() + (c1 & c4).sum() == df.shape[0]
adj_rev_in += (c0 & c2).sum() * 160
adj_rev_in += (c0 & c3).sum() * 200
adj_rev_in += (c0 & c4).sum() * 240
adj_rev_in += (c1 & c2).sum() * 800
adj_rev_in += (c1 & c3).sum() * 1000
adj_rev_in += (c1 & c4).sum() * 1200

# show
plt.show()
