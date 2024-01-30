
"""
bokeh app to explore and compare collision prediction model artifacts
"""

import os
import utils
import numpy as np
import pandas as pd
import pickle
from scipy.stats import ks_2samp
from functools import partial
from glob import glob
from bokeh.io import show, curdoc
from bokeh.models.widgets import MultiSelect, Slider, Select, Div, CheckboxGroup, TextInput
from bokeh.models.annotations import LegendItem, Title
from bokeh.models.layouts import Tabs, Panel
from bokeh.models.sources import ColumnDataSource
from bokeh.events import MouseMove, MouseLeave
from bokeh.layouts import column, row
from sklearn.metrics import average_precision_score, roc_auc_score
from pyrb.bokeh import MapInterface, MultiLineInterface, LearningCurveInterface, HorizontalBarChartInterface
from pyrb.bokeh import MetricDistributionInterface, ShapValuesWaterfallInterface
from pyrb.bokeh import str_axis_labels, link_axes, console_str_objects, update_fig_range1d
from ipdb import set_trace

# Learning Curve
def mlc_callback(attr, old, new, sender):
    """
    update single model learning curve interface
    """

    # model artifacts folder, descriptions, learning curve data
    adir = mlc['data-select'].value
    with open(os.path.join(datadir, adir, 'data desc.txt'), 'r') as fid:
        data_desc = ''.join(fid.readlines()[1:]).strip().replace('\n', '<br>')
    with open(os.path.join(datadir, adir, 'model desc.txt'), 'r') as fid:
        model_desc = ''.join(fid.readlines()[1:]).strip().replace('\n', '<br>')
    dlc = pd.read_pickle(os.path.join(datadir, adir, 'learning-curve-data.p'))

    # update descriptions
    c0, c1 = console_str_objects(280)
    mlc['data-desc'].text = c0 + data_desc + c1
    mlc['model-desc'].text = c0 + model_desc + c1

    # reset slider on initialization and dataset change
    if (sender is None) or (sender == mlc['data-select']):
        smin = np.concatenate(dlc['proba'].values).min()
        smax = np.concatenate(dlc['proba'].values).max()
        mlc['slider'].start = smin
        mlc['slider'].end = smax
        mlc['slider'].step = (smax - smin) / 20
        mlc['slider'].remove_on_change('value', mlc_slider_callback)
        mlc['slider'].value = (smin + smax) / 2
        mlc['slider'].on_change('value', mlc_slider_callback)

    # extract and decode selected metric
    metric = {
        'accuracy': 'acc',
        'true positive rate': 'tpr',
        'false positive rate': 'fpr',
        'precision': 'precision',
        'balanced accuracy': 'bacc'}[mlc['metric-select'].value]

    # initialize ylims for learning curve
    ymin, ymax = np.inf, -np.inf

    # extract metric values for learning curve based on predicted probability and threshold
    thresh = mlc['slider'].value
    dlc['metric'] = [utils.get_classification_metrics(ytrue=row['true'], ypred=row['proba'] >= thresh)[metric] for _, row in dlc.iterrows()]

    # train data learning curve
    train = dlc.loc[dlc['component'] == 'train', ['train frac all data', 'metric']]
    train = train.groupby('train frac all data')['metric']
    mlc['learning-curve'].train.data = {
        'x': np.array(list(train.groups.keys())),
        'mean': train.mean().values,
        'min': train.min().values,
        'max': train.max().values}
    ymin, ymax = min(ymin, train.min().min()), max(ymax, train.max().max())

    # test data learning curve
    test = dlc.loc[dlc['component'] == 'test', ['train frac all data', 'metric']]
    test = test.groupby('train frac all data')['metric']
    mlc['learning-curve'].test.data = {
        'x': np.array(list(test.groups.keys())),
        'mean': test.mean().values,
        'min': test.min().values,
        'max': test.max().values}
    ymin, ymax = min(ymin, test.min().min()), max(ymax, test.max().max())

    # clean up learning curve
    y0 = (ymin + ymax) / 2
    yr = ymax - ymin
    mlc['learning-curve'].fig.title.text = f"""learning curve, {pd.unique(dlc['split']).size}-fold cv"""
    mlc['learning-curve'].fig.y_range.start = max(0, y0 - 1.4 * yr)
    mlc['learning-curve'].fig.y_range.end = min(1, y0 + 1.4 * yr)
    mlc['learning-curve'].fig.yaxis.axis_label = mlc['metric-select'].value
    mlc['learning-curve'].fig.x_range.start = dlc['train frac all data'].min()
    mlc['learning-curve'].fig.x_range.end = 1
    mlc['learning-curve'].legend.visible = True

# Model Eval
def mev_callback(attr, old, new, sender):
    """
    update single model evaluation interface
    """

    # model artifacts folder, descriptions, eval, feature importance data
    adir = mev['data-select'].value
    with open(os.path.join(datadir, adir, 'data desc.txt'), 'r') as fid:
        data_desc = ''.join(fid.readlines()[1:]).strip().replace('\n', '<br>')
    with open(os.path.join(datadir, adir, 'model desc.txt'), 'r') as fid:
        model_desc = ''.join(fid.readlines()[1:]).strip().replace('\n', '<br>')
    with open(os.path.join(datadir, adir, 'eval desc.txt'), 'r') as fid:
        eval_desc = ''.join(fid.readlines()[1:]).strip().replace('\n', '<br>')
    yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
    with open(os.path.join(datadir, adir, 'feature-importance.p'), 'rb') as fid:
        dfm = pickle.load(fid)

    # update descriptions
    c0, c1 = console_str_objects(300)
    mev['data-desc'].text = c0 + data_desc + c1
    mev['model-desc'].text = c0 + model_desc + c1
    mev['eval-desc'].text = c0 + eval_desc + c1

    # reset slider on initialization and dataset change
    if (sender is None) or (sender == mev['data-select']):
        smin = yp['prediction probability'].min()
        smax = yp['prediction probability'].max()
        mev['slider'].start = smin
        mev['slider'].end = smax
        mev['slider'].step = (smax - smin) / 50
        mev['slider'].remove_on_change('value', mev_slider_callback)
        mev['slider'].value = (smin + smax) / 2
        mev['slider'].on_change('value', mev_slider_callback)

    # classification metrics and confusion matrix
    thresh = mev['slider'].value
    metrics = utils.get_classification_metrics(ytrue=yp['actual outcome'].values, ypred=yp['prediction probability'].values >= thresh)
    tn, tp, fn, fp = metrics.pop('tn'), metrics.pop('tp'), metrics.pop('fn'), metrics.pop('fp')
    assert tn + tp + fn + fp == yp.shape[0]

    # classification metrics bar chart
    height = 0.8
    right = np.array(list(metrics.values()))
    labels = np.array(list(metrics.keys()))
    mev['classification metrics'].data.data = {
        'y': np.arange(right.size),
        'right': right,
        'height': np.tile(height, right.size)}
    mev['classification metrics'].fig.yaxis.ticker.ticks = list(range(right.size))
    str_axis_labels(axis=mev['classification metrics'].fig.yaxis, labels=labels)
    mev['classification metrics'].fig.x_range.start = 0
    mev['classification metrics'].fig.x_range.end = 1.1
    mev['classification metrics'].fig.y_range.start = - height
    mev['classification metrics'].fig.y_range.end = right.size - 1 + height
    mev['classification metrics'].label_source.data = {
        'x': right,
        'y': np.arange(right.size),
        'text': [f'{x:0.3f}' for x in right]}

    # confusion matrix Div object
    cs, ce = console_str_objects(300)
    c0 = f"""{yp.shape[0]} total instances<br>"""
    c1 = f"""{yp['actual outcome'].sum()} positive instances<br>"""
    c2 = f"""{(yp['actual outcome'] == 0).sum()} negative instances<br>"""
    c3 = f"""true positive&nbsp&nbsp&nbsp&nbspfalse positive<br>"""
    c4 = f"""&nbsp&nbsp{tp:.0f}&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp{fp:.0f}<br>"""
    c5 = f"""false negative&nbsp&nbsp&nbsp&nbsptrue negative<br>"""
    c6 = f"""&nbsp&nbsp{fn:.0f}&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp{tn:.0f}<br>"""
    mev['confusion matrix'].text = cs + c0 + c1 + c2 + c3 + c4 + c5 + c6 + ce

    # reset feature importance method select menu on initialization and dataset change
    if (sender is None) or (sender == mev['data-select']):

        # build menu of feature importance metrics
        assert 'features' in dfm.keys()
        assert 'model feature importance' in dfm.keys()
        options = ['model feature importance']
        if ('roc_auc' in dfm.keys()) and ('average_precision' in dfm.keys()):
            options += ['permutation feature importance, roc_auc', 'permutation feature importance, average_precision']
        if 'distribution feature importance' in dfm.keys():
            options += ['distribution feature importance']

        # update feature importance menu and manage callback
        mev['feature-importance-select'].remove_on_change('value', mev_feature_importance_callback)
        mev['feature-importance-select'].options = options
        mev['feature-importance-select'].value = 'model feature importance'
        mev['feature-importance-select'].on_change('value', mev_feature_importance_callback)

    # update feature importance bar chart on initialization, feature importance change, and dataset change
    if (sender is None) or (sender == mev['feature-importance-select']) or (sender == mev['data-select']):
        height = 0.8
        method = mev['feature-importance-select'].value

        # permutation feature importance
        if method[:11] == 'permutation':
            metric = method.split(', ')[1]
            right = dfm[metric]['importances_mean']
            mev['feature importance'].fig.xaxis.axis_label = 'permutation importance'

        # distribution feature importance
        elif method[:12] == 'distribution':
            right = dfm['distribution feature importance']
            mev['feature importance'].fig.xaxis.axis_label = 'KS test metric'

        # model feature importance
        else:
            assert method == 'model feature importance'
            right = dfm['model feature importance']
            mev['feature importance'].fig.xaxis.axis_label = 'model feature importance'

        # remove nans, sort, get feature labels
        labels = dfm['features']
        ok = ~np.isnan(right)
        right, labels = right[ok], labels[ok]
        ok = np.argsort(right)
        right, labels = right[ok], labels[ok]

        # feature importance bar chart
        mev['feature importance'].data.data = {
            'y': np.arange(right.size),
            'right': right,
            'height': np.tile(height, right.size)}
        n = 10
        mev['feature importance'].fig.yaxis.ticker.ticks = list(np.arange(right.size))
        str_axis_labels(axis=mev['feature importance'].fig.yaxis, labels=labels)
        mev['feature importance'].fig.x_range.start = 0
        mev['feature importance'].fig.x_range.end = 1.05 * right.max()
        mev['feature importance'].fig.y_range.start = right.size - n - height
        mev['feature importance'].fig.y_range.end = right.size - 1 + height

# Model Eval by Company
def mdx_callback(attr, old, new, sender):
    """
    update single model evaluation by company interface
    """

    # reset fig interfaces
    mdx['roc-curve'].reset_interface()
    mdx['pr-curve'].reset_interface()
    mdx['negative prediction probabilities'].reset_interface()
    mdx['positive prediction probabilities'].reset_interface()
    mdx['pdfs'].reset_interface()
    mdx['cdfs'].reset_interface()

    # model artifacts folder, prediction probabilities, population DataFrame
    adir = mdx['data-select'].value
    yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
    dp = pd.read_pickle(os.path.join(datadir, adir, 'population-data.p'))
    dp = dp.loc[~dp['oversampled']]
    assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))

    # update company multi-select on initialization and dataset change
    if (sender is None) or (sender == mdx['data-select']):
        companies = sorted(pd.unique(dp['CompanyName']))
        mdx['company-select'].remove_on_change('value', mdx_company_select_callback)
        mdx['company-select'].options = [(x, x) for x in companies]
        mdx['company-select'].value = companies
        mdx['company-select'].on_change('value', mdx_company_select_callback)

    # extract companies, update description of selected companies
    companies = np.array(mdx['company-select'].value)
    ok = dp['CompanyName'].isin(companies)
    c0, c1 = console_str_objects(300)
    desc = f"""
        {companies.size} company(s)
        {ok.sum()} vehicle evaluations
        {dp.loc[ok, 'outcome'].sum()} positive instances""".strip().replace('\n', '<br>')
    mdx['companies-desc'].text = c0 + desc + c1

    # distribution of negative prediction probabilities
    bins = np.linspace(0, 1, 60)
    px = yp.loc[ok & ~(dp['outcome'].astype('bool')), 'prediction probability'].values
    if px.size > 0:
        px = np.digitize(px, bins)
        assert ((px == 0).sum() == 0) and ((px == 60).sum() == 0)
        top = np.array([(px == xi).sum() for xi in range(1, bins.size + 1)])
        width = np.diff(bins)[0]
        mdx['negative prediction probabilities'].data.data = {'x': bins + width / 2, 'width': np.tile(width, top.size), 'top': top}
        mdx['negative prediction probabilities'].fig.x_range.start = 0
        mdx['negative prediction probabilities'].fig.x_range.end = 1
        mdx['negative prediction probabilities'].fig.xaxis.ticker.ticks = np.arange(0, 1.01, 0.1)
        mdx['negative prediction probabilities'].fig.y_range.start = 0
        mdx['negative prediction probabilities'].fig.y_range.end = 1.05 * top.max()
        mdx['negative prediction probabilities'].fig.title.text = f"""negative prediction probabilities, x{(ok & ~dp['outcome']).sum()}"""
    else:
        mdx['negative prediction probabilities'].fig.title.text = ''

    # distribution of positive prediction probabilities
    px = yp.loc[ok & dp['outcome'].astype('bool'), 'prediction probability'].values
    if px.size > 0:
        px = np.digitize(px, bins)
        assert ((px == 0).sum() == 0) and ((px == 60).sum() == 0)
        top = np.array([(px == xi).sum() for xi in range(1, bins.size + 1)])
        mdx['positive prediction probabilities'].data.data = {'x': bins + width / 2, 'width': np.tile(width, top.size), 'top': top}
        mdx['positive prediction probabilities'].fig.x_range.start = 0
        mdx['positive prediction probabilities'].fig.x_range.end = 1
        mdx['positive prediction probabilities'].fig.xaxis.ticker.ticks = np.arange(0, 1.01, 0.1)
        mdx['positive prediction probabilities'].fig.y_range.start = 0
        mdx['positive prediction probabilities'].fig.y_range.end = 1.05 * top.max()
        mdx['positive prediction probabilities'].fig.title.text = f"""positive prediction probabilities, x{(ok & dp['outcome']).sum()}"""
    else:
        mdx['positive prediction probabilities'].fig.title.text = ''

    # return on no positive instances for selected companies
    if ~np.any(yp.loc[ok, 'actual outcome'].values):
        return

    # get roc-pr-curve data, auc, ap
    dml = utils.get_roc_pr_data(ytrue=yp.loc[ok, 'actual outcome'].values, yprob=yp.loc[ok, 'prediction probability'].values, size=100)
    auc = roc_auc_score(y_true=yp.loc[ok, 'actual outcome'].values, y_score=yp.loc[ok, 'prediction probability'].values)
    ap = average_precision_score(y_true=yp.loc[ok, 'actual outcome'].values, y_score=yp.loc[ok, 'prediction probability'].values)

    # update ROC Curve
    dml['x'] = dml['fpr'].values
    dml['y'] = dml['tpr'].values
    mdx['roc-curve'].data_sources[0].data = dml
    label = f'{adir}, AUC={auc:.3f}'.replace('artifacts-', '')
    mdx['roc-curve'].legend.items.append(LegendItem(label=label, renderers=list(mdx['roc-curve'].renderers[0, :])))

    # update PR Curve
    dml['x'] = dml['tpr'].values
    dml['y'] = dml['precision'].values
    mdx['pr-curve'].data_sources[0].data = dml
    label = f'{adir}, AP={ap:.3f}'.replace('artifacts-', '')
    mdx['pr-curve'].legend.items.append(LegendItem(label=label, renderers=list(mdx['pr-curve'].renderers[0, :])))

    # update ROC/PR hover tooltips
    mdx['roc-curve'].hover.tooltips = [('threshold', '@thresh'), ('false positive rate', '@fpr'), ('true positive rate', '@tpr')]
    mdx['pr-curve'].hover.tooltips = [('threshold', '@thresh'), ('recall (true positive rate)', '@tpr'), ('precision', '@precision')]

    # calculate pdfs
    pos = yp.loc[ok & dp['outcome'].astype('bool'), 'prediction probability'].values
    neg = yp.loc[ok & ~(dp['outcome'].astype('bool')), 'prediction probability'].values
    bins = np.linspace(0, 1, 60)
    width = np.diff(bins)[0]
    centers = (bins[1:] + bins[:-1]) / 2
    posx = np.digitize(pos, bins)
    negx = np.digitize(neg, bins)
    posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
    negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
    assert (posx[-1] == 0) and (negx[-1] == 0)
    posx, negx = posx[:-1], negx[:-1]
    posx = posx / posx.sum()
    negx = negx / negx.sum()

    # update pdf data sources and legend
    mdx['pdfs'].data_sources[0].data = {'x': centers, 'y': negx}
    label = f"""{(ok & (~dp['outcome'].astype('bool'))).sum()} non-collision probabilities"""
    mdx['pdfs'].legend.items.append(LegendItem(label=label, renderers=list(mdx['pdfs'].renderers[0, :])))
    mdx['pdfs'].data_sources[1].data = {'x': centers, 'y': posx}
    label = f"""{(ok & dp['outcome']).sum()} collision probabilities"""
    mdx['pdfs'].legend.items.append(LegendItem(label=label, renderers=list(mdx['pdfs'].renderers[1, :])))

    # cdfs and update cdf data sources and legend
    posx = np.cumsum(posx)
    negx = np.cumsum(negx)
    ks = ks_2samp(pos, neg)
    mdx['cdfs'].data_sources[0].data = {'x': centers, 'y': negx}
    label = f"""{(ok & (~dp['outcome'].astype('bool'))).sum()} non-collision probabilities"""
    mdx['cdfs'].legend.items.append(LegendItem(label=label, renderers=list(mdx['cdfs'].renderers[0, :])))
    mdx['cdfs'].data_sources[1].data = {'x': centers, 'y': posx}
    label = f"""{(ok & dp['outcome']).sum()} collision probabilities"""
    mdx['cdfs'].legend.items.append(LegendItem(label=label, renderers=list(mdx['cdfs'].renderers[1, :])))
    x = np.argmax(np.abs(posx - negx))
    mdx['cdfs'].data_sources[2].data = {'x': np.tile(centers[x], 2), 'y': np.array([negx[x], posx[x]])}
    mdx['cdfs'].legend.items.append(LegendItem(label='max distance', renderers=list(mdx['cdfs'].renderers[2, :])))
    mdx['ks-title'].text = f'KS Statistic {ks.statistic:.2f}, KS p-value {100 * ks.pvalue:.2f}%'

# Model Eval by Industry
def mde_callback(attr, old, new, sender):
    """
    update single model evaluation by industry interface
    """

    # reset fig interfaces
    mde['roc-curve'].reset_interface()
    mde['pr-curve'].reset_interface()
    mde['negative prediction probabilities'].reset_interface()
    mde['positive prediction probabilities'].reset_interface()
    mde['pdfs'].reset_interface()
    mde['cdfs'].reset_interface()

    # model artifacts folder, prediction probabilities, population DataFrame
    adir = mde['data-select'].value
    yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
    dp = pd.read_pickle(os.path.join(datadir, adir, 'population-data.p'))
    dp = dp.loc[~dp['oversampled']]
    assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))

    # update industry multi-select on initialization and dataset change
    if (sender is None) or (sender == mde['data-select']):
        industries = sorted(pd.unique(dp['IndustryDesc']))
        mde['industry-select'].remove_on_change('value', mde_industry_select_callback)
        mde['industry-select'].options = [(x, x) for x in industries]
        mde['industry-select'].value = industries
        mde['industry-select'].on_change('value', mde_industry_select_callback)

    # extract industries, update description of selected industries
    industries = np.array(mde['industry-select'].value)
    ok = dp['IndustryDesc'].isin(industries)
    c0, c1 = console_str_objects(300)
    desc = f"""
        {industries.size} industry(s)
        {ok.sum()} vehicle evaluations
        {dp.loc[ok, 'outcome'].sum()} positive instances""".strip().replace('\n', '<br>')
    mde['industries-desc'].text = c0 + desc + c1

    # distribution of negative prediction probabilities
    bins = np.linspace(0, 1, 60)
    px = yp.loc[ok & ~(dp['outcome'].astype('bool')), 'prediction probability'].values
    if px.size > 0:
        px = np.digitize(px, bins)
        assert ((px == 0).sum() == 0) and ((px == 60).sum() == 0)
        top = np.array([(px == xi).sum() for xi in range(1, bins.size + 1)])
        width = np.diff(bins)[0]
        mde['negative prediction probabilities'].data.data = {'x': bins + width / 2, 'width': np.tile(width, top.size), 'top': top}
        mde['negative prediction probabilities'].fig.x_range.start = 0
        mde['negative prediction probabilities'].fig.x_range.end = 1
        mde['negative prediction probabilities'].fig.xaxis.ticker.ticks = np.arange(0, 1.01, 0.1)
        mde['negative prediction probabilities'].fig.y_range.start = 0
        mde['negative prediction probabilities'].fig.y_range.end = 1.05 * top.max()
        mde['negative prediction probabilities'].fig.title.text = f"""negative prediction probabilities, x{(ok & ~dp['outcome']).sum()}"""
    else:
        mde['negative prediction probabilities'].fig.title.text = ''

    # distribution of positive prediction probabilities
    px = yp.loc[ok & dp['outcome'].astype('bool'), 'prediction probability'].values
    if px.size > 0:
        px = np.digitize(px, bins)
        assert ((px == 0).sum() == 0) and ((px == 60).sum() == 0)
        top = np.array([(px == xi).sum() for xi in range(1, bins.size + 1)])
        mde['positive prediction probabilities'].data.data = {'x': bins + width / 2, 'width': np.tile(width, top.size), 'top': top}
        mde['positive prediction probabilities'].fig.x_range.start = 0
        mde['positive prediction probabilities'].fig.x_range.end = 1
        mde['positive prediction probabilities'].fig.xaxis.ticker.ticks = np.arange(0, 1.01, 0.1)
        mde['positive prediction probabilities'].fig.y_range.start = 0
        mde['positive prediction probabilities'].fig.y_range.end = 1.05 * top.max()
        mde['positive prediction probabilities'].fig.title.text = f"""positive prediction probabilities, x{(ok & dp['outcome']).sum()}"""
    else:
        mde['positive prediction probabilities'].fig.title.text = ''

    # return on no positive instances for selected industries
    if ~np.any(yp.loc[ok, 'actual outcome'].values):
        return

    # get roc-pr-curve data, auc, ap
    dml = utils.get_roc_pr_data(ytrue=yp.loc[ok, 'actual outcome'].values, yprob=yp.loc[ok, 'prediction probability'].values, size=100)
    auc = roc_auc_score(y_true=yp.loc[ok, 'actual outcome'].values, y_score=yp.loc[ok, 'prediction probability'].values)
    ap = average_precision_score(y_true=yp.loc[ok, 'actual outcome'].values, y_score=yp.loc[ok, 'prediction probability'].values)

    # update ROC Curve
    dml['x'] = dml['fpr'].values
    dml['y'] = dml['tpr'].values
    mde['roc-curve'].data_sources[0].data = dml
    label = f'{adir}, AUC={auc:.3f}'.replace('artifacts-', '')
    mde['roc-curve'].legend.items.append(LegendItem(label=label, renderers=list(mde['roc-curve'].renderers[0, :])))

    # update PR Curve
    dml['x'] = dml['tpr'].values
    dml['y'] = dml['precision'].values
    mde['pr-curve'].data_sources[0].data = dml
    label = f'{adir}, AP={ap:.3f}'.replace('artifacts-', '')
    mde['pr-curve'].legend.items.append(LegendItem(label=label, renderers=list(mde['pr-curve'].renderers[0, :])))

    # update ROC/PR hover tooltips
    mde['roc-curve'].hover.tooltips = [('threshold', '@thresh'), ('false positive rate', '@fpr'), ('true positive rate', '@tpr')]
    mde['pr-curve'].hover.tooltips = [('threshold', '@thresh'), ('recall (true positive rate)', '@tpr'), ('precision', '@precision')]

    # calculate pdfs
    pos = yp.loc[ok & dp['outcome'].astype('bool'), 'prediction probability'].values
    neg = yp.loc[ok & ~(dp['outcome'].astype('bool')), 'prediction probability'].values
    bins = np.linspace(0, 1, 60)
    width = np.diff(bins)[0]
    centers = (bins[1:] + bins[:-1]) / 2
    posx = np.digitize(pos, bins)
    negx = np.digitize(neg, bins)
    posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
    negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
    assert (posx[-1] == 0) and (negx[-1] == 0)
    posx, negx = posx[:-1], negx[:-1]
    posx = posx / posx.sum()
    negx = negx / negx.sum()

    # update pdf data sources and legend
    mde['pdfs'].data_sources[0].data = {'x': centers, 'y': negx}
    label = f"""{(ok & (~dp['outcome'].astype('bool'))).sum()} non-collision probabilities"""
    mde['pdfs'].legend.items.append(LegendItem(label=label, renderers=list(mde['pdfs'].renderers[0, :])))
    mde['pdfs'].data_sources[1].data = {'x': centers, 'y': posx}
    label = f"""{(ok & dp['outcome']).sum()} collision probabilities"""
    mde['pdfs'].legend.items.append(LegendItem(label=label, renderers=list(mde['pdfs'].renderers[1, :])))

    # cdfs and update cdf data sources and legend
    posx = np.cumsum(posx)
    negx = np.cumsum(negx)
    ks = ks_2samp(pos, neg)
    mde['cdfs'].data_sources[0].data = {'x': centers, 'y': negx}
    label = f"""{(ok & (~dp['outcome'].astype('bool'))).sum()} non-collision probabilities"""
    mde['cdfs'].legend.items.append(LegendItem(label=label, renderers=list(mde['cdfs'].renderers[0, :])))
    mde['cdfs'].data_sources[1].data = {'x': centers, 'y': posx}
    label = f"""{(ok & dp['outcome']).sum()} collision probabilities"""
    mde['cdfs'].legend.items.append(LegendItem(label=label, renderers=list(mde['cdfs'].renderers[1, :])))
    x = np.argmax(np.abs(posx - negx))
    mde['cdfs'].data_sources[2].data = {'x': np.tile(centers[x], 2), 'y': np.array([negx[x], posx[x]])}
    mde['cdfs'].legend.items.append(LegendItem(label='max distance', renderers=list(mde['cdfs'].renderers[2, :])))
    mde['ks-title'].text = f'KS Statistic {ks.statistic:.2f}, KS p-value {100 * ks.pvalue:.2f}%'

# Shap Values
def svx_callback(attr, old, new, sender):
    """
    update shap values by company interface
    """
    global shap_status
    global shap_data

    # reset fig interfaces, get console objects
    svx['shap curve'].reset_interface()
    svx['shap curve'].data_sources[0].selected.remove_on_change('indices', svx_selected_callback)
    svx['shap curve'].data_sources[0].selected.indices = []
    svx['shap curve'].data_sources[0].selected.on_change('indices', svx_selected_callback)
    svx['feature distribution'].reset_interface()
    svx['waterfall'].reset_interface()
    c0, c1 = console_str_objects(270)

    # model artifacts folder, prediction probabilities, feature importance, population DataFrame
    adir = svx['data-select'].value
    fs = os.path.join(datadir, adir, 'shap.p')
    if not os.path.isfile(fs):
        svx['vehicle-eval-desc'].text = c0 + 'No shap values data' + c1
        shap_status = False
        return
    shap_status = True
    yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
    dp = pd.read_pickle(os.path.join(datadir, adir, 'population-data.p'))
    assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))
    with open(os.path.join(datadir, adir, 'feature-importance.p'), 'rb') as fid:
        dfm = pickle.load(fid)
    with open(fs, 'rb') as fid:
        ds = pickle.load(fid)
    with open(os.path.join(datadir, adir, 'ml-data.p'), 'rb') as fid:
        df = pickle.load(fid)
    assert ds['base'].size == ds['values'].shape[0]
    assert df.shape == ds['values'].shape

    # update feature-select on initialization and dataset change
    if (sender is None) or (sender == svx['data-select']):
        svx['feature-select'].remove_on_change('value', svx_feature_select_callback)
        svx['feature-select'].options = dfm['features'][np.argsort(dfm['model feature importance'])[::-1]].tolist()
        svx['feature-select'].value = svx['feature-select'].options[0]
        svx['feature-select'].on_change('value', svx_feature_select_callback)

    # filter vehicle evals by checkbox and range objects current selection
    ok = np.zeros(df.shape[0]).astype('bool')
    if 0 in svx['include-checkbox'].active:
        ok = np.logical_or(ok, yp['actual outcome'].values == 0)
    if 1 in svx['include-checkbox'].active:
        ok = np.logical_or(ok, yp['actual outcome'].values == 1)
    vmin = float(svx['include-range-min'].value)
    vmax = float(svx['include-range-max'].value)
    ok = np.logical_and(ok,
        np.logical_and(yp['prediction probability'].values > vmin, yp['prediction probability'].values < vmax))

    # extract feature data and shap base / values, update shap_data dict
    feature = svx['feature-select'].value
    xf = dfm['features'] == feature
    assert xf.sum() == 1
    shap_data = {
        'xmin': df[:, xf].flatten().min(),
        'xmax': df[:, xf].flatten().max(),
        'ymin': ds['values'][:, xf].flatten().min(),
        'ymax': ds['values'][:, xf].flatten().max(),
        'shap curve x': df[ok, xf].flatten(),
        'shap curve y': ds['values'][ok, xf].flatten(),
        'shap base': ds['base'][ok],
        'shap values': ds['values'][ok, :],
        'feature data': df[ok, :],
        'features': dfm['features'],
        'prediction probabilities': yp['prediction probability'].values[ok],
        'actual outcome': yp['actual outcome'].values[ok],
        'company': dp['CompanyName'][ok].values,
        'industry': dp['IndustryDesc'][ok].values,
        'vehicle-id': dp['VehicleId'][ok].values,
        'time0': dp['time0'][ok].values,
        'time1':dp['time1'][ok].values,
        'time2':dp['time2'][ok].values}

    # null case
    if shap_data['shap curve x'].size == 0:
        svx['shap curve'].reset_interface()
        svx['feature distribution'].reset_interface()
        svx['waterfall'].reset_interface()
        svx['vehicle-eval-desc'].text = c0 + 'No shap values data' + c1
        shap_status = False
        return

    # shap values for selected feature and manually set x/y lims
    svx['shap curve'].data_sources[0].selected.remove_on_change('indices', svx_selected_callback)
    svx['shap curve'].data_sources[0].data = {'x': shap_data['shap curve x'], 'y': shap_data['shap curve y']}
    svx['shap curve'].tap.renderers = [x for x in svx['shap curve'].fig.renderers if x.name == 'circles']
    svx['shap curve'].data_sources[0].selected.on_change('indices', svx_selected_callback)
    svx['shap curve'].fig.xaxis.axis_label = feature
    svx['shap curve'].fig.title.text = f'shap values for {feature}'
    update_fig_range1d(svx['shap curve'].fig.x_range, shap_data['shap curve x'])
    update_fig_range1d(svx['shap curve'].fig.y_range, shap_data['shap curve y'])

    # distribution of prediction probabilities
    bins = np.linspace(shap_data['xmin'], shap_data['xmax'], 300)
    xd = np.digitize(shap_data['shap curve x'], bins)
    top = np.array([(xd == xi).sum() for xi in range(1, bins.size + 1)])
    assert (xd == 0).sum() == 0
    width = np.diff(bins)[0]
    svx['feature distribution'].data.data = {'x': bins + width / 2, 'width': np.tile(width, top.size), 'top': top}
    svx['feature distribution'].fig.xaxis.axis_label = feature
    svx['feature distribution'].fig.y_range.start = 0
    svx['feature distribution'].fig.y_range.end = 1.005 * top.max()
    svx['feature distribution'].fig.title.text = f'Distribution of {feature}'

    # initialize vehicle eval div
    svx['vehicle-eval-desc'].text = c0 + f'Select one of {ok.sum()} shap values' + c1

def svx_selected_callback(attr, old, new):
    """
    callback on selected shap value data points
    """

    # validate, clear interface, get console objects
    assert shap_status
    svx['waterfall'].reset_interface()
    c0, c1 = console_str_objects(270)

    # null cases
    if new == []:
        svx['vehicle-eval-desc'].text = c0 + f"""Select one of {shap_data['time0'].size} shap values""" + c1
        return
    if len(new) > 1:
        svx['shap curve'].data_sources[0].selected.remove_on_change('indices', svx_selected_callback)
        svx['shap curve'].data_sources[0].selected.indices = []
        svx['shap curve'].data_sources[0].selected.on_change('indices', svx_selected_callback)
        svx['vehicle-eval-desc'].text = c0 + f"""Select one of {shap_data['time0'].size} shap values""" + c1
        return
    x = new[0]

    # vehicle eval description div
    time0 = pd.Timestamp(shap_data['time0'][x]).strftime('%m/%d/%Y')
    time1 = pd.Timestamp(shap_data['time1'][x]).strftime('%m/%d/%Y')
    time2 = pd.Timestamp(shap_data['time2'][x]).strftime('%m/%d/%Y')
    desc = f"""{shap_data['vehicle-id'][x]}
        company, {shap_data['company'][x]}
        industry, {shap_data['industry'][x]}
        predictor interval, {time0} to {time1}
        collision interval, {time1} to {time2}
        prediction probability, {shap_data['prediction probabilities'][x]:.2f}
        actual collision outcome, {bool(shap_data['actual outcome'][x])}""".replace('\n', '<br>')
    svx['vehicle-eval-desc'].text = c0 + desc + c1

    # shap waterfall chart data
    sdata = utils.get_shap_chart_data(
        base=shap_data['shap base'][x],
        values=shap_data['shap values'][x, :],
        xr=shap_data['feature data'][x, :],
        n_features=6,
        cols=shap_data['features'])

    # shap waterfall chart column data sources
    yp, positive, pcols, yn, negative, ncols = sdata
    base = positive[0, 0] if positive.size > 0 else negative[0, 1]
    svx['waterfall'].base.data = {'x': np.tile(base, 2), 'y': np.array([-0.5, pcols.size + ncols.size - 0.5])}
    svx['waterfall'].positive.data = {'x0': positive[:, 0], 'x1': positive[:, 1], 'y0': yp[:, 0], 'y1': yp[:, 1],
        'labels': [f'+{x:.3f}' for x in np.diff(positive).flatten()]}
    svx['waterfall'].negative.data = {'x0': negative[:, 1], 'x1': negative[:, 0], 'y0': yn[:, 0], 'y1': yn[:, 1],
        'labels': [f'-{x:.3f}' for x in np.diff(negative).flatten()]}

    # clean up
    vs = np.hstack((negative.flatten(), positive.flatten()))
    svx['waterfall'].fig.x_range.start = 0.9 * vs.min()
    svx['waterfall'].fig.x_range.end = 1.1 * vs.max()
    svx['waterfall'].fig.y_range.start = -0.5
    svx['waterfall'].fig.y_range.end = pcols.size + ncols.size - 0.5
    svx['waterfall'].fig.yaxis.ticker.ticks = list(range(pcols.size + ncols.size))
    str_axis_labels(axis=svx['waterfall'].fig.yaxis, labels=np.hstack((pcols, ncols))[::-1])

# PDF/CDF
def cdf_callback(attr, old, new, sender):
    """
    PDF and CDF for a single collision prediction model
    """

    # reset fig interfaces
    cdf['pdfs'].reset_interface()
    cdf['cdfs'].reset_interface()

    # prediction probabilities and population data
    adir = cdf['data-select'].value
    yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
    dp = pd.read_pickle(os.path.join(datadir, adir, 'population-data.p'))
    dp = dp.loc[~dp['oversampled']]
    assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))

    # calculate pdfs
    pos = yp.loc[dp['outcome'].astype('bool'), 'prediction probability'].values
    neg = yp.loc[~(dp['outcome'].astype('bool')), 'prediction probability'].values
    bins = np.linspace(0, 1, 60)
    width = np.diff(bins)[0]
    centers = (bins[1:] + bins[:-1]) / 2
    posx = np.digitize(pos, bins)
    negx = np.digitize(neg, bins)
    posx = np.array([(posx == xi).sum() for xi in range(1, bins.size + 1)])
    negx = np.array([(negx == xi).sum() for xi in range(1, bins.size + 1)])
    assert (posx[-1] == 0) and (negx[-1] == 0)
    posx, negx = posx[:-1], negx[:-1]
    posx = posx / posx.sum()
    negx = negx / negx.sum()

    # update pdf data sources and legend
    cdf['pdfs'].data_sources[0].data = {'x': centers, 'y': negx}
    label = f"""{(~dp['outcome'].astype('bool')).sum()} non-collision probabilities"""
    cdf['pdfs'].legend.items.append(LegendItem(label=label, renderers=list(cdf['pdfs'].renderers[0, :])))
    cdf['pdfs'].data_sources[1].data = {'x': centers, 'y': posx}
    label = f"""{dp['outcome'].sum()} collision probabilities"""
    cdf['pdfs'].legend.items.append(LegendItem(label=label, renderers=list(cdf['pdfs'].renderers[1, :])))

    # cdfs and update cdf data sources and legend
    posx = np.cumsum(posx)
    negx = np.cumsum(negx)
    ks = ks_2samp(pos, neg)
    cdf['cdfs'].data_sources[0].data = {'x': centers, 'y': negx}
    label = f"""{(~dp['outcome'].astype('bool')).sum()} non-collision probabilities"""
    cdf['cdfs'].legend.items.append(LegendItem(label=label, renderers=list(cdf['cdfs'].renderers[0, :])))
    cdf['cdfs'].data_sources[1].data = {'x': centers, 'y': posx}
    label = f"""{dp['outcome'].sum()} collision probabilities"""
    cdf['cdfs'].legend.items.append(LegendItem(label=label, renderers=list(cdf['cdfs'].renderers[1, :])))
    x = np.argmax(np.abs(posx - negx))
    cdf['cdfs'].data_sources[2].data = {'x': np.tile(centers[x], 2), 'y': np.array([negx[x], posx[x]])}
    cdf['cdfs'].legend.items.append(LegendItem(label='max distance', renderers=list(cdf['cdfs'].renderers[2, :])))
    cdf['ks-title'].text = f'KS Statistic {ks.statistic:.2f}, KS p-value {100 * ks.pvalue:.2f}%'

# Prediction Probability vs Time
def vt_callback(attr, old, new, sender):
    """
    update prediction probability vs time interface
    """
    global pvt_data
    assert sender in [None, vt['data-select'], vt['include-checkbox'], vt['include-range-min'], vt['include-range-max'], vt['vehicle-select']]
    c0, c1 = console_str_objects(300)

    # initialize pvt_data dict and figure limits
    if sender in [None, vt['data-select']]:

        # model artifacts dir and initialize pvt_data dict
        adir = vt['data-select'].value
        pvt_data = {}

        # population data and prediction probabilities
        dp = pd.read_pickle(os.path.join(datadir, adir, 'population-data.p'))
        pvt_data['dp'] = dp
        yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
        pvt_data['yp'] = yp
        assert (dp.shape[0] == yp.shape[0]) and (all(dp['outcome'] == yp['actual outcome']))
        yp['VehicleId'] = dp['VehicleId']
        dpx = pd.merge(
            left=yp.groupby('VehicleId')['prediction probability'].min().to_frame().rename(columns={'prediction probability': 'pmin'}),
            right=yp.groupby('VehicleId')['prediction probability'].max().to_frame().rename(columns={'prediction probability': 'pmax'}),
            how='inner', left_index=True, right_index=True)
        dpx = pd.merge(left=dpx,
            right=yp.groupby('VehicleId')['actual outcome'].any().to_frame().rename(columns={'actual outcome': 'any collision'}),
            how='inner', left_index=True, right_index=True)
        pvt_data['dpx'] = dpx

        # feature importance, ml-data, collisions
        with open(os.path.join(datadir, adir, 'feature-importance.p'), 'rb') as fid:
            dfm = pickle.load(fid)
        pvt_data['dfm'] = dfm
        with open(os.path.join(datadir, adir, 'ml-data.p'), 'rb') as fid:
            df = pickle.load(fid)
        pvt_data['df'] = df
        pvt_data['dc'] = pd.read_pickle(os.path.join(datadir, adir, 'collisions.p'))

        # shap values data if exists
        fs = os.path.join(datadir, adir, 'shap.p')
        if not os.path.isfile(fs):
            pvt_data['shap_status'] = False
            vt['vehicle-eval-desc'].text = c0 + 'No shap values data' + c1
        else:
            pvt_data['shap_status'] = True
            vt['vehicle-eval-desc'].text = c0 + c1
            with open(fs, 'rb') as fid:
                ds = pickle.load(fid)
                pvt_data['ds'] = ds
            assert ds['base'].size == ds['values'].shape[0]
            assert df.shape == ds['values'].shape

        # figure limits
        vt['pvt'].fig.x_range.start = dp['time1'].min() - pd.Timedelta(days=1)
        vt['pvt'].fig.x_range.end = dp['time1'].max() + pd.Timedelta(days=1)
        vt['pvt'].fig.y_range.start = 0
        vt['pvt'].fig.y_range.end = min(1, 1.1 * pvt_data['yp']['prediction probability'].max())

    # reset vehicle-select and pvt interface
    if sender in [None, vt['data-select'], vt['include-checkbox'], vt['include-range-min'], vt['include-range-max']]:

        # clear interfaces
        vt['vehicle-select'].remove_on_change('value', vt_vs_callback)
        vt['vehicle-select'].options = []
        vt['vehicle-select'].value = []
        vt['pvt'].reset_interface()
        vt['waterfall'].reset_interface()
        for ds in vt['pvt'].data_sources:
            ds.selected.remove_on_change('indices', vt_ds_selected_callback[ds])
            ds.selected.indices = []
            ds.selected.on_change('indices', vt_ds_selected_callback[ds])
        for ds in vt['pvt'].hds:
            ds.data = {'x': np.array([]), 'y': np.array([])}

        # vid filters
        active = vt['include-checkbox'].active
        pmin = float(vt['include-range-min'].value)
        pmax = float(vt['include-range-max'].value)

        # null case
        if active == []:
            vt['vehicle-select'].on_change('value', vt_vs_callback)
            return

        # vids based on filters
        dpx = pvt_data['dpx']
        c0 = dpx['pmin'] < pmin
        c1 = dpx['pmax'] > pmax
        ok = \
            (c0 & c1 & dpx['any collision']) if active == [1] else \
            (c0 & c1 & ~dpx['any collision']) if active == [0] else (c0 & c1)
        vids = dpx.loc[ok].index.values
        assert np.unique(vids).size == vids.size

        # update vehicle-select multi-select
        vt['vehicle-select'].options = [(str(x), vid) for x, vid in enumerate(vids)]
        vt['vehicle-select'].title = f"""Select one or more of {vids.size} vehicle-id"""
        vt['vehicle-select'].on_change('value', vt_vs_callback)
        return

    # validate vehicle-select as sender and clear interface
    assert sender == vt['vehicle-select']
    vt['pvt'].reset_interface()
    vt['waterfall'].reset_interface()
    for ds in vt['pvt'].data_sources:
        ds.selected.remove_on_change('indices', vt_ds_selected_callback[ds])
        ds.selected.indices = []
        ds.selected.on_change('indices', vt_ds_selected_callback[ds])
    for ds in vt['pvt'].hds:
        ds.data = {'x': np.array([]), 'y': np.array([])}

    # handle null case and too-many-selected case
    if new == []:
        return
    if len(new) > 4:
        new = [new[-1]]
        vt['vehicle-select'].remove_on_change('value', vt_vs_callback)
        vt['vehicle-select'].value = new
        vt['vehicle-select'].on_change('value', vt_vs_callback)

    # update interface for each vid
    vids = np.array(vt['vehicle-select'].options)
    vids = vids[:, 1][np.in1d(vids[:, 0], np.array(new))]
    assert vids.size == len(new)
    assert len(vt['pvt'].legend.items) == 0
    for xv, vid in enumerate(vids):

        # population data and prediction probabilities for vid
        dp = pvt_data['dp'].loc[pvt_data['dp']['VehicleId'] == vid]
        yp = pvt_data['yp'].loc[dp.index]

        # data source for vid
        x = dp['time1'].values
        y = yp['prediction probability'].values
        ok = np.argsort(x)
        vt['pvt'].data_sources[xv].data = {'x': x[ok], 'y': y[ok]}

        # hack data source (hds) for vid
        dct = pvt_data['dc'].loc[(pvt_data['dc']['VehicleId'] == vid) & (pvt_data['dc']['BehaviorId'] == 47), 'RecordDate'].values
        x = np.array([], dtype='datetime64[ns]')
        y = np.array([])
        for dc in dct:
            x = np.hstack((x, dc))
            before = dp.loc[dp['time1'] < dc].index[-1]
            x0 = dp.loc[before, 'time1']
            y0 = yp.loc[before, 'prediction probability']
            after = dp.loc[dp['time1'] > dc].index
            if after.size > 0:
                after = after[0]
                x1 = dp.loc[after, 'time1']
                y1 = yp.loc[after, 'prediction probability']
                m = (y1 - y0) / ((x1 - x0).total_seconds())
            else:
                m = 0
            y = np.hstack((y, m * (dc - x0).total_seconds() + y0))
        vt['pvt'].hds[xv].data = {'x': x, 'y': y}

        # legend for vid
        vt['pvt'].legend.items.append(LegendItem(label=vid, renderers=list(vt['pvt'].renderers[xv, :])))

    # tap tool renderers
    vt['pvt'].tap.renderers = [x for x in vt['pvt'].fig.renderers if x.name == 'circles'][:xv + 1]

def vt_selected_callback(attr, old, new, ds):

    # validate and assert one selected index
    c0, c1 = console_str_objects(300)
    assert ds.selected.indices == new
    if not pvt_data['shap_status']:
        ds.selected.indices = []
        vt['vehicle-eval-desc'].text = c0 + c1
        return
    if new == []:
        vt['vehicle-eval-desc'].text = c0 + c1
        return
    if len(new) > 1:
        ds.selected.indices = []
        vt['vehicle-eval-desc'].text = c0 + c1
        return

    # selected data point and vehicle-id
    x = ds.data['x'][new[0]]
    y = ds.data['y'][new[0]]
    vids = np.array(vt['vehicle-select'].options)
    vids = vids[:, 1][np.in1d(vids[:, 0], np.array(np.array(vt['vehicle-select'].value)))]
    vid = vids[vt['pvt'].data_sources.index(ds)]

    # data for vid
    dp = pvt_data['dp']
    dpv = dp.loc[(dp['VehicleId'] == vid) & (dp['time1'] == pd.Timestamp(x))]
    assert dpv.shape[0] == 1
    dpv = dpv.squeeze()
    ypv = pvt_data['yp'].loc[dpv.name]
    assert (ypv['VehicleId'] == dpv['VehicleId']) and (ypv['actual outcome'] == dpv['outcome'])

    # vehicle eval description div
    time0 = dpv['time0'].strftime('%m/%d/%Y')
    time1 = dpv['time1'].strftime('%m/%d/%Y')
    time2 = dpv['time2'].strftime('%m/%d/%Y')
    desc = f"""{vid}
        company, {dpv['CompanyName']}
        industry, {dpv['IndustryDesc']}
        predictor interval, {time0} to {time1}
        collision interval, {time1} to {time2}
        prediction probability, {ypv['prediction probability']:.2f}
        actual collision outcome, {bool(ypv['actual outcome'])}""".replace('\n', '<br>')
    vt['vehicle-eval-desc'].text = c0 + desc + c1

    # shap waterfall chart data
    sdata = utils.get_shap_chart_data(
        base=pvt_data['ds']['base'][dpv.name],
        values=pvt_data['ds']['values'][dpv.name, :],
        xr=pvt_data['df'][dpv.name, :],
        n_features=6,
        cols=pvt_data['dfm']['features'])

    # shap waterfall chart column data sources
    yp, positive, pcols, yn, negative, ncols = sdata
    base = positive[0, 0] if positive.size > 0 else negative[0, 1]
    vt['waterfall'].base.data = {'x': np.tile(base, 2), 'y': np.array([-0.5, pcols.size + ncols.size - 0.5])}
    vt['waterfall'].positive.data = {'x0': positive[:, 0], 'x1': positive[:, 1], 'y0': yp[:, 0], 'y1': yp[:, 1],
        'labels': [f'+{x:.3f}' for x in np.diff(positive).flatten()]}
    vt['waterfall'].negative.data = {'x0': negative[:, 1], 'x1': negative[:, 0], 'y0': yn[:, 0], 'y1': yn[:, 1],
        'labels': [f'-{x:.3f}' for x in np.diff(negative).flatten()]}

    # clean up
    vs = np.hstack((negative.flatten(), positive.flatten()))
    vt['waterfall'].fig.x_range.start = 0.9 * vs.min()
    vt['waterfall'].fig.x_range.end = 1.1 * vs.max()
    vt['waterfall'].fig.y_range.start = -0.5
    vt['waterfall'].fig.y_range.end = pcols.size + ncols.size - 0.5
    vt['waterfall'].fig.yaxis.ticker.ticks = list(range(pcols.size + ncols.size))
    str_axis_labels(axis=vt['waterfall'].fig.yaxis, labels=np.hstack((pcols, ncols))[::-1])

# ROC/PR Curves
def mx_callback(attr, old, new):
    """
    ROC and PR curves for one or more models
    """

    # reset fig interfaces
    mx['roc-curve'].reset_interface()
    mx['pr-curve'].reset_interface()
    if not new:
        return

    # scan over selected model artifacts, update data source/legend for each
    for key in new:

        # get adir and key as int
        key = [x for x in mx['multi-select'].options if x[0] == key]
        assert len(key) == 1
        key, adir = key[0]
        key = int(key)

        # metrics for ROC and PR curves
        yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
        auc = roc_auc_score(y_true=yp['actual outcome'].values, y_score=yp['prediction probability'].values)
        ap = average_precision_score(y_true=yp['actual outcome'].values, y_score=yp['prediction probability'].values)

        # load roc-pr-curve data
        dml = pd.read_pickle(os.path.join(datadir, adir, 'roc-pr-curve-data.p'))

        # update ROC Curve
        dml['x'] = dml['fpr'].values
        dml['y'] = dml['tpr'].values
        mx['roc-curve'].data_sources[key].data = dml
        label = f'{adir}, AUC={auc:.3f}'.replace('artifacts-', '')
        mx['roc-curve'].legend.items.append(LegendItem(label=label, renderers=list(mx['roc-curve'].renderers[key, :])))

        # update PR Curve
        dml['x'] = dml['tpr'].values
        dml['y'] = dml['precision'].values
        mx['pr-curve'].data_sources[key].data = dml
        label = f'{adir}, AP={ap:.3f}'.replace('artifacts-', '')
        mx['pr-curve'].legend.items.append(LegendItem(label=label, renderers=list(mx['pr-curve'].renderers[key, :])))

    # update hover tooltips once
    mx['roc-curve'].hover.tooltips = [('threshold', '@thresh'), ('false positive rate', '@fpr'), ('true positive rate', '@tpr')]
    mx['pr-curve'].hover.tooltips = [('threshold', '@thresh'), ('recall (true positive rate)', '@tpr'), ('precision', '@precision')]

# Segmentation Curves
def seg_callback(attr, old, new):
    """
    vehicle eval segmentation curves for one or more models
    """

    # reset fig interfaces
    seg['seg'].reset_interface()
    if not new:
        return

    # scan over selected model artifacts, update data source/legend for each
    for key in new:

        # get adir and key as int
        key = [x for x in seg['multi-select'].options if x[0] == key]
        assert len(key) == 1
        key, adir = key[0]
        key = int(key)

        # segmentation data for adir
        yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
        x = 100 * np.arange(1, yp.shape[0] + 1) / yp.shape[0]
        y = 100 * yp.sort_values('prediction probability', ascending=False)['actual outcome'].values.cumsum() / yp['actual outcome'].sum()

        # update data source and legend
        seg['seg'].data_sources[key].data = {'x': x, 'y': y}
        seg['seg'].legend.items.append(LegendItem(label=adir.replace('artifacts-', ''), renderers=list(seg['seg'].renderers[key, :])))

    # hover tooltips
    seg['seg'].hover.tooltips = [('x', '@x'), ('y', '@y')]

# Segmentation Curves based on prediction probability and miles
def segm_callback(attr, old, new):
    """
    vehicle eval segmentation curve for single model based on prediction probability and miles
    """

    # reset fig interfaces
    segm['seg'].reset_interface()
    if not new:
        return

    # segmentation curve based on model prediction probability
    adir = segm['data-select'].value
    yp = pd.read_pickle(os.path.join(datadir, adir, 'model-prediction-probabilities.p'))
    x = 100 * np.arange(1, yp.shape[0] + 1) / yp.shape[0]
    y = 100 * yp.sort_values('prediction probability', ascending=False)['actual outcome'].values.cumsum() / yp['actual outcome'].sum()
    segm['seg'].data_sources[0].data = {'x': x, 'y': y}
    segm['seg'].legend.items.append(LegendItem(label='prediction probability', renderers=list(segm['seg'].renderers[0, :])))

    # segmentation curve based on miles
    df = pd.read_pickle(os.path.join(datadir, adir, 'ml-data.p'))
    with open(os.path.join(datadir, adir, 'feature-importance.p'), 'rb') as fid:
        dfm = pickle.load(fid)
    features = dfm['features']
    assert df.shape == (yp.shape[0], features.size)
    f0 = 'gps_miles'
    f1 = 'gpse_travel_distance_meters_sum'
    feature = f0 if f0 in features else f1 if f1 in features else None
    assert feature is not None
    y = 100 * yp.loc[np.argsort(df[:, features == feature].flatten())[::-1], 'actual outcome'].values.cumsum() / yp['actual outcome'].sum()
    segm['seg'].data_sources[1].data = {'x': x, 'y': y}
    segm['seg'].legend.items.append(LegendItem(label=feature, renderers=list(segm['seg'].renderers[1, :])))

    # hover tooltips
    segm['seg'].hover.tooltips = [('x', '@x'), ('y', '@y')]

# Decoder table
def get_decoder_text(df):

    # ids and names
    ids = df['Id'].values.astype('str')
    names = df['Name'].values
    max_id = max([len(x) for x in ids])

    # build text table
    desc = ''
    for xid, name in zip(ids, names):
        spaces = 4 + max_id - len(xid)
        spaces *= '&nbsp'
        desc += xid + spaces + name + '<br>'
    c0, c1 = console_str_objects(300)

    return c0 + desc + c1

# datadir and list of available model artifact directories
datadir = r'/mnt/home/russell.burdt/data/collision-model/app'
assert os.path.isdir(datadir)
adirs = sorted([os.path.split(x)[1] for x in glob(os.path.join(datadir, 'artifacts*'))])
assert adirs

# single model learning curve
mlc = {}
mlc['data-select'] = Select(
    title='select model artifacts', width=280, options=adirs, value=adirs[0])
mlc['data-select'].on_change('value', partial(mlc_callback, sender=mlc['data-select']))
mlc['metric-select'] = Select(
    title='select learning curve metric', width=280,
    options=['accuracy', 'true positive rate', 'false positive rate', 'precision', 'balanced accuracy'], value='balanced accuracy')
mlc['metric-select'].on_change('value', partial(mlc_callback, sender=mlc['metric-select']))
mlc['slider'] = Slider(start=0, end=1, value=0, step=0.01, title='learning curve positive threshold', width=280)
mlc_slider_callback = partial(mlc_callback, sender=mlc['slider'])
mlc['slider'].on_change('value', mlc_slider_callback)
mlc['data-desc-title'] = Div(text='<strong>Data Description</strong>', width=280)
mlc['data-desc'] = Div()
mlc['model-desc-title'] = Div(text='<strong>Model Description</strong>', width=280)
mlc['model-desc'] = Div()
mlc['learning-curve'] = LearningCurveInterface(
    width=540, height=320, hover=False, xlabel='fraction of all data in training set', size=13)

# single model evaluation
mev = {}
mev['data-select'] = Select(
    title='select model artifacts', width=300, options=adirs, value=adirs[0])
mev['data-select'].on_change('value', partial(mev_callback, sender=mev['data-select']))
mev['data-desc-title'] = Div(text='<strong>Data Description</strong>', width=300)
mev['data-desc'] = Div()
mev['model-desc-title'] = Div(text='<strong>Model Description</strong>', width=300)
mev['model-desc'] = Div()
mev['slider'] = Slider(start=0, end=1, value=0, step=0.01, title='positive threshold', width=300)
mev_slider_callback = partial(mev_callback, sender=mev['slider'])
mev['slider'].on_change('value', mev_slider_callback)
mev['eval-desc-title'] = Div(text='<strong>Evaluation Description</strong>', width=300)
mev['eval-desc'] = Div()
mev['classification metrics'] = HorizontalBarChartInterface(
    width=640, height=260, title='classification metrics', size=13, include_nums=True, pan_dimensions='height')
mev['feature importance'] = HorizontalBarChartInterface(
    width=640, height=260, title='feature importance', size=13)
mev['feature-importance-select'] = Select(title='feature importance method', width=300)
mev_feature_importance_callback = partial(mev_callback, sender=mev['feature-importance-select'])
mev['feature-importance-select'].on_change('value', mev_feature_importance_callback)
mev['confusion-matrix-title'] = Div(text='<strong>Confusion Matrix</strong>', width=300)
mev['confusion matrix'] = Div()

# single model evaluation by company
mdx = {}
mdx['data-select'] = Select(
    title='select model artifacts', width=300, options=adirs, value=adirs[0])
mdx['data-select'].on_change('value', partial(mdx_callback, sender=mdx['data-select']))
mdx['company-select'] = MultiSelect(
    title='select company name', width=300, height=240)
mdx_company_select_callback = partial(mdx_callback, sender=mdx['company-select'])
mdx['company-select'].on_change('value', mdx_company_select_callback)
mdx['companies-desc-title'] = Div(text='<strong>Description of Selected Companies</strong>', width=300)
mdx['companies-desc'] = Div()
mdx['roc-curve'] = MultiLineInterface(
    title='ROC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate', width=440, height=260, hover=True, size=12,
    n=1, legend_location='bottom_right')
mdx['pr-curve'] = MultiLineInterface(
    title='PR Curve', xlabel='Recall', ylabel='Precision', width=440, height=260, hover=True, size=12,
    n=1, legend_location='top_right')
link_axes([mdx['roc-curve'].fig, mdx['pr-curve'].fig], axis='x')
mdx['negative prediction probabilities'] = MetricDistributionInterface(
    width=650, height=260, xlabel='prediction probability', ylabel='bin count', cross=True,
    size=12, logscale=False, bar_color='darkblue', fixed_ticker=True)
mdx['positive prediction probabilities'] = MetricDistributionInterface(
    width=650, height=260, xlabel='prediction probability', ylabel='bin count', cross=True,
    size=12, logscale=False, bar_color='darkred', fixed_ticker=True)
mdx['positive prediction probabilities'].fig.tools[-1] = mdx['negative prediction probabilities'].cross
mdx['pdfs'] = MultiLineInterface(
    xlabel='collision-prediction model probability', ylabel='PDF', width=690, height=270, size=12, n=3,
    circle=False, line=True, line_width=4, legend_location='top_right', palette='Category20_5',
    title='Probability density functions of collision-prediction model probabilities')
mdx['pdfs'].fig.title.text_font_size = '12pt'
mdx['cdfs'] = MultiLineInterface(
    xlabel='collision-prediction model probability', ylabel='CDF', width=690, height=270, size=12, n=3,
    circle=False, line=True, line_width=4, legend_location='bottom_right', palette='Category20_5')
mdx['ks-title'] = Title(text_font_style='italic', text_font_size='12pt')
mdx['cdfs'].fig.add_layout(mdx['ks-title'], 'above')
mdx['cdfs'].fig.add_layout(Title(text='Cumulative distribution functions of collision-prediction model probabilities', text_font_size='12pt'), 'above')

# single model evaluation by industry
mde = {}
mde['data-select'] = Select(
    title='select model artifacts', width=300, options=adirs, value=adirs[0])
mde['data-select'].on_change('value', partial(mde_callback, sender=mde['data-select']))
mde['industry-select'] = MultiSelect(
    title='select industry', width=300, height=240)
mde_industry_select_callback = partial(mde_callback, sender=mde['industry-select'])
mde['industry-select'].on_change('value', mde_industry_select_callback)
mde['industries-desc-title'] = Div(text='<strong>Description of Selected Industries</strong>', width=300)
mde['industries-desc'] = Div()
mde['roc-curve'] = MultiLineInterface(
    title='ROC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate', width=440, height=260, hover=True, size=12,
    n=1, legend_location='bottom_right')
mde['pr-curve'] = MultiLineInterface(
    title='PR Curve', xlabel='Recall', ylabel='Precision', width=440, height=260, hover=True, size=12,
    n=1, legend_location='top_right')
link_axes([mde['roc-curve'].fig, mde['pr-curve'].fig], axis='x')
mde['negative prediction probabilities'] = MetricDistributionInterface(
    width=650, height=260, xlabel='prediction probability', ylabel='bin count', cross=True,
    size=12, logscale=False, bar_color='darkblue', fixed_ticker=True)
mde['positive prediction probabilities'] = MetricDistributionInterface(
    width=650, height=260, xlabel='prediction probability', ylabel='bin count', cross=True,
    size=12, logscale=False, bar_color='darkred', fixed_ticker=True)
mde['positive prediction probabilities'].fig.tools[-1] = mde['negative prediction probabilities'].cross
mde['pdfs'] = MultiLineInterface(
    xlabel='collision-prediction model probability', ylabel='PDF', width=690, height=270, size=12, n=3,
    circle=False, line=True, line_width=4, legend_location='top_right', palette='Category20_5',
    title='Probability density functions of collision-prediction model probabilities')
mde['pdfs'].fig.title.text_font_size = '12pt'
mde['cdfs'] = MultiLineInterface(
    xlabel='collision-prediction model probability', ylabel='CDF', width=690, height=270, size=12, n=3,
    circle=False, line=True, line_width=4, legend_location='bottom_right', palette='Category20_5')
mde['ks-title'] = Title(text_font_style='italic', text_font_size='12pt')
mde['cdfs'].fig.add_layout(mde['ks-title'], 'above')
mde['cdfs'].fig.add_layout(Title(text='Cumulative distribution functions of collision-prediction model probabilities', text_font_size='12pt'), 'above')

# single model shap values
svx = {}
svx['data-select'] = Select(
    title='select model artifacts', width=300, options=adirs, value=adirs[0])
svx['data-select'].on_change('value', partial(svx_callback, sender=svx['data-select']))
svx['feature-select'] = Select(title='select model feature (sorted by model feature importance)', width=300)
svx_feature_select_callback = partial(svx_callback, sender=svx['feature-select'])
svx['feature-select'].on_change('value', svx_feature_select_callback)
svx['feature distribution'] = MetricDistributionInterface(
    width=460, height=260, ylabel='bin count', size=12, logscale=False, bar_color='darkblue')
svx['shap curve'] = MultiLineInterface(
    n=1, width=460, height=260, ylabel='shap value', size=12,
    circle=True, line=False, manual_xlim=True, manual_ylim=True, tap=True)
link_axes([svx['shap curve'].fig, svx['feature distribution'].fig], axis='x')
svx['shap curve'].tap.renderers = [x for x in svx['shap curve'].fig.renderers if x.name == 'circles']
svx['shap curve'].data_sources[0].selected.on_change('indices', svx_selected_callback)
svx['vehicle-eval-desc-title'] = Div(text='<strong>Vehicle Eval Description</strong>', width=270)
svx['vehicle-eval-desc'] = Div()
svx['include-desc'] = Div(text='<strong>Include in Shap Values Chart</strong>', width=270)
svx['include-checkbox'] = CheckboxGroup(labels=['negative actual outcomes', 'positive actual outcomes'], active=[1], width=270)
svx['include-checkbox'].on_change('active', partial(svx_callback, sender=svx['include-checkbox']))
svx['include-range-min'] = TextInput(title='Min prediction probability', value='0.00', width=270)
svx['include-range-max'] = TextInput(title='Max prediction probability', value='1.00', width=270)
svx['include-range-min'].on_change('value', partial(svx_callback, sender=svx['include-range-min']))
svx['include-range-max'].on_change('value', partial(svx_callback, sender=svx['include-range-max']))
svx['waterfall'] = ShapValuesWaterfallInterface(
    width=700, height=520, size=14, xlabel='prediction probability', title='shap values for vehicle eval')

# single model pdf / cdf
cdf = {}
cdf['data-select'] = Select(
    title='select model artifacts', width=300, options=adirs, value=adirs[0])
cdf['data-select'].on_change('value', partial(cdf_callback, sender=cdf['data-select']))
cdf['pdfs'] = MultiLineInterface(
    xlabel='collision-prediction model probability', ylabel='PDF', width=690, height=270, size=12, n=3,
    circle=False, line=True, line_width=4, legend_location='top_right', palette='Category20_5',
    title='Probability density functions of collision-prediction model probabilities')
cdf['pdfs'].fig.title.text_font_size = '12pt'
cdf['cdfs'] = MultiLineInterface(
    xlabel='collision-prediction model probability', ylabel='CDF', width=690, height=270, size=12, n=3,
    circle=False, line=True, line_width=4, legend_location='bottom_right', palette='Category20_5')
cdf['ks-title'] = Title(text_font_style='italic', text_font_size='12pt')
cdf['cdfs'].fig.add_layout(cdf['ks-title'], 'above')
cdf['cdfs'].fig.add_layout(Title(text='Cumulative distribution functions of collision-prediction model probabilities', text_font_size='12pt'), 'above')

# vehicle prediction probability vs time
vt = {}
vt['data-select'] = Select(
    title='select model artifacts', width=300, options=adirs, value=adirs[0])
vt['data-select'].on_change('value', partial(vt_callback, sender=vt['data-select']))
vt['vehicle-select'] = MultiSelect(width=300, height=120)
vt_vs_callback = partial(vt_callback, sender=vt['vehicle-select'])
vt['vehicle-select'].on_change('value', vt_vs_callback)
vt['include-desc'] = Div(text='<strong>Include in Vehicle-Id List</strong>', width=300)
vt['include-checkbox'] = CheckboxGroup(labels=['negative actual outcomes', 'positive actual outcomes'], active=[1], width=300)
vt['include-checkbox'].on_change('active', partial(vt_callback, sender=vt['include-checkbox']))
vt['include-range-min'] = TextInput(title='Min prediction probability threshold', value='1.00', width=300)
vt['include-range-max'] = TextInput(title='Max prediction probability threshold', value='0.00', width=300)
vt['include-range-min'].on_change('value', partial(vt_callback, sender=vt['include-range-min']))
vt['include-range-max'].on_change('value', partial(vt_callback, sender=vt['include-range-max']))
vt['pvt'] = MultiLineInterface(
    width=1000, height=520, size=12, n=5, circle=True, line=True, datetime=True, manual_xlim=True, manual_ylim=True,
    xlabel='end of predictor interval', ylabel='prediction probability', title='prediction probability vs time',
    legend_layout_location='right', legend_location='center', toolbar_location='above', tap=True)
colors = [x.glyph.line_color for x in vt['pvt'].renderers[:, 0]]
vt['pvt'].hds = [ColumnDataSource(data={'x': np.array([]), 'y': np.array([])}) for _ in range(vt['pvt'].n)]
for src, color in zip(vt['pvt'].hds, colors):
    vt['pvt'].fig.x('x', 'y', source=src, size=16, line_width=3, color=color, alpha=1)
vt_ds_selected_callback = {}
for ds in vt['pvt'].data_sources:
    vt_ds_selected_callback[ds] = partial(vt_selected_callback, ds=ds)
    ds.selected.on_change('indices', vt_ds_selected_callback[ds])
vt['vehicle-eval-desc-title'] = Div(text='<strong>Vehicle Eval Description</strong>', width=300)
vt['vehicle-eval-desc'] = Div()
vt['waterfall'] = ShapValuesWaterfallInterface(
    width=700, height=520, size=14, xlabel='prediction probability', title='shap values for vehicle eval')

# all models ROC/PR curves
mx = {}
mx['multi-select'] = MultiSelect(
    title='select model artifacts', width=240, height=240, options=[(str(x), adir) for x, adir in enumerate(adirs)])
mx['multi-select'].on_change('value', mx_callback)
mx['roc-curve'] = MultiLineInterface(
    title='ROC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate', width=540, height=280, hover=True, n=20,
    size=12, legend_location='bottom_right', circle=False, line_width=4, dimensions='height')
mx['pr-curve'] = MultiLineInterface(
    title='PR Curve', xlabel='Recall', ylabel='Precision', width=540, height=280, hover=True, n=20,
        size=12, legend_location='top_right', circle=False, line_width=4, dimensions='height')
link_axes([mx['roc-curve'].fig, mx['pr-curve'].fig], axis='x')

# all models segmentation curves
seg = {}
seg['multi-select'] = MultiSelect(
    title='select model artifacts', width=240, height=240, options=[(str(x), adir) for x, adir in enumerate(adirs)])
seg['multi-select'].on_change('value', seg_callback)
seg['seg'] = MultiLineInterface(
    width=600, height=400, size=14, n=20, circle=False, line=True, line_width=4,
    xlabel='percentage of vehicle evals',
    ylabel='cumulative percentage of collisions',
    legend_location='top_left', hover=True)
seg['seg'].fig.add_layout(Title(text='vehicle evals sorted by descending prediction probability',
    text_font_style='italic', text_font_size='14pt'), 'above')
seg['seg'].fig.add_layout(Title(text='cumulative percentage of collisions', text_font_size='16pt'), 'above')

# single model segmentation curves for prediction probability vs miles
segm = {}
segm['data-select'] = Select(
    title='select model artifacts', width=300, options=[''] + adirs, value='')
segm['data-select'].on_change('value', segm_callback)
segm['seg'] = MultiLineInterface(
    width=600, height=400, size=14, n=2, circle=False, line=True, line_width=4,
    xlabel='percentage of vehicle evals',
    ylabel='cumulative percentage of collisions',
    legend_location='top_left', hover=True)
segm['seg'].fig.add_layout(Title(text='vehicle evals sorted by descending metric value',
    text_font_style='italic', text_font_size='14pt'), 'above')
segm['seg'].fig.add_layout(Title(text='cumulative percentage of collisions', text_font_size='16pt'), 'above')

# decoder interface
assert os.path.isfile(os.path.join(datadir, 'decoder.p'))
dd = pd.read_pickle(os.path.join(datadir, 'decoder.p'))
decoder = {}
decoder['event-type-title'] = Div(text='<strong>Event Types</strong>', width=300)
decoder['event-type'] = Div(text=get_decoder_text(dd['event-type']))
decoder['event-sub-title'] = Div(text='<strong>Event Sub Types</strong>', width=300)
decoder['event-sub'] = Div(text=get_decoder_text(dd['event-sub-type']))
decoder['behavior-title'] = Div(text='<strong>Behavior Types</strong>', width=300)
decoder['behavior'] = Div(text=get_decoder_text(dd['behaviors']))

# app layout based on tabs/panels
layout_mlc = row(
    column(mlc['data-select'], mlc['metric-select'], mlc['slider']),
    column(mlc['learning-curve'].fig),
    column(mlc['data-desc-title'], mlc['data-desc'], mlc['model-desc-title'], mlc['model-desc']))
layout_mev = row(
    column(mev['data-select'], mev['feature-importance-select'], mev['slider'], mev['confusion-matrix-title'], mev['confusion matrix'], mev['eval-desc-title'], mev['eval-desc']),
    column(mev['classification metrics'].fig, mev['feature importance'].fig),
    column(mev['data-desc-title'], mev['data-desc'], mev['model-desc-title'], mev['model-desc']))
layout_mdx = row(
    column(mdx['data-select'], mdx['company-select'], mdx['companies-desc-title'], mdx['companies-desc']),
    column(mdx['roc-curve'].fig, mdx['pr-curve'].fig),
    column(mdx['negative prediction probabilities'].fig, mdx['positive prediction probabilities'].fig),
    column(mdx['pdfs'].fig, mdx['cdfs'].fig))
layout_mde = row(
    column(mde['data-select'], mde['industry-select'], mde['industries-desc-title'], mde['industries-desc']),
    column(mde['roc-curve'].fig, mde['pr-curve'].fig),
    column(mde['negative prediction probabilities'].fig, mde['positive prediction probabilities'].fig),
    column(mde['pdfs'].fig, mde['cdfs'].fig))
layout_svx = row(
    column(
        svx['data-select'],
        svx['feature-select'],
        svx['include-desc'],
        svx['include-checkbox'],
        svx['include-range-min'],
        svx['include-range-max'],
        svx['vehicle-eval-desc-title'],
        svx['vehicle-eval-desc']),
    column(svx['shap curve'].fig, svx['feature distribution'].fig),
    svx['waterfall'].fig)
layout_cdf = row(
    cdf['data-select'],
    column(cdf['pdfs'].fig, cdf['cdfs'].fig))
layout_vt = row(
    column(vt['data-select'], vt['vehicle-select'],
        vt['include-desc'], vt['include-checkbox'], vt['include-range-min'], vt['include-range-max'],
        vt['vehicle-eval-desc-title'], vt['vehicle-eval-desc']),
    vt['pvt'].fig, vt['waterfall'].fig)
layout_mx = row(
    mx['multi-select'],
    column(mx['roc-curve'].fig, mx['pr-curve'].fig))
layout_seg = row(
    seg['multi-select'],
    seg['seg'].fig)
layout_segm = row(
    segm['data-select'],
    segm['seg'].fig)
layout_decoder = row(
    column(decoder['event-type-title'], decoder['event-type']),
    column(decoder['event-sub-title'], decoder['event-sub']),
    column(decoder['behavior-title'], decoder['behavior']))
layout = Tabs(tabs=[
    Panel(child=layout_mev, title='Classification Metrics'),
    Panel(child=layout_mlc, title='Learning Curve'),
    Panel(child=layout_cdf, title='PDF/CDF'),
    Panel(child=layout_mdx, title='Eval by Company'),
    Panel(child=layout_mde, title='Eval by Industry'),
    Panel(child=layout_svx, title='Shap Values'),
    Panel(child=layout_vt, title='Prediction Probability vs Time'),
    Panel(child=layout_mx, title='ROC/PR'),
    Panel(child=layout_seg, title='Segmentation'),
    Panel(child=layout_segm, title='Segmentation vs miles'),
    Panel(child=layout_decoder, title='Decoder')])

# deploy and initialize state
curdoc().add_root(layout)
curdoc().title = 'collision model eval app'
mlc_callback(None, None, None, sender=None)
mev_callback(None, None, None, sender=None)
mdx_callback(None, None, None, sender=None)
mde_callback(None, None, None, sender=None)
cdf_callback(None, None, None, sender=None)
vt_callback(None, None, None, sender=None)
svx_callback(None, None, None, sender=None)
