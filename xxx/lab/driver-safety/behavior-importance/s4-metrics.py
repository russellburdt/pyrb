
"""
behavior analysis metrics, without ML
- percentage of vehicle evals with any individual behavior, method A
- average count of individual behaviors for vehicle evals, method B
behavior analysis metrics, with ML
- logistic regression and random forest models
- model and permutation feature importance, method C and D
"""

import os
import lytx
import pickle
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from ipdb import set_trace
from collections import defaultdict
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.inspection import permutation_importance
from tqdm import tqdm


# datadir and spark session
datadir = r'/mnt/home/russell.burdt/data/driver-safety/behavior-importance/dist300'
assert os.path.isdir(datadir)
spark = lytx.spark_session(memory='32g', cores='*')

# data objects
dp = pd.read_pickle(os.path.join(datadir, 'population-dataframe.p'))
db = pd.read_pickle(os.path.join(datadir, 'behavior-counts-mod.p'))
decoder = pd.read_sql_query(f'SELECT Id, Name FROM hs.Behaviors_i18n', con=lytx.get_conn('edw'))

# remove admin behaviors
# 90(ER Obstruction), 91(Camera Issue), 142(Suspected Collision), 143(Lens Obstruction), 144(Driver Tagged)
for x in [90, 91, 143, 144]:
    del db[f'nbehaviors_{x}']

# join population dataframe and behavior counts dataframe
dx = pd.merge(left=dp, right=db, left_index=True, right_index=True, how='inner')
assert pd.isnull(dx).sum().sum() == 0
bxs = db.columns.to_list()
dx = dx[['VehicleId', 'ta'] + bxs]
dx['month'] = [x.month for x in dx['ta']]
dx['year'] = [x.year for x in dx['ta']]
del dx['ta']
dx['collision'] = dx.pop('nbehaviors_47').astype('bool')
bxs.remove('nbehaviors_47')
dx = dx[['VehicleId', 'month', 'year', 'collision'] + bxs].copy()

# generate table for report
# dx = pd.merge(left=dp, right=db, left_index=True, right_index=True, how='left').fillna(0)
# dx.loc[[334,335,336,344,345,346], ['VehicleId', 'month', 'year', 'collision', 'nbehaviors_11', 'nbehaviors_69', 'nbehaviors_80']]

def cs(xs):
    rv = {}
    for key, value in xs.items():
        rv[decoder.loc[decoder['Id'] == int(key.split('_')[1]), 'Name'].iloc[0]] = value
    return pd.Series(rv)

# percentage of vehicle evals with any individual behavior and discrepancy
x1 = dx.loc[dx['collision'], bxs].astype('bool').sum(axis=0) / dx['collision'].sum()
x1 = cs(x1).sort_values(ascending=False).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})
x1['num evals'] = dx['collision'].sum()
x2 = dx.loc[~dx['collision'], bxs].astype('bool').sum(axis=0) / (~dx['collision']).sum()
x2 = cs(x2).sort_values(ascending=False).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})
x2['num evals'] = (~dx['collision']).sum()
x12 = pd.merge(on='behavior', how='inner', left=x1, right=x2, suffixes=('-collision', '-no-collision'))
x12['abs-diff'] = np.abs(x12['value-collision'] - x12['value-no-collision'])
x12 = x12.sort_values('abs-diff', ascending=False).reset_index(drop=True)

# average count of individual behaviors for vehicle evals and discrepancy
x3 = dx.loc[dx['collision'], bxs].mean(axis=0)
x3 = cs(x3).sort_values(ascending=False).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})
x3['num evals'] = dx['collision'].sum()
x4 = dx.loc[~dx['collision'], bxs].mean(axis=0)
x4 = cs(x4).sort_values(ascending=False).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})
x4['num evals'] = (~dx['collision']).sum()
x34 = pd.merge(on='behavior', how='inner', left=x3, right=x4, suffixes=('-collision', '-no-collision'))
x34['abs-diff'] = np.abs(x34['value-collision'] - x34['value-no-collision'])
x34 = x34.sort_values('abs-diff', ascending=False).reset_index(drop=True)

# non-ML metrics
metrics = {}
metrics['x1'] = x1
metrics['x2'] = x2
metrics['x12'] = x12
metrics['x3'] = x3
metrics['x4'] = x4
metrics['x34'] = x34

# ml data
dp = dp.loc[dp.index.isin(db.index)]
y = db.pop('nbehaviors_47').values
X = db.values.astype('float')
cols = db.columns.to_numpy()
assert cols.size == X.shape[1]
y = y.astype('bool').astype('float')

# cross-validation, rfc
class_weight = {0: 0.01, 1: 0.99}
model = RandomForestClassifier(max_depth=4, n_estimators=300, class_weight=class_weight)
kf = StratifiedKFold(n_splits=4)
yp = np.full(y.size, np.nan)
df = defaultdict(list)
for split, (train, test) in enumerate(tqdm(kf.split(X=np.zeros(y.size), y=y), desc='cross-val-rfc', total=kf.n_splits)):

    # model train
    mx = clone(model)
    mx.fit(X[train], y[train])
    y_pred_train = mx.predict_proba(X[train])[:, mx.classes_ == 1].flatten()
    y_pred_test = mx.predict_proba(X[test])[:, mx.classes_ == 1].flatten()

    # model eval
    df['split'].append(split)
    df['train score mean'].append(y_pred_train.mean())
    df['test score mean'].append(y_pred_test.mean())
    df['train auc'].append(roc_auc_score(y_true=y[train], y_score=y_pred_train))
    df['test auc'].append(roc_auc_score(y_true=y[test], y_score=y_pred_test))
    df['train ap'].append(average_precision_score(y_true=y[train], y_score=y_pred_train))
    df['test ap'].append(average_precision_score(y_true=y[test], y_score=y_pred_test))
rfc = pd.DataFrame(df)

# feature importance, rfc
model.fit(X, y)
mx = np.argsort(model.feature_importances_)[::-1]
dp = permutation_importance(model, X, y, n_jobs=-1, scoring='roc_auc', n_repeats=5, random_state=0)
px = np.argsort(dp['importances_mean'])[::-1]
metrics['rfc-model'] = cs(pd.Series(index=cols[mx],
    data=model.feature_importances_[mx])).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})
metrics['rfc-permutation'] = cs(pd.Series(index=cols[px],
    data=dp['importances_mean'][px])).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})

# cross-validation, logistic regression
model = LogisticRegression(class_weight=class_weight)
yp = np.full(y.size, np.nan)
df = defaultdict(list)
for split, (train, test) in enumerate(tqdm(kf.split(X=np.zeros(y.size), y=y), desc='cross-val-lr', total=kf.n_splits)):

    # model train
    mx = clone(model)
    preprocessor = MinMaxScaler()
    preprocessor.fit(X[train])
    Xs = preprocessor.transform(X[train])
    mx.fit(Xs, y[train])
    y_pred_train = mx.predict_proba(Xs)[:, mx.classes_ == 1].flatten()
    Xs = preprocessor.transform(X[test])
    y_pred_test = mx.predict_proba(Xs)[:, mx.classes_ == 1].flatten()

    # model eval
    df['split'].append(split)
    df['train score mean'].append(y_pred_train.mean())
    df['test score mean'].append(y_pred_test.mean())
    df['train auc'].append(roc_auc_score(y_true=y[train], y_score=y_pred_train))
    df['test auc'].append(roc_auc_score(y_true=y[test], y_score=y_pred_test))
    df['train ap'].append(average_precision_score(y_true=y[train], y_score=y_pred_train))
    df['test ap'].append(average_precision_score(y_true=y[test], y_score=y_pred_test))
lr = pd.DataFrame(df)

# feature importance, lr
Xs = preprocessor.fit_transform(X)
model.fit(Xs, y)
mx = np.argsort(np.abs(model.coef_.flatten()))[::-1]
dp = permutation_importance(model, X, y, n_jobs=-1, scoring='roc_auc', n_repeats=5, random_state=0)
px = np.argsort(dp['importances_mean'])[::-1]
metrics['lr-model'] = cs(pd.Series(index=cols[mx],
    data=np.abs(model.coef_.flatten())[mx])).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})
metrics['lr-permutation'] = cs(pd.Series(index=cols[px],
    data=dp['importances_mean'][px])).reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'})

# feature relative importance
rx = cs(pd.Series(index=db.columns.to_numpy(), data=100 * db.values.sum(axis=0) / db.values.sum()))
rx = rx.reset_index(drop=False).rename(columns={'index': 'behavior', 0: 'value'}).sort_values('value', ascending=False)
metrics['relative occurrence'] = rx

# save metrics
with open(os.path.join(datadir, 'metrics.p'), 'wb') as fid:
    pickle.dump(metrics, fid)
