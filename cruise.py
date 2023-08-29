
"""
process data from 'Ryde Automation'
- clean and validate
- develop models for price, rating, and num_near_misses
- identify useful business actions based on model results
- validate null hypothesis - user_id does not impact price
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl  # matplotlib utils, https://github.com/russellburdt/pyrb/blob/master/mpl.py
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from ipdb import set_trace
from tqdm import tqdm
plt.style.use('bmh')


def read_and_validate_same_data():

    # read csv and xlsx files, assert same shape
    fn1 = r'c:/Users/russe/Downloads/cruise/russell_burdt_final_analytics_takehome.csv'
    fn2 = r'c:/Users/russe/Downloads/cruise/russell_burdt_final_analytics_takehome.xlsx'
    assert os.path.isfile(fn1) and os.path.isfile(fn2)
    df1, df2 = pd.read_csv(fn1), pd.read_excel(fn2)
    assert df1.shape == df2.shape

    # object to datetime for csv data
    df1['start_time'] = pd.to_datetime(df1['start_time'])
    df1['end_time'] = pd.to_datetime(df1['end_time'])

    # validate same datatypes and column-names
    assert all(df1.dtypes == df2.dtypes)
    assert sorted(df1.columns) == sorted(df2.columns)
    df1 = df1.sort_values('start_time').reset_index(drop=True)
    df2 = df2.sort_values('start_time').reset_index(drop=True)

    # validate same data, all non-null, return single DataFrame
    assert df1.equals(df2)
    assert all(~pd.isnull(df1))
    return df1.copy()

# read and validate same data in each provided file
df = read_and_validate_same_data()
df0 = df.copy()

# get ride duration, filter negative ride duration (mean duration for duration < 0 indicates bad data)
df['duration'] = [x.total_seconds() / 60 for x in df['end_time'] - df['start_time']]
x = df['duration'].values
mean_duration_raw = pd.Series({
    'mean ride duration in minutes, duration < 0': np.abs(x[x < 0]).mean(),
    'mean ride duration in minutes, duration >= 0': x[x >= 0].mean()})
df = df.loc[x >= 0].reset_index(drop=True)

# assert all rides start and end on 10/2, ie day-of-week/year not useful
assert all([x.strftime('%m/%d') == '10/02' for x in df['start_time']])
assert all([x.strftime('%m/%d') == '10/02' for x in df['end_time']])

# create start_time_key feature, no further use for start_time / end_time
df['start_time_key'] = [int(x.strftime('%H')) for x in df['start_time']]
df['start_time_key'] = np.digitize(df['start_time_key'].values, bins=[0, 6, 9, 15, 18, 24])
df['start_time_key'] = [f'r{x}' for x in df['start_time_key']]
del df['start_time']
del df['end_time']

# remove but keep user_id
user_id = df.pop('user_id')

# dataset properties
dp = defaultdict(list)
for col in df.columns:
    dp['column'].append(col)
    dp['num unique'].append(pd.unique(df[col]).size)
    dp['min value'].append(df[col].min())
    dp['max value'].append(df[col].max())
dp = pd.DataFrame(dp)

def distribution_ride_duration_by_column(col):

    # ride duration distribution for each unique value in column
    for value in pd.unique(df[col]):
        x = df.loc[df[col] == value, 'duration'].values
        mpl.metric_distribution(x=x, bins=bins, legend=f'{col}, {value}', figsize=(8, 5), size=18, xlabel=xlabel,
            title=f'distribution of ride duration by {col}', loc='upper right', bbox_to_anchor=None)

# distribution of ride-duration in minutes, explore splits by column to explain missing data in full distribution
bins = np.linspace(df['duration'].min(), df['duration'].max(), 60)
xlabel = 'ride duration in minutes'
mpl.metric_distribution(x=df['duration'].values, bins=bins, figsize=(9, 5), size=18, xlabel=xlabel, title=f'distribution of ride duration')
distribution_ride_duration_by_column(col='region')      # indicates split, but likely not useful
distribution_ride_duration_by_column(col='num_riders')  # not a useful split, did not find other useful splits

# distributions of price, rating, num_near_misses (to determine appropriate model definitions)
for outcome in ['price', 'rating', 'num_near_misses']:
    x = df[outcome].values
    bins = np.linspace(x.min(), x.max(), 60)
    logscale = True if outcome == 'num_near_misses' else False
    mpl.metric_distribution(x=x, bins=bins, figsize=(9, 4), size=18, xlabel=outcome, title=f'distribution of {outcome}', logscale=logscale)

# generic regression model
def regression_model_train_eval(regressor, X, y, features):

    # validate
    assert X.shape == (y.size, features.size)

    # initialize metrics for train and test splits, scan over splits
    rmse_train, rmse_test = [], []
    r2_train, r2_test = [], []
    kf = KFold(n_splits=4, shuffle=True)
    for xtrain, xtest in kf.split(X, y):
        regressor.fit(X[xtrain], y[xtrain])
        y_pred_train = regressor.predict(X[xtrain])
        y_pred_test = regressor.predict(X[xtest])
        rmse_train.append(mean_squared_error(y_true=y[xtrain], y_pred=y_pred_train)**0.5)
        rmse_test.append(mean_squared_error(y_true=y[xtest], y_pred=y_pred_test)**0.5)
        r2_train.append(r2_score(y_true=y[xtrain], y_pred=y_pred_train))
        r2_test.append(r2_score(y_true=y[xtest], y_pred=y_pred_test))

    # train on all data to get full feature importance
    regressor.fit(X, y)

    # return mean metrics for all splits and feature importance
    return pd.Series({
        'rmse_train_mean': np.mean(rmse_train),
        'rmse_test_mean': np.mean(rmse_test),
        'r2_train': np.mean(r2_train),
        'r2_test': np.mean(r2_test),
        'feature_importance': features[np.argsort(regressor.feature_importances_)[::-1]]})

# generic classification model
def classification_model_train_eval(classifier, X, y, features):

    # validate
    assert X.shape == (y.size, features.size)

    # initialize metrics for train and test splits, scan over splits
    auc_train, auc_test = [], []
    ap_train, ap_test = [], []
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    for xtrain, xtest in skf.split(X, y):
        classifier.fit(X[xtrain], y[xtrain])
        y_pred_train = classifier.predict_proba(X[xtrain])[:, classifier.classes_].flatten()
        y_pred_test = classifier.predict_proba(X[xtest])[:, classifier.classes_].flatten()
        auc_train.append(roc_auc_score(y_true=y[xtrain], y_score=y_pred_train)**0.5)
        auc_test.append(roc_auc_score(y_true=y[xtest], y_score=y_pred_test)**0.5)
        ap_train.append(average_precision_score(y_true=y[xtrain], y_score=y_pred_train))
        ap_test.append(average_precision_score(y_true=y[xtest], y_score=y_pred_test))

    # train on all data to get full feature importance
    classifier.fit(X, y)

    # return mean metrics for all splits and feature importance
    return pd.Series({
        'auc_train_mean': np.mean(auc_train),
        'auc_test_mean': np.mean(auc_test),
        'ap_train': np.mean(ap_train),
        'ap_test': np.mean(ap_test),
        'feature_importance': features[np.argsort(classifier.feature_importances_)[::-1]]})

# regression model for price (model A)
y = df['price'].values
X = pd.get_dummies(df[[x for x in df.columns if x != 'price']].copy()).astype('float')
features, X = X.columns, X.values
ra = RandomForestRegressor(n_estimators=100, max_depth=3)
ma = regression_model_train_eval(regressor=ra, X=X, y=y, features=features)

# classification model for rating (model B)
y = df['rating'].values.copy()
y[y < 5] = 0
y = y.astype('bool')
X = pd.get_dummies(df[[x for x in df.columns if x != 'rating']].copy()).astype('float')
features, X = X.columns, X.values
cb = RandomForestClassifier(n_estimators=600, max_depth=3, class_weight={0:0.30, 1:0.70})
mb = classification_model_train_eval(classifier=cb, X=X, y=y, features=features)

# classification model for num_near_misses (model C)
y = df['num_near_misses'].values.copy().astype('bool')
X = pd.get_dummies(df[[x for x in df.columns if x != 'num_near_misses']].copy()).astype('float')
features, X = X.columns, X.values
cc = RandomForestClassifier(n_estimators=200, max_depth=3, class_weight={0:0.06, 1:0.94})
mc = classification_model_train_eval(classifier=cc, X=X, y=y, features=features)

# ride price vs top-feature from model A - number of riders
fig, ax = mpl.open_figure('ride price vs number of riders', figsize=(7, 3))
ax.plot(df['num_riders'], df['price'], 'x')
mpl.format_axes('num_riders', 'price', 'ride price vs number of riders', ax)
mpl.largefonts(18)
fig.tight_layout()

# explore top-features from model C - duration and num_near_misses grouped by car_id
left = df.groupby('car_id')['num_near_misses'].sum()
right = df.groupby('car_id')['duration'].mean()
dc = pd.merge(left=left, right=right, left_index=True, right_index=True).rename(columns={'duration': 'mean duration'})

# run price model without user_id, n times
n = 100
y = df['price'].values
X = pd.get_dummies(df[[x for x in df.columns if x != 'price']].copy()).astype('float')
features, X = X.columns, X.values
ra = RandomForestRegressor(n_estimators=100, max_depth=3)
h0 = np.full(n, np.nan)
for x in tqdm(range(n), desc='model without user_id'):
    ma = regression_model_train_eval(regressor=ra, X=X, y=y, features=features)
    h0[x] = ma['r2_test']

# run price model with user_id, n times
df['user_id'] = [f'user{x}' for x in user_id]
y = df['price'].values
X = pd.get_dummies(df[[x for x in df.columns if x != 'price']].copy()).astype('float')
features, X = X.columns, X.values
h1 = np.full(n, np.nan)
for x in tqdm(range(n), desc='model with user_id'):
    ma = regression_model_train_eval(regressor=ra, X=X, y=y, features=features)
    h1[x] = ma['r2_test']

# combine h0 and h1
hx = np.hstack((h0, h1))

# bootstrap differences of sample means
nt = 10000
bootstrap = np.full(nt, np.nan)
idx = range(hx.size)
for x in tqdm(range(nt), desc='bootstrap hypothesis test'):
    x0 = np.random.choice(idx, replace=False, size=n)
    x1 = np.array(list(set(idx).difference(x0)))
    bootstrap[x] = hx[x0].mean() - hx[x1].mean()

# p-value for null hypothesis - user_id does not impact R2 of price model (reject if low p-value)
m0 = np.abs(h0.mean() - h1.mean())
pv = (np.abs(bootstrap) > m0).sum() / nt

# bootstrap hypothesis test results
bins = np.linspace(bootstrap.min(), bootstrap.max(), 400)
title = f'bootstrap hypothesis test, p-value {100 * pv:.1f}%'
fig, ax = mpl.metric_distribution(x=bootstrap, bins=bins, figsize=(10, 6), size=18, legend=f'{nt} tests', xlabel='mean difference', title=title)
ax.plot(np.tile(m0, 2), ax.get_ylim(), '-', lw=4, label='natural mean difference')
ax.legend(loc='upper left', fontsize=16)
mpl.largefonts(18)

# show figures
plt.show()
