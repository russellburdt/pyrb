
"""
run collision model
- preprocessing
- learning curve
- evaluation (cross-validation, train-test split, test-dataset)
- save model artifacts
"""

import os
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shap
from pyspark import SparkConf
from pyspark.sql import SparkSession
from datetime import datetime
from pytz import timezone
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from pyrb.mpl import open_figure, largefonts, format_axes, save_pngs
from lytx import get_conn
from tqdm import tqdm
from ipdb import set_trace


# load train data
datadir = r'/mnt/home/russell.burdt/data/collision-model/dft'
assert os.path.isdir(datadir)
dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
dp = pd.read_pickle(os.path.join(datadir, 'metadata', 'positive_instances.p'))
dcm = pd.read_pickle(os.path.join(datadir, 'dcm.p'))
df = pd.read_pickle(os.path.join(datadir, 'df.p'))

# drop oversampled records
ok = ~dcm['oversampled']
dcm = dcm.loc[ok].reset_index(drop=True)
df = df.loc[ok].reset_index(drop=True)

# filter by columns
# with open(r'/mnt/home/russell.burdt/data/collision-model/app/artifacts-01/feature-importance.p', 'rb') as fid:
#     dfm = pickle.load(fid)
# cok = np.argsort(dfm['model feature importance'])[::-1][:12]
# cok = dfm['features'][cok]
# cok = [x for x in df.columns if x[:4] != 'gpse']
cok = [x for x in df.columns if 'trig' not in x]
df = df[cok].copy()

# filter by row condition
ds = pd.read_pickle(os.path.join(datadir, 'gps_segmentation_metrics.p'))
dx = dcm[['VehicleId', 'time0', 'time1']].copy()
dx['time0'] = [(pd.Timestamp(x) - pd.Timestamp('1970-1-1')).total_seconds() for x in dx['time0'].values]
dx['time1'] = [(pd.Timestamp(x) - pd.Timestamp('1970-1-1')).total_seconds() for x in dx['time1'].values]
dx = dx.reset_index(drop=False)
dx = pd.merge(dx, ds, on=['VehicleId', 'time0', 'time1'], how='inner')
assert np.all(np.isclose(np.full(dx.shape[0], 90).astype('float'), dx['total_days'].values))
dx = dx.loc[dx['n_days_segments'] > 30, ['index', 'n_days_segments']]
dx['sf'] = 90 / dx['n_days_segments'].values
dcm = dcm.loc[dx['index'].values].reset_index(drop=True)
df = df.loc[dx['index'].values].reset_index(drop=True)

# validate and get population metadata
assert len(set(dcm.columns).intersection(df.columns)) == 0
assert dcm.shape[0] == df.shape[0]
df0 = df.copy()
dm = utils.get_population_metadata(dcm, dc, datadir=None)
utils.validate_dcm_dp(dcm, dp)

# set model evaluation and shap values
meval = 'test-dataset'
include_shap = True
assert meval in ['cross-val', 'cross-val-oversampled', 'train-test-split', 'test-dataset']

# load test data for test-dataset evaluation
if meval == 'test-dataset':

    # load test dataset
    datadir_test = r'/mnt/home/russell.burdt/data/collision-model/gwcc/v0'
    assert os.path.isdir(datadir_test)
    dc_test = pd.read_pickle(os.path.join(datadir_test, 'metadata', 'model_params.p'))
    dp_test = pd.read_pickle(os.path.join(datadir_test, 'metadata', 'positive_instances.p'))
    dcm_test = pd.read_pickle(os.path.join(datadir_test, 'dcm-gwcc.p'))
    df_test = pd.read_pickle(os.path.join(datadir_test, 'df-gwcc.p'))

    # drop oversampled records
    ok = ~dcm_test['oversampled']
    dcm_test = dcm_test.loc[ok].reset_index(drop=True)
    df_test = df_test.loc[ok].reset_index(drop=True)

    # filter by columns
    # cok = [x for x in df_test.columns if x[:4] != 'gpse']
    # df_test = df_test[cok].copy()

    # filter by row condition
    ds_test = pd.read_pickle(os.path.join(datadir_test, 'gps_segmentation_metrics.p'))
    dx = dcm_test[['VehicleId', 'time0', 'time1']].copy()
    dx['time0'] = [(pd.Timestamp(x) - pd.Timestamp('1970-1-1')).total_seconds() for x in dx['time0'].values]
    dx['time1'] = [(pd.Timestamp(x) - pd.Timestamp('1970-1-1')).total_seconds() for x in dx['time1'].values]
    dx = dx.reset_index(drop=False)
    dx = pd.merge(dx, ds_test, on=['VehicleId', 'time0', 'time1'], how='inner')
    assert np.all(np.isclose(np.full(dx.shape[0], 90).astype('float'), dx['total_days'].values))
    dx = dx.loc[dx['n_days_segments'] > 30, ['index', 'n_days_segments']]
    dx['sf'] = 90 / dx['n_days_segments'].values
    dcm_test = dcm_test.loc[dx['index'].values].reset_index(drop=True)
    df_test = df_test.loc[dx['index'].values].reset_index(drop=True)

    # validate and get population metadata
    assert len(set(dcm_test.columns).intersection(df_test.columns)) == 0
    assert dcm_test.shape[0] == df_test.shape[0]
    dm_test = utils.get_population_metadata(dcm_test, dc_test, datadir=None)
    # utils.validate_dcm_dp(dcm_test, dp_test)

    # identify same rows in test-dataset and train-dataset
    on = ['VehicleId', 'time0', 'time1']
    left = dcm[on].reset_index(drop=False)
    right = dcm_test[on].reset_index(drop=False)
    same = pd.merge(left=left, right=right, on=on, how='inner', suffixes=['_train', '_test'])

    # remove rows in test-dataset from train-dataset
    if same.size > 0:
        nok = same['index_train'].values
        dcm = dcm.loc[~dcm.index.isin(nok)].reset_index(drop=True)
        df = df.loc[~df.index.isin(nok)].reset_index(drop=True)
        df0 = df0.loc[~df0.index.isin(nok)].reset_index(drop=True)
        assert len(set(dcm.columns).intersection(df.columns)) == 0
        assert dcm.shape[0] == df.shape[0]
        dm = utils.get_population_metadata(dcm, dc, datadir=None)

    # common features for train and test sets, filter df and df_test
    common = list(set(df.columns).intersection(df_test.columns))
    df = df[common].copy()
    df_test = df_test[common].copy()

# train data pre-processing
df = df.fillna(0)
df['industry'] = dcm['IndustryDesc'].values

# create and validate preprocessor object
cat_features = ['industry']
num_features = [x for x in df.columns if x not in cat_features]
cat_pipe = Pipeline(steps=[
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'))])
num_pipe = Pipeline(steps=[
    ('standard-scaler', StandardScaler())])
preprocessor = ColumnTransformer(
    transformers=[
        # ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)],
    verbose_feature_names_out=False,
    remainder='passthrough')
assert np.isnan(preprocessor.fit_transform(df)).sum() == 0

# identify column categories
cxs = preprocessor.get_feature_names_out()
assert np.unique(cxs).size == cxs.size
columns = {}
columns['trip'] = np.array([x for x in cxs if 'trip' in x.lower()])
columns['event'] = np.array([x for x in cxs if 'events' in x.lower()])
columns['behavior'] = np.array([x for x in cxs if 'behaviors' in x.lower()])
columns['dce model'] = np.array([x for x in cxs if 'dce_model' in x.lower()])
columns['trigger'] = np.array([x for x in cxs if 'triggers' in x.lower()])
columns['industry'] = np.array([x for x in cxs if 'industry' in x.lower()])
columns['gps'] = np.array([x for x in cxs if 'gps' in x.lower()])
columns['imn'] = np.array([x for x in cxs if 'imn' in x.lower()])
columns['gps-enriched'] = np.array([x for x in cxs if 'dc_' in x.lower()])
cxa = np.concatenate(list(columns.values()))
columns['all others'] = np.array([x for x in cxs if x not in cxa])
cxa = np.concatenate(list(columns.values()))
assert np.unique(cxa).size == cxa.size
assert np.all(np.sort(cxa) == np.sort(cxs))

# train data outcome
dcm['outcome'] = dcm['collision-47'].values.astype('int')
# dcm['outcome'] = np.logical_or(
#     dcm['collision-45'].values,
#     dcm['collision-46'].values,
#     dcm['collision-47'].values).astype('int')
y = dcm['outcome'].values

# RandomForest model parameters
max_depth = 6
n_estimators = 600
class_weight = {0: 0.010, 1: 0.990}
# max_depth = 10
# n_estimators = 600
# class_weight = {0: 0.100, 1: 0.900}

# data description
with open(os.path.join(datadir, 'metadata', 'desc.txt')) as fid:
    data_desc = fid.readlines()
data_desc = ''.join(data_desc)
data_desc += f"""
number of rows - {y.size}
number of positive instances - {y.sum():.0f} ({100 * y.sum() / y.size:.2f})%"""
data_desc += f"""\n- {cxs.size}x total metrics"""
for category, cx in columns.items():
    data_desc += f"""\n- {cx.size}x {category} metrics"""

# model and model description
assert sum(class_weight.values()) == 1
model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, class_weight=class_weight)
model_desc = f"""
    Random Forest Classifier
    - max_depth, {max_depth}
    - n_estimators, {n_estimators}
    - positive class weight, {100 * class_weight[1]:.1f}%"""

# learning curve data
n_splits = 4
dlc = utils.get_learning_curve_data(df=df, y=y, model=model, preprocessor=preprocessor, n_splits=n_splits, train_sizes=np.logspace(-1, 0, 20))

# initialize model artifacts
now = datetime.now().astimezone(timezone('US/Pacific')).strftime(r'%Y-%m-%d-%H-%M-%S')
adir = os.path.join(datadir, f'artifacts-{now}')
os.mkdir(adir)
dc.to_pickle(os.path.join(adir, 'configuration.p'))
dlc.to_pickle(os.path.join(adir, 'learning-curve-data.p'))
with open(os.path.join(adir, 'data desc.txt'), 'w') as fid:
    fid.write(data_desc)
with open(os.path.join(adir, 'model desc.txt'), 'w') as fid:
    fid.write(model_desc)

"""
The following code blocks under each of (meval == '...') all save at least the following artifacts
* ml-data.p
* population-data.p
* ml-metadata.p
* collisions.p
* eval desc.txt
"""

# model evaluation via cross-validation
if meval == 'cross-val':

    # cross-validation description
    n_splits = 4
    eval_desc = f"""
        - {n_splits}-fold cross-validation
        - prediction probabilities from combined test sets
        - model feature importance from final train split
        - permutation feature importance from final test split
        - distribution feature importance from all data
        """

    # cross-validation
    kf = StratifiedKFold(n_splits=n_splits)
    yp = np.full(y.size, np.nan)
    xtest = np.full((y.size, preprocessor.get_feature_names_out().size), np.nan)
    if include_shap:
        base = np.full(y.size, np.nan)
        values = np.full((y.size, preprocessor.get_feature_names_out().size), np.nan)
    for split, (train, test) in enumerate(tqdm(kf.split(X=np.zeros(y.size), y=y), desc='cross-val all data', total=n_splits)):

        # model and preprocessor for split
        model = clone(model)
        preprocessor = clone(preprocessor)

        # train and test data for split
        df_train = df.loc[train]
        df_test = df.loc[test]
        y_train = y[train]
        y_test = y[test]

        # features for split
        X_train = preprocessor.fit_transform(df_train)
        X_test = preprocessor.transform(df_test)
        xtest[test] = X_test

        # fit on train set, predict on test set
        model.fit(X_train, y_train)
        yp[test] = model.predict_proba(X_test)[:, model.classes_ == 1].flatten()

        # shap values for X_test
        if include_shap:
            ex = shap.TreeExplainer(model)
            shap_values = ex(X_test)
            base[test] = ex.expected_value[1]
            values[test, :] = shap_values.values[:, :, 1]
    assert np.isnan(yp).sum() == 0
    assert np.isnan(xtest).sum() == 0
    if include_shap:
        assert np.isnan(base).sum() == 0
        assert np.isnan(values).sum() == 0

    # permutation feature importance
    now = datetime.now()
    print('permutation importance')
    scoring = ['average_precision', 'roc_auc']
    dfm = permutation_importance(model, X_test, y_test, n_jobs=-1, scoring=scoring, n_repeats=5, random_state=0)
    print(f'duration, {(datetime.now() - now).total_seconds() / 60:.1f} min')

    # model feature importance
    dfm['features'] = preprocessor.get_feature_names_out()
    dfm['model feature importance'] = model.feature_importances_

    # shap values
    if include_shap:
        assert all(np.isclose(base + values.sum(axis=1), yp))
        shap_dict = {'base': base, 'values': values}
        with open(os.path.join(adir, 'shap.p'), 'wb') as fid:
            pickle.dump(shap_dict, fid)

    # model artifacts
    with open(os.path.join(adir, 'ml-data.p'), 'wb') as fid:
        pickle.dump(xtest, fid)
    dcm.to_pickle(os.path.join(adir, 'population-data.p'))
    dm.to_pickle(os.path.join(adir, 'ml-metadata.p'))
    dp.to_pickle(os.path.join(adir, 'collisions.p'))
    with open(os.path.join(adir, 'eval desc.txt'), 'w') as fid:
        fid.write(eval_desc)

# model evaluation via cross-validation and oversampled data
if meval == 'cross-val-oversampled':

    # cross-validation description
    n_splits = 4
    eval_desc = f"""
        - {n_splits}-fold cross-validation
        - train splits use oversampled data
        - test splits do not use oversampled data
        - prediction probabilities from combined test sets
        - model feature importance from final train split
        - permutation feature importance from final test split
        - distribution feature importance from all data
        """

    # validate oversampled records in dcm, get true outcome without oversampled records
    assert dcm['oversampled'].sum() > 0
    assert all(dcm.reset_index().index == dcm.index)
    assert dcm.loc[dcm['oversampled']].index.min() > dcm.loc[~dcm['oversampled']].index.max()
    yx = dcm.loc[~dcm['oversampled'], 'outcome'].values

    # cross-validation
    kf = StratifiedKFold(n_splits=n_splits)
    yp = np.full(yx.size, np.nan)
    for split, (train, test) in enumerate(tqdm(kf.split(X=np.zeros(yx.size), y=yx), desc='cross-val all data', total=n_splits)):

        # model and preprocessor for split
        model = clone(model)
        preprocessor = clone(preprocessor)

        # train and test data for split
        df_train = df.loc[train]
        df_test = df.loc[test]
        y_train = y[train]
        y_test = y[test]

        # augment df_train and y_train with oversampled data
        for tx, row in dcm.loc[dcm.index.isin(train) & dcm['outcome']].iterrows():
            dx = dcm.loc[(dcm['oversampled']) & (dcm['oversampled index'] == tx)]
            if dx.size == 0:
                continue
            assert all(y[dx.index] == 1)
            df_train = pd.concat((df_train, df.loc[dx.index]))
            y_train = np.hstack((y_train, y[dx.index]))

        # features for split
        X_train = preprocessor.fit_transform(df_train)
        X_test = preprocessor.transform(df_test)

        # fit on train set, predict on test set
        model.fit(X_train, y_train)
        yp[test] = model.predict_proba(X_test)[:, model.classes_ == 1].flatten()
    assert np.isnan(yp).sum() == 0

    # permutation feature importance
    now = datetime.now()
    print('permutation importance')
    scoring = ['average_precision', 'roc_auc']
    dfm = permutation_importance(model, X_test, y_test, n_jobs=-1, scoring=scoring, n_repeats=5, random_state=0)
    print(f'duration, {(datetime.now() - now).total_seconds() / 60:.1f} min')

    # model feature importance
    dfm['features'] = preprocessor.get_feature_names_out()
    dfm['model feature importance'] = model.feature_importances_

    # save outcome as yx
    y = yx

    # model artifacts
    df.to_pickle(os.path.join(adir, 'ml-data.p'))
    dcm.to_pickle(os.path.join(adir, 'population-data.p'))
    dm.to_pickle(os.path.join(adir, 'ml-metadata.p'))
    dp.to_pickle(os.path.join(adir, 'collisions.p'))
    with open(os.path.join(adir, 'eval desc.txt'), 'w') as fid:
        fid.write(eval_desc)

# model evaluation via train-test-split
if meval == 'train-test-split':

    # split by company-id repeatedly until test dataset size appx aligns with target test size
    test_size = 0.20
    print(f'train test split evaluation by company-id, {100 * test_size:.0f}% test size')
    test_size_margin = 0.01
    cid = pd.unique(dcm['CompanyId'])
    while True:

        # split by company-id
        cid_train = np.random.choice(range(cid.size), size=int((1 - test_size) * cid.size), replace=False)
        cid_test = np.array(list(set(range(cid.size)).difference(cid_train)))
        cid_train = cid[cid_train]
        cid_test = cid[cid_test]

        # get associated dcm split
        dcm_train = dcm.loc[dcm['CompanyId'].isin(cid_train)]
        dcm_test = dcm.loc[dcm['CompanyId'].isin(cid_test)]
        actual_test_size = dcm_test.shape[0] / dcm.shape[0]

        # get associated df split and break if actual test size within margin
        if test_size - test_size_margin < actual_test_size < test_size + test_size_margin:
            df_train = df.loc[dcm_train.index].reset_index(drop=True)
            df_test = df.loc[dcm_test.index].reset_index(drop=True)
            y_train = y[dcm_train.index]
            y_test = y[dcm_test.index]
            dcm_train = dcm_train.reset_index(drop=True)
            dcm_test = dcm_test.reset_index(drop=True)
            dm_train = utils.get_population_metadata(dcm_train, dc, datadir=None)
            dm_test = utils.get_population_metadata(dcm_test, dc, datadir=None)
            break

    # holdout description
    eval_desc = f"""
        - train / test split, test size is {100 * test_size:.0f}%
        - {df_train.shape[0]} rows in train set
        - {df_test.shape[0]} rows in test set
        - {y_train.sum()} positive instances in train set
        - {y_test.sum()} positive instances in test set
        - model feature importance from train set
        - permutation feature importance from test set
        - distribution feature importance from all data
        """

    # model and preprocessor
    model = clone(model)
    preprocessor = clone(preprocessor)

    # features for train and test sets
    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)

    # fit on train set, predict on test set
    model.fit(X_train, y_train)
    yp = model.predict_proba(X_test)[:, model.classes_ == 1].flatten()
    y = y_test

    # permutation feature importance
    now = datetime.now()
    print('permutation importance')
    scoring = ['average_precision', 'roc_auc']
    dfm = permutation_importance(model, X_test, y, n_jobs=-1, scoring=scoring, n_repeats=5, random_state=0)
    print(f'duration, {(datetime.now() - now).total_seconds() / 60:.1f} min')

    # model feature importance
    dfm['features'] = preprocessor.get_feature_names_out()
    dfm['model feature importance'] = model.feature_importances_

    # model artifacts
    df_train.to_pickle(os.path.join(adir, 'ml-train-data.p'))
    df_test.to_pickle(os.path.join(adir, 'ml-data.p'))
    dcm_train.to_pickle(os.path.join(adir, 'population-train-data.p'))
    dcm_test.to_pickle(os.path.join(adir, 'population-data.p'))
    dm_train.to_pickle(os.path.join(adir, 'ml-train-metadata.p'))
    dm_test.to_pickle(os.path.join(adir, 'ml-metadata.p'))
    dp.to_pickle(os.path.join(adir, 'collisions.p'))
    with open(os.path.join(adir, 'eval desc.txt'), 'w') as fid:
        fid.write(eval_desc)

# model evaluation via test-dataset
if meval == 'test-dataset':

    # test data pre-processing
    df_test = df_test.fillna(0)
    df_test['industry'] = dcm_test['IndustryDesc'].values
    assert np.isnan(preprocessor.fit_transform(df_test)).sum() == 0

    # features data
    X_train = preprocessor.fit_transform(df)
    X_test = preprocessor.transform(df_test)

    # clone model, fit on train set, predict on test set
    model = clone(model)
    model.fit(X_train, y)
    yp = model.predict_proba(X_test)[:, model.classes_ == 1].flatten()

    # test data outcome
    # dcm_test['outcome'] = dcm_test['collision-47'].values.astype('int')
    dcm_test['outcome'] = dcm_test['collision-gwcc'].values.astype('int')
    # dcm_test['outcome'] = np.logical_or(
    #     dcm_test['collision-47'].values,
    #     dcm_test['collision-gwcc'].values).astype('int')
    # dcm_test['outcome'] = np.logical_or(
    #     dcm_test['collision-45'].values,
    #     dcm_test['collision-46'].values,
    #     dcm_test['collision-47'].values).astype('int')
    y = dcm_test['outcome'].values
    # np.random.shuffle(y)

    # shap values for X_test
    if include_shap:
        ex = shap.TreeExplainer(model)
        shap_values = ex(X_test)
        base = np.tile(ex.expected_value[1], yp.size)
        values = shap_values.values[:, :, 1]
        assert np.isnan(base).sum() == 0
        assert np.isnan(values).sum() == 0
        assert all(np.isclose(base + values.sum(axis=1), yp))
        shap_dict = {'base': base, 'values': values}
        with open(os.path.join(adir, 'shap.p'), 'wb') as fid:
            pickle.dump(shap_dict, fid)

    # text eval description
    eval_desc = f"""
        --- train data ---
        - {dc['desc']}
        - num rows, {df.shape[0]}
        - num positive instances, {dcm['outcome'].sum()}
        - positive class percentage, {100 * dcm['outcome'].sum() / dcm.shape[0]:.3f}%
        - model feature importance from train data
        - distribution feature importance from train data
        --- test data ---
        - {dc_test['desc']}
        - num rows, {df_test.shape[0]}
        - num positive instances, {dcm_test['outcome'].sum()}
        - positive class percentage, {100 * dcm_test['outcome'].sum() / dcm_test.shape[0]:.3f}%
        - permutation feature importance from test data
        """

    # permutation feature importance
    now = datetime.now()
    print('permutation importance')
    scoring = ['average_precision', 'roc_auc']
    dfm = permutation_importance(model, X_test, y, n_jobs=-1, scoring=scoring, n_repeats=5, random_state=0)
    print(f'duration, {(datetime.now() - now).total_seconds() / 60:.1f} min')

    # model feature importance
    dfm['features'] = preprocessor.get_feature_names_out()
    dfm['model feature importance'] = model.feature_importances_

    # model artifacts
    with open(os.path.join(adir, 'ml-train-data.p'), 'wb') as fid:
        pickle.dump(X_train, fid)
    with open(os.path.join(adir, 'ml-data.p'), 'wb') as fid:
        pickle.dump(X_test, fid)
    dcm.to_pickle(os.path.join(adir, 'population-train-data.p'))
    dcm_test.to_pickle(os.path.join(adir, 'population-data.p'))
    dp_test.to_pickle(os.path.join(adir, 'collisions.p'))
    dm.to_pickle(os.path.join(adir, 'ml-train-metadata.p'))
    dm_test.to_pickle(os.path.join(adir, 'ml-metadata.p'))
    with open(os.path.join(adir, 'eval desc.txt'), 'w') as fid:
        fid.write(eval_desc)

# distribution feature importance
dfm['distribution feature importance'] = np.full(dfm['features'].size, np.nan)
for x, feature in enumerate(dfm['features']):
    if feature not in df0.columns:
        continue
    xf = df0[feature].values
    positive = xf[dcm['outcome'].values.astype('bool')]
    negative = xf[~dcm['outcome'].values.astype('bool')]
    positive = positive[~np.isnan(positive)]
    negative = negative[~np.isnan(negative)]
    if (positive.size == 0) or (negative.size == 0):
        continue
    dfm['distribution feature importance'][x] = ks_2samp(positive, negative).statistic
with open(os.path.join(adir, 'feature-importance.p'), 'wb') as fid:
    pickle.dump(dfm, fid)

# ROC / PR data
dml = utils.get_roc_pr_data(y, yp, size=100)
dml.to_pickle(os.path.join(adir, 'roc-pr-curve-data.p'))

# outcome vs prediction probability
dx = pd.DataFrame({'actual outcome': y, 'prediction probability': yp})
dx.to_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
