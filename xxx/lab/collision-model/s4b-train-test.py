
"""
train/test collision prediction model
"""

import os
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from pyrb.mpl import open_figure, largefonts, format_axes, save_pngs
from tqdm import tqdm
from ipdb import set_trace


# train and test datasets
dcm, dm, y = {}, {}, {}
for ds, population in zip(['train', 'test'], ['dft', 'nst']):
    dcm[ds], dm[ds] = utils.cpm_artifacts(population=population, min_days_segments=60)
    y[ds] = dm[ds].pop('outcome').fillna(0).astype('bool').values
    dm[ds] = dm[ds].fillna(0)
    dm[ds]['industry'] = dcm[ds]['IndustryDesc'].values
    print(f"""{ds}: {dm[ds].shape[0]} evals, {y[ds].sum()} collisions, {100 * y[ds].sum() / dm[ds].shape[0]:.1f}% of evals""")

# remove rows in test set from train set
on = ['VehicleId', 'ta', 'tb', 'tc']
nok = pd.merge(dcm['train'][on], dcm['test'][on], how='left', on=on, indicator=True)
nok = nok.loc[nok['_merge'] == 'both']
dcm['train'] = dcm['train'].loc[~dcm['train'].index.isin(nok.index)].reset_index(drop=True)
assert pd.merge(dcm['train'][on], dcm['test'][on], how='inner', on=on).size == 0

# validate same columns in train and test sets
ok = list(set(dm['train'].columns).intersection(dm['test'].columns))
for ds in ['train', 'test']:
    dm[ds] = dm[ds][ok].copy()

# preprocessing
pipe = Pipeline(steps=[('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('cat', pipe, ['industry'])], verbose_feature_names_out=False, remainder='passthrough')
preprocessor.fit(dm['train'])
cols = preprocessor.get_feature_names_out()
dx = {}
for ds in ['train', 'test']:
    dx[ds] = preprocessor.transform(dm[ds])
    assert np.isnan(dx[ds]).sum() == 0
    assert dx[ds].shape[1] == cols.size

# RandomForest model
model = RandomForestClassifier(max_depth=6, n_estimators=600, class_weight={0: 0.010, 1: 0.990})

# fit and predict
model.fit(X=dx['train'], y=y['train'])
yp = model.predict_proba(dx['test'])[:, model.classes_ == 1].flatten()

# save results
dx = pd.DataFrame({'outcome': y['test']})
dx['pred'] = yp
dx.to_pickle(os.path.join(r'/mnt/home/russell.burdt/data/collision-model', population, 'train-test.p'))
