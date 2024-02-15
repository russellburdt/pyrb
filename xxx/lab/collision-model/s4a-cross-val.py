
"""
cross-validated collision prediction model
"""

import os
import utils
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from ipdb import set_trace


# train data
population = 'dft'
dcm, dm = utils.cpm_artifacts(population=population, min_days_segments=60)
print(f"""{dm.shape[0]} evals, {dm['outcome'].sum()} collisions, {100 * dm['outcome'].sum() / dm.shape[0]:.1f}% of evals""")

# pre-processing
y = dm.pop('outcome').fillna(0).astype('bool')
dm = dm.fillna(0)
dm['industry'] = dcm['IndustryDesc'].values
pipe = Pipeline(steps=[('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('cat', pipe, ['industry'])], verbose_feature_names_out=False, remainder='passthrough')
preprocessor.fit(dm)
cols = preprocessor.get_feature_names_out()
assert np.isnan(preprocessor.transform(dm)).sum() == 0

# RandomForest model
model = RandomForestClassifier(max_depth=6, n_estimators=600, class_weight={0: 0.010, 1: 0.990})

# cross-validation
kf = StratifiedKFold(n_splits=4)
yp = np.full(y.size, np.nan)
for split, (train, test) in enumerate(tqdm(kf.split(X=np.zeros(y.size), y=y), desc='cross-val all data', total=4)):

    # train and test data for split
    X_train = preprocessor.fit_transform(dm.loc[train])
    X_test = preprocessor.transform(dm.loc[test])
    y_train = y[train]
    y_test = y[test]

    # fit on train set, predict on test set
    model.fit(X_train, y_train)
    yp[test] = model.predict_proba(X_test)[:, model.classes_ == 1].flatten()
assert np.isnan(yp).sum() == 0

# save results
dx = y.to_frame()
dx['pred'] = yp
dx.to_pickle(os.path.join(r'/mnt/home/russell.burdt/data/collision-model', population, 'cross-val.p'))
