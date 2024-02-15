
# """
# shap values for collision prediction model
# """
# import os
# import pickle
# import utils
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pyrb.mpl import open_figure, format_axes, largefonts
# plt.style.use('bmh')

# # load data
# adir = r'c:/Users/russell.burdt/Downloads/artifacts-2022-06-09-16-52-31'
# with open(os.path.join(adir, 'shap.p'), 'rb') as fid:
#     shap = pickle.load(fid)
# try:
#     df = pd.read_pickle(os.path.join(adir, 'ml-data.p'))
# except FileNotFoundError:
#     df = pd.read_pickle(os.path.join(adir, 'ml-test-data.p'))
# with open(os.path.join(adir, 'feature-importance.p'), 'rb') as fid:
#     dfm = pickle.load(fid)
# try:
#     dcm = pd.read_pickle(os.path.join(adir, 'population-data.p'))
# except FileNotFoundError:
#     dcm = pd.read_pickle(os.path.join(adir, 'population-test-data.p'))
# yp = pd.read_pickle(os.path.join(adir, 'model-prediction-probabilities.p'))
# base = shap['base']
# values = shap['values']
# cols = dfm['features']
# assert all(np.isclose(base + values.sum(axis=1), yp['prediction probability'].values))
# assert all(yp['actual outcome'].values == dcm['outcome'].values)
# assert all(dcm['IndustryDesc'] == df['industry'])

# # shap waterfall chart for individual row in X_test
# n_features = 8
# row = yp.loc[(yp['actual outcome'] == 0)].sort_values('prediction probability').index[-1]
# vr = values[row, :]
# xr = np.hstack((df.loc[row].values[:-1], np.array([0, 0])))
# y_proba = yp.loc[row, 'prediction probability']
# y_true = yp.loc[row, 'actual outcome']
# title = f'shap values for row {row}, prediction probability {y_proba:.3f}, actual outcome {y_true}'
# sdata = utils.get_shap_chart_data(base=base, values=vr, xr=xr, n_features=n_features, cols=cols)
# utils.mpl_shap_waterfall_chart(*sdata, title=title)

# # # shap dependence plot for individual feature in X_test
# # feature = 'nevents_30_52_all'
# # x = df[feature].values
# # sx = values[:, cols == feature].flatten()
# # fig, ax = open_figure(f'shap dependence plot for {feature}', figsize=(12, 6))
# # ax.plot(x, sx, 'o', ms=6, mew=1, color='darkblue', alpha=0.05)
# # format_axes(feature, f'shap value for {feature}', f'shap dependence plot for {feature}', ax)
# # largefonts(16)
# # fig.tight_layout()

# plt.show()



"""
shap demo, Random Forest binary classification, Titanic dataset
"""

import os
import utils
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from pyrb.mpl import open_figure, format_axes, largefonts
plt.style.use('bmh')


def load_titanic_ml():

    # load data - https://www.kaggle.com/competitions/titanic/data?select=train.csv
    data = pd.read_csv(r'c:/Users/russell.burdt/Downloads/train.csv')
    data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

    # encode categorical
    sex = pd.get_dummies(data.pop('Sex'))['male'].to_frame()
    data = pd.concat((data, sex), axis=1)
    embarked = pd.get_dummies(data.pop('Embarked'))
    data = pd.concat((data, embarked), axis=1)

    # fill na as column mean value
    for col in data.columns:
        x = data[col].values
        x[np.isnan(x)] = np.nanmean(x)
        data[col] = x
    assert np.isnan(data.values).sum() == 0

    # train and test data and outcomes
    outcome = 'Survived'
    features = [x for x in data.columns if x != outcome]
    train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data[outcome])
    X_train = train[features].copy().reset_index(drop=True)
    y_train = train[outcome].values
    X_test = test[features].copy().reset_index(drop=True)
    y_test = test[outcome].values
    assert all(np.in1d(np.hstack((y_train, y_test)), np.array([0, 1])))
    y_train, y_test = y_train.astype('bool'), y_test.astype('bool')
    print(f"""train num rows, {X_train.shape[0]}, positive class percentage, {100 * y_train.sum() / X_train.shape[0]:.2f}%""")
    print(f"""test num rows, {X_test.shape[0]}, positive class percentage, {100 * y_test.sum() / X_test.shape[0]:.2f}%""")

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_titanic_ml()

# Random Forest model
rf = RandomForestClassifier(max_depth=6, n_estimators=100)
rf.fit(X_train, y_train)
bacc_train = balanced_accuracy_score(y_true=y_train, y_pred=rf.predict(X_train))
bacc_test = balanced_accuracy_score(y_true=y_test, y_pred=rf.predict(X_test))
print(f"""train balanced accuracy, {100 * bacc_train:.2f}%""")
print(f"""test balanced accuracy, {100 * bacc_test:.2f}%""")

# shap values for X_test, validate additivity
ex = shap.TreeExplainer(rf)
shap_values = ex(X_test)
base = ex.expected_value[1]
values = shap_values.values[:, :, 1]
cols = X_test.columns.to_numpy()
assert all(cols == shap_values.feature_names)
assert all(np.isclose(base + values.sum(axis=1), rf.predict_proba(X_test)[:, 1]))
yp = pd.DataFrame({'survived': y_test, 'proba': rf.predict_proba(X_test)[:, 1]})

# shap waterfall chart for individual row in X_test
row = yp.loc[yp['survived']].sort_values('proba').index[30]
title = f"""shap values for row {row}, prediction probability {yp.loc[row, 'proba']:.3f}, actual outcome {yp.loc[row, 'survived']}"""
sdata = utils.get_shap_chart_data(
    base=base,
    values=values[row, :],
    xr=X_test.loc[row].values,
    n_features=10,
    cols=cols)
utils.mpl_shap_waterfall_chart(*sdata, title=title)

# shap dependence plot for individual feature
feature = 'Age'
title = f"""shap dependence plot for {feature}"""
utils.mpl_shap_dependence_chart(
    feature=feature,
    x=X_test[feature].values,
    xs=base + values[:, cols == feature].flatten(),
    x0=yp['survived'].values,
    xp=yp['proba'].values,
    title=title)

plt.show()
