
"""
multiple regression toy problem and visualizations
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from pyrb import datasets
from pyrb import largefonts, save_pngs, format_axes
plt.style.use('bmh')


# load regression datasets
rbo = datasets.regression_boston()
rca = datasets.regression_california()
rdi = datasets.regression_diabetes()
rco = datasets.regression_concrete()
rde = datasets.regression_demand()
rtr = datasets.regression_traffic()

# test that residuals of regression follow a normal distribution
for ds in [rbo, rca, rdi, rco, rde, rtr]:

    # load dataset name and clean data
    name = ds['name']
    X = ds['X'].values
    y = ds['y'].values
    assert X.shape[0] == y.shape[0]

    # create train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # instantiate, run, and score a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    res_train = y_train_pred - y_train
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    y_test_pred = model.predict(X_test)
    res_test = y_test_pred - y_test
    r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    # create fig and ax objects
    fig = plt.figure(figsize=(12, 6))
    fig.canvas.set_window_title('Residuals Chart, {} dataset'.format(ds['name']))
    ax1 = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=3, fig=fig)
    ax2 = plt.subplot2grid((1, 4), (0, 3), rowspan=1, colspan=1, fig=fig, sharey=ax1)

    # plot residuals for all data
    ax1.plot(y_train_pred, res_train, 'o', label='Train $r^2$ = {:.3f}'.format(r2_train))
    ax1.plot(y_test_pred, res_test, 'o', label='Test $r^2$ = {:.3f}'.format(r2_test))

    # plot distribution of residuals
    bins = np.linspace(min(res_train.min(), res_test.min()), max(res_train.max(), res_test.max()), 50)
    htrain = plt.hist(x=res_train, bins=bins, orientation='horizontal', alpha=0.8)
    assert htrain[0].sum() == res_train.size
    htest = plt.hist(x=res_test, bins=bins, orientation='horizontal', alpha=0.8)
    assert htest[0].sum() == res_test.size

    # clean up residuals chart
    ax1.legend(loc='upper left', numpoints=3)
    format_axes('Predicted Value', 'Residual Value', 'Residuals for Linear Model', ax1)
    largefonts(14)
    fig.tight_layout()

plt.show()
