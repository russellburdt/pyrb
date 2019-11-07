
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
from pyrb import largefonts, save_pngs, format_axes, open_figure
plt.style.use('bmh')


# load regression datasets
rbo = datasets.regression_boston()
rca = datasets.regression_california()
rdi = datasets.regression_diabetes()
rco = datasets.regression_concrete()
rde = datasets.regression_demand()
rtr = datasets.regression_traffic()
rrs = datasets.regression_random_sum(features=20, instances=1000)
with open(r'c:\Users\rburdt\Desktop\data.p', 'rb') as fid:
    rma = pickle.load(fid)

# test that residuals of regression follow a normal distribution
for ds in [rco]:

    # load dataset name and clean data
    name = ds['name']
    X = ds['X'].values
    y = ds['y'].values
    assert X.shape[0] == y.shape[0]

    # create train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # instantiate, run, and score a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    res_train = y_train_pred - y_train
    r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)
    y_test_pred = model.predict(X_test)
    res_test = y_test_pred - y_test
    r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)

    # create a residuals chart
    if True:
        title = 'Residuals Chart, {} dataset'.format(ds['name'])
        fig = plt.figure(figsize=(12, 6))
        fig.canvas.set_window_title(title)
        ax1 = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=3, fig=fig)
        ax2 = plt.subplot2grid((1, 4), (0, 3), rowspan=1, colspan=1, fig=fig, sharey=ax1)

        # plot residuals
        ax1.plot(y_train_pred, res_train, 'o', label='Train $r^2$ = {:.3f}'.format(r2_train))
        ax1.plot(y_test_pred, res_test, 'o', label='Test $r^2$ = {:.3f}'.format(r2_test))

        # plot distribution of residuals
        bins = np.linspace(min(res_train.min(), res_test.min()), max(res_train.max(), res_test.max()), 50)
        htrain = plt.hist(x=res_train, bins=bins, orientation='horizontal', alpha=0.8)
        assert htrain[0].sum() == res_train.size
        htest = plt.hist(x=res_test, bins=bins, orientation='horizontal', alpha=0.8)
        assert htest[0].sum() == res_test.size

        # clean up
        ax1.legend(loc='upper left', numpoints=3)
        format_axes('Predicted Value', 'Residual Value', 'Residuals Chart', ax1)
        largefonts(14)
        fig.tight_layout()

    # create a prediction error chart
    if True:
        title = 'Prediction Error Chart, {} dataset'.format(ds['name'])
        fig, ax = open_figure(title, figsize=(12, 6))

        # plot prediction error
        train = ax.plot(y_train, y_train_pred, 'o', label='Train $r^2$ = {:.3f}'.format(r2_train))[0]
        test = ax.plot(y_test, y_test_pred, 'o', label='Test $r^2$ = {:.3f}'.format(r2_test))[0]
        x = ax.get_xlim()
        ax.plot(x, np.polyval(np.polyfit(y_train, y_train_pred, deg=1), x), '-', lw=3, label='Train best fit', color=train.get_color())
        ax.plot(x, np.polyval(np.polyfit(y_test, y_test_pred, deg=1), x), '-', lw=3, label='Test best fit', color=test.get_color())
        ax.plot(x, x, '--', lw=3, label='identity', color='black')

        # clean up
        ax.set_xlim(x)
        ax.legend(loc='upper left', numpoints=3)
        format_axes('Actual target data', 'Predicted target data', 'Prediction Error Chart', ax)
        largefonts(14)
        fig.tight_layout()

    # create a feature correlation chart
    if True:
        pass

plt.show()
