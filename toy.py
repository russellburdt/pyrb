
"""
multiple regression toy problem and visualizations
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pyrb import datasets
from pyrb import largefonts, save_pngs, format_axes, open_figure
from ipdb import set_trace
plt.style.use('bmh')
size = 14


# load regression datasets
rbo = datasets.regression_boston()
rca = datasets.regression_california()
rdi = datasets.regression_diabetes()
rco = datasets.regression_concrete()
rde = datasets.regression_demand()
rtr = datasets.regression_traffic()
rrs = datasets.regression_random_sum(features=10, instances=600)
with open(r'c:\Users\rburdt\Desktop\data.p', 'rb') as fid:
    rma = pickle.load(fid)

# test that residuals of regression follow a normal distribution
for ds in [rma]:

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
        largefonts(size)
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
        largefonts(size)
        fig.tight_layout()

    # create a feature correlation chart
    if True:
        df = ds['X'].copy()
        df[ds['y'].name] = ds['y'].values
        corr = np.flipud(df.corr().values)
        n = corr.shape[0]
        cols = list(df.columns)
        cols[-1] = cols[-1] + '\n(target)'

        title = 'Feature Correlation Chart, {} dataset'.format(ds['name'])
        fig = plt.figure(figsize=(12, 6))
        fig.canvas.set_window_title(title)
        axm = plt.subplot2grid(shape=(1, 10), loc=(0, 0), rowspan=1, colspan=9)
        axb = plt.subplot2grid(shape=(1, 10), loc=(0, 9), rowspan=1, colspan=1)
        cmap = axm.pcolor(corr, cmap=plt.cm.seismic, vmin=-1, vmax=1)
        cbar = fig.colorbar(cmap, cax=axb)

        axm.fill_between(x=[0, n, n, n - 1, n - 1, 0], y1=[0, 0, n, n, 1, 1], fc='None', ec='black', lw=4)

        ticks = np.arange(0.5, n, 1)
        axm.set_xlim(-0.1, n + 0.1)
        axm.set_ylim(-0.1, n + 0.1)
        axm.set_xticks(ticks)
        axm.set_yticks(ticks)
        axm.set_xticklabels(cols, rotation=90)
        axm.set_yticklabels(cols[::-1])
        axm.set_title('Regression Feature Correlation Matrix')
        for xi, x in enumerate(ticks):
            for yyi, yy in enumerate(ticks):
                axm.text(x, yy, s='{:.2f}'.format(corr[yyi, xi]), fontweight='bold', fontsize=10, ha='center', va='center')
        largefonts(size)
        fig.tight_layout()

    # create a learning curve chart
    if True:

        # generate data for learning curve
        model_lc = LinearRegression()
        sample_fracs = np.linspace(0.2, 1, 20)      # fractions of full dataset to use in learning curve
        n_splits = 3                                # number of cross-validation splits to use for each fraction
        r2_scores = {}                              # dictionary for r2 scores
        for sample_frac in sample_fracs:

            # get Xs and ys and subsets of X and y defined by 'sample_frac' samples
            n_samples = int(X.shape[0] * sample_frac)
            idx_n_samples = np.random.choice(range(X.shape[0]), size=n_samples, replace=False)
            Xs = X[idx_n_samples, :]
            ys = y[idx_n_samples]
            r2_scores[sample_frac] = {}
            r2_scores[sample_frac]['train'] = []
            r2_scores[sample_frac]['test'] = []

            # initialize a KFold iterator and scan over folds
            kfold = KFold(n_splits=n_splits)
            for train_index, test_index in kfold.split(Xs):
                Xs_train, Xs_test = Xs[train_index], Xs[test_index]
                ys_train, ys_test = ys[train_index], ys[test_index]
                model_lc.fit(Xs_train, ys_train)
                r2_scores[sample_frac]['train'].append(r2_score(y_true=ys_train, y_pred=model_lc.predict(Xs_train)))
                r2_scores[sample_frac]['test'].append(r2_score(y_true=ys_test, y_pred=model_lc.predict(Xs_test)))

        # extract leaning curve data as numpy arrays
        x = np.array(sorted(r2_scores.keys()))
        y_train_mean = np.array([np.mean(r2_scores[xi]['train']) for xi in x])
        y_train_min = np.array([np.min(r2_scores[xi]['train']) for xi in x])
        y_train_max = np.array([np.max(r2_scores[xi]['train']) for xi in x])
        y_test_mean = np.array([np.mean(r2_scores[xi]['test']) for xi in x])
        y_test_min = np.array([np.min(r2_scores[xi]['test']) for xi in x])
        y_test_max = np.array([np.max(r2_scores[xi]['test']) for xi in x])

        # plot learning curve data
        fig, ax = open_figure('Regression Learning Curve Chart', figsize=(12, 6))
        train = ax.fill_between(x=np.hstack((x, x[::-1])), y1=np.hstack((y_train_min, y_train_max[::-1])), alpha=0.5, label='train data bounds')
        test = ax.fill_between(x=np.hstack((x, x[::-1])), y1=np.hstack((y_test_min, y_test_max[::-1])), alpha=0.5, label='test data bounds')
        ax.plot(x, y_train_mean, '.-', color=train.get_facecolor()[0], ms=12, lw=3, label='train data mean')[0]
        ax.plot(x, y_test_mean, '.-', color=test.get_facecolor()[0], ms=12, lw=3, label='test data mean')[0]
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3)
        format_axes('Fraction of original dataset', 'r2 score, %', '{}-fold cross-validation learning curve'.format(n_splits), ax)
        largefonts(size)
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)

    # create a model coefficient chart
    if True:

        ab = {a: b for a, b in zip(ds['X'].columns, model.coef_)}
        ab['intercept'] = model.intercept_
        ab = pd.Series(data=ab)

        # title = 'Regression Model Coefficient Chart, {} dataset'.format(ds['name'])
        # fig, ax = open_figure(title, figsize=(12, 6))

plt.show()
