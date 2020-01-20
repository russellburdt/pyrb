
"""
multiple regression toy problem and visualizations
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, normaltest, jarque_bera
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.linear_model import LinearRegression as MLM
# from sklearn.linear_model import Ridge as MLM
# from sklearn.svm import LinearSVR as MLM
# from sklearn.linear_model import ElasticNet as MLM
# from sklearn.linear_model import Lasso as MLM
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from pyrb import datasets
from pyrb import largefonts, save_pngs, format_axes, open_figure
from pyrb import get_bounds_of_data_within_interval
from ipdb import set_trace

# model hyper-parameters and cross-validation parameters
plt.style.use('bmh')
size = 14
figsize = (14, 8)
mlm = {}
test_size = 0.4

# load regression datasets
rbo = datasets.regression_boston()
rca = datasets.regression_california()
rdi = datasets.regression_diabetes()
rco = datasets.regression_concrete()
rde = datasets.regression_demand()
rtr = datasets.regression_traffic()
rad = datasets.regression_advertising()
rrs = datasets.regression_random_sum(features=10, instances=600)
with open(r'c:\Users\rburdt\Desktop\data.p', 'rb') as fid:
    rma = pickle.load(fid)

# test that residuals of regression follow a normal distribution
for ds in [rma]:

    # create a target correlation chart - does not use any MLM
    if False:
        title = 'Target Correlation Chart, {} dataset'.format(ds['name'])
        fig, ax = open_figure(title, 1, ds['X'].shape[1], sharey=True, figsize=(16, 4))
        for idx, col in enumerate(ds['X'].iteritems()):
            ax[idx].plot(col[1].values, ds['y'].values, 'o')
            format_axes(col[0], '', 'corrcoef - {:.2f}'.format(pearsonr(col[1].values, ds['y'].values)[0]), ax[idx])
        largefonts(2)
        fig.tight_layout()

    # create a feature correlation chart - does not use any MLM
    if True:
        df = ds['X'].copy()
        # pca = PCA()
        # pca.fit(df.values)
        # df = pd.DataFrame(data=pca.transform(df.values), columns=df.columns)
        df[ds['y'].name] = ds['y'].values
        corr = np.flipud(df.corr().values)
        n = corr.shape[0]
        cols = list(df.columns)
        cols[-1] = cols[-1] + '\n(target)'

        title = 'Feature Correlation Chart, {} dataset'.format(ds['name'])
        fig = plt.figure(figsize=figsize)
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
        cols = [x.replace('total hours of ', '') for x in cols]
        axm.set_xticklabels(cols, rotation=90)
        axm.set_yticklabels(cols[::-1])
        axm.set_title('Correlation Matrix for Component Duration Metrics')
        for xi, x in enumerate(ticks):
            for yyi, yy in enumerate(ticks):
                axm.text(x, yy, s='{:.2f}'.format(corr[yyi, xi]), fontweight='bold', fontsize=10, ha='center', va='center')
        largefonts(size)
        fig.tight_layout()

    # create a residuals chart - uses many trained MLMs
    if False:

        # set up residual chart
        N = 100
        title = 'Residuals Chart, {} dataset'.format(ds['name'])
        fig = plt.figure(figsize=figsize)
        fig.canvas.set_window_title(title)
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=1, colspan=3, fig=fig)
        ax2 = plt.subplot2grid((2, 4), (0, 3), rowspan=1, colspan=1, fig=fig, sharey=ax1)
        ax3 = plt.subplot2grid((2, 4), (1, 0), rowspan=1, colspan=4, fig=fig)

        for idx in range(N):

            # train and score MLM
            model = MLM(**mlm)
            X_train, X_test, y_train, y_test = train_test_split(ds['X'].values, ds['y'].values, test_size=test_size)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            res_train = y_train_pred - y_train
            res_test = y_test_pred - y_test

            # plot detailed residuals and distributions on first iteration only
            if idx == 0:
                train_label = 'Train $r^2$ = {:.3f}\np-value = {:.2g}'.format(r2_score(y_true=y_train, y_pred=y_train_pred), normaltest(res_train).pvalue)
                test_label = 'Test $r^2$ = {:.3f}\np-value = {:.2g}'.format(r2_score(y_true=y_test, y_pred=y_test_pred), normaltest(res_test).pvalue)
                ax1.plot(y_train_pred, res_train, 'o', label=train_label)
                p = ax1.plot(y_test_pred, res_test, 'o', label=test_label)[0]
                bins = np.linspace(min(res_train.min(), res_test.min()), max(res_train.max(), res_test.max()), 50)
                htrain = ax2.hist(x=res_train, bins=bins, orientation='horizontal', alpha=0.8, label=train_label)
                assert htrain[0].sum() == res_train.size
                htest = ax2.hist(x=res_test, bins=bins, orientation='horizontal', alpha=0.8, label=test_label)
                assert htest[0].sum() == res_test.size
                ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot test residuals on every iteration
            interval = get_bounds_of_data_within_interval(res_test, x=0.95)
            ax3.errorbar(idx, sum(interval) / 2, yerr=np.diff(interval) / 2, lw=2, capsize=6, elinewidth=2, markeredgewidth=1, color=p.get_color())

        # clean up
        format_axes('Predicted Value', 'Residual Value', 'Cross validation train ({:.0f}%) and test ({:.0f}%) residuals for 1 model'.format(100*(1-test_size), 100*test_size), ax1)
        format_axes('bin count', '', 'distributions')
        format_axes('Model iteration', 'Residual Value', 'Range with 95% of test residuals for {} models'.format(N), ax3)
        largefonts(size)
        fig.tight_layout()

    # create a prediction error chart - uses 1 trained MLM
    if False:

        # train and score MLM
        model = MLM(**mlm)
        X_train, X_test, y_train, y_test = train_test_split(ds['X'].values, ds['y'].values, test_size=test_size)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # set up chart
        title = 'Prediction Error Chart, {} dataset'.format(ds['name'])
        fig, ax = open_figure(title, figsize=figsize)

        # plot prediction error
        train = ax.plot(y_train, y_train_pred, 'o', label='Train $r^2$ = {:.3f}'.format(r2_score(y_true=y_train, y_pred=y_train_pred)))[0]
        test = ax.plot(y_test, y_test_pred, 'o', label='Test $r^2$ = {:.3f}'.format(r2_score(y_true=y_test, y_pred=y_test_pred)))[0]
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

    # create a learning curve chart - uses many trained MLMs
    if False:

        # generate data for learning curve
        model = MLM(**mlm)
        sample_fracs = np.linspace(0.2, 1, 20)      # fractions of full dataset to use in learning curve
        n_splits = 3                                # number of cross-validation splits to use for each fraction
        r2_scores = {}                              # dictionary for r2 scores
        for sample_frac in sample_fracs:

            # get Xs and ys and subsets of X and y defined by 'sample_frac' samples
            n_samples = int(ds['X'].shape[0] * sample_frac)
            idx_n_samples = np.random.choice(range(ds['X'].shape[0]), size=n_samples, replace=False)
            Xs = ds['X'].values[idx_n_samples, :]
            ys = ds['y'].values[idx_n_samples]
            r2_scores[sample_frac] = {}
            r2_scores[sample_frac]['train'] = []
            r2_scores[sample_frac]['test'] = []

            # initialize a KFold iterator and scan over folds
            kfold = KFold(n_splits=n_splits)
            for train_index, test_index in kfold.split(Xs):
                Xs_train, Xs_test = Xs[train_index], Xs[test_index]
                ys_train, ys_test = ys[train_index], ys[test_index]
                model.fit(Xs_train, ys_train)
                r2_scores[sample_frac]['train'].append(r2_score(y_true=ys_train, y_pred=model.predict(Xs_train)))
                r2_scores[sample_frac]['test'].append(r2_score(y_true=ys_test, y_pred=model.predict(Xs_test)))

        # extract leaning curve data as numpy arrays
        x = np.array(sorted(r2_scores.keys()))
        y_train_mean = np.array([np.mean(r2_scores[xi]['train']) for xi in x])
        y_train_min = np.array([np.min(r2_scores[xi]['train']) for xi in x])
        y_train_max = np.array([np.max(r2_scores[xi]['train']) for xi in x])
        y_test_mean = np.array([np.mean(r2_scores[xi]['test']) for xi in x])
        y_test_min = np.array([np.min(r2_scores[xi]['test']) for xi in x])
        y_test_max = np.array([np.max(r2_scores[xi]['test']) for xi in x])

        # plot learning curve data
        fig, ax = open_figure('Regression Learning Curve Chart', figsize=figsize)
        train = ax.fill_between(x=np.hstack((x, x[::-1])), y1=np.hstack((y_train_min, y_train_max[::-1])), alpha=0.5, label='train data bounds')
        test = ax.fill_between(x=np.hstack((x, x[::-1])), y1=np.hstack((y_test_min, y_test_max[::-1])), alpha=0.5, label='test data bounds')
        ax.plot(x, y_train_mean, '.-', color=train.get_facecolor()[0], ms=12, lw=3, label='train data mean')[0]
        ax.plot(x, y_test_mean, '.-', color=test.get_facecolor()[0], ms=12, lw=3, label='test data mean')[0]
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3)
        ax.set_ylim(0, 1)
        format_axes('Fraction of original dataset', 'r2 score, %', '{}-fold cross-validation learning curve'.format(n_splits), ax)
        largefonts(size)
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)

    # create a model coefficient chart - uses many trained MLMs
    if False:

        # extract model coefficients for repeated runs
        N = 1000                            # number of times to run the model and identify coefficient ranges
        frac = 0.8                          # fraction of full dataset to use for training in each model
        model = MLM(**mlm)
        coefs = []
        intercepts = []
        for _ in range(N):
            n_samples = int(ds['X'].shape[0] * frac)
            idx_n_samples = np.random.choice(range(ds['X'].shape[0]), size=n_samples, replace=False)
            Xs = ds['X'].values[idx_n_samples, :]
            ys = ds['y'].values[idx_n_samples]
            model.fit(Xs, ys)
            coefs.append(model.coef_)
            intercepts.append(model.intercept_)
        coefs, intercepts = np.array(coefs).T, np.array(intercepts)
        cols = np.array(ds['X'].columns)
        assert cols.size == ds['X'].shape[1]

        # calculate coef and intercept metrics
        interval = 0.95
        intercepts_interval = get_bounds_of_data_within_interval(intercepts, x=0.95)
        coefs_intervals = np.array([get_bounds_of_data_within_interval(x, x=0.95) for x in coefs])

        # plot results
        title = 'Regression Model Coefficient Chart, {} dataset'.format(ds['name'])
        fig = plt.figure(title, figsize=figsize)
        fig.canvas.set_window_title(title)
        ax1 = plt.subplot2grid(shape=(5, 1), loc=(0, 0), rowspan=4, colspan=1)
        ax2 = plt.subplot2grid(shape=(5, 1), loc=(4, 0), rowspan=1, colspan=1)

        y = np.arange(cols.size)
        p = ax1.plot(np.mean(coefs, axis=1), y, 'o', label='mean coefficient')[0]
        for yi, yinterval in zip(y, coefs_intervals):
            x = ax1.errorbar(sum(yinterval) / 2, yi, xerr=np.diff(yinterval) / 2, lw=2, capsize=6, elinewidth=2, markeredgewidth=1, color=p.get_color())
            if yi == 0:
                x.set_label('interval with 95% of coefficients')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=1)
        ax1.set_yticks(y)
        ax1.set_yticklabels(cols)
        ax1.set_ylim(-0.5, y.size - 0.5)
        format_axes('', '', 'Coefficient ranges for {} models, each trained with random {:.0f}% of instances'.format(N, frac * 100), ax1)

        ax2.plot(np.mean(intercepts), 1, 'o', color=p.get_color())
        ax2.errorbar(sum(intercepts_interval) / 2, 1, xerr=np.diff(intercepts_interval) / 2, lw=2, capsize=6, elinewidth=2, markeredgewidth=1, color=p.get_color())
        ax2.set_ylim(0, 2)
        ax2.set_yticks([1])
        ax2.set_yticklabels(['intercept'])
        format_axes('', '', '', ax2)

        largefonts(size)
        fig.tight_layout()

    # statsmodel implementation on full dataset
    if False:
        endog = ds['y']
        exog = pd.DataFrame(data=np.hstack((ds['X'].values, np.expand_dims(np.ones(ds['X'].shape[0]), axis=1))), columns=list(ds['X'].columns) + ['intercept'])
        ols = sm.OLS(endog=endog, exog=exog)
        model = ols.fit()
        print(model.summary())

plt.show()
