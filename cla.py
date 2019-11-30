
"""
classification toy problems and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pyrb import open_figure, format_axes, largefonts
from pyrb import datasets
from ipdb import set_trace
from tqdm import tqdm

# plotting
plt.style.use('bmh')
figsize = (12, 6)
size = 12

# load classification datasets
cad = datasets.supervised_adult(size=3000, random_state=0)
css = datasets.supervised_skin_segmentation()
cbn = datasets.supervised_banknote()
cio = datasets.supervised_ionosphere()
cbl = datasets.supervised_blobs()

# model hyper-parameters
MLM = SVC
mlm = {'gamma': 'auto', 'C': 1.0, 'kernel': 'rbf'}

# scan over datasets
for ds in [cbl]:

    # create a learning curve chart - uses many trained MLMs
    if True:

        # load data
        name = ds['name']
        X = ds['X'].values
        y = ds['y'].values
        assert X.shape[0] == y.shape[0]

        # generate data for learning curve
        model = MLM(**mlm)
        n_splits = 3                                # cross-val splits
        sample_fracs = np.linspace(0.2, 1, 10)      # fractions of full dataset to use in learning curve
        scores = {}                                 # dictionary for model scores
        for sample_frac in tqdm(sample_fracs, desc='scanning dataset fractions'):

            # get Xs and ys and subsets of X and y defined by 'sample_frac' samples
            n_samples = int(X.shape[0] * sample_frac)
            idx_n_samples = np.random.choice(range(X.shape[0]), size=n_samples, replace=False)
            Xs = X[idx_n_samples, :]
            ys = y[idx_n_samples]
            scores[sample_frac] = {}
            scores[sample_frac]['train'] = []
            scores[sample_frac]['test'] = []

            # initialize a KFold iterator and scan over folds
            kfold = KFold(n_splits=n_splits)
            for train_index, test_index in kfold.split(Xs):
                Xs_train, Xs_test = Xs[train_index], Xs[test_index]
                ys_train, ys_test = ys[train_index], ys[test_index]
                model.fit(Xs_train, ys_train)
                scores[sample_frac]['train'].append(accuracy_score(y_true=ys_train, y_pred=model.predict(Xs_train)))
                scores[sample_frac]['test'].append(accuracy_score(y_true=ys_test, y_pred=model.predict(Xs_test)))

        # extract leaning curve data as numpy arrays
        x = np.array(sorted(scores.keys()))
        y_train_mean = np.array([np.mean(scores[xi]['train']) for xi in x])
        y_train_min = np.array([np.min(scores[xi]['train']) for xi in x])
        y_train_max = np.array([np.max(scores[xi]['train']) for xi in x])
        y_test_mean = np.array([np.mean(scores[xi]['test']) for xi in x])
        y_test_min = np.array([np.min(scores[xi]['test']) for xi in x])
        y_test_max = np.array([np.max(scores[xi]['test']) for xi in x])

        # plot learning curve data
        fig, ax = open_figure('Classification Accuracy Learning Curve Chart, {}'.format(name), figsize=figsize)
        train = ax.fill_between(x=np.hstack((x, x[::-1])), y1=np.hstack((y_train_min, y_train_max[::-1])), alpha=0.5, label='train data bounds')
        test = ax.fill_between(x=np.hstack((x, x[::-1])), y1=np.hstack((y_test_min, y_test_max[::-1])), alpha=0.5, label='test data bounds')
        ax.plot(x, y_train_mean, '.-', color=train.get_facecolor()[0], ms=12, lw=3, label='train data mean')[0]
        ax.plot(x, y_test_mean, '.-', color=test.get_facecolor()[0], ms=12, lw=3, label='test data mean')[0]
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3)
        ax.set_ylim(0, 1)
        format_axes('Fraction of original dataset', 'accuracy score, %', '{}-fold cross-validation learning curve'.format(n_splits), ax)
        largefonts(size)
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)

    # plot linear SVC decision boundary for two input variables of binary classification problems
    # i.e. always uses SVC(kernel='linear') and y must only have two classes
    if True:

        # define dimensions for chart
        dims = ['x1', 'x2']

        # load data
        name = ds['name']
        X = ds['X'].values
        y = ds['y'].values
        assert X.shape[0] == y.shape[0]
        assert pd.unique(y).size == 2
        assert all([x in ds['X'].columns for x in dims])
        d0 = np.where(ds['X'].columns == dims[0])[0][0]
        d1 = np.where(ds['X'].columns == dims[1])[0][0]

        # create chart and add raw data
        fig, ax = open_figure('Linear SVC decision boundaries, {}'.format(name), 1, 2, figsize=figsize)
        for yi in pd.unique(y):
            data0 = X[y == yi][:, d0]
            data1 = X[y == yi][:, d1]
            ax[0].plot(data0, data1, 'o', label='class - {}'.format(y))
        format_axes(dims[0], dims[1], 'raw data', ax[0])

        # fit model to data
        X = X[:, [d0, d1]].astype(np.float)
        svm = SVC(kernel='linear')
        svm.fit(X, y)

        # plot X mapped to SVM decision function values
        for x in svm.support_vectors_:
            assert x in X
        cmap = ax[1].scatter(X[:, 0], X[:, 1], c=svm.decision_function(X), cmap=plt.cm.inferno)
        cbar = plt.colorbar(cmap, ax=ax[1])
        cbar.ax.yaxis.set_label_position('left')
        format_axes(dims[0], dims[1], 'SVM decision function values', ax[1])

        # plot contour lines representing the decision boundary and margin on each axes
        xx = np.linspace(X[:, 0].min(), X[:, 0].max(), num=100)
        yy = np.linspace(X[:, 1].min(), X[:, 1].max(), num=100)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = svm.decision_function(xy).reshape(XX.shape)
        ax[0].contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'], linewidths=[2, 2, 2])
        ax[1].contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'], linewidths=[2, 2, 2])
        largefonts(size)
        fig.tight_layout()
