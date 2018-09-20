
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from collections import Counter, defaultdict
from pyrb import open_figure, format_axes, largefonts, save_pngs
from pyrb.knuth import algorithm_u_permutations
from ipdb import set_trace
from tqdm import tqdm

from sklearn import cluster
from sklearn.externals.six import StringIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import datasets
plt.style.use('bmh')


"""
improve classification accuracy with a clustering pre-processing step
"""

# get valid sklearn classification datasets
valid = ['iris', 'digits', 'wine', 'breast_cancer']
df_datasets = defaultdict(list)
for ds in valid:
    data = getattr(datasets, 'load_' + ds)()
    df_datasets['name'].append(ds)
    df_datasets['Xshape'].append(data.data.shape)
    df_datasets['Xtype'].append(data.data.dtype)
    df_datasets['yshape'].append(data.target.shape)
    df_datasets['ytype'].append(data.target.dtype)
    df_datasets['# unique y'].append(np.unique(data.target).size)
    try:
        df_datasets['y names'].append(', '.join(data.target_names))
    except TypeError:
        df_datasets['y names'].append(', '.join(data.target_names.astype(np.str)))
df_datasets = pd.DataFrame(index=df_datasets.pop('name'), data=df_datasets)

# apply classifiers to each dataset
classifiers = [GaussianNB, SVC, RandomForestClassifier, KNeighborsClassifier, LogisticRegression]
for ds in valid:
    print('\n')
    data = getattr(datasets, 'load_' + ds)()
    X = data.data
    y = data.target
    for classifier in classifiers:

        classifier = classifier()
        scores = cross_val_score(classifier, X, y, cv=10)
        print('10-fold cross-validation accuracy, mean {:.2f}%, stdev {:.2f}%'.format(
            100 * scores.mean(), 100 * scores.std()))




# # make blob data
# centers = 4
# X, y = datasets.make_blobs(n_samples=20000, n_features=5, centers=centers, cluster_std=0.2)

# # map blog data to 2D space
# pca = PCA(n_components=2)
# pca.fit(X)
# X = pca.transform(X)

# # plot blob data
# fig, ax = open_figure('blobs', figsize=(8, 5))
# for center in range(centers):
#     idx = (y == center)
#     ax.plot(X[idx, 0], X[idx, 1], 'x', label='blob {}'.format(center))
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3)
# largefonts(14)
# fig.tight_layout()
# plt.show()

# # convert y into 'centers' or fewer classes
# yclass = np.empty(y.size, dtype=np.object)
# yclass[y == 0] = 'class 0 or 1'
# yclass[y == 1] = 'class 0 or 1'
# yclass[y == 2] = 'class 2 or 3'
# yclass[y == 3] = 'class 2 or 3'
# # yclass[y == 4] = 'class 4 or 5'
# # yclass[y == 5] = 'class 4 or 5'

# # convert yclass into encoded data
# le = LabelEncoder()
# le.fit(yclass)
# yclass = le.transform(yclass)

# # fit a decision tree to the results
# classifier = DecisionTreeClassifier(max_depth=3)
# scores = cross_val_score(classifier, X, y, cv=5)
# print('10-fold cross-validation accuracy, mean {:.2f}%, stdev {:.2f}%'.format(
#     100 * scores.mean(), 100 * scores.std()))
