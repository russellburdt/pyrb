
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datasets
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
plt.style.use('bmh')


"""
improve classification accuracy with a clustering pre-processing step
"""

data = datasets.supervised_mammographic_mass()
X, y = data['X'].values, data['y'].values

# get baseline classification accuracy
classifiers = [GaussianNB,
               SVC,
               RandomForestClassifier,
               KNeighborsClassifier,
               LogisticRegression,
               DecisionTreeClassifier]
accuracy = []
for classifier in classifiers:
    classifier = classifier()
    accuracy.append(cross_val_score(classifier, X, y, cv=5).mean())

# determine best number of clusters with the elbow method
elbow = []
for n in range(1, 20):
    kmm = cluster.MiniBatchKMeans(n_clusters=n)
    kmm.fit(X)
    elbow.append(kmm.inertia_)
n_clusters = 5

# get cluster assignments based on optimal n_clusters in previous step
kmm = cluster.KMeans(n_clusters=n_clusters)
kmm.fit(X)
ycluster = kmm.predict(X)





# accuracy = []
# groups = list(algorithm_u_permutations(ns=np.unique(df['bq type cluster']), m=np.unique(df['bq type int']).size))
# for group in tqdm(groups, desc='scanning groups'):
#     tmp = np.full(df['bq type int'].size, np.nan)
#     for idx, items in enumerate(group):
#         tmp[df['bq type cluster'].isin(items)] = idx
#     accuracy.append(metrics.accuracy_score(df['bq type int'], tmp))




