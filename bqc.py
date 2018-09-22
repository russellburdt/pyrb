
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

data = datasets.supervised_ecoli()
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
    algo = classifier()
    acc = cross_val_score(algo, X, y, cv=5).mean()
    accuracy.append(acc)
    print('{} accuracy, {:.1f}%'.format(classifier.__name__, 100 * acc))
print('max classification accuracy, {:.1f}%'.format(100 * max(accuracy)))

# determine best number of clusters with the elbow method
elbow = []
for n in range(1, 20):
    kmm = cluster.KMeans(n_clusters=n)
    kmm.fit(X)
    elbow.append(kmm.inertia_)
plt.plot(elbow, 'o-'); plt.show()
assert False
n_clusters = 5

# get cluster assignments based on optimal n_clusters in previous step
scores = []
models = []
nruns = 100
for _ in tqdm(range(nruns), 'clustering'):

  kmm = cluster.MiniBatchKMeans(n_clusters=n_clusters)
  # kmm = cluster.KMeans(n_clusters=n_clusters)
  # kmm = cluster.Birch(n_clusters=n_clusters)
  kmm.fit(X)
  ycluster = kmm.predict(X)

  # find optimal mapping of ycluster to y for best 'clustering accuracy'
  accuracy = []
  groups = list(algorithm_u_permutations(ns=np.unique(ycluster), m=np.unique(y).size))
  for group in tqdm(groups, desc='scanning groups'):
      tmp = np.full(y.size, np.nan)
      for idx, items in enumerate(group):
          tmp[np.in1d(ycluster, items)] = idx
      accuracy.append(metrics.accuracy_score(y, tmp))
  # group = groups[np.argmax(accuracy)]
  scores.append(max(accuracy))
  models.append(kmm)

print('best {}-cluster accuracy, {:.1f}%'.format(n_clusters, 100 * max(accuracy)))
