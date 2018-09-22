
"""
Retrieve data from machine learning dataset repositories in a consistent structure

Three types of datasets are supported
    - supervised
    - regression
    - unsupervised

Each function name begins with one of the three types of supported datasets,
followed by the name of the dataset.  For example, 'supervised_ecoli' represents
the ecoli dataset for a supervised machine learning problem.  Each type of supported
dataset returns the following consistent information.

--- supervised datasets return a dictionary of
    * text description - description of data from source location
    * X0 - DataFrame of original feature data and column names
    * y0 - Series of original class data with name
    (X0 and y0 represent unclean, original data - categorical, numeric, missing, etc. all as in original source)
    * X - DataFrame with all data cleaning steps applied
    * y - Series with all data cleaning steps applied
    (see code to understand cleaning steps)
    * encoders - dict of any LabelEncoder objects used in data cleaning

--- regression datasets return ...

--- unsupervised datasets return ...

Current data sources are
    1) scikit-learn built-in datasets
    2) UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/index.php

R Burdt, v0.01
20 Sep 2018
"""

import requests
import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_wine, load_digits
from ipdb import set_trace


def supervised_blank():
    """
    retrieve information from UCI Machine Learning Repository Ecoli dataset

    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = ''
    out['text description'] = \
        requests.get(r'').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'').text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 9999999999
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=[])
    out['y0'] = pd.Series(data=y, name='')

    # create LabelEncoder objects, save to out
    # le1 = LabelEncoder()
    # le1.fit(out['X0']['Sequence Name'])
    # le2 = LabelEncoder()
    # le2.fit(out['y0'])
    # out['encoders'] = {}
    # out['encoders']['X0, Sequence Name'] = le1
    # out['encoders']['y0'] = le2

    # create clean X, y data
    # out['X'] = out['X0'].copy()
    # out['X']['Sequence Name'] = le1.transform(out['X']['Sequence Name'])
    # out['X'] = out['X'].apply(pd.to_numeric)
    # out['y'] = pd.Series(data=le2.transform(out['y0'].values), name=out['y0'].name)
    # assert pd.isnull(out['X'].values).sum() == 0
    # assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_a():
    """
    retrieve information from UCI Machine Learning Repository Ecoli dataset

    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = ''
    out['text description'] = \
        requests.get(r'').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'').text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 9999999999
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=[])
    out['y0'] = pd.Series(data=y, name='')

    # create LabelEncoder objects, save to out
    # le1 = LabelEncoder()
    # le1.fit(out['X0']['Sequence Name'])
    # le2 = LabelEncoder()
    # le2.fit(out['y0'])
    # out['encoders'] = {}
    # out['encoders']['X0, Sequence Name'] = le1
    # out['encoders']['y0'] = le2

    # create clean X, y data
    # out['X'] = out['X0'].copy()
    # out['X']['Sequence Name'] = le1.transform(out['X']['Sequence Name'])
    # out['X'] = out['X'].apply(pd.to_numeric)
    # out['y'] = pd.Series(data=le2.transform(out['y0'].values), name=out['y0'].name)
    # assert pd.isnull(out['X'].values).sum() == 0
    # assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_b():
    """
    retrieve information from UCI Machine Learning Repository Ecoli dataset

    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = ''
    out['text description'] = \
        requests.get(r'').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'').text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 9999999999
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=[])
    out['y0'] = pd.Series(data=y, name='')

    # create LabelEncoder objects, save to out
    # le1 = LabelEncoder()
    # le1.fit(out['X0']['Sequence Name'])
    # le2 = LabelEncoder()
    # le2.fit(out['y0'])
    # out['encoders'] = {}
    # out['encoders']['X0, Sequence Name'] = le1
    # out['encoders']['y0'] = le2

    # create clean X, y data
    # out['X'] = out['X0'].copy()
    # out['X']['Sequence Name'] = le1.transform(out['X']['Sequence Name'])
    # out['X'] = out['X'].apply(pd.to_numeric)
    # out['y'] = pd.Series(data=le2.transform(out['y0'].values), name=out['y0'].name)
    # assert pd.isnull(out['X'].values).sum() == 0
    # assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_c():
    """
    retrieve information from UCI Machine Learning Repository Ecoli dataset

    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = ''
    out['text description'] = \
        requests.get(r'').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'').text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 9999999999
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=[])
    out['y0'] = pd.Series(data=y, name='')

    # create LabelEncoder objects, save to out
    # le1 = LabelEncoder()
    # le1.fit(out['X0']['Sequence Name'])
    # le2 = LabelEncoder()
    # le2.fit(out['y0'])
    # out['encoders'] = {}
    # out['encoders']['X0, Sequence Name'] = le1
    # out['encoders']['y0'] = le2

    # create clean X, y data
    # out['X'] = out['X0'].copy()
    # out['X']['Sequence Name'] = le1.transform(out['X']['Sequence Name'])
    # out['X'] = out['X'].apply(pd.to_numeric)
    # out['y'] = pd.Series(data=le2.transform(out['y0'].values), name=out['y0'].name)
    # assert pd.isnull(out['X'].values).sum() == 0
    # assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_ecoli():
    """
    retrieve information from UCI Machine Learning Repository Ecoli dataset
    https://archive.ics.uci.edu/ml/datasets/Ecoli
    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = 'ECOLI'
    out['text description'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data').text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 9
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'])
    out['y0'] = pd.Series(data=y, name='localization site')

    # create LabelEncoder objects, save to out
    le1 = LabelEncoder()
    le1.fit(out['X0']['Sequence Name'])
    le2 = LabelEncoder()
    le2.fit(out['y0'])
    out['encoders'] = {}
    out['encoders']['X0, Sequence Name'] = le1
    out['encoders']['y0'] = le2

    # create clean X, y data
    out['X'] = out['X0'].copy()
    out['X']['Sequence Name'] = le1.transform(out['X']['Sequence Name'])
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.Series(data=le2.transform(out['y0'].values), name=out['y0'].name)
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_lung_cancer():
    """
    retrieve information from UCI Machine Learning Lung Cancer dataset
    https://archive.ics.uci.edu/ml/datasets/Lung+Cancer
    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = 'LUNG CANCER'
    out['text description'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data').text
    data = data.split('\n')[:-2]
    data = [x.split(',') for x in data]
    assert np.unique([len(x) for x in data]) == 57
    data = np.array(data)
    X = data[:, 1:]
    y = data[:, 0]

    # original data, including all original numeric and categorical values
    # (there are no feature labels for this dataset)
    out['X0'] = pd.DataFrame(data=X, columns=['feature{}'.format(x) for x in range(56)])
    out['y0'] = pd.Series(data=y, name='type of lung cancer')

    # create clean X, y data
    out['X'] = out['X0'].copy()
    out['X'][out['X'] == '?'] = -1
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.to_numeric(out['y0'].copy())
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_mammographic_mass():
    """
    retrieve information from UCI Machine Learning Mammographic Mass dataset
    https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass
    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = 'MAMMOGRAPHIC MASS'
    out['text description'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data').text
    data = data.split('\n')[:-2]
    data = [x.split(',') for x in data]
    data = [[x.strip() for x in y] for y in data]
    assert np.unique([len(x) for x in data]) == 6
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=['bi-rads', 'age', 'shape', 'margin', 'density'])
    out['y0'] = pd.Series(data=y, name='benign or malignant')

    # create clean X, y data
    # this dataset has mostly a missing value problem
    out['X'] = out['X0'].copy()
    for name, col in out['X'].iteritems():
        cavg = np.array(col)
        cavg = cavg[cavg != '?'].astype(np.float).mean()
        out['X'].loc[col == '?', name] = cavg
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.to_numeric(out['y0'].copy())
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_ionosphere():
    """
    retrieve information from UCI Machine Learning Repository Ionosphere dataset
    https://archive.ics.uci.edu/ml/datasets/Ionosphere
    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = 'IONOSPHERE'
    out['text description'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data').text
    data = data.split('\n')[:-1]
    data = [x.split(',') for x in data]
    assert np.unique([len(x) for x in data]) == 35
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    columns = list(chain(*[['pulse {} real'.format(x), 'pulse {} imag'.format(x)] for x in range(17)]))
    out['X0'] = pd.DataFrame(data=X, columns=columns)
    out['y0'] = pd.Series(data=y, name='reading good or bad')

    # convert categorical data to numeric, save LabelEncoders in a dict
    le = LabelEncoder()
    le.fit(out['y0'])
    out['encoders'] = {}
    out['encoders']['y0'] = le

    # create X, y data with only numeric data
    out['X'] = out['X0'].copy()
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.Series(data=out['encoders']['y0'].transform(out['y0'].values), name=out['y0'].name)
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_car_evaluation():
    """
    retrieve information from UCI Machine Learning Repository Car Evaluation dataset
    https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = 'CAR EVALUATION'
    out['text description'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data').text
    data = data.split('\n')[:-1]
    data = [x.split(',') for x in data]
    assert np.unique([len(x) for x in data]) == 7
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    out['X0'] = pd.DataFrame(data=X, columns=columns)
    out['y0'] = pd.Series(data=y, name='reading good or bad')

    # this dataset requires LabelEncoders for all features and for class data
    le_features = [LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()]
    le = LabelEncoder()
    out['encoders'] = {}
    for le_feature, column in zip(le_features, columns):
        le_feature.fit(out['X0'][column])
        out['encoders']['X0, {}'.format(column)] = le_feature
    le.fit(out['y0'])
    out['encoders']['y0'] = le

    # create clean X, y data
    out['X'] = out['X0'].copy()
    for le_feature, column in zip(le_features, columns):
        out['X'][column] = le_feature.transform(out['X0'][column])
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.Series(data=le.transform(out['y0'].values), name=out['y0'].name)
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_digits():
    """
    retrieve information from scikit-learn built-in dataset
    sklearn.datasets.load_digits
    """

    # load sklearn dataset
    data = load_digits()

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = 'DIGITS'
    out['text description'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=['pixel {}'.format(x) for x in range(64)])
    out['y0'] = pd.Series(data=data.target, name='digit in range 0-9')

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def supervised_wine():
    """
    retrieve information from scikit-learn built-in dataset
    sklearn.datasets.load_digits
    """

    # load sklearn dataset
    data = load_wine()

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['name'] = 'WINE'
    out['text description'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=data.feature_names)
    out['y0'] = pd.Series(data=data.target, name='wine type')

    # create X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out
