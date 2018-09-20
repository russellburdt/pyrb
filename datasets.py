
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
     (categorical and numeric formats if applicable)
    * y0 - Series of original class data with name
    * X - DataFrame with all categorical data encoded to numeric data
    * y - Series with categorical data encoded to numeric
    * encoders - dict of all LabelEncoder objects used in X0 to X and y0 to y transformations

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
from sklearn.preprocessing import LabelEncoder
from ipdb import set_trace


def supervised_ecoli():
    """
    retrieve information from UCI Machine Learning Repository Ecoli dataset
    https://archive.ics.uci.edu/ml/datasets/Ecoli
    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
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

    # convert categorical data to numeric, save LabelEncoders in a dict
    out['encoders'] = {}

    le = LabelEncoder()
    le.fit(out['X0']['Sequence Name'])
    out['encoders']['X0, Sequence Name'] = le

    le = LabelEncoder()
    le.fit(out['y0'])
    out['encoders']['y0'] = le

    # create X, y data with only numeric data
    out['X'] = out['X0'].copy()
    out['X']['Sequence Name'] = out['encoders']['X0, Sequence Name'].transform(out['X0']['Sequence Name'])
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.Series(data=out['encoders']['y0'].transform(out['y0'].values), name=out['y0'].name)

    return out

def supervised_ionosphere():
    """
    retrieve information from UCI Machine Learning Repository Ionosphere dataset
    https://archive.ics.uci.edu/ml/datasets/Ionosphere
    """

    # initialize an output dictionary, assign a text description to the dataset
    out = {}
    out['text description'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data').text
    data = data.split('\n')[:-1]
    data = [x.split(',') for x in data]
    assert np.unique([len(x) for x in data]) == 35
    data = np.array(data)

    set_trace()

    X = data[:, :-1]
    y = data[:, -1]







