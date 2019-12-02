
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

--- supervised and regression datasets return a dictionary of
    * desc - description of data from source location
    * X0 - DataFrame of original feature data and column names
    * y0 - Series of original class data with name
    (X0 and y0 represent unclean, original data - categorical, numeric, missing, etc. all as in original source)
    * X - DataFrame with all data cleaning steps applied
    * y - Series with all data cleaning steps applied
    (see code to understand cleaning steps)
    * encoders - dict of any LabelEncoder objects used in data cleaning

--- unsupervised datasets return ...

--- use 'size' (absolute units) and 'random_state' keyword arguments to return a fraction of the dataset

Current data sources are
    1) scikit-learn built-in datasets
    2) UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/index.php
    3) yellowbrick (districtdatalabs channel of conda)
    4) misc URLs

Author - Russell Burdt
"""

import requests
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from zipfile import ZipFile
from itertools import chain
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from yellowbrick import datasets as ydatasets
from ipdb import set_trace


def regression_random_sum(features=10, instances=200, target_percentage_min=0.8, target_percentage_max=1.2):
    """
    return artificial X, y data for regression with shape set by keyword arguments
    - each row of X is random numbers that exactly sum to the equivalent row of y
    """

    # create artifical X, y data for regression
    y = np.random.rand(instances)
    X = np.full((instances, features), np.nan)
    for idx in range(instances):
        # create data for features that randomly sums to class data
        frac = target_percentage_min + np.random.rand() * (target_percentage_max - target_percentage_min)
        total = frac * y[idx]
        data = []
        for _ in range(features):
            data.append(np.random.rand() * total)
            total -= data[-1]
        data[-1] += total
        data = np.array(data)
        # data = np.random.rand(features)
        np.random.shuffle(data)
        assert np.isclose(sum(data), frac * y[idx])
        X[idx, :] = data

    # convert to pandas objects
    X = pd.DataFrame(data=X, columns=['f{}'.format(x) for x in range(features)])
    y = pd.Series(data=y, name='class data')

    # convert to expected output structure
    out = {}
    out['name'] = 'RANDOM SUM'
    out['desc'] = 'each row of X sums randomly to each row of y'
    out['X0'] = X.copy()
    out['y0'] = y.copy()
    out['X'] = X.copy()
    out['y'] = y.copy()

    return out

def regression_boston():
    """
    retrieve information from scikit-learn built-in dataset
    sklearn.datasets.load_boston
    """

    # load sklearn dataset
    data = datasets.load_boston()

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'BOSTON'
    out['desc'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=data.feature_names)
    out['y0'] = pd.Series(data=data.target, name='home price')

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def regression_california():
    """
    retrieve information from scikit-learn built-in dataset
    sklearn.datasets.california_housing
    """

    # load sklearn dataset
    data = datasets.california_housing.fetch_california_housing()

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'CALIFORNIA'
    out['desc'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=data.feature_names)
    out['y0'] = pd.Series(data=data.target, name='home price')

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def regression_diabetes():
    """
    retrieve information from scikit-learn built-in dataset
    sklearn.datasets.load_diabetes
    """

    # load sklearn dataset
    data = datasets.load_diabetes()

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'DIABETES'
    out['desc'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=data.feature_names)
    out['y0'] = pd.Series(data=data.target, name='One Year Disease Progression')

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def regression_concrete():
    """
    retrieve information from yellowbrick built in dataset
    yellowbrick.datasets.load_concrete
    """

    # load yellowbrick dataset
    data = ydatasets.load_concrete()

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'CONCRETE'
    out['desc'] = ydatasets.load_concrete.__doc__

    # load training and class data into DataFrames
    out['X0'] = data[0]
    out['y0'] = data[1]

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def regression_demand():
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/Daily+Demand+Forecasting+Orders
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'DEMAND'
    out['desc'] = """
        Data Set Information:
        The database was collected during 60 days, this is a real database of a Brazilian company of large logistics. Twelve predictive attributes and a target that is the total of orders for daily. treatment

        Attribute Information:
        The dataset was collected during 60 days, this is a real database of a brazilian logistics company. The dataset has twelve predictive attributes and a target that is the total of orders for daily treatment.
        The database was used in academic research at the Universidade Nove de Julho.
        .arff header for Weka:
        @relation Daily_Demand_Forecasting_Orders
        @attribute Week_of_the_month {1.0, 2.0, 3.0, 4.0, 5.0}
        @attribute Day_of_the_week_(Monday_to_Friday) {2.0, 3.0, 4.0, 5.0, 6.0}
        @attribute Non_urgent_order integer
        @attribute Urgent_order integer
        @attribute Order_type_A integer
        @attribute Order_type_B integer
        @attribute Order_type_C integer
        @attribute Fiscal_sector_orders integer
        @attribute Orders_from_the_traffic_controller_sector integer
        @attribute Banking_orders_(1) integer
        @attribute Banking_orders_(2) integer
        @attribute Banking_orders_(3) integer
        @attribute Target_(Total_orders) integer
        @data
        """

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00409/Daily_Demand_Forecasting_Orders.csv').text
    data = [x.strip() for x in data.split('\n')]
    hdr = data[0].split(';')
    data = data[1:-1]
    assert len(data) == 60
    data = np.array([np.array([float(xi) for xi in x.split(';')]) for x in data])
    assert data.shape[0] == 60
    assert data.shape[1] == len(hdr)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=hdr[:-1])
    out['y0'] = pd.Series(data=y, name=hdr[-1])

    # create clean X, y data
    cdict = {
        'WoM': 'Week of the month (first week, second, third, fourth or fifth week',
        'DoW': 'Day of the week (Monday to Friday)',
        'NUO': 'Non-urgent order',
        'UO': 'Urgent order',
        'OTA': 'Order type A',
        'OTB': 'Order type B',
        'OTC': 'Order type C',
        'FSO': 'Fiscal sector orders',
        'OTCS': 'Orders from the traffic controller sector',
        'BO1': 'Banking orders (1)',
        'BO2': 'Banking orders (2)',
        'BO3': 'Banking orders (3)'}
    out['X'] = out['X0'].copy().rename(columns={b: a for a, b in cdict.items()})
    out['y'] = out['y0'].copy().rename('Orders')
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def regression_traffic():
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'TRAFFIC'
    out['desc'] = """
        Data Set Information:
        The database was created with records of behavior of the urban traffic of the city of Sao Paulo in Brazil from December 14, 2009 to December 18, 2009 (From Monday to Friday). Registered from 7:00 to 20:00 every 30 minutes. The data set Behavior of the urban traffic of the city of Sao Paulo in Brazil was used in academic research at the Universidade Nove de Julho - Postgraduate Program in Informatics and Knowledge Management.

        Attribute Information:
        1. Hour
        2. Immobilized bus
        3. Broken Truck
        4. Vehicle excess
        5. Accident victim
        6. Running over
        7. Fire Vehicles
        8. Occurrence involving freight
        9. Incident involving dangerous freight
        10. Lack of electricity
        11. Fire
        12. Point of flooding
        13. Manifestations
        14. Defect in the network of trolleybuses
        15. Tree on the road
        16. Semaphore off
        17. Intermittent Semaphore
        18. Slowness in traffic (%) (Target)
        """

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00483/Behavior%20of%20the%20urban%20traffic%20of%20the%20city%20of%20Sao%20Paulo%20in%20Brazil.zip')
    data = ZipFile(BytesIO(data.content))
    csv = [x for x in data.namelist() if '.csv' in x]
    assert len(csv) == 1
    with data.open(csv[0]) as fid:
        data = fid.readlines()
    data = [x.decode().strip() for x in data]
    hdr = data[0]
    hdr = hdr.split(';')
    data = data[1:]
    assert len(data) == 135
    data = [x.split(';') for x in data]
    data = [x[:-1] + ['.'.join(x[-1].split(','))] for x in data]    # floats use ',' in Brazil instead of '.'
    data = np.array([np.array([float(xi) for xi in x]) for x in data])
    assert data.shape[0] == 135
    assert data.shape[1] == len(hdr)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=hdr[:-1])
    out['y0'] = pd.Series(data=y, name=hdr[-1])

    # create clean X, y data
    cdict = {
        'HR': 'Hour (Coded)',
        'IB': 'Immobilized bus',
        'BT': 'Broken Truck',
        'VE': 'Vehicle excess',
        'AV': 'Accident victim',
        'RO': 'Running over',
        'FV': 'Fire vehicles',
        'OIF': 'Occurrence involving freight',
        'IIDF': 'Incident involving dangerous freight',
        'LOE': 'Lack of electricity',
        'FIRE': 'Fire',
        'POF': 'Point of flooding',
        'MANS': 'Manifestations',
        'DNT': 'Defect in the network of trolleybuses',
        'TOR': 'Tree on the road',
        'SO': 'Semaphore off',
        'IS': 'Intermittent Semaphore'}
    out['X'] = out['X0'].copy().rename(columns={b: a for a, b in cdict.items()})
    out['y'] = out['y0'].copy().rename('Orders')
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def regression_advertising():
    """
    retrieve information from Regression Advertising dataset
    http://faculty.marshall.usc.edu/gareth-james/ISL/Advertising.csv
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'ADVERTISING'
    out['desc'] = """
        regression dataset supporting this useful blog:
        https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
        """

    # read data from url; parse to X, y numpy arrays
    data = pd.read_csv(r'http://faculty.marshall.usc.edu/gareth-james/ISL/Advertising.csv')
    data = data[['TV', 'radio', 'newspaper', 'sales']]

    # load raw training and class data
    out['X0'] = data[['TV', 'radio', 'newspaper']]
    out['y0'] = data['sales']

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def supervised_banknote():
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'BANKNOTE'
    out['desc'] = \
        'Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.'

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt').text
    data = data.split('\n')
    data = [x.split(',') for x in data]
    data = [[x.strip() for x in y] for y in data]
    assert np.unique([len(x) for x in data]) == 5
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=['wavelet{}'.format(x) for x in range(4)])
    out['y0'] = pd.Series(data=y, name='banknote authenticity')

    # create clean X, y data
    out['X'] = out['X0'].copy()
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.to_numeric(out['y0'].copy())
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_adult(size=None, random_state=None):
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/Adult
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'ADULT'
    out['desc'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data').text
    data = data.split('\n')[:-2]
    data = [x.split(',') for x in data]
    data = [[x.strip() for x in y] for y in data]
    assert np.unique([len(x) for x in data]) == 15
    data = np.array(data)
    if size is not None:
        if random_state is not None:
            np.random.seed(random_state)
        x = np.random.choice(range(data.shape[0]), size, replace=False)
        data = data[x, :]
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    columns = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    out['X0'] = pd.DataFrame(data=X, columns=columns)
    out['y0'] = pd.Series(data=y, name='income, more or less than 50k')

    # create LabelEncoder objects, save to out
    encoders = []
    encoder_columns = []
    for col, cdata in out['X0'].iteritems():
        try:
            pd.to_numeric(cdata)
        except ValueError:
            le = LabelEncoder()
            le.fit(cdata)
            encoders.append(le)
            encoder_columns.append(col)
    le = LabelEncoder()
    le.fit(out['y0'])

    # create clean X, y data
    out['X'] = out['X0'].copy()
    for col, encoder in zip(encoder_columns, encoders):
        out['X'][col] = encoder.transform(out['X'][col])
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.Series(data=le.transform(out['y0'].values), name=out['y0'].name)
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_skin_segmentation(size=None, random_state=None):
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'SKIN SEGMENTATION'
    out['desc'] = 'This dataset is of the dimension 245057 * 4 where first three columns are B,G,R (x1,x2, and x3 features) values and fourth column is of the class labels (decision variable y).'

    # read data from url; parse to X, y numpy arrays
    fn = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    data = requests.get(fn).text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 4
    data = np.array(data)
    if size is not None:
        if random_state is not None:
            np.random.seed(random_state)
        x = np.random.choice(range(data.shape[0]), size, replace=False)
        data = data[x, :]
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=['B', 'G', 'R'])
    out['y0'] = pd.Series(data=y, name='skin region')

    # create clean X, y data
    out['X'] = out['X0'].copy()
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.to_numeric(out['y0'].copy())
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_yeast():
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/Yeast
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'YEAST'
    out['desc'] = \
        requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.names').text

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data').text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 10
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    columns = ['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
    out['X0'] = pd.DataFrame(data=X, columns=columns)
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

def supervised_wireless_localization():
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'WIRELESS LOCALIZATION'
    out['desc'] = 'Each attribute is wifi signal strength observed on smartphone'

    # read data from url; parse to X, y numpy arrays
    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt').text
    data = data.split('\n')[:-1]
    data = [x.split() for x in data]
    assert np.unique([len(x) for x in data]) == 8
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]

    # original data, including all original numeric and categorical values
    out['X0'] = pd.DataFrame(data=X, columns=['signal{}'.format(x) for x in range(1, 8)])
    out['y0'] = pd.Series(data=y, name='location index')

    # create clean X, y data
    out['X'] = out['X0'].copy()
    out['X'] = out['X'].apply(pd.to_numeric)
    out['y'] = pd.to_numeric(out['y0'].copy())
    assert pd.isnull(out['X'].values).sum() == 0
    assert pd.isnull(out['y']).sum() == 0

    return out

def supervised_ecoli():
    """
    retrieve information from UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/Ecoli
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'ECOLI'
    out['desc'] = \
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

def supervised_mammographic_mass():
    """
    retrieve information from UCI Machine Learning Mammographic Mass dataset
    https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'MAMMOGRAPHIC MASS'
    out['desc'] = \
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

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'IONOSPHERE'
    out['desc'] = \
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

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'CAR EVALUATION'
    out['desc'] = \
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
    data = datasets.load_digits()

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'DIGITS'
    out['desc'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=['pixel {}'.format(x) for x in range(64)])
    out['y0'] = pd.Series(data=data.target, name='digit in range 0-9')

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def supervised_blobs(n_samples=1000, n_features=2, centers=2, center_box=(-10, 10), cluster_std=2):
    """
    retrieve information from scikit-learn built-in dataset generator
    sklearn.datasets.make_blobs
    """

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'BLOBS'
    out['desc'] = datasets.make_blobs.__doc__

    # create blobs dataset
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, center_box=center_box, cluster_std=cluster_std)
    out['X0'] = pd.DataFrame(data=X, columns=['x{}'.format(x) for x in range(1, n_features + 1)])
    out['y0'] = pd.Series(data=y, name='centers')

    # create clean X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out

def supervised_iris():
    """
    retrieve information from scikit-learn built-in dataset
    sklearn.datasets.load_iris
    """

    # load sklearn dataset
    data = datasets.load_iris()

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'IRIS'
    out['desc'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=data.feature_names)
    out['y0'] = pd.Series(data=data.target, name='iris type')

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
    data = datasets.load_wine()

    # initialize an output dictionary, assign a desc to the dataset
    out = {}
    out['name'] = 'WINE'
    out['desc'] = data.DESCR

    # load training and class data into DataFrames
    out['X0'] = pd.DataFrame(data=data.data, columns=data.feature_names)
    out['y0'] = pd.Series(data=data.target, name='wine type')

    # create X, y data - no data cleaning required in this case
    assert pd.isnull(out['X0'].values).sum() == 0
    assert pd.isnull(out['y0']).sum() == 0
    out['X'] = out['X0'].copy()
    out['y'] = out['y0'].copy()

    return out
