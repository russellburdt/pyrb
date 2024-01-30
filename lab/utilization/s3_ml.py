
"""
baseline and DL vehicle utilization prediction models
"""

import os
import lytx
import utils
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow import keras
from tensorflow.keras import layers
from ipdb import set_trace
plt.style.use('bmh')


# model datadir
datadir = r'/mnt/home/russell.burdt/data/utilization/amt'
assert os.path.isdir(datadir)

# metadata and population
dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
dp = pd.read_pickle(os.path.join(datadir, 'dp.p'))
dm = utils.get_population_metadata(dp, dc, datadir=datadir)
ds = pd.read_pickle(os.path.join(datadir, 'coverage', 'gps_segmentation_metrics.p'))
with open(os.path.join(datadir, 'datasets.p'), 'rb') as fid:
    data = pickle.load(fid)

# data for company
company = 'Beacon Transport LLC'
vids = data[company]['vids']
df = data[company]['data']
d0 = data.copy()
data = df.values.copy()

# train (first 50%), val (next 25%), and test (last 25%) indices
xtrain = np.arange(int(0.5 * data.shape[0]))
xval = np.arange(int(0.25 * data.shape[0])) + xtrain[-1] + 1
xtest = np.arange(xval[-1] + 1, data.shape[0])

# build DataFrame of sequences in train, val, and test sets
sequence_length = 30
dx = np.expand_dims(np.arange(data.shape[0]), axis=1)
targets = dx[sequence_length:, 0]
sequences = defaultdict(list)
kws = {'sequence_length': sequence_length, 'batch_size': 512, 'shuffle': False}
train = timeseries_dataset_from_array(data=dx, targets=targets, start_index=xtrain[0], end_index=xtrain[-1], **kws)
val = timeseries_dataset_from_array(data=dx, targets=targets, start_index=xval[0], end_index=xval[-1], **kws)
test = timeseries_dataset_from_array(data=dx, targets=targets, start_index=xtest[0], end_index=xtest[-1], **kws)
for dataset, desc in zip([train, val, test], ['train', 'val', 'test']):
    for t0, t1 in dataset:
        for x, row in enumerate(t0):
            sequences['seq start day'].append(df.index[row.numpy().flatten()][0])
            sequences['seq end day'].append(df.index[row.numpy().flatten()][-1])
            sequences['seq start index'].append(row.numpy().flatten()[0])
            sequences['seq end index'].append(row.numpy().flatten()[-1])
            sequences['target day'].append(df.index[t1[x].numpy()])
            sequences['target index'].append(t1[x].numpy())
            sequences['dataset'].append(desc)
sequences = pd.DataFrame(sequences)

# preprocess
mean = data[xtrain].mean(axis=0)
std = data[xtrain].std(axis=0)
data = (data - mean) / std

stest = sequences.loc[sequences['dataset'] == 'test'].reset_index(drop=True)

# initialize baseline metrics dict, scan over vehicles
baseline = {}
baseline['mae'] = np.full(data.shape[1], np.nan)
baseline['mse'] = np.full(data.shape[1], np.nan)
kws = {'sequence_length': sequence_length, 'batch_size': 512, 'shuffle': False}
for vx in tqdm(range(data.shape[1]), desc='baseline evaluation'):

    # timeseries dataset generators for vx
    targets = data[sequence_length:, vx]
    train = timeseries_dataset_from_array(data=data, targets=targets, start_index=xtrain[0], end_index=xtrain[-1], **kws)
    val = timeseries_dataset_from_array(data=data, targets=targets, start_index=xval[0], end_index=xval[-1], **kws)
    test = timeseries_dataset_from_array(data=data, targets=targets, start_index=xtest[0], end_index=xtest[-1], **kws)

    # scan over test dataset
    pred = np.array([])
    actual = np.array([])
    for t0, t1 in test:

        # prediction is last item in each sequence for target column (tc)
        pred = np.hstack((pred, t0[:, -1, vx].numpy() * std[vx] + mean[vx]))
        actual = np.hstack((actual, t1.numpy() * std[vx] + mean[vx]))

    if vx == 0:
        stest['actual'] = actual
        stest['baseline'] = pred

    # metrics for vx
    baseline['mae'][vx] = mean_absolute_error(y_true=actual, y_pred=pred)
    baseline['mse'][vx] = mean_squared_error(y_true=actual, y_pred=pred)

# dropout-regularized LSTM model
inputs = keras.Input(shape=(sequence_length, data.shape[-1]))
x = layers.LSTM(128, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# compile model
callbacks = [keras.callbacks.ModelCheckpoint(r'/mnt/home/russell.burdt/data/utilization/amt/model.keras', save_best_only=True)]
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# initialize DL metrics dict, scan over vehicles
dl = {}
dl['mae'] = np.full(data.shape[1], np.nan)
dl['mse'] = np.full(data.shape[1], np.nan)
kws = {'sequence_length': sequence_length, 'batch_size': 512}
for vx in tqdm(range(data.shape[1]), desc='DL evaluation'):

    # timeseries dataset generators for vx
    targets = data[sequence_length:, vx]
    train = timeseries_dataset_from_array(data=data, targets=targets, start_index=xtrain[0], end_index=xtrain[-1], shuffle=True, **kws)
    val = timeseries_dataset_from_array(data=data, targets=targets, start_index=xval[0], end_index=xval[-1], shuffle=True, **kws)
    test = timeseries_dataset_from_array(data=data, targets=targets, start_index=xtest[0], end_index=xtest[-1], shuffle=False, **kws)

    # fit model
    history = model.fit(train, epochs=40, verbose=1, validation_data=val, callbacks=callbacks)
    best_model = keras.models.load_model(r'/mnt/home/russell.burdt/data/utilization/amt/model.keras')

    # scan over test dataset
    pred = np.array([])
    actual = np.array([])
    for t0, t1 in test:

        # prediction is last item in each sequence for target column (tc)
        pred = np.hstack((pred, best_model.predict(t0, verbose=1).flatten() * std[vx] + mean[vx]))
        actual = np.hstack((actual, t1.numpy() * std[vx] + mean[vx]))

    if vx == 0:
        assert all(np.isclose(stest['actual'].values, actual))
        stest['dl'] = pred

    assert False

    # metrics for vx
    dl['mae'][vx] = mean_absolute_error(y_true=actual, y_pred=pred)
    dl['mse'][vx] = mean_squared_error(y_true=actual, y_pred=pred)

# train/val model loss curves
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# fig, ax = open_figure('train/val model loss curves', figsize=(10, 4))
# ax.plot(np.array(history.history['loss']), 'o-', ms=8, lw=2, label='Training MAE')
# ax.plot(np.array(history.history['val_loss']), 'x-', ms=8, lw=2, label='Validation MAE')
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3)
# format_axes('epoch index', 'Loss (MSE)', 'Training and Valiation Loss', ax)
# largefonts(14)
# fig.tight_layout()

# plt.show()
