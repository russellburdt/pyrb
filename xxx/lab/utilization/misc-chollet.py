
"""
timeseries analysis
- data at https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
- meant to be weather related data at a single location 1/1/2009 to 1/1/2017 at 10 min intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrb.mpl import open_figure, format_axes, largefonts
from datetime import datetime
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from ipdb import set_trace
plt.style.use('bmh')

# read and validate data, extract numpy arrays for ML
# df = pd.read_csv(r'/mnt/home/russell.burdt/data/jena_climate_2009_2016.csv')
df = pd.read_csv(r'c:/Users/russell.burdt/Downloads/jena_climate_2009_2016.csv')
df.index = [pd.Timestamp(datetime.strptime(x, r'%d.%m.%Y %H:%M:%S')) for x in df.pop('Date Time').values]
df = df[~df.duplicated()].sort_index()
# df['index'] = range(df.shape[0])
minutes = pd.value_counts(((1e-9) * np.diff(df.index) / 60).astype('int'))
assert all(minutes.index >= 10)
print(f'{100 * minutes[10] / (df.shape[0] - 1):.3f}% of intervals are 10 min, others all greater than 10 min')
temp = df['T (degC)'].values
# temp = df['index'].values
data = df.values.copy()

# train (first 50%), val (next 25%), and test (last 25%) indices
xtrain = np.arange(int(0.5 * data.shape[0]))
xval = np.arange(int(0.25 * data.shape[0])) + xtrain[-1] + 1
xtest = np.arange(xval[-1] + 1, data.shape[0])

# preprocess
mean = data[xtrain].mean(axis=0)
std = data[xtrain].std(axis=0)
# mean = np.zeros(data.shape[1])
# std = np.ones(data.shape[1])
data = (data - mean) / std

# consistent timeseries dataset parameters for train, test, val datasets
kws = {
    # timeseries data sampled once per hour
    'sampling_rate': 6,
    # observations go back 5 days, ie 120 hours
    'sequence_length': 120,
    # batch size
    'batch_size': 256,
    # shuffle
    'shuffle': False}
# target is 24 hours after end of sequence
delay = kws['sampling_rate'] * (kws['sequence_length'] + 24 - 1)

# timeseries dataset generators
train = timeseries_dataset_from_array(data=data[:-delay], targets=temp[delay:], end_index=xtrain[-1], **kws)
val = timeseries_dataset_from_array(data=data[:-delay], targets=temp[delay:], start_index=xval[0], end_index=xval[-1], **kws)
test = timeseries_dataset_from_array(data=data[:-delay], targets=temp[delay:], start_index=xtest[0], **kws)

# baseline model - always predict same temp 24 hours in advance
def baseline_model_mae(dataset):
    total_err = 0
    total_samples = 0
    for t0, t1 in dataset:
        # prediction is last temperature (column 1) in each sequence
        prediction = t0[:, -1, 1].numpy() * std[1] + mean[1]
        actual = t1.numpy()
        total_err += np.sum(np.abs(prediction - actual))
        total_samples += t0.shape[0]
    return total_err / total_samples
for dataset, desc in zip([train, val, test], ['train', 'val', 'test']):
    print(f'mean absolute error of baseline model for {desc} dataset, {baseline_model_mae(dataset):.2f}')

# fully connected NN model
# inputs = keras.Input(shape=(kws['sequence_length'], data.shape[-1]))
# x = layers.Flatten()(inputs)
# x = layers.Dense(16, activation='relu')(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)

# 1d convolutional model
# inputs = keras.Input(shape=(kws['sequence_length'], data.shape[-1]))
# x = layers.Conv1D(8, 24, activation="relu")(inputs)
# x = layers.MaxPooling1D(2)(x)
# x = layers.Conv1D(8, 12, activation="relu")(x)
# x = layers.MaxPooling1D(2)(x)
# x = layers.Conv1D(8, 6, activation="relu")(x)
# x = layers.GlobalAveragePooling1D()(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)

# simple LSTM
inputs = keras.Input(shape=(kws['sequence_length'], data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# dropout-regularized LSTM model
# inputs = keras.Input(shape=(kws['sequence_length'], data.shape[-1]))
# x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)

# dropout-regularized, stacked GRU model
# inputs = keras.Input(shape=(kws['sequence_length'], data.shape[-1]))
# x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
# x = layers.GRU(32, recurrent_dropout=0.5)(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)

# compile and fit
callbacks = [keras.callbacks.ModelCheckpoint(r'c:/Users/russell.burdt/Downloads/model.keras', save_best_only=True)]
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(train, epochs=10, validation_data=val, callbacks=callbacks)

# mae for test set
print(f'final model mae on test set, {model.evaluate(test)[1]:.2f}')
print(f"""best model mae on test set, {keras.models.load_model(r'c:/Users/russell.burdt/Downloads/model.keras').evaluate(test)[1]:.2f}""")

# train/val model loss curves
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
fig, ax = open_figure('train/val model loss curves', figsize=(10, 4))
ax.plot(history.history['mae'], 'o-', ms=8, lw=2, label='Training MAE')
ax.plot(history.history['val_mae'], 'x-', ms=8, lw=2, label='Validation MAE')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3)
format_axes('epoch index', 'Mean Absolute Error, deg C', 'Training and Valiation MAE', ax)
largefonts(14)
fig.tight_layout()

plt.show()
