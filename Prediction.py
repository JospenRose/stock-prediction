from tensorflow.keras.models import Sequential
from sklearn import metrics
import numpy as np
from math import sqrt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, Flatten, Dense, LSTM, GRU, \
    concatenate, Bidirectional


def error(actual, pred):
    # 'MSE', 'MAE', 'NMSE', 'RMSE'
    err=np.empty(4)
    err[0] = metrics.mean_squared_error(actual, pred)
    err[1] = metrics.mean_absolute_error(actual, pred)
    err[2] = metrics.mean_squared_log_error(actual, pred)
    rms = metrics.mean_squared_error(actual, pred)
    err[3] = sqrt(rms)

    return err


def proposed(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    input_shape = x_train[1].shape

    # Input layer
    inputs = Input(shape=input_shape)

    # Multi-Path 1D-CNN Feature Extraction
    cnn1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn1)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = Dropout(0.3)(cnn1)

    cnn2 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(cnn2)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = Dropout(0.3)(cnn2)

    cnn3 = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(inputs)
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = Conv1D(filters=128, kernel_size=7, padding='same', activation='relu')(cnn3)
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = Dropout(0.3)(cnn3)

    # Merge CNN paths
    merged_cnn = concatenate([cnn1, cnn2, cnn3])

    # Deep Sequential Prediction Model (DSPM) with LSTM (feed forward neural network) + GRU
    lstm = Bidirectional(LSTM(128, return_sequences=True))(merged_cnn)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(LSTM(64, return_sequences=True))(lstm)
    lstm = Dropout(0.3)(lstm)

    gru = GRU(128, return_sequences=True)(merged_cnn)
    gru = Dropout(0.3)(gru)
    gru = GRU(64, return_sequences=True)(gru)
    gru = Dropout(0.3)(gru)

    # Merge LSTM and GRU outputs
    merged_rnn = concatenate([lstm, gru])
    flattened = Flatten()(merged_rnn)

    # Fully Connected Layers
    fc = Dense(128, activation='relu')(flattened)
    fc = Dropout(0.3)(fc)
    fc = Dense(64, activation='relu')(fc)
    fc = Dropout(0.3)(fc)

    out = Dense(1, activation='linear')(fc)  # Single neuron for regression

    # Build Model
    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE as loss, MAE as additional metric
    # Summary of Model
    model.summary()

    history = model.fit(x_train, y_train, epochs=100, batch_size=25, validation_data=(x_test, y_test))

    predicted = model.predict(x_test).squeeze()

    met = error(y_test, predicted)

    return predicted, met, history


def lstm(X_train, y_train, X_test, y_test):

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    input_shape = X_train[1].shape

    # LSTM model for Stock Price Prediction
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=25, validation_data=(X_test, y_test))

    # Predict on test data
    y_pred = model.predict(X_test)

    met = error(y_test, y_pred)
    return y_pred, met


def cnn(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    input_shape = x_train[1].shape

    inputs = Input(shape=input_shape)

    cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.3)(cnn)

    flat = Flatten()(cnn)

    fc = Dense(64, activation='relu')(flat)

    out = Dense(1, activation='linear')(fc)

    # Build Model
    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(x_train, y_train, epochs=100, batch_size=25, validation_data=(x_test, y_test))

    predicted = model.predict(x_test)

    met = error(y_test, predicted)

    return predicted, met


def dnn(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    input_shape = x_train[1].shape
    inputs = Input(shape=input_shape)
    flattened = Flatten()(inputs)
    # Fully Connected Layers (Deep Neural Network)
    fc = Dense(256, activation='relu')(flattened)
    fc = Dense(128, activation='relu')(fc)
    fc = Dense(64, activation='relu')(fc)

    out = Dense(1, activation='linear')(fc)

    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(x_train, y_train, epochs=100, batch_size=25, validation_data=(x_test, y_test))

    predicted = model.predict(x_test)

    met = error(y_test, predicted)

    return predicted, met, history


def gru(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    input_shape = x_train[1].shape

    inputs = Input(shape=input_shape)

    # GRU Layers
    gru1 = GRU(128, return_sequences=True)(inputs)
    gru2 = GRU(64, return_sequences=True)(gru1)
    gru3 = GRU(32)(gru2)

    flat = Flatten()(gru3)

    fc = Dense(128, activation='relu')(flat)
    fc = Dense(64, activation='relu')(fc)

    output = Dense(1, activation='linear')(fc)  # Regression

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(x_train, y_train, epochs=100, batch_size=25, validation_data=(x_test, y_test))

    predicted = model.predict(x_test)

    met = error(y_test, predicted)

    return predicted, met

