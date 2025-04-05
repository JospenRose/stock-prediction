from save_load import load
import numpy as np
from Prediction import proposed


def fit_func_70(x):
    x_train = load('x_train_70')
    y_train = load('y_train_70')
    x_test = load('x_test_70')
    y_test = load('y_test_70')

    soln = np.round(x)
    selected_indices = np.where(soln == 1)[0]
    if len(selected_indices) == 0:
        selected_indices = np.where(soln == 0)[0]

    x_train = x_train[:, selected_indices]
    x_test = x_test[:, selected_indices]

    pred, met, history = proposed(x_train, y_train, x_test, y_test)

    fit = met[0]

    return fit


def fit_func_80(x):
    x_train = load('x_train_80')
    y_train = load('y_train_80')
    x_test = load('x_test_80')
    y_test = load('y_test_80')

    soln = np.round(x)
    selected_indices = np.where(soln == 1)[0]
    if len(selected_indices) == 0:
        selected_indices = np.where(soln == 0)[0]

    x_train = x_train[:, selected_indices]
    x_test = x_test[:, selected_indices]

    pred, met, history = proposed(x_train, y_train, x_test, y_test)

    fit = met[0]

    return fit

