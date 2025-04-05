import os

import os
os.makedirs('Dataset', exist_ok=True)
os.makedirs('Data visualization', exist_ok=True)
os.makedirs('Saved Data', exist_ok=True)
os.makedirs('Results', exist_ok=True)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datagen import datagen
from save_load import *
from Prediction import lstm, proposed, cnn, dnn, gru
from plot_result import plotres
import matplotlib.pyplot as plt
from feature_selection import WSGOA
from Objective_func import fit_func_70, fit_func_80
import numpy as np


def full_analysis():
    datagen()

    # 70 training, 30 testing

    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')

    # 80 training, 20 testing

    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    training_data = [(x_train_70, y_train_70, x_test_70, y_test_70, fit_func_70), (x_train_80, y_train_80, x_test_80, y_test_80, fit_func_80)]

    i = 70

    for train_data in training_data:
        X_train, y_train, X_test, y_test, fit_func = train_data

        # Feature Selection

        lb = np.zeros(X_train.shape[1])
        ub = np.ones(X_train.shape[1])

        pop_size = 10
        prob_size = len(lb)

        max_iter = 500
        best_solution, best_fitness = WSGOA(fit_func, prob_size, pop_size, max_iter, lb, ub)

        soln = np.round(best_solution)
        selected_indices = np.where(soln == 1)[0]
        if len(selected_indices) == 0:
            selected_indices = np.where(soln == 0)[0]

        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

        pred, met, history = proposed(X_train, y_train, X_test, y_test)
        save(f'proposed_{i}', met)
        save(f'predicted_{i}', pred)

        plt.figure(figsize=(10, 4))

        # Plot MAE (Mean Absolute Error)
        plt.subplot(121)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error (MAE)', fontweight='bold', fontname='Serif')
        plt.xlabel('Epoch', fontweight='bold', fontname='Serif')
        plt.ylabel('MAE', fontweight='bold', fontname='Serif')
        plt.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Serif'})

        # Plot Loss
        plt.subplot(122)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss', fontweight='bold', fontname='Serif')
        plt.xlabel('Epoch', fontweight='bold', fontname='Serif')
        plt.ylabel('Loss', fontweight='bold', fontname='Serif')
        plt.legend(loc='upper right', prop={'weight': 'bold', 'family': 'Serif'})

        plt.tight_layout()
        plt.savefig(f'Results/MAE Loss Graph Learning rate {i}.png')
        plt.show()

        pred, met = lstm(X_train, y_train, X_test, y_test)
        save(f'lstm_{i}', met)

        pred, met = cnn(X_train, y_train, X_test, y_test)
        save(f'cnn_{i}', met)

        pred, met, history = dnn(X_train, y_train, X_test, y_test)
        save(f'dnn_{i}', met)

        pred, met = gru(X_train, y_train, X_test, y_test)
        save(f'gru_{i}', met)

        i += 10


a = 0
if a == 1:
    full_analysis()

plotres()
plt.show()



