import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from save_load import load
import joblib


def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learn Rate(%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learn Rate(%)',
            kind='bar',
            stacked=False)

    plt.xticks(rotation=0, fontweight='bold', fontname='Serif')
    plt.yticks(fontweight='bold', fontname='Serif')
    plt.xlabel('Learn Rate %', fontweight='bold', fontname='Serif')
    plt.legend(loc='center', prop={'weight': 'bold', 'family': 'Serif', 'size': 10})
    plt.title(metric, fontweight='bold', fontname='Serif')

    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)


def plotres():

    # learning rate -  70  and 30
    cnn_70 = load('cnn_70')
    dnn_70 = load('dnn_70')
    gru_70 = load('gru_70')
    lstm_70 = load('lstm_70')
    proposed_70 = load('proposed_70')

    data1 = {
        'CNN': cnn_70,
        'GRU': gru_70,
        'DNN': dnn_70,
        'LSTM': lstm_70,
        'PROPOSED': proposed_70
    }

    ind = ['MSE', 'MAE', 'NMSE', 'RMSE']
    table = pd.DataFrame(data1, index=ind)
    print('---------- Metrics for 70 training 30 testing ----------')
    print(table)

    table.to_excel('./Results/table_70.xlsx')

    val1 = np.array(table)

    # learning rate -  80  and 20

    cnn_80 = load('cnn_80')
    dnn_80 = load('dnn_80')
    gru_80 = load('gru_80')
    lstm_80 = load('lstm_80')
    proposed_80 = load('proposed_80')

    data2 = {
        'CNN': cnn_80,
        'GRU': gru_80,
        'DNN': dnn_80,
        'LSTM': lstm_80,
        'PROPOSED': proposed_80
    }

    ind = ['MSE', 'MAE', 'NMSE', 'RMSE']
    table1 = pd.DataFrame(data2, index=ind)
    print('---------- Metrics for 80 training 20 testing ----------')
    print(table1)

    val2 = np.array(table1)
    table1.to_excel('./Results/table_80.xlsx')

    metrices = [val1, val2]

    mthod = ['CNN', 'GRU', 'DNN', 'LSTM', 'PROPOSED']
    metrices_plot = ['MSE', 'MAE', 'NMSE', 'RMSE']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, metrices[0][i, :], metrices[1][i, :], metrices_plot[i])

    for k in [70, 80]:
        y_test = load(f'y_test_{k}')
        y_pred = load(f'predicted_{k}')

        scaler = joblib.load('Saved Data/Label Scaler.joblib')
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label="Actual Prices", color='blue', linewidth=2)
        plt.plot(y_pred_inv, label="Predicted Prices", color='red', linestyle='dashed', linewidth=2)
        plt.xlabel("Time", fontweight='bold', fontfamily='Serif')
        plt.ylabel("Stock Price", fontweight='bold', fontfamily='Serif')
        plt.xticks(fontweight="bold", fontfamily="Serif")
        plt.yticks(fontweight="bold", fontfamily="Serif")
        plt.title("Actual vs Predicted Stock Prices", fontweight='bold', fontfamily='Serif')
        plt.legend(prop={'weight': 'bold', 'family': 'Serif', 'size': 10})
        plt.savefig(f'Results/Actual vs Predicted Stock Prices_{k}.png')
        plt.show()

