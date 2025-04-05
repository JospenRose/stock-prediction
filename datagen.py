import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from feature_extraction import *
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from save_load import save
import os
import joblib



def load_dataset():
    # Apple Inc. (AAPL)
    # Amazon (AMZN)
    # Google (Alphabet) (GOOGL)
    # Microsoft (MSFT)
    # Meta (Facebook) (META)
    # NVIDIA (NVDA)
    # Tesla (TSLA)

    tech_list = ['AMZN', 'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META', 'TSLA']  # Add more tickers as needed
    start = datetime(2020, 1, 1)
    end = datetime(2025, 1, 1)

    company_data = []

    # Loop through each company ticker in the list
    for ticker in tech_list:
        # Download the stock data for each ticker
        data = yf.download(ticker, start=start, end=end)

        # Flatten the MultiIndex columns and rename them
        data.columns = [f'{col[0]}_{ticker}' for col in data.columns]
        data.reset_index(inplace=True)

        data['Ticker'] = ticker

        # Reorganize columns: Date, Close, High, Low, Open, Volume, Ticker
        data = data[
            ['Date', f'Close_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Open_{ticker}', f'Volume_{ticker}',
             'Ticker']]

        data.rename(columns={
            f'Close_{ticker}': 'Close',
            f'High_{ticker}': 'High',
            f'Low_{ticker}': 'Low',
            f'Open_{ticker}': 'Open',
            f'Volume_{ticker}': 'Volume'
        }, inplace=True)

        company_data.append(data)

    final_data = pd.concat(company_data, ignore_index=True)
    final_data.to_excel('Dataset/stock market.xlsx')
    print(final_data.head(25))


def datagen():
    Dataset_path = 'Dataset/stock market.xlsx'

    if not os.path.exists(Dataset_path):
        print('Dataset does not exist')
        load_dataset()

    else:
        data = pd.read_excel('Dataset/stock market.xlsx')

        # Convert 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'])

        # Drop unnecessary columns
        data = data.drop(columns=['Unnamed: 0'])

        # Set Seaborn style
        sns.set_style("darkgrid")

        # Get unique tickers
        tickers = data['Ticker'].unique()

        # Plot separate graphs for Closing Price and Trading Volume
        for ticker in tickers:
            subset = data[data['Ticker'] == ticker]

            # # Plot Closing Price
            # plt.figure(figsize=(8, 4))
            # plt.plot(subset['Date'], subset['Close'], label=f'{ticker} Closing Price', marker='o', markersize=2)
            # plt.xlabel('Date', fontweight='bold', fontfamily='Serif')
            # plt.ylabel('Closing Price', fontweight='bold', fontfamily='Serif')
            # plt.title(f'Stock Closing Prices for {ticker}', fontweight='bold', fontfamily='Serif')
            # plt.legend( prop={'weight': 'bold', 'family': 'Serif'})
            # plt.xticks(fontweight='bold', fontfamily='Serif')
            # plt.yticks(fontweight='bold', fontfamily='Serif')
            # plt.grid(True)
            # plt.savefig(f'Data visualization/Closing Prices for {ticker}.png')
            # plt.show()
            #
            # # Plot Trading Volume
            # plt.figure(figsize=(8, 4))
            # plt.plot(subset['Date'], subset['Volume'], label=f'{ticker} Trading Volume', marker='o', color='red', markersize=2)
            # plt.xlabel('Date', fontweight='bold', fontfamily='Serif')
            # plt.ylabel('Trading Volume', fontweight='bold', fontfamily='Serif')
            # plt.title(f'Trading Volume for {ticker}', fontweight='bold', fontfamily='Serif')
            # plt.legend( prop={'weight': 'bold', 'family': 'Serif'})
            # plt.xticks(fontweight='bold', fontfamily='Serif')
            # plt.yticks(fontweight='bold', fontfamily='Serif')
            # plt.grid(True)
            # plt.savefig(f'Data visualization/Trading Volume for {ticker}.png')
            # plt.show()
        #
        # # Candlestick-like visualization for each ticker
        #
        # for ticker in tickers:
        #     stock_data = data[data["Ticker"] == ticker]
        #
        #     plt.figure(figsize=(8, 4))
        #     plt.plot(stock_data['Date'], stock_data['High'], label="High", linestyle="dashed", alpha=0.7)
        #     plt.plot(stock_data['Date'], stock_data['Low'], label="Low", linestyle="dashed", alpha=0.7)
        #     plt.fill_between(stock_data['Date'], stock_data['Low'], stock_data['High'], alpha=0.2, color="blue")
        #
        #     plt.xlabel("Date", fontweight='bold', fontfamily='Serif')
        #     plt.ylabel("Price", fontweight='bold', fontfamily='Serif')
        #     plt.title(f"{ticker} High-Low Price Range", fontweight='bold', fontfamily='Serif')
        #     plt.xticks(fontweight='bold', fontfamily='Serif')
        #     plt.yticks(fontweight='bold', fontfamily='Serif')
        #     plt.legend(prop={'weight': 'bold', 'family': 'Serif'})
        #     plt.savefig(f'Data visualization/High-Low Price Range for {ticker}.png')
        #     plt.show()

        # Step 1: Handle Missing Data
        final_data = data.interpolate(method='linear', axis=0)  # Linear interpolation for missing data

        # Step 2: Normalize Data (Feature Scaling)
        scaler = MinMaxScaler()
        final_data[['High', 'Low', 'Open', 'Volume']] = scaler.fit_transform(final_data[['High', 'Low', 'Open', 'Volume']])

        scaler = MinMaxScaler()
        final_data['Close'] = scaler.fit_transform(final_data[['Close']])  # Normalize between 0 and 1
        joblib.dump(scaler, 'Saved Data/Label Scaler.joblib')

        labels = final_data['Close']
        labels = np.array(labels)
        final_data.drop(columns=['Close', 'Date', 'Ticker'], inplace=True)

        # Step 3: Apply Simple Moving Average (SMA) for smoothing
        final_data['SMA_50'] = final_data['High'].rolling(window=50).mean()  # 50-day moving average
        final_data['SMA_200'] = final_data['High'].rolling(window=200).mean()  # 200-day moving average

        # Feature Extraction
        final_data['RSI'] = calculate_rsi(final_data)
        final_data['MACD'], final_data['MACD_Signal'] = calculate_macd(final_data)
        final_data['Upper_Band'], final_data['Lower_Band'] = calculate_bollinger_bands(final_data)
        final_data['OBV'] = calculate_obv(final_data)

        X = abs(final_data)  # Absolute

        X = X / np.max(X, axis=0)  # Normalization

        X = X.fillna(0)  # Nan to Num Conversion

        X = np.array(X)

        # Train-Test Split
        train_sizes = [0.7, 0.8]
        for train_size in train_sizes:
            x_train, x_test, y_train, y_test = train_test_split(X, labels, train_size=train_size, random_state=42)

            save('x_train_' + str(int(train_size * 100)), x_train)
            save('y_train_' + str(int(train_size * 100)), y_train)
            save('x_test_' + str(int(train_size * 100)), x_test)
            save('y_test_' + str(int(train_size * 100)), y_test)

