import os
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch


import numpy as np

def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])  
    return np.array(X), np.array(y)


def load_btc_data(train_start, train_end, test_start, test_end, save_csv=True):
    '''
    start/end params only work when data is downloading, not loading
    :param train_start:
    :param train_end:
    :param test_start:
    :param test_end:
    :param save_csv:
    :return:
    '''

    train_path = "./data/btc_train_data.csv"
    test_path = "./data/btc_test_data.csv"

    # pobieranie danych jesli nie istnieja w pliku
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("[DATA_LOADER] Pobieranie danych z yfinance...")
        train_data = yf.download("BTC-USD", start=train_start, end=train_end)
        test_data = yf.download("BTC-USD", start=test_start, end=test_end)

        if save_csv:
            os.makedirs("./data", exist_ok=True)
            train_data.to_csv(train_path)
            test_data.to_csv(test_path)
            print("[DATA_LOADER] Zapisano dane do CSV")

    # wczytywanie danych
    columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    train_data = pd.read_csv(train_path, index_col=0,  parse_dates=True, skiprows=3,  names=columns)
    test_data = pd.read_csv(test_path, index_col=0, parse_dates=True, skiprows=3,  names=columns)

    print("[DATA_LOADER] Wczytano dane z plików CSV")

    return train_data, test_data

def get_all():
    train_data, test_data = load_btc_data("2015-01-01", "2023-01-01",
                                          "2023-01-02", "2024-01-01")

    features = ["Close"]  # lub ["Open", "High", "Low", "Close", "Volume"]
    train = train_data[features]
    test = test_data[features]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    x_train, y_train = create_sequences(train_scaled)
    x_test, y_test = create_sequences(test_scaled)

    print(x_train, y_train)

    X_trainT = torch.tensor(x_train, dtype=torch.float32)
    y_trainT = torch.tensor(y_train, dtype=torch.float32)
    X_testT = torch.tensor(x_test, dtype=torch.float32)
    y_testT = torch.tensor(y_test, dtype=torch.float32)

    return X_trainT, y_trainT, X_testT, y_testT, scaler