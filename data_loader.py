import os
import pandas as pd
import yfinance as yf


def load_btc_data(train_start, train_end, test_start, test_end, save_csv=True):
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

if __name__ == "__main__":
    train_data, test_data = load_btc_data("2015-01-01", "2023-01-01",
                                          "2023-01-02", "2024-01-01")
    print(train_data)