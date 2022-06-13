import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class ProductionPipeline:
    def __init__(self, scalers_path: Path) -> None:
        self.X_scaler = self.load_x_scaler(scalers_path)

    def load_x_scaler(self, scalers_path) -> MinMaxScaler:
        with open(scalers_path, "rb") as handle:
            scalers = pickle.load(handle)
        return scalers["X_scaler"]

    def scale_data(self, X):
        X_cols = [col for col in X.columns if col not in ["close", "date"]]
        return pd.DataFrame(
            self.X_scaler.fit_transform(X.loc[:, X_cols]), columns=X_cols
        )

    @staticmethod
    def get_bollinger_bands(prices, peroid):
        sma = prices.rolling(window=peroid).mean()
        std = prices.rolling(peroid).std()
        bollinger_up = sma + std * 2
        bollinger_down = sma - std * 2
        return bollinger_up, bollinger_down

    @staticmethod
    def get_macd(prices):
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        return macd, signal_line

    def get_moving_averages(self, X, peroid_list):
        df = X.copy()
        
        for peroid in peroid_list:
            df[f"sma_{peroid}"] = df["close"].rolling(window=peroid).mean()
            df[f"ema_{peroid}"] = df["close"].ewm(span=peroid).mean()

            weights = np.arange(1, peroid + 1)
            df[f"wma_{peroid}"] = (
                df["close"]
                .rolling(window=peroid)
                .apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
            )

            bollinger_up, bollinger_down = self.get_bollinger_bands(df["close"], peroid)
            df[f"bb_{peroid}_up"] = bollinger_up
            df[f"bb_{peroid}_down"] = bollinger_down
        return df

    def perform_technical_analysis(self, X, peroid_list=[5, 10, 20, 50, 100]):
        df = X.copy()
        df = self.get_moving_averages(X, peroid_list)
        df["macd"], df["signal_line"] = self.get_macd(df["close"])
        return df

    def transform(self, X):
        stock_data = self.perform_technical_analysis(X)
        stock_data = stock_data.drop(
            columns="excess_column"
            if not "Unnamed: 0" in stock_data.columns
            else ["Unnamed: 0", "excess_column"]
        )
        stock_data = stock_data.tail(60)
        X_scaled = self.scale_data(stock_data)
        return X_scaled
