import logging
from msilib.schema import Error
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from data_pipeline.production_pipeline import ProductionPipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from transformers import pipeline
from utils.log import prepare_logger

from bot.binance_connector import BinanceConnector
from bot.get_data import get_data

logger = prepare_logger(logging.INFO)


class CryptoBot:
    def __init__(
        self,
        scalers_path: Path,
        model_path: Path,
        save_results_path: Path,
        pipeline: ProductionPipeline,
        connector=BinanceConnector(),
    ) -> None:
        self.y_scaler = self.load_y_scaler(scalers_path)
        self.model = self.load_model(model_path)
        self.save_results_path = save_results_path
        self.pipeline = pipeline
        self.connector = connector
        self.results_history = pd.DataFrame()
        self.initial_btc_amount = self.get_initial_btc_amount()

    def get_initial_btc_amount(self):
        exchange_rate = self.connector.get_exchange_rate()
        total_inital_balance = self.connector.balance.total_balance
        return self.connector.round_quantity(total_inital_balance / exchange_rate)

    def update_results_history(self, last_price, action, quantity, pred):
        self.connector.get_balance(verbose=False)
        results_dict = self.connector.balance.__dict__ | {
            "buy_and_hold_balance": self.initial_btc_amount * last_price,
            "action": action,
            "quantity": quantity,
            "last_price": last_price,
            "pred": pred,
        }
        self.results_history = self.results_history.append(
            results_dict, ignore_index=True
        )
        self.results_history.to_csv(
            self.save_results_path,
            index=False,
        )

    def load_model(self, load_path):
        return keras.models.load_model(load_path)

    def load_y_scaler(self, scalers_path) -> MinMaxScaler:
        with open(scalers_path, "rb") as handle:
            scalers = pickle.load(handle)
        return scalers["y_scaler"]

    def inverse_scale(self, data):
        return self.y_scaler.inverse_transform(data)

    @staticmethod
    def get_action(buy, sell):
        if buy:
            return "buy"
        elif sell:
            return "sell"
        elif buy and sell:
            raise ValueError("Error! Two operations in one time peroid")
        else:
            return None

    def get_price_diff(self):
        df = get_data()
        last_price = float(df.tail(1)["close"].values[0])
        df_t = pipeline.transform(df)
        df_t = np.array(df_t).reshape([-1, 60, 37])
        pred = self.inverse_scale(self.model.predict(df_t))[0][0]
        price_diff = pred - last_price
        logger.info(f"Pred: {pred}")
        logger.info(f"Actual: {last_price}")
        return price_diff, last_price, pred

    def trade(self, top_cut_off=0, down_cut_off=0):
        while True:
            if datetime.now().second == 0:
                buy = None
                sell = None
                quantity = None
                price_diff, last_price, pred = self.get_price_diff()

                if price_diff > last_price * top_cut_off / 100:
                    buy, quantity = self.connector.buy()

                elif price_diff < -(last_price * down_cut_off / 100):
                    sell, quantity = self.connector.sell()

                action = self.get_action(buy=buy, sell=sell)

                self.update_results_history(
                    last_price=last_price, action=action, quantity=quantity, pred=pred
                )


if __name__ == "__main__":
    MODEL_VERSION = "0.3"
    MODEL_NAME = "lstm"

    base_path = Path(__file__).parents[1]
    data_path = base_path / "data"
    save_path = data_path / "preprocessed"
    scaler_path = save_path / "scalers_BTC.pickle"
    model_path = (
        base_path / "models" / "lstm" / "versions" / f"{MODEL_NAME}_{MODEL_VERSION}"
    )

    pipeline = ProductionPipeline(scalers_path=scaler_path)

    bot = CryptoBot(
        scalers_path=scaler_path,
        model_path=model_path,
        save_results_path=data_path / "results.csv",
        pipeline=pipeline,
    )
    bot.trade()
