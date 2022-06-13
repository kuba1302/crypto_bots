from data_pipeline.production_pipeline import ProductionPipeline
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from transformers import pipeline
from bot.get_data import get_data
from datetime import datetime
from bot.binance_connector import BinanceConnector
from utils.log import prepare_logger
import logging

logger = prepare_logger(logging.INFO)


class CryptoBot:
    def __init__(
        self,
        scalers_path: Path,
        model_path: Path,
        pipeline: ProductionPipeline,
        connector=BinanceConnector(),
    ) -> None:
        self.y_scaler = self.load_y_scaler(scalers_path)
        self.model = self.load_model(model_path)
        self.pipeline = pipeline
        self.connector = connector

    def load_model(self, load_path):
        return keras.models.load_model(load_path)

    def load_y_scaler(self, scalers_path) -> MinMaxScaler:
        with open(scalers_path, "rb") as handle:
            scalers = pickle.load(handle)
        return scalers["y_scaler"]

    def inverse_scale(self, data):
        return self.y_scaler.inverse_transform(data)

    def get_price_diff(self):
        df = get_data()
        last_price = float(df.tail(1)["close"].values[0])
        df_t = pipeline.transform(df)
        df_t = np.array(df_t).reshape([-1, 60, 37])
        pred = self.inverse_scale(self.model.predict(df_t))[0][0]
        price_diff = pred - last_price
        return price_diff, last_price

    def trade(self, top_cut_off=0, down_cut_off=0):
        while True:
            # if datetime.now().second == 0:
                
            price_diff, last_price = self.get_price_diff()
            logger.info(f"!!!LAST PRICE get_data: {last_price}")

            if price_diff > last_price * top_cut_off / 100:
                self.connector.buy()

            elif price_diff < -(last_price * down_cut_off / 100):
                self.connector.sell()


if __name__ == "__main__":
    MODEL_VERSION = "0.2"
    MODEL_NAME = "lstm"

    base_path = Path(__file__).parents[1]
    data_path = base_path / "data"
    save_path = data_path / "preprocessed"
    scaler_path = save_path / "scalers_BTC.pickle"
    model_path = (
        base_path / "models" / "lstm" / "versions" / f"{MODEL_NAME}_{MODEL_VERSION}"
    )

    pipeline = ProductionPipeline(scalers_path=scaler_path)

    bot = CryptoBot(scalers_path=scaler_path, model_path=model_path, pipeline=pipeline)
    bot.trade()
