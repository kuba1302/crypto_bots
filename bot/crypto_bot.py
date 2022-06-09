import os
from data_pipeline.production_pipeline import ProductionPipeline
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from transformers import pipeline
from bot.get_data import get_data
from datetime import datetime


class CryptoBot:
    def __init__(
        self,
        scalers_path: Path,
        model_path: Path,
        pipeline: ProductionPipeline,
        ticker="BTC",
    ) -> None:
        self.y_scaler = self.load_y_scaler(scalers_path)
        self.model = self.load_model(model_path)
        self.pipeline = pipeline

    def load_model(self, load_path):
        return keras.models.load_model(load_path)

    def load_y_scaler(self, scalers_path) -> MinMaxScaler:
        with open(scalers_path, "rb") as handle:
            scalers = pickle.load(handle)
        return scalers["y_scaler"]

    def inverse_scale(self, data): 
        return self.y_scaler.inverse_transform(data)

    def generate_decision(self, X):
        pass

    def buy(self):
        pass 
    
    def sell(self):
        pass 

    def trade(self, top_cut_off=0, down_cut_off=0):
        while True: 
            # if datetime.now().second == 0: 

            df = get_data()
            last_price = float(df.tail(1)["close"].values[0])
            df_t = pipeline.transform(df)
            df_t = np.array(df_t).reshape([-1, 60, 11])
            pred = self.inverse_scale(self.model.predict(df_t))[0][0]

            price_diff = pred - last_price

            if price_diff > last_price * top_cut_off / 100:
                # buy_amount = self.calculate_asset_amount_by_price(
                #     price=last_price, curr_amount=self.currency_count
                # )
                print("BUY")
                # self.buy(buy_amount, last_price)

            elif price_diff < -(last_price * down_cut_off / 100):
                sell_amout = self.asset_count
                print("sell")
                # self.sell(sell_amout, last_price)


if __name__ == "__main__":
    MODEL_VERSION = "0.1"
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
"""
What have to be done?: 
1. Choose time interval, ticker and number of klines
2. Collect that time with proper cols 
3. Move it to pipeline
4. Make pred 
5. Make transaction or not 
"""
