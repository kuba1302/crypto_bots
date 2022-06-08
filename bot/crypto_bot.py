import requests
import datetime
from data_pipeline.production_pipeline import ProductionPipeline
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle

from transformers import pipeline

class DownloadData:

    def get_data():
        return pd.read_csv()


class CryptoBot:
    def __init__(
        self,
        data_downloader,
        scalers_path: Path,
        model_path: Path,
        pipeline: ProductionPipeline,
        ticker="BTC",
    ) -> None:
        self.y_scaler = self.load_y_scaler(scalers_path)
        self.model = self.load_model(model_path)
        self.data_downloader = data_downloader
        self.pipeline = pipeline

    def load_model(self, load_path):
        return keras.models.load_model(load_path)

    def load_y_scaler(self, scalers_path) -> MinMaxScaler:
        with open(scalers_path, "rb") as handle:
            scalers = pickle.load(handle)
        return scalers["y_scaler"]

    
    def generate_decision(self, X):
        pass 


    def trade(self):
        df = self.data_downloader.get_data()
        X_cols = [col for col in df.columns if col != "close"]
        y_col = "col"



"""
What have to be done?: 
1. Choose time interval, ticker and number of klines
2. Collect that time with proper cols 
3. Move it to pipeline
4. Make pred 
5. Make transaction or not 
"""
