from binance.client import Client
import pandas as pd
import os
from dotenv import load_dotenv


def get_data():
        """Gets data with binance.client"""
        
        #Loads values from .env file
        load_dotenv()
        api_key = os.environ['API_KEY']
        api_secret = os.environ['SECRET_KEY']

        client = Client(api_key, api_secret)
        
        column_names = [
                "open_tstmp",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "close_tstmp",
                "quote_vol",
                "trades",
                "taker_base",
                "taker_quote",
                "excess_column",
                ]

        df = pd.DataFrame(client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "2 hours ago UTC"), columns = column_names)
        print(df)
        
        return df