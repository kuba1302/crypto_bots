from binance.client import Client
import pandas as pd
from bot.load_env import API_KEY, SECRET_KEY

def get_data():
    client = Client(api_key=API_KEY, api_secret=SECRET_KEY, testnet=True)
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

    df = pd.DataFrame(
        client.get_historical_klines(
            "BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "3 hours ago UTC"
        ),
        columns=column_names,
    )

    return df
