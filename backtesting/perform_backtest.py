import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_class import BackTest

if __name__ == "__main__":
    TICKER = "BTC"
    MODEL_VERSION = "0.1"
    MODEL_NAME = "lstm"
    scaled_path = Path(os.path.abspath("")).parents[0] / "data" / "preprocessed"
    scalers_path = scaled_path / f"scalers_{TICKER}.pickle"
    data_path = scaled_path / f"data_{TICKER}.pickle"
    preds_path = Path(os.path.abspath("")) / f"preds_{TICKER}_{MODEL_VERSION}.pickle"
    model_path = (
        Path(os.path.abspath("")).parents[0]
        / "models"
        / "lstm"
        / "versions"
        / f"{MODEL_NAME}_{MODEL_VERSION}"
    )

    bot = BackTest(
        transaction_cost_percent=0.001,
        currency_count=1000000,
        ticker=TICKER,
        scalers_path=scalers_path,
        model_path=model_path,
    )
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    with open(preds_path, "rb") as handle:
        preds = pickle.load(handle)

    y = data["Y_preds_real_list_test"]

    bot.simulate(preds=preds, y=y, top_cut_off=1, down_cut_off=1, if_short=False)