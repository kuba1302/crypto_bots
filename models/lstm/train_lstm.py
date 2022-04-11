import os
import numpy as np
import pickle
import os
from pathlib import Path
from models.lstm.lstm import lstm_nn


if __name__ == "__main__":
    MODEL_VERSION = "0.1"
    EPOCHS = 2000
    BATCH_SIZE = 100
    TICKER = "BTC"
    load_path = Path(os.path.abspath("")) / "data" / "preprocessed"
    save_path = Path(os.path.abspath("")) / "models" / "lstm" / "versions"
    os.makedirs(save_path, exist_ok=True)
    with open(load_path / f"data_{TICKER}.pickle", "rb") as test:
        data = pickle.load(test)
    print(
        "-----------------------------------------------"
        f'TRAIN DATA SHAPE: {data["X_list_train"].shape}'
        " ----------------------------------------------"
    )
    lstm = lstm_nn(
        input_dim=data["X_list_train"].shape[1],
        feature_size=data["X_list_train"].shape[2],
        optimizer="Adam",
        loss="mse",
    )
    lstm.fit(data["X_list_train"], data["Y_preds_real_list_train"], epochs=EPOCHS, batch_size=BATCH_SIZE)
    lstm.save(save_path)