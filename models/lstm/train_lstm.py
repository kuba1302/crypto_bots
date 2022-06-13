import os
import numpy as np
import pickle
import os
from pathlib import Path
from models.lstm.lstm import lstm_nn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == "__main__":
    MODEL_VERSION = "0.2"
    EPOCHS = 10
    BATCH_SIZE = 32
    TICKER = "BTC"

    load_path = Path(__file__).parents[2] / "data" / "preprocessed"
    save_path = Path(__file__).parents[2] / "models" / "lstm" / "versions"
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
    mc = ModelCheckpoint(
        save_path / f"lstm_{MODEL_VERSION}",
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=2)
    lstm.fit(
        data["X_list_train"],
        data["Y_preds_real_list_train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, mc],
    )
    lstm.save(save_path / f"lstm_{MODEL_VERSION}")
