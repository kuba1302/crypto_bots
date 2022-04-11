import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pathlib import Path
import os 

cuda_path = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory(str(cuda_path))

def lstm_nn(input_dim, feature_size, output_dim=1, optimizer='Adam', loss='rmse'):
    model = Sequential()
    model.add(
        LSTM(
            units=512,
            return_sequences=True,
            input_shape=(input_dim, feature_size),
            recurrent_dropout=0.2,
            activation="tanh",
            kernel_regularizer="l2",
        )
    )
    model.add(
        LSTM(
            units=256,
            return_sequences=True,
            recurrent_dropout=0.2,
            activation="tanh",
            kernel_regularizer="l2",
        )
    )
    model.add(
        LSTM(
            units=128,
            return_sequences=False,
            recurrent_dropout=0.2,
            activation="tanh",
            kernel_regularizer="l2",
        )
    )
    model.add(Dense(units=output_dim))
    model.compile(optimizer=optimizer, metrics=["mae"], loss=loss)
    return model