from __future__ import annotations

from tensorflow import keras


def build_lstm(window: int, units: int, dropout: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window, 1)),
            keras.layers.LSTM(units, return_sequences=True),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(units),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model
