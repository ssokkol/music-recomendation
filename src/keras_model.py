from __future__ import annotations

import numpy as np
import tensorflow as tf

from data_prep import load_dataset
from utils_io import save_recommendations


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu", input_shape=(input_dim,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    x, y, rows = load_dataset("data/sample_tracks.csv")
    model = build_model(x.shape[1])
    model.fit(x, y, epochs=40, verbose=0)

    scores = model.predict(x, verbose=0).reshape(-1)
    ranked = sorted(
        zip((r.track_id for r in rows), (r.title for r in rows), (r.artist for r in rows), scores),
        key=lambda t: t[3],
        reverse=True,
    )
    save_recommendations(ranked[:5], "data/recommendations_keras.csv")
    print("Top-5 recommendations saved to data/recommendations_keras.csv")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
