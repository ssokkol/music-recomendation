from __future__ import annotations

import numpy as np

from data_prep import load_dataset
from utils_io import save_recommendations


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def train_logreg(x: np.ndarray, y: np.ndarray, lr: float = 0.2, epochs: int = 300) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.1, size=(x.shape[1], 1)).astype(np.float32)
    bias = 0.0
    for _ in range(epochs):
        logits = x @ weights + bias
        preds = sigmoid(logits)
        error = preds - y
        grad_w = (x.T @ error) / x.shape[0]
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def predict_scores(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(x @ weights + bias).reshape(-1)


def main() -> None:
    x, y, rows = load_dataset("data/sample_tracks.csv")
    weights, bias = train_logreg(x, y)
    scores = predict_scores(x, weights, bias)

    ranked = sorted(
        zip((r.track_id for r in rows), (r.title for r in rows), (r.artist for r in rows), scores),
        key=lambda t: t[3],
        reverse=True,
    )
    top = ranked[:5]
    save_recommendations(top, "data/recommendations_simple.csv")
    print("Top-5 recommendations saved to data/recommendations_simple.csv")


if __name__ == "__main__":
    main()
