from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import numpy as np

from utils_io import TrackRow, load_tracks


def to_feature_matrix(rows: list[TrackRow]) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in rows:
        features.append([row.tempo, row.energy, row.danceability, row.valence])
        labels.append(row.favorite)
    x = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)
    return x, y


def normalize_features(x: np.ndarray) -> np.ndarray:
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (x - mins) / denom


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, list[TrackRow]]:
    rows = load_tracks(path)
    x, y = to_feature_matrix(rows)
    x = normalize_features(x)
    return x, y, rows


def demo_print(rows: list[TrackRow]) -> None:
    for row in rows[:3]:
        print(asdict(row))


if __name__ == "__main__":
    x, y, rows = load_dataset("data/sample_tracks.csv")
    print("x shape:", x.shape, "y shape:", y.shape)
    demo_print(rows)
