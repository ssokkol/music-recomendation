import numpy as np

from data_prep import load_dataset, normalize_features
from utils_io import load_tracks


def test_load_tracks():
    rows = load_tracks("data/sample_tracks.csv")
    assert len(rows) > 0
    assert rows[0].title != ""


def test_load_dataset_shapes():
    x, y, rows = load_dataset("data/sample_tracks.csv")
    assert x.shape[0] == len(rows)
    assert x.shape[1] == 4
    assert y.shape == (len(rows), 1)


def test_normalize_features_range():
    x = np.array([[0.0, 2.0], [1.0, 4.0]], dtype=np.float32)
    out = normalize_features(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
