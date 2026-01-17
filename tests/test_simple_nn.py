from data_prep import load_dataset
from simple_nn import predict_scores, train_logreg


def test_train_logreg_shapes():
    x, y, _ = load_dataset("data/sample_tracks.csv")
    weights, bias = train_logreg(x, y, epochs=5)
    assert weights.shape == (x.shape[1], 1)
    assert isinstance(bias, float)


def test_predict_scores_range():
    x, y, _ = load_dataset("data/sample_tracks.csv")
    weights, bias = train_logreg(x, y, epochs=5)
    scores = predict_scores(x, weights, bias)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
