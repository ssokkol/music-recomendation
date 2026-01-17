import pytest

tf = pytest.importorskip("tensorflow")

from keras_model import build_model


def test_build_model_output_shape():
    model = build_model(4)
    assert model.output_shape == (None, 1)
