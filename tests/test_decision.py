import numpy as np
import pytest
from src.decision import apply_threshold


def test_apply_threshold_basic():
    probs = np.array([0.2, 0.6, 0.8])
    preds = apply_threshold(probs, 0.5)
    assert (preds == np.array([0, 1, 1])).all()


def test_apply_threshold_edge_cases():
    probs = np.array([0.5])
    preds = apply_threshold(probs, 0.5)
    assert preds[0] == 1


def test_invalid_threshold():
    with pytest.raises(ValueError):
        apply_threshold(np.array([0.5]), 1.5)