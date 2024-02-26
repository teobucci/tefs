import numpy as np
import pytest

from tefs import TEFS
from tefs.core import score_features

@pytest.fixture
def create_test_data():
    def _create_test_data(shape):
        return np.random.rand(*shape)
    return _create_test_data

def test_score_features_forward_direction(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    conditioning = create_test_data((100, 3))
    scores = score_features(features, target, conditioning, k=1, lag_features=[1], lag_target=[1], direction="forward", n_jobs=1)
    assert scores.shape == (5,), "Scores should have the same length as number of features."

def test_score_features_backward_direction(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    conditioning = create_test_data((100, 5))
    scores = score_features(features, target, conditioning, k=1, lag_features=[1], lag_target=[1], direction="backward", n_jobs=1)
    assert scores.shape == (5,), "Scores should have the same length as number of features."

def is_iteration_result(obj):
    """Check if obj matches the IterationResult structure."""
    if not isinstance(obj, dict):
        return False

    for key, value in obj.items():
        if not isinstance(key, str):
            return False
        if isinstance(value, dict):
            if not all(isinstance(k, int) and isinstance(v, float) for k, v in value.items()):
                return False
        elif not isinstance(value, float) and not isinstance(value, int):
            return False

    return True

def test_te_fs_forward_basic(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    tefs = TEFS(features, target, k=1, lag_features=[1], lag_target=[1], direction="forward", verbose=2, n_jobs=1)
    tefs.fit()
    results = tefs.get_result()
    assert isinstance(results, list), "Results should be a list."
    assert all(is_iteration_result(result) for result in results), "Each item in results should be an instance of IterationResult."

def test_te_fs_backward_basic(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    tefs = TEFS(features, target, k=1, lag_features=[1], lag_target=[1], direction="backward", verbose=2, n_jobs=1)
    tefs.fit()
    results = tefs.get_result()
    assert all(is_iteration_result(result) for result in results), "Each item in results should be an instance of IterationResult."

def test_fs_invalid_direction(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    with pytest.raises(ValueError):
        TEFS(features, target, k=1, lag_features=[1], lag_target=[1], direction="invalid", verbose=2, n_jobs=1).fit()
