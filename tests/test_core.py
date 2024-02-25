import numpy as np
import pytest

from tefs.core import compute_transfer_entropy, score_features, te_fs_forward, fs, te_fs_backward

@pytest.fixture
def create_test_data():
    def _create_test_data(shape):
        return np.random.rand(*shape)
    return _create_test_data


def test_compute_transfer_entropy_1d_inputs(create_test_data):
    X = create_test_data((100,))
    Y = create_test_data((100,))
    Z = create_test_data((100,))
    result = compute_transfer_entropy(X, Y, Z, k=1)
    assert isinstance(result, float), "The result should be a float."

def test_compute_transfer_entropy_different_lags(create_test_data):
    X = create_test_data((100, 2))
    Y = create_test_data((100, 1))
    Z = create_test_data((100, 3))
    result = compute_transfer_entropy(X, Y, Z, k=1, lag_features=[2], lag_target=[1], lag_conditioning=[3])
    assert isinstance(result, float), "The result should be a float."
    
def test_compute_transfer_entropy_with_no_lag_conditioning(create_test_data):
    X = create_test_data((100, 2))
    Y = create_test_data((100, 1))
    Z = create_test_data((100, 3))
    # Omit lag_conditioning to use default
    result = compute_transfer_entropy(X, Y, Z, k=1, lag_features=[2], lag_target=[1])
    assert isinstance(result, float), "The result should be a float."


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


def test_te_fs_forward_basic(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    results = te_fs_forward(features, target, k=1, lag_features=[1], lag_target=[1], verbose=0, n_jobs=1)
    assert isinstance(results, list), "Results should be a list."
    assert all("feature_scores" in result for result in results), "Each result should contain 'feature_scores'."

def test_te_fs_backward_basic(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    results = te_fs_backward(features, target, k=1, lag_features=[1], lag_target=[1], verbose=0, n_jobs=1)
    assert isinstance(results, list), "Results should be a list."
    assert all("feature_scores" in result for result in results), "Each result should contain 'feature_scores'."

def test_fs_forward_direction(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    results = fs(features, target, k=1, lag_features=[1], lag_target=[1], direction="forward", verbose=0, n_jobs=1)
    assert isinstance(results, list), "Results should be a list."

def test_fs_invalid_direction(create_test_data):
    features = create_test_data((100, 5))
    target = create_test_data((100, 1))
    with pytest.raises(ValueError):
        fs(features, target, k=1, lag_features=[1], lag_target=[1], direction="invalid", verbose=0, n_jobs=1)
