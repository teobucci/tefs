import pytest
import numpy as np
from tefs.estimation import estimate_mi, estimate_cmi, estimate_conditional_transfer_entropy

@pytest.fixture
def create_test_data():
    def _create_test_data(shape):
        return np.random.rand(*shape)
    return _create_test_data

@pytest.mark.parametrize("method", ["digamma", "log"])
def test_estimate_mi_methods(method):
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 1)
    assert isinstance(estimate_mi(X, Y, 5, method), float)

def test_estimate_mi_invalid_k():
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 1)
    with pytest.raises(AssertionError):
        estimate_mi(X, Y, -1)

def test_estimate_mi_invalid_method():
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 1)
    with pytest.raises(AssertionError):
        estimate_mi(X, Y, 5, "invalid_method")

def test_estimate_mi_mismatched_lengths():
    X = np.random.rand(10, 2)
    Y = np.random.rand(9, 1)
    with pytest.raises(ValueError):
        estimate_mi(X, Y)

def test_estimate_mi_empty_arrays():
    X = np.array([])
    Y = np.array([])
    with pytest.raises(ValueError):
        estimate_mi(X, Y)

@pytest.mark.parametrize("X,Y,Z,k,expected", [
    (np.array([[0], [1]]), np.array([[0], [1]]), np.array([[1], [0]]), 1, pytest.approx(0.0)),
])
def test_estimate_cmi_basic(X, Y, Z, k, expected):
    assert estimate_cmi(X, Y, Z, k) == expected


def test_estimate_conditional_transfer_entropy_1d_inputs(create_test_data):
    X = create_test_data((100,))
    Y = create_test_data((100,))
    Z = create_test_data((100,))
    result = estimate_conditional_transfer_entropy(X, Y, Z, k=1)
    assert isinstance(result, float), "The result should be a float."

def test_estimate_conditional_transfer_entropy_different_lags(create_test_data):
    X = create_test_data((100, 2))
    Y = create_test_data((100, 1))
    Z = create_test_data((100, 3))
    result = estimate_conditional_transfer_entropy(X, Y, Z, k=1, lag_features=[2], lag_target=[1], lag_conditioning=[3])
    assert isinstance(result, float), "The result should be a float."
    
def test_estimate_conditional_transfer_entropy_with_no_lag_conditioning(create_test_data):
    X = create_test_data((100, 2))
    Y = create_test_data((100, 1))
    Z = create_test_data((100, 3))
    # Omit lag_conditioning to use default
    result = estimate_conditional_transfer_entropy(X, Y, Z, k=1, lag_features=[2], lag_target=[1])
    assert isinstance(result, float), "The result should be a float."
