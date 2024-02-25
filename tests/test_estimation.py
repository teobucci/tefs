import pytest
import numpy as np
from tefs.estimation import estimate_mi, estimate_cmi  # Adjust the import path as necessary

# @pytest.mark.parametrize("X,Y,k,expected", [
#     (np.array([[0, 0], [1, 1]]), np.array([[0], [1]]), 1, pytest.approx(0.0)),
#     (np.random.rand(10, 2), np.random.rand(10, 1), 5, pytest.approx(0.0, abs=1e-1)),
# ])
# def test_estimate_mi_basic(X, Y, k, expected):
#     assert estimate_mi(X, Y, k) == expected

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