import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


# TODO might be worth checking
# - https://github.com/syanga/pycit/blob/master/pycit/estimators/mixed_cmi.py
# - https://github.com/wgao9/knnie/blob/master/knnie.py
def estimate_mi(X, Y, k=5, estimation_method="digamma"):
    """
    Mutual Information (MI) Estimator of I(X;Y) based on Mixed Random Variable Mutual Information Estimator - Gao et al.

    This function estimates the mutual information between two arrays, X and Y, using the method described by Gao et al.
    The arrays are first checked to ensure they are two-dimensional, and reshaped if necessary.
    A KDTree is then used to find the k nearest neighbors for each point in the combined dataset.
    The mutual information is then estimated using either the digamma function or a logarithmic function, depending on the 'estimate' parameter.

    :param X: The first input array.
    :type X: numpy.ndarray
    :param Y: The second input array.
    :type Y: numpy.ndarray
    :param k: The number of nearest neighbors to consider, defaults to 5.
    :type k: int, optional
    :param estimation_method: The estimation method to use, defaults to "digamma".
    :type estimation_method: str, optional
    :return: The estimated mutual information.
    :rtype: float
    """

    assert k > 0, "k must be greater than 0"
    assert k % 1 == 0, "k must be an integer"
    assert estimation_method in ["digamma", "log"], "Invalid estimation method"

    num_samples = len(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("Input arrays must have the same number of samples")
    
    if X.shape[0] == 0:
        raise ValueError("Input arrays must not be empty")

    dataset = np.concatenate((X, Y), axis=1)

    tree_xy = cKDTree(dataset)
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)

    # rho
    knn_distances = [tree_xy.query(sample, k + 1, p=float("inf"))[0][k] for sample in dataset]

    res = 0

    for i in range(num_samples):
        
        k_hat, n_xi, n_yi = k, k, k
        
        if knn_distances[i] <= 1e-15:
            # punti a distanza inferiore o uguale a (quasi) 0
            k_hat = len(tree_xy.query_ball_point(dataset[i], 1e-15, p=float("inf")))
            n_xi = len(tree_x.query_ball_point(X[i], 1e-15, p=float("inf")))
            n_yi = len(tree_y.query_ball_point(Y[i], 1e-15, p=float("inf")))
        else:
            # punti a distanza inferiore o uguale a rho
            k_hat = k
            n_xi = len(tree_x.query_ball_point(X[i], knn_distances[i] - 1e-15, p=float("inf")))
            n_yi = len(tree_y.query_ball_point(Y[i], knn_distances[i] - 1e-15, p=float("inf")))
        
        if estimation_method == "digamma":
            res += (digamma(k_hat) + np.log(num_samples) - digamma(n_xi) - digamma(n_yi)) / num_samples
        else:
            res += (digamma(k_hat) + np.log(num_samples) - np.log(n_xi + 1) - np.log(n_yi + 1)) / num_samples
    
    return res


def estimate_cmi(X, Y, Z, k=5, estimation_method="digamma"):
    """
    Estimate the Conditional Mutual Information (CMI) between X and Y given Z.
    Uses the formula: I(X;Y|Z) = I(X,Z;Y) - I(Z;Y)

    Note that I(X;Y|Z) = I(Y;X|Z)

    :param X: The input variable X.
    :type X: numpy.ndarray
    :param Y: The input variable Y.
    :type Y: numpy.ndarray
    :param Z: The input variable Z.
    :type Z: numpy.ndarray
    :param k: The number of nearest neighbors for k-nearest neighbor estimation (default is 5).
    :type k: int
    :param estimation_method: The estimation method to use (default is "digamma").
    :type estimation_method: str
    :return: The estimated CMI between X and Y given Z.
    :rtype: float
    """

    assert estimation_method in ["digamma", "log"], "Invalid estimation method"

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    XZ = np.hstack((X, Z))

    return estimate_mi(XZ, Y, k, estimation_method) - estimate_mi(Z, Y, k, estimation_method)
