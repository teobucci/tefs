import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


# TODO might be worth checking
# - https://github.com/syanga/pycit/blob/master/pycit/estimators/mixed_cmi.py
# - https://github.com/wgao9/knnie/blob/master/knnie.py
def estimate_mi(
        X: np.ndarray,
        Y: np.ndarray,
        k: int = 5,
        estimation_method: str = "digamma",
        ) -> float:
    """
    Estimate the Mutual Information (MI) between :math:`X` and :math:`Y`, i.e. :math:`I(X;Y)`, based on *Mixed Random Variable Mutual Information Estimator - Gao et al.*.

    :param X: The first input array.
    :type X: numpy.ndarray
    :param Y: The second input array.
    :type Y: numpy.ndarray
    :param k: The number of nearest neighbors to consider, defaults to 5.
    :type k: int, optional
    :param estimation_method: The estimation method to use, can be either 'digamma' or 'log', defaults to 'digamma'.
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
            # Points at a distance less than or equal to (nearly) 0
            k_hat = len(tree_xy.query_ball_point(dataset[i], 1e-15, p=float("inf")))
            n_xi = len(tree_x.query_ball_point(X[i], 1e-15, p=float("inf")))
            n_yi = len(tree_y.query_ball_point(Y[i], 1e-15, p=float("inf")))
        else:
            # Points at distances less than or equal to rho
            k_hat = k
            n_xi = len(tree_x.query_ball_point(X[i], knn_distances[i] - 1e-15, p=float("inf")))
            n_yi = len(tree_y.query_ball_point(Y[i], knn_distances[i] - 1e-15, p=float("inf")))
        
        if estimation_method == "digamma":
            res += (digamma(k_hat) + np.log(num_samples) - digamma(n_xi) - digamma(n_yi)) / num_samples
        elif estimation_method == "log":
            res += (digamma(k_hat) + np.log(num_samples) - np.log(n_xi + 1) - np.log(n_yi + 1)) / num_samples
    
    return res


def estimate_cmi(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        k: int = 5,
        estimation_method: str = "digamma",
        ) -> float:
    """
    Estimate the Conditional Mutual Information (CMI) between :math:`X` and :math:`Y` given :math:`Z`, i.e. :math:`I(X;Y \mid Z)`, using the equivalance

    .. math::
        I(X;Y \mid Z) = I(X,Z;Y) - I(Z;Y)

    Note that :math:`I(X;Y \mid Z) = I(Y;X \mid Z)`.

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


def estimate_conditional_transfer_entropy(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        k: int,
        lag_features: list[int] = [1],
        lag_target: list[int] = [1],
        lag_conditioning: list[int] = None,
        ) -> float:
    """
    Computes the conditional transfer entropy from X to Y given Z, using the specified lags.

    :param X: Sample of a (multivariate) random variable representing the input
    :type X: np.ndarray of shape (n_samples, n_features)
    :param Y: Sample of a (multivariate) random variable representing the target
    :type Y: np.ndarray of shape (n_samples, n_targets)
    :param Z: Sample of a (multivariate) random variable representing the conditioning
    :type Z: np.ndarray of shape (n_samples, n_conditioning)
    :param lag_features: the lag applied on X
    :type lag_features: List[int]
    :param lag_target: the lag applied on Y
    :type lag_target: List[int]
    :param lag_conditioning: the lag applied on Z, if None it is set to lag_features
    :type lag_conditioning: List[int]
    :return: a scalar of the value of the transfer entropy
    """

    if lag_conditioning is None:
        lag_conditioning = lag_features
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    max_lag = max(max(lag_features), max(lag_target), max(lag_conditioning))

    # Filling member1
    member1 = np.hstack([X[max_lag - lag : X.shape[0]-lag, :] for lag in lag_features])

    # Filling member2
    member2 = Y[max_lag:, :]

    # Filling member3
    member3 = np.hstack([
        # Filling the part relative the past of the target
        *[Y[max_lag - lag : Y.shape[0]-lag, :] for lag in lag_target],
        # Filling the part relative the past of the conditioning features
        *[Z[max_lag - lag : Z.shape[0]-lag, :] for lag in lag_conditioning],
    ])

    return estimate_cmi(member1, member2, member3, k)
