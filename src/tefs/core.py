import numpy as np
from .utils import shift_data_up  # Util functions might be needed for internal operations

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import pandas as pd

from .estimation import estimate_cmi


def compute_transfer_entropy(
    X,
    Y,
    Z,
    k,
    lag_features=1,
    lag_target=1,
    lag_conditioning=None
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
    :param lag_target: the lag applied on Y
    :param lag_conditioning: the lag applied on Z, if None it is set to lag_features
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

    max_lag = max(lag_features, lag_target, lag_conditioning)

    n_timesteps = X.shape[0]
    n_features = X.shape[1]
    n_targets = Y.shape[1]
    n_conditioning = Z.shape[1]

    # Filling member1
    member1 = np.zeros((n_timesteps - max_lag, n_features * lag_features))
    for feature in range(n_features):
        for lag in range(lag_features):
            member1[:, feature * lag_features + lag] = X[max_lag - (lag + 1) : -(lag + 1), feature]

    # Filling member2
    member2 = np.zeros((n_timesteps - max_lag, n_targets))
    for target in range(n_targets):
        member2[:, target] = Y[max_lag:, target]

    # Filling member3 (the conditioning set)
    member3 = np.zeros((n_timesteps - max_lag, n_targets * lag_target + n_conditioning * lag_conditioning))
    # Filling the part relative the past of the target
    for target in range(n_targets):
        for lag in range(lag_target):
            member3[:, target * lag_target + lag] = Y[max_lag - (lag + 1) : -(lag + 1), target]
    # filling the part relative the past of the conditioning features
    for conditioning_feature in range(n_conditioning):
        for lag in range(lag_conditioning):
            member3[:, n_targets * lag_target + conditioning_feature * lag_conditioning + lag] = Z[max_lag - (lag + 1) : -(lag + 1), conditioning_feature]

    return estimate_cmi(member1, member2, member3, k)

def score_features(
    features,
    target,
    conditioning,
    k,
    lag_features,
    lag_target,
    direction,
    n_jobs=1
) -> np.ndarray:
    """
    Computes the transfer entropy score for each feature :math:`X_i` in :math:`X`, to the target :math:`Y`, given the conditioning set :math:`X_A`.

    :param features: ndarray of shape N x p
    :param target: ndarray of shape N x 1
    :param conditioning: ndarray of shape N x q
    :param k: number of bins for the discretization
    :param lag_features: the lag applied on X and conditioning
    :param lag_target: the lag applied on Y
    :param direction: "forward" or "backward"
    :param n_jobs: number of parallel jobs to run
    :return: a dictionary with key = feature index and value = transfer entropy score
    """
    
    assert direction in ["forward", "backward"], "direction must be forward or backward"
    n_features = features.shape[1]
    args = []

    for col in range(n_features):
        # The only difference in the two cases is the conditioning set,
        # in the backward case I must make sure to exclude the feature X_i
        if direction == "forward":
            conditioning_set = conditioning
        elif direction == "backward":
            conditioning_set = np.delete(conditioning, col, axis=1)  # \ X_i

        args.append((
            features[:, col],  # X_i
            target,  # Y
            conditioning_set,  # X_A
            k,
            lag_features,
            lag_target,
        ))

    with ThreadPool(n_jobs) as pool:
        scores = pool.map(score_features_parallel, args)

    return np.array(scores)

def score_features_parallel(args):
    """
    Helper function to compute the score of a single feature in parallel.
    """
    return compute_transfer_entropy(*args)

def te_fs_forward(
    features,
    target,
    k,
    lag_features=1,
    lag_target=1,
    verbose=1,
    var_names=None,
    n_jobs=1,
):
    """
    This function returns the selected features starting from an empty array
    and adding features that have the best CMI score.
    :param features: features numpy.ndarray object
    :param target: target numpy.ndarray object
    :param k: number of bins for the discretization
    :return: list of indexes of the selected features
    """

    df = pd.DataFrame(features)
    selected_features = list()
    candidate_features = list(range(features.shape[1]))
    TE_cumulated = 0
    results = []

    while True:
        # check that there are still features to add
        if len(candidate_features) == 0:
            break

        if verbose >= 2:
            print(f"candidate_features: {candidate_features}")
            print(f"selected_features: {selected_features}")

        # compute the TE scores for each feature
        feature_scores = score_features(
            features=df[candidate_features].values,
            target=target,
            conditioning=df[selected_features].values,
            k=k,
            lag_features=lag_features,
            lag_target=lag_target,
            direction="forward",
            n_jobs=n_jobs,
        )

        # i assume features_scores to be a dict wit
        # key = feature index and value = TE score
        # ex: {0: -0.5, 1: 0.5, 2: 0.7}

        feature_scores = dict(zip(df[candidate_features].columns, feature_scores))

        # print the scores
        if verbose >= 2:
            for key, value in feature_scores.items():
                print(f"Feature {key} has Transfer Entropy score on the target: {value}")

        # sort the scores in descending order by value
        feature_scores = dict(sorted(feature_scores.items(), key=lambda item: item[1], reverse=True))

        # find the first key value pair in the dictionary
        max_feature_index = int(next(iter(feature_scores)))
        max_TE = next(iter(feature_scores.values()))

        # increase the cumulative loss of information
        TE_cumulated += max(max_TE, 0)

        # by checking before selection of the feature
        # I DON'T make sure that at least one feature is selected

        if verbose >= 2:
            print(f"TE_cumulated: {TE_cumulated}")
            print("-" * 50)

        results.append({"feature_scores": feature_scores, "TE": TE_cumulated})

        if verbose >= 2:
            print("Details of the maximum feature:")
            print(f"max_feature_index: {max_feature_index}")
            print(f"max_TE: {max_TE}")

        if verbose >= 1 and var_names is not None:
            print(f"Adding feature: {var_names[max_feature_index]} with TE score: {max_TE}")

        # add the feature to the selected features list
        selected_features.append(max_feature_index)

        # remove the feature from the candidate features list
        candidate_features.remove(max_feature_index)

    return results

def te_fs_backward(
    features,
    target,
    k,
    lag_features=1,
    lag_target=1,
    verbose=1,
    var_names=None,
    n_jobs=1,
):
    """
    This function returns the selected features starting from the full dataset
    and removing features keeping the loss of information smaller than the threshold.
    :param features: features numpy.ndarray object
    :param target: target numpy.ndarray object
    :param k: number of bins for the discretization
    :return: list of indexes of the selected features
    """

    df = pd.DataFrame(features)
    selected_features = list()
    candidate_features = list(range(features.shape[1]))
    TE_loss = 0
    results = []

    while True:
        # check that there are still features to remove
        if len(candidate_features) == 0:
            break

        if verbose >= 2:
            print(f"candidate_features: {candidate_features}")
            print(f"selected_features: {selected_features}")

        # compute the TE scores for each feature
        feature_scores = score_features(
            features=df[candidate_features].values,
            target=target,
            conditioning=df[candidate_features].values,
            k=k,
            lag_features=lag_features,
            lag_target=lag_target,
            direction="backward",
            n_jobs=n_jobs,
        )

        # i assume features_scores to be a dict wit
        # key = feature index and value = TE score
        # ex: {0: -0.5, 1: 0.5, 2: 0.7}

        feature_scores = dict(zip(df[candidate_features].columns, feature_scores))

        # sort the scores in descending order by value
        feature_scores = dict(sorted(feature_scores.items(), key=lambda item: item[1], reverse=False))

        # print the scores
        if verbose >= 2:
            for key, value in feature_scores.items():
                print(f"Feature {key} has Transfer Entropy score on the target: {value}")

        # find the first key value pair in the dictionary
        min_feature_index = int(next(iter(feature_scores)))
        min_TE = next(iter(feature_scores.values()))

        # increase the cumulative loss of information
        TE_loss += max(min_TE, 0)

        if verbose >= 2:
            print(f"TE_loss: {TE_loss}")
            print("-" * 50)

        results.append({"feature_scores": feature_scores, "TE": TE_loss})

        # by checking after the removal of the feature
        # I make it possible to not remove any feature

        if verbose >= 2:
            print("Details of the minimum feature:")
            print(f"min_feature_index: {min_feature_index}")
            print(f"min_TE: {min_TE}")

        if verbose >= 1 and var_names is not None:
            print(f"Removing feature: {var_names[min_feature_index]} with TE score: {min_TE}")

        # add the feature to the selected features list
        selected_features.append(min_feature_index)

        # remove the feature from the candidate features list
        candidate_features.remove(min_feature_index)

    return results

def fs(
        features,
        target,
        k,
        direction,
        lag_features=1,
        lag_target=1,
        verbose=1,
        var_names=None,
        n_jobs=1,
    ):
    """
    This function selects features either by forward or backward selection based on the Transfer Entropy score.
    
    :param features: features numpy.ndarray object
    :param target: target numpy.ndarray object
    :param k: number of bins for the discretization
    :param direction: 'forward' or 'backward' selection
    :param lag_features: lag of features for TE calculation
    :param lag_target: lag of target for TE calculation
    :param verbose: verbosity level
    :param var_names: names of the variables/features
    :param n_jobs: number of jobs for parallel computation
    :return: list of indexes of the selected features
    """
    
    # Validate direction argument
    if direction not in ['forward', 'backward']:
        raise ValueError("direction must be either 'forward' or 'backward'")
    
    if not len(var_names) == features.shape[1]:
        raise ValueError("var_names must have the same length as the number of features")

    # Prepare common arguments for both functions
    common_args = {
        "features": features,
        "target": target,
        "k": k,
        "lag_features": lag_features,
        "lag_target": lag_target,
        "verbose": verbose,
        "var_names": var_names,
        "n_jobs": n_jobs
    }
    
    # Call the appropriate function based on the direction
    if direction == 'forward':
        return te_fs_forward(**common_args)
    else:  # direction == 'backward'
        return te_fs_backward(**common_args)
