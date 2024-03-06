from multiprocessing.pool import ThreadPool
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from .estimation import estimate_conditional_transfer_entropy
from .types import IterationResult


class TEFS:
    def __init__(
            self,
            features: np.ndarray,
            target: np.ndarray,
            k: int,
            lag_features: list[int],
            lag_target: list[int],
            direction: str,
            verbose: int = 1,
            var_names: list[str] = None,
            n_jobs: int = 1,
            ) -> List[IterationResult]:
        """
        Perform the forward or backward feature selection based on the Transfer Entropy score.

        :param features: Sample of a (multivariate) random variable representing the input
        :type features: np.ndarray of shape (n_samples, n_features)
        :param target: Sample of a (multivariate) random variable representing the target
        :type target: np.ndarray of shape (n_samples, n_targets)
        :param k: number of nearest neighbors for the CMI estimation
        :type k: int
        :param lag_features: the lags applied on features and conditioning
        :type lag_features: list[int]
        :param lag_target: the lags applied on the target
        :type lag_target: list[int]
        :param direction: the direction of the transfer entropy, either "forward" or "backward"
        :type direction: str
        :param verbose: verbosity level
        :type verbose: int
        :param var_names: names of the variables/features
        :type var_names: list[str]
        :param n_jobs: number of parallel jobs to run
        :type n_jobs: int
        :return: list of indexes of the selected features
        :rtype: List[IterationResult]
        """

        # Validate direction argument
        if direction not in ['forward', 'backward']:
            raise ValueError("direction must be either 'forward' or 'backward'")

        if var_names and not len(var_names) == features.shape[1]:
            raise ValueError("var_names must have the same length as the number of features")
        
        if var_names is None:
            var_names = [f"{i+1}" for i in range(features.shape[1])]
        
        self.features = features
        self.target = target
        self.k = k
        self.lag_features = lag_features
        self.lag_target = lag_target
        self.direction = direction
        self.verbose = verbose
        self.var_names = var_names
        self.n_jobs = n_jobs
        self.result = None
    
    def get_result(self) -> List[IterationResult]:
        """
        Return the result of the feature selection algorithm.
        """
        if self.result is None:
            raise ValueError("The feature selection algorithm has not been run yet")
        
        return self.result
    
    def __repr__(self) -> str:
        return f"TEFS(features={self.features}, target={self.target}, k={self.k}, lag_features={self.lag_features}, lag_target={self.lag_target}, direction={self.direction}, verbose={self.verbose}, var_names={self.var_names}, n_jobs={self.n_jobs})"

    def fit(self) -> None:
        """
        Run the feature selection algorithm and return the results.
        """

        if self.result is not None:
            raise ValueError("The feature selection algorithm has already been run")
        
        # Call the appropriate function based on the direction
        if self.direction == 'forward':
            return self.__tefs_forward()
        else:
            return self.__tefs_backward()
        
    def __tefs_forward(self) -> None:
        """
        Perform the forward selection of features based on the Transfer Entropy score.

        :return: list of indexes of the selected features
        :rtype: List[IterationResult]
        """

        df = pd.DataFrame(self.features)
        selected_features = list()
        candidate_features = list(range(self.features.shape[1]))
        TE_cumulated = 0
        results = []
        iteration_count = 1

        while True:

            # check that there are still features to add
            if len(candidate_features) == 0:
                break

            if self.verbose >= 2:
                print(f"Iteration {iteration_count}")
                iteration_count += 1

            if self.verbose >= 2:
                print(f"Candidate Features: {[self.var_names[i] for i in candidate_features]}")
                print(f"Selected Features: {[self.var_names[i] for i in selected_features]}")

            # compute the TE scores for each feature
            feature_scores = score_features(
                features=df[candidate_features].values,
                target=self.target,
                conditioning=df[selected_features].values,
                k=self.k,
                lag_features=self.lag_features,
                lag_target=self.lag_target,
                direction="forward",
                n_jobs=self.n_jobs,
            )

            # i assume features_scores to be a dict wit
            # key = feature index and value = TE score
            # ex: {0: -0.5, 1: 0.5, 2: 0.7}

            feature_scores = dict(zip(df[candidate_features].columns, feature_scores))

            # print the scores
            if self.verbose >= 2:
                print(f"TE_cumulated: {TE_cumulated}")
                for key, value in feature_scores.items():
                    print(f"TE score of feature {self.var_names[key]}: {value}")

            # sort the scores in descending order by value
            feature_scores = dict(sorted(feature_scores.items(), key=lambda item: item[1], reverse=True))

            # find the first key value pair in the dictionary
            max_feature_index = int(next(iter(feature_scores)))
            max_TE = next(iter(feature_scores.values()))

            # increase the cumulative loss of information
            TE_cumulated += max(max_TE, 0)

            # by checking before selection of the feature
            # I DON'T make sure that at least one feature is selected

            results.append({"feature_scores": feature_scores, "TE": TE_cumulated})

            if self.verbose >= 1 and self.var_names is not None:
                print(f"Adding feature {self.var_names[max_feature_index]} with TE score: {max_TE}")

            # add the feature to the selected features list
            selected_features.append(max_feature_index)

            # remove the feature from the candidate features list
            candidate_features.remove(max_feature_index)

            if self.verbose >= 2:
                print("-" * 80)

        self.result = results
    
    def __tefs_backward(self) -> None:
        """
        Perform the backward selection of features based on the Transfer Entropy score.
        """

        df = pd.DataFrame(self.features)
        selected_features = list()
        candidate_features = list(range(self.features.shape[1]))
        TE_loss = 0
        results = []
        iteration_count = 1

        while True:
            # check that there are still features to remove
            if len(candidate_features) == 0:
                break

            if self.verbose >= 2:
                print(f"Iteration {iteration_count}")
                iteration_count += 1

            if self.verbose >= 2:
                print(f"Candidate Features: {[self.var_names[i] for i in candidate_features]}")
                print(f"Selected Features: {[self.var_names[i] for i in selected_features]}")

            # compute the TE scores for each feature
            feature_scores = score_features(
                features=df[candidate_features].values,
                target=self.target,
                conditioning=df[candidate_features].values,
                k=self.k,
                lag_features=self.lag_features,
                lag_target=self.lag_target,
                direction="backward",
                n_jobs=self.n_jobs,
            )

            # i assume features_scores to be a dict wit
            # key = feature index and value = TE score
            # ex: {0: -0.5, 1: 0.5, 2: 0.7}

            feature_scores = dict(zip(df[candidate_features].columns, feature_scores))

            # sort the scores in descending order by value
            feature_scores = dict(sorted(feature_scores.items(), key=lambda item: item[1], reverse=False))

            # print the scores
            if self.verbose >= 2:
                print(f"TE_loss: {TE_loss}")
                for key, value in feature_scores.items():
                    print(f"TE score of feature {self.var_names[key]}: {value}")

            # find the first key value pair in the dictionary
            min_feature_index = int(next(iter(feature_scores)))
            min_TE = next(iter(feature_scores.values()))

            # increase the cumulative loss of information
            TE_loss += max(min_TE, 0)

            results.append({"feature_scores": feature_scores, "TE": TE_loss})

            # by checking after the removal of the feature
            # I make it possible to not remove any feature

            if self.verbose >= 1 and self.var_names is not None:
                print(f"Removing feature {self.var_names[min_feature_index]} with TE score: {min_TE}")

            # add the feature to the selected features list
            selected_features.append(min_feature_index)

            # remove the feature from the candidate features list
            candidate_features.remove(min_feature_index)

            if self.verbose >= 2:
                print("-" * 80)

        self.result = results

    def plot_te_results(
            self,
            ax: matplotlib.axes.Axes,
            ) -> None:
        """
        Plot the results of the TE estimation for each iteration.

        :param scores_iterations: A list of results of the TE scores, one per iteration.
        :type scores_iterations: list[IterationResult]
        :param var_names: A list of variable names.
        :type var_names: list[str]
        :param ax: The axis to plot the results on.
        :type ax: matplotlib.axes.Axes
        """

        if self.result is None:
            raise ValueError("The feature selection algorithm has not been run yet")

        # Get a list of unique keys to create columns in the DataFrame
        unique_keys = set(key for d in self.result for key in d["feature_scores"].keys())

        # Initialize an empty list to store records with NaN values
        records_with_nan = []

        # Iterate through the dictionaries and create records with NaN values
        for d in self.result:
            record = {}
            for key in unique_keys:
                record[key] = d["feature_scores"].get(key, np.nan)
            records_with_nan.append(record)

        # Create the DataFrame from the records list
        df = pd.DataFrame.from_records(records_with_nan)

        # Add a column with the iteration number
        df["iteration"] = df.index + 1

        # Melt the DataFrame to convert columns to a 'variable' and 'value' format
        melted_data = pd.melt(df, id_vars=["iteration"], var_name="Variable", value_name="Value")

        # Map variable names to more readable names
        if self.var_names is not None:
            melted_data["Variable"] = melted_data["Variable"].map(dict(enumerate(self.var_names)))

        # Get the unique categories in the "Variable" column
        unique_categories = melted_data["Variable"].unique()

        # Create a line plot for each line using Seaborn
        sns.set_theme(style="whitegrid")
        sns.lineplot(
            data=melted_data,
            x="iteration",
            y="Value",
            hue="Variable",
            markers=["o"] * len(unique_categories),
            style="Variable",
            dashes=False,
            ax=ax,
        )

        # Add labels and title
        ax.set_xlabel("Iteration")
        ax.set_ylabel("TE score on target")
        ax.set_title("TE score on target for each iteration")

        # Add horizontal line in 0, thick and red
        ax.axhline(y=0, color="r", linestyle="--", linewidth=3)

        # Get unique x values with data
        x_values_with_data = melted_data["iteration"].unique()

        # Customize x-axis ticks to display only when there is data
        ax.set_xticks(x_values_with_data)

        # Show the plot
        ax.legend(title="Variables", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)


    def map_output_to_var_names(method):
        """
        A decorator that modifies the output of the decorated method. 
        Assumes the method returns a list of indices and uses these 
        indices to return the corresponding values from self.var_names.
        """
        def wrapper(self, *args, **kwargs):
            # Call the original method and get its output (list of indices)
            out = method(self, *args, **kwargs)
            
            # Map the output indices to the values in self.var_names
            mapped_output = [self.var_names[i] for i in out]
            
            return mapped_output
        
        return wrapper

    @map_output_to_var_names
    def select_features(
            self,
            threshold: float,
            ) -> List[int]:
        """
        Selects the features based on the transfer entropy scores and a threshold.

        :param threshold: The threshold to use for feature selection.
        :type threshold: float
        :return: A list of feature indexes representing the selected features.
        """

        stop = False
        initial_features = None

        for iteration in self.result:
            assert isinstance(iteration["TE"], (float, int)), "TE score must be a number"
            assert isinstance(iteration["feature_scores"], dict), "feature_scores must be a dictionary"

            if initial_features is None:
                initial_features = iteration["feature_scores"].keys()

            # First check if the threshold is reached
            if iteration["TE"] > threshold:
                if self.verbose >= 1:
                    print(f"Stopping condition reached: TE score {iteration['TE']} > threshold {threshold}")
                stop = True

            # Then check if the feature scores are all positive or negative
            if self.direction == "backward" and min(iteration["feature_scores"].values()) > 0:
                if self.verbose >= 1:
                    print("Stopping condition reached: all feature scores are positive")
                stop = True
            elif self.direction == "forward" and max(iteration["feature_scores"].values()) < 0:
                if self.verbose >= 1:
                    print("Stopping condition reached: all feature scores are negative")
                stop = True

            # If the stop condition is reached, return the list of selected features indexes
            if stop:
                if self.direction == "backward":
                    # In the backward case the remaining features are the ones that were not removed
                    return list(iteration["feature_scores"].keys())
                elif self.direction == "forward":
                    # In the forward case the remaining features are the ones that were not added yet
                    # so I need to take the initial set of features and remove the remaining ones
                    return list(set(initial_features) - set(iteration["feature_scores"].keys()))

        # If the loop reaches the end, no stop condition was reached
        if self.verbose >= 1:
            print("Stopping condition not reached")
        if self.direction == "backward":
            # Didn't reach the stop condition, I should return no features, but I decide to return the last feature left
            return list(self.result[-1]["feature_scores"].keys())
        elif self.direction == "forward":
            # Didn't reach the stop condition, I return all the features
            return list(initial_features)

    @map_output_to_var_names
    def select_n_features(
            self,
            n: int,
            ) -> List[int]:
        """
        Selects the `n` features with the highest or lowest score, regardless of the threshold and the transfer entropy scores.

        :param n: The number of features to select.
        :type n: int
        :param direction: The direction of selection. Can be either "forward" or "backward".
        :type direction: str
        :param verbose: Verbosity level, defaults to 0.
        :type verbose: int, optional
        :return: A list of feature indexes representing the selected features.
        """

        num_total_features = self.features.shape[1]
        assert 0 <= n <= num_total_features, f"n must be between 0 and {num_total_features}"

        if n == 0:
            return []

        if self.direction == "backward":

            for iteration in self.result:
                features_indexes = iteration["feature_scores"].keys()
                if len(features_indexes) == n:
                    return list(features_indexes)
            
        elif self.direction == "forward":

            initial_features = None

            for iteration in self.result:

                features_indexes = iteration["feature_scores"].keys()

                if initial_features is None:
                    initial_features = features_indexes

                if len(features_indexes) == num_total_features - n:
                    return list(set(initial_features) - set(features_indexes))
            
            # If the loop reaches the end, it means that n = num_total_features, so I return all the features
            return list(initial_features)



def score_features(
        features: np.ndarray,
        target: np.ndarray,
        conditioning: np.ndarray,
        k: int,
        lag_features: list[int],
        lag_target: list[int],
        direction: str,
        n_jobs=1,
        ) -> np.ndarray:
    """
    Computes the transfer entropy score for each feature :math:`X_i` in :math:`X`, to the target :math:`Y`, given the conditioning set :math:`X_A`.

    :param features: Sample of a (multivariate) random variable representing the input
    :type features: np.ndarray of shape (n_samples, n_features)
    :param target: Sample of a (multivariate) random variable representing the target
    :type target: np.ndarray of shape (n_samples, n_targets)
    :param conditioning: Sample of a (multivariate) random variable representing the conditioning
    :type conditioning: np.ndarray of shape (n_samples, n_conditioning)
    :param k: number of nearest neighbors for the CMI estimation
    :type k: int
    :param lag_features: the lags applied on features and conditioning
    :type lag_features: list[int]
    :param lag_target: the lags applied on the target
    :type lag_target: list[int]
    :param direction: the direction of the transfer entropy, either "forward" or "backward"
    :type direction: str
    :param n_jobs: number of parallel jobs to run
    :type n_jobs: int
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
    return estimate_conditional_transfer_entropy(*args)

