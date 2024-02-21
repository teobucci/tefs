from .types import IterationResult
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns


def list_of_dicts_to_dataframe(
        data: List[IterationResult],
        ) -> pd.DataFrame:
    """
    Helper function to convert a list of dictionaries to a pandas DataFrame, used to plot the results of the TE estimation.

    :param data: A list of results of the TE scores, one per iteration.
    :type data: List[IterationResult]
    :return: A pandas DataFrame with the results of the TE estimation.
    :rtype: pd.DataFrame
    """
    # Get a list of unique keys to create columns in the DataFrame
    unique_keys = set(key for d in data for key in d["feature_scores"].keys())

    # Initialize an empty list to store records with NaN values
    records_with_nan = []

    # Iterate through the dictionaries and create records with NaN values
    for d in data:
        record = {}
        for key in unique_keys:
            record[key] = d["feature_scores"].get(key, np.nan)
        records_with_nan.append(record)

    # Create the DataFrame from the records list
    df = pd.DataFrame.from_records(records_with_nan)

    # Add a column with the iteration number
    df["iteration"] = df.index + 1

    return df


def plot_te_results(
        scores_iterations: List[IterationResult],
        var_names: List[str],
        ax: matplotlib.axes.Axes,
        ) -> None:
    """
    Plot the results of the TE estimation for each iteration.

    :param scores_iterations: A list of results of the TE scores, one per iteration.
    :type scores_iterations: List[IterationResult]
    :param var_names: A list of variable names.
    :type var_names: List[str]
    :param ax: The axis to plot the results on.
    :type ax: matplotlib.axes.Axes
    """
    df = list_of_dicts_to_dataframe(scores_iterations)

    # Melt the DataFrame to convert columns to a 'variable' and 'value' format
    melted_data = pd.melt(df, id_vars=["iteration"], var_name="Variable", value_name="Value")

    # Map variable names to more readable names
    melted_data["Variable"] = melted_data["Variable"].map(dict(enumerate(var_names)))

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
