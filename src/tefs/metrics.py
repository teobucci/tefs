from typing import Any, Dict, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import BaseCrossValidator, cross_val_score

inputs_names_lags_doc = """
:param inputs_names_lags: A dictionary mapping input feature names to their corresponding list of lags. 
    For example, {'feature1': [1, 2], 'feature2': [1]} indicates 'feature1' should be lagged by 1 and 2 periods, 
    and 'feature2' by 1 period.
"""

target_name_doc = """
:param target_name: The name of the target variable in the DataFrame.
"""

def prepare_data_with_lags(
    df: pd.DataFrame, 
    inputs_names_lags: Dict[str, list[int]],
    target_name: str,
) -> pd.DataFrame:
    f"""
    Prepares data for regression by generating lagged features for specified variables and targets.
    
    :param df: The pandas DataFrame containing the time series data.
    {inputs_names_lags_doc}
    {target_name_doc}
    :return: A tuple containing the lagged features DataFrame and the target variable Series.
    """

    required_columns = set([*inputs_names_lags.keys(), target_name])
    if not required_columns.issubset(set(df.columns)):
        raise ValueError("DataFrame 'df' must contain all the columns specified in 'features_names' and 'targets_names'.")

    for lags in inputs_names_lags.values():
        if lags and min(lags) < 0:
            raise ValueError("Lag for independent variables must be a non-negative integer.")
    
    # Initialize a list to hold all DataFrame chunks
    lagged_chunks = []
    
    # Generate lagged inputs for the independent variables
    for input, lags in inputs_names_lags.items():
        for lag in lags:
            lagged_chunk = df[input].shift(lag).to_frame(f"{input}_t-{lag}")
            lagged_chunks.append(lagged_chunk)
    
    # Adding target column
    lagged_chunks.append(df[target_name].to_frame(target_name))

    # Concatenate chunks
    df_lagged = pd.concat(lagged_chunks, axis=1)
    
    # Dropping rows with NaN values caused by shifting
    df_lagged = df_lagged.dropna()
    
    return df_lagged.drop(columns=target_name), df_lagged[target_name]




def regression_analysis(
    inputs_names_lags: Dict[str, list[int]],
    target_name: str,
    df: Optional[pd.DataFrame] = None,
    cv_scheme: Optional[BaseCrossValidator] = None,
    df_train: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None
) -> Any:
    f"""
    Performs regression analysis with support for either cross-validation or a train-test split,
    based on the arguments provided.
    
    {inputs_names_lags_doc}
    {target_name_doc}
    :param df: DataFrame for cross-validation mode. If specified, cv_scheme must also be provided.
    :param cv_scheme: Cross-validator object for cross-validation mode. If specified, df must also be provided.
    :param df_train: Training DataFrame for train-test split mode. Required if df_test is provided.
    :param df_test: Testing DataFrame for train-test split mode. Requires df_train to be specified.
    :return: Cross-validated scores or R-squared scores from train-test evaluation.
    """

    # Check that exactly one mode is specified
    cross_val_mode = bool(df is not None and cv_scheme is not None)
    train_test_mode = bool(df_train is not None and df_test is not None)
    if not (cross_val_mode ^ train_test_mode):
        raise ValueError("Specify either cross-validation with 'cv_scheme' and 'df', or a train-test split with 'df_train' and 'df_test', not both.")
    
    if cross_val_mode:

        X, y = prepare_data_with_lags(
            df,
            inputs_names_lags,
            target_name,
        )
        
        model = LinearRegression()
        return cross_val_score(model, X, y, cv=cv_scheme)
    
    elif train_test_mode:
        
        X_train, y_train = prepare_data_with_lags(
            df_train,
            inputs_names_lags,
            target_name,
        )

        X_test, y_test = prepare_data_with_lags(
            df_test,
            inputs_names_lags,
            target_name,
        )
        
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)
