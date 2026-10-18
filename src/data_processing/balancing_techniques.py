import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def plot_class_distribution(df, target_column, title="Class Distribution"):
    """Helper function to visualize the class balance."""
    counts = df[target_column].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()


def plot_target_by_month(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    title: str = "Average Target % by Month",
    figsize: tuple = (12, 6),
    date_format: str = None,
    ylabel: str = "Target Rate (%)",
    color: str = "#1f77b4",
    show_grid: bool = True,
    show_values: bool = True,
):
    """Plot the average target percentage per month.

    Creates a line plot showing the monthly average of the target variable, useful for
    visualizing temporal trends in the target rate (e.g., default rate, fraud rate).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing date and target columns.
    date_column : str
        Name of the column containing datetime information. Will be converted to datetime
        if not already in datetime format.
    target_column : str
        Name of the binary target column (0/1 values).
    title : str, default="Average Target % by Month"
        Title for the plot.
    figsize : tuple, default=(12, 6)
        Figure size as (width, height) in inches.
    date_format : str, optional
        Date format string for parsing dates (e.g., '%Y-%m-%d'). If None, pandas will
        infer the format.
    ylabel : str, default="Target Rate (%)"
        Label for the y-axis.
    color : str, default="#1f77b4"
        Color for the line plot (matplotlib color code).
    show_grid : bool, default=True
        Whether to display grid lines.
    show_values : bool, default=True
        Whether to display percentage values on data points.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly aggregated statistics:
        - month: Month period
        - target_rate: Average target rate (0-1)
        - target_pct: Average target percentage (0-100)
        - count: Number of records per month

    Examples
    --------
    >>> # Basic usage
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=100, freq='D'),
    ...     'target': [0, 1, 0, 1, ...]
    ... })
    >>> stats = plot_target_by_month(df, 'date', 'target')

    >>> # Custom styling
    >>> stats = plot_target_by_month(
    ...     df, 'date', 'default_flag',
    ...     title='Monthly Default Rate',
    ...     ylabel='Default Rate (%)',
    ...     color='red',
    ...     show_values=False
    ... )

    >>> # With specific date format
    >>> stats = plot_target_by_month(
    ...     df, 'date_str', 'target',
    ...     date_format='%Y-%m-%dT%H:%M:%SZ'
    ... )
    """
    # Create a copy to avoid modifying original
    df_plot = df[[date_column, target_column]].copy()

    # Convert to datetime if not already
    if date_format:
        df_plot[date_column] = pd.to_datetime(df_plot[date_column], format=date_format)
    else:
        df_plot[date_column] = pd.to_datetime(df_plot[date_column])

    # Extract month period
    df_plot["month"] = df_plot[date_column].dt.to_period("M")

    # Calculate monthly statistics
    monthly_stats = (
        df_plot.groupby("month")
        .agg(target_rate=(target_column, "mean"), count=(target_column, "count"))
        .reset_index()
    )

    # Convert to percentage
    monthly_stats["target_pct"] = monthly_stats["target_rate"] * 100

    # Convert period to timestamp for plotting
    monthly_stats["month_dt"] = monthly_stats["month"].dt.to_timestamp()

    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(
        monthly_stats["month_dt"],
        monthly_stats["target_pct"],
        marker="o",
        linewidth=2,
        markersize=8,
        color=color,
        label="Target %",
    )

    # Add value labels on points if requested
    if show_values:
        for idx, row in monthly_stats.iterrows():
            plt.annotate(
                f"{row['target_pct']:.1f}%",
                (row["month_dt"], row["target_pct"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                alpha=0.8,
            )

    # Formatting
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if show_grid:
        plt.grid(True, alpha=0.3, linestyle="--")

    plt.legend(fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Print summary statistics
    print("\nMonthly Target Rate Summary:")
    print(f"  Average: {monthly_stats['target_pct'].mean():.2f}%")
    print(
        f"  Min: {monthly_stats['target_pct'].min():.2f}% ({monthly_stats.loc[monthly_stats['target_pct'].idxmin(), 'month']})"
    )
    print(
        f"  Max: {monthly_stats['target_pct'].max():.2f}% ({monthly_stats.loc[monthly_stats['target_pct'].idxmax(), 'month']})"
    )
    print(f"  Std Dev: {monthly_stats['target_pct'].std():.2f}%")
    print(f"\nTotal months: {len(monthly_stats)}")
    print(f"Total records: {monthly_stats['count'].sum()}")

    # Return the statistics dataframe
    return monthly_stats[["month", "target_rate", "target_pct", "count"]]


def undersample_data(df: pd.DataFrame, target_column: str, random_state: int = 42):
    """Perform random undersampling of the majority class."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    undersampler = RandomUnderSampler(random_state=random_state)
    X_res, y_res = undersampler.fit_resample(X, y)
    df_res = pd.concat([X_res, y_res], axis=1)

    plot_class_distribution(df_res, target_column, "After Undersampling")
    return df_res


def oversample_data(df: pd.DataFrame, target_column: str, random_state: int = 42):
    """Perform random oversampling of the minority class."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    oversampler = RandomOverSampler(random_state=random_state)
    X_res, y_res = oversampler.fit_resample(X, y)
    df_res = pd.concat([X_res, y_res], axis=1)

    plot_class_distribution(df_res, target_column, "After Oversampling")
    return df_res


def smote_data(
    df: pd.DataFrame, target_column: str, random_state: int = 42, k_neighbors: int = 5
):
    """Perform SMOTE (Synthetic Minority Over-sampling Technique)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    df_res = pd.concat([X_res, y_res], axis=1)

    plot_class_distribution(df_res, target_column, "After SMOTE")
    return df_res


def smotenc_data(
    df: pd.DataFrame, target_column: str, random_state: int = 42, k_neighbors: int = 5
):
    """Perform SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and
    Continuous).

    Automatically detects categorical and numerical features.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Automatically detect categorical features
    categorical_features = []

    for i, col in enumerate(X.columns):
        # Check if column is categorical based on dtype or unique value ratio
        if (
            X[col].dtype == "object"
            or X[col].dtype.name == "category"
            or X[col].dtype == "bool"
            or (
                X[col].dtype in ["int64", "int32"]
                and X[col].nunique() <= 20
                and X[col].min() >= 0
            )
        ):
            categorical_features.append(i)

    print(
        f"Detected {len(categorical_features)} categorical features out of {len(X.columns)} total features"
    )
    print(f"Categorical feature indices: {categorical_features}")

    # Print categorical column names for verification
    if categorical_features:
        cat_column_names = [X.columns[i] for i in categorical_features]
        print(f"Categorical columns: {cat_column_names}")

    # Use SMOTENC if there are categorical features, otherwise use regular SMOTE
    if categorical_features:
        smotenc = SMOTENC(
            categorical_features=categorical_features,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        print("Using SMOTENC for mixed data types")
    else:
        # Fall back to regular SMOTE if no categorical features detected
        from imblearn.over_sampling import SMOTE

        smotenc = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        print("No categorical features detected, using regular SMOTE")

    X_res, y_res = smotenc.fit_resample(X, y)
    df_res = pd.concat(
        [pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_column)],
        axis=1,
    )

    plot_class_distribution(df_res, target_column, "After SMOTENC")
    return df_res


def adasyn_data(
    df: pd.DataFrame, target_column: str, random_state: int = 42, n_neighbors: int = 5
):
    """Perform ADASYN (Adaptive Synthetic Sampling)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    adasyn = ADASYN(random_state=random_state, n_neighbors=n_neighbors)
    X_res, y_res = adasyn.fit_resample(X, y)
    df_res = pd.concat([X_res, y_res], axis=1)

    plot_class_distribution(df_res, target_column, "After ADASYN")
    return df_res


def balance_visual_demo(df, target_column):
    """Show original vs resampled distributions using all three methods."""
    plot_class_distribution(df, target_column, "Original Distribution")

    _ = undersample_data(df, target_column)
    _ = oversample_data(df, target_column)
    _ = smote_data(df, target_column)


def balance_data(
    df: pd.DataFrame,
    target_column: str,
    method: str = "oversample",
    random_state: int = 42,
    k_neighbors: int = 5,
    show_plot: bool = True,
    numeric_only: bool = True,
    no_nan_columns: bool = False,
):
    """Wrapper function to perform data balancing using different resampling techniques.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target.
    target_column : str
        Name of the target column.
    method : str, default="oversample"
        Resampling method to use. Options: "undersample", "oversample", "smote", "smotenc", "adasyn".
    random_state : int, default=42
        Random state for reproducibility.
    k_neighbors : int, default=5
        Number of neighbors for SMOTE/SMOTENC (only used when method="smote" or "smotenc").
        For ADASYN, this parameter is called n_neighbors.
    show_plot : bool, default=True
        Whether to display the class distribution plot.
    numeric_only : bool, default=False
        If True, only numeric columns will be used for SMOTE and ADASYN methods.
        This parameter is ignored for other methods.
    no_nan_columns : bool, default=False
        If True, only columns with no NaN values will be used (excludes RequestDateTime column).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame.

    Raises
    ------
    ValueError
        If method is not one of the supported options.

    Examples
    --------
    >>> # Undersample the data
    >>> df_balanced = balance_data(df, "target", method="undersample")

    >>> # Oversample the data without showing plot
    >>> df_balanced = balance_data(df, "target", method="oversample", show_plot=False)

    >>> # Apply SMOTE with custom parameters
    >>> df_balanced = balance_data(df, "target", method="smote", k_neighbors=3)

    >>> # Apply SMOTENC for mixed data types
    >>> df_balanced = balance_data(df, "target", method="smotenc", k_neighbors=3)

    >>> # Apply ADASYN with numeric columns only
    >>> df_balanced = balance_data(df, "target", method="adasyn", k_neighbors=5, numeric_only=True)

    >>> # Apply SMOTE with only non-null columns
    >>> df_balanced = balance_data(df, "target", method="smote", no_nan_columns=True)
    """
    valid_methods = ["undersample", "oversample", "smote", "smotenc", "adasyn"]

    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")

    # Start with original dataframe
    df_to_use = df

    # Filter to non-null columns if requested
    if no_nan_columns:
        selected_columns = []
        for col in df.columns:
            if col.lower() == "requestdatetime":
                continue
            if df[col].isnull().sum() == 0:
                selected_columns.append(col)

        # Ensure target column is included
        if target_column not in selected_columns:
            selected_columns.append(target_column)

        print(
            f"Using non-null columns only: {len(selected_columns)} out of {len(df.columns)} columns"
        )
        print(f"Non-null columns: {selected_columns}")

        df_to_use = df[selected_columns].copy()
        print(f"Filtered DataFrame shape: {df_to_use.shape}")

    # Filter to numeric columns only for SMOTE and ADASYN if requested
    if numeric_only and method in ["smote", "adasyn"]:
        X = df_to_use.drop(columns=[target_column])
        numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
        print(
            f"Using numeric columns only: {len(numeric_columns)} out of {len(X.columns)} columns"
        )
        print(f"Numeric columns: {numeric_columns}")

        # Create DataFrame with only numeric columns plus target
        df_to_use = df_to_use[numeric_columns + [target_column]].copy()
        print(f"Final filtered DataFrame shape: {df_to_use.shape}")

    if show_plot:
        plot_class_distribution(df_to_use, target_column, "Original Distribution")

    if method == "undersample":
        return undersample_data(df_to_use, target_column, random_state)
    elif method == "oversample":
        return oversample_data(df_to_use, target_column, random_state)
    elif method == "smote":
        return smote_data(df_to_use, target_column, random_state, k_neighbors)
    elif method == "smotenc":
        return smotenc_data(df_to_use, target_column, random_state, k_neighbors)
    elif method == "adasyn":
        return adasyn_data(df_to_use, target_column, random_state, k_neighbors)


def calculate_class_ratio(data, target_column: str = None):
    """Calculate the ratio of negative instances to positive instances.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input DataFrame containing the target column, or Series with target values.
    target_column : str, optional
        Name of the target column. Required if data is a DataFrame.
        Ignored if data is a Series.

    Returns
    -------
    float
        Ratio of negative instances (0) to positive instances (1).
        Returns inf if there are no positive instances.
        Returns 0.0 if there are no negative instances.

    Examples
    --------
    >>> df = pd.DataFrame({'target': [0, 0, 0, 1, 1]})
    >>> calculate_class_ratio(df, 'target')
    1.5

    >>> series = pd.Series([0, 0, 0, 1, 1])
    >>> calculate_class_ratio(series)
    1.5
    """
    if isinstance(data, pd.Series):
        value_counts = data.value_counts()
    elif isinstance(data, pd.DataFrame):
        if target_column is None:
            raise ValueError("target_column must be specified when data is a DataFrame")
        value_counts = data[target_column].value_counts()
    else:
        raise TypeError("data must be a pandas DataFrame or Series")

    negative_count = value_counts.get(0, 0)
    positive_count = value_counts.get(1, 0)

    if positive_count == 0:
        return float("inf")

    return negative_count / positive_count