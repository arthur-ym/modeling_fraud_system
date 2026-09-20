"""Utilities for model evaluation and visualization."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    set_name: str = "validation",
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for model predictions.

    Computes accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and KS statistic
    for the given predictions and true labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).
    y_pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    set_name : str, default="validation"
        Name of the dataset being evaluated (for logging purposes).

    Returns
    -------
    dict
        Dictionary containing the following metrics:
        - accuracy: float
        - precision: float
        - recall: float
        - f1: float
        - auc: float (ROC-AUC)
        - pr_auc: float (Precision-Recall AUC)
        - ks_statistic: float (Kolmogorov-Smirnov statistic)
        - ks_pvalue: float (p-value for KS test)
        - precision_curve: ndarray
        - recall_curve: ndarray
        - thresholds: ndarray

    Examples
    --------
    >>> from evaluation_utils import evaluate_model
    >>> metrics = evaluate_model(y_true, y_pred, y_pred_proba, "test")
    >>> print(f"Test AUC: {metrics['auc']:.4f}")
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Calculate precision-recall curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(
        y_true, y_pred_proba
    )
    pr_auc = auc(recall_curve, precision_curve)

    # Calculate KS statistic
    pos_proba = y_pred_proba[y_true == 1]
    neg_proba = y_pred_proba[y_true == 0]
    ks_statistic, ks_pvalue = ks_2samp(pos_proba, neg_proba)

    log.info(
        f"{set_name.capitalize()} Metrics - AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, KS: {ks_statistic:.4f}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc,
        "pr_auc": pr_auc,
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "thresholds": thresholds,
    }


def get_max_f1_threshold(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Find the threshold that maximizes the F1 score.

    Calculates predicted probabilities from the model and evaluates F1 scores
    across all possible thresholds from the precision-recall curve to find
    the optimal threshold.

    Parameters
    ----------
    model : estimator object
        Trained classifier with predict_proba method that returns probabilities
        for binary classification.
    X : array-like of shape (n_samples, n_features)
        Feature matrix for predictions.
    y : array-like of shape (n_samples,)
        True binary labels (0 or 1).

    Returns
    -------
    float
        Threshold value that maximizes the F1 score.

    Examples
    --------
    >>> from evaluation_utils import get_max_f1_threshold
    >>> optimal_threshold = get_max_f1_threshold(model, X_val, y_val)
    >>> print(f"Optimal threshold: {optimal_threshold:.4f}")
    >>> y_pred = (model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
    """
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * recall * precision / (recall + precision)

    # Return threshold with maximum F1 score
    return thresholds[np.argmax(f1_scores)]


def plot_pr_curve(
    metrics: Dict[str, Any],
    set_name: str = "validation",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Create precision-recall curve plot.

    Generates a matplotlib figure showing the precision-recall curve
    for model predictions.

    Parameters
    ----------
    metrics : dict
        Dictionary containing 'recall_curve', 'precision_curve', and 'pr_auc' keys
        as returned by evaluate_model function.
    set_name : str, default="validation"
        Name of the dataset (used in label).
    title : str, optional
        Custom title for the plot. If None, generates title from set_name.
    figsize : tuple of float, default=(8, 6)
        Figure size as (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> from evaluation_utils import evaluate_model, plot_pr_curve
    >>> metrics = evaluate_model(y_true, y_pred, y_pred_proba, "test")
    >>> fig = plot_pr_curve(metrics, set_name="test")
    >>> fig.savefig("pr_curve.png")
    >>> plt.close(fig)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        metrics["recall_curve"],
        metrics["precision_curve"],
        label=f"{set_name.capitalize()} (PR-AUC={metrics['pr_auc']:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    if title is None:
        title = f"Precision-Recall Curve - {set_name.capitalize()}"
    ax.set_title(title)

    ax.legend()
    ax.grid(True)

    return fig


def plot_ks_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    set_name: str = "validation",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Create Kolmogorov-Smirnov (KS) curve plot.

    Generates a matplotlib figure showing the cumulative distribution functions
    for positive and negative classes, along with the KS statistic.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    set_name : str, default="validation"
        Name of the dataset (used in title and label).
    title : str, optional
        Custom title for the plot. If None, generates title from set_name.
    figsize : tuple of float, default=(8, 6)
        Figure size as (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Notes
    -----
    The KS curve shows the cumulative distribution of predicted probabilities
    for both positive and negative classes. The KS statistic is the maximum
    vertical distance between these two curves, indicating the model's ability
    to discriminate between classes.

    Examples
    --------
    >>> from evaluation_utils import plot_ks_curve
    >>> fig = plot_ks_curve(y_true, y_pred_proba, set_name="test")
    >>> fig.savefig("ks_curve.png")
    >>> plt.close(fig)
    """
    import pandas as pd
    import seaborn as sns

    # Prepare data for plotting
    df = pd.DataFrame()
    df["real"] = y_true
    df["proba"] = y_pred_proba

    # Calculate KS statistic
    class0 = df[df["real"] == 0]
    class1 = df[df["real"] == 1]
    ks_statistic, ks_pvalue = ks_2samp(class0["proba"], class1["proba"])

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.ecdfplot(df, x="proba", hue="real", ax=ax)

    # Add KS statistic to plot
    ax.text(
        0.02,
        0.98,
        f"KS={ks_statistic:.3f} (p={ks_pvalue:.4f})",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Cumulative Distribution")

    if title is None:
        title = f"KS Curve - {set_name.capitalize()}"
    ax.set_title(title)

    ax.grid(True)

    return fig


def plot_probability_histogram(
    y_pred_proba: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    set_name: str = "validation",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    bins: int = 50,
) -> plt.Figure:
    """Plot histogram of predicted probabilities.

    Generates a matplotlib figure showing the distribution of predicted probabilities.
    Optionally can show separate distributions for positive and negative classes.

    Parameters
    ----------
    y_pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    y_true : array-like of shape (n_samples,), optional
        True binary labels (0 or 1). If provided, will show separate histograms
        for each class.
    set_name : str, default="validation"
        Name of the dataset (used in title).
    title : str, optional
        Custom title for the plot. If None, generates a default title.
    figsize : tuple of float, default=(10, 6)
        Figure size as (width, height) in inches.
    bins : int, default=50
        Number of bins for the histogram.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> from evaluation_utils import plot_probability_histogram
    >>> # Simple histogram
    >>> fig = plot_probability_histogram(y_pred_proba, set_name="test")
    >>> fig.savefig("prob_histogram.png")
    >>> plt.close(fig)

    >>> # Histogram split by true class
    >>> fig = plot_probability_histogram(y_pred_proba, y_true=y_true, set_name="test")
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    if y_true is not None:
        # Plot separate histograms for each class
        pos_proba = y_pred_proba[y_true == 1]
        neg_proba = y_pred_proba[y_true == 0]

        ax.hist(
            neg_proba,
            bins=bins,
            alpha=0.6,
            label=f"Negative Class (n={len(neg_proba)})",
            color="blue",
            edgecolor="black",
        )
        ax.hist(
            pos_proba,
            bins=bins,
            alpha=0.6,
            label=f"Positive Class (n={len(pos_proba)})",
            color="red",
            edgecolor="black",
        )
        ax.legend()
    else:
        # Plot single histogram
        ax.hist(
            y_pred_proba, bins=bins, alpha=0.7, color="steelblue", edgecolor="black"
        )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")

    if title is None:
        title = f"Predicted Probability Distribution - {set_name.capitalize()}"
    ax.set_title(title)

    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"Mean: {y_pred_proba.mean():.3f}\nMedian: {np.median(y_pred_proba):.3f}\nStd: {y_pred_proba.std():.3f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return fig


def plot_calibration_bins(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    set_name: str = "validation",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    n_bins: int = 10,
) -> plt.Figure:
    """Plot calibration curve showing actual vs predicted probability by bins.

    Creates a bar plot where x-axis shows probability bins and y-axis shows
    the actual target rate (percentage) within each bin. This helps visualize
    model calibration.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred_proba : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    set_name : str, default="validation"
        Name of the dataset (used in title).
    title : str, optional
        Custom title for the plot. If None, generates a default title.
    figsize : tuple of float, default=(10, 6)
        Figure size as (width, height) in inches.
    n_bins : int, default=10
        Number of bins to divide probabilities into (e.g., 10 = deciles:
        0-0.1, 0.1-0.2, etc.).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Notes
    -----
    A well-calibrated model will show bars close to a diagonal line from 0 to 1.
    If bars are consistently below the bin centers, the model is overconfident.
    If bars are consistently above, the model is underconfident.

    Examples
    --------
    >>> from evaluation_utils import plot_calibration_bins
    >>> fig = plot_calibration_bins(y_true, y_pred_proba, set_name="test")
    >>> fig.savefig("calibration_bins.png")
    >>> plt.close(fig)

    >>> # Use 20 bins for finer granularity
    >>> fig = plot_calibration_bins(y_true, y_pred_proba, n_bins=20)
    >>> plt.show()
    """
    import pandas as pd

    # Create dataframe
    df = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_pred_proba})

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(n_bins)]

    # Assign each prediction to a bin
    df["bin"] = pd.cut(
        df["y_pred_proba"], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    # Calculate average target rate per bin
    bin_stats = (
        df.groupby("bin", observed=True)
        .agg(
            avg_target_rate=("y_true", "mean"),
            count=("y_true", "size"),
            bin_center=("y_pred_proba", "mean"),
        )
        .reset_index()
    )

    # Convert to percentage
    bin_stats["avg_target_rate_pct"] = bin_stats["avg_target_rate"] * 100

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    x_pos = np.arange(len(bin_stats))
    bars = ax.bar(
        x_pos,
        bin_stats["avg_target_rate_pct"],
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )

    # Add percentage labels on top of bars
    for i, (bar, pct) in enumerate(zip(bars, bin_stats["avg_target_rate_pct"])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Predicted Probability Bin")
    ax.set_ylabel("Actual Target Rate (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats["bin"], rotation=45, ha="right")

    if title is None:
        title = f"Calibration Curve (Binned) - {set_name.capitalize()}"
    ax.set_title(title)

    ax.grid(True, alpha=0.3, axis="y")

    # Set y-axis to 0-100%
    ax.set_ylim(0, 100)

    plt.tight_layout()

    return fig


def plot_learning_curves(
    model: Any,
    metric: str = "logloss",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot learning curves from XGBoost or CatBoost model training history.

    Generates a matplotlib figure showing the evolution of a specified metric
    across training iterations for both training and validation sets. If the
    specified metric is not found, automatically uses the first available metric.

    Parameters
    ----------
    model : XGBClassifier, XGBRegressor, CatBoostClassifier, or CatBoostRegressor
        Trained model with evaluation results. Must have been trained with
        eval_set parameter.
    metric : str, default='logloss'
        The metric to plot. Common options:

        - XGBoost: 'logloss', 'error', 'auc', 'aucpr', 'rmse', 'mae'
        - CatBoost: 'Logloss', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1'

        If not found, will automatically use the first available metric.
    title : str, optional
        Custom title for the plot. If None, generates a default title.
    figsize : tuple of float, default=(10, 6)
        Figure size as (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Raises
    ------
    ValueError
        If the model has no evaluation results or no evaluation sets are found.

    Notes
    -----
    The model must have been trained with eval_set parameter to have evaluation
    results available.

    For XGBoost:
        - Expects evaluation sets named 'validation_0' (training set)
          and 'validation_1' (validation/test set).
        - Access results via model.evals_result()

    For CatBoost:
        - Expects 'learn' (training set) and 'validation' (validation set).
        - Access results via model.evals_result_

    Examples
    --------
    >>> from evaluation_utils import plot_learning_curves
    >>> from xgboost import XGBClassifier
    >>>
    >>> # Train model with evaluation set
    >>> model = XGBClassifier()
    >>> model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    >>>
    >>> # Plot learning curves
    >>> fig = plot_learning_curves(model, metric='logloss')
    >>> fig.savefig("learning_curves.png")
    >>> plt.close(fig)

    >>> # Plot AUC curves with custom title
    >>> fig = plot_learning_curves(model, metric='auc',
    ...                            title='Model AUC Over Iterations')
    >>> plt.show()
    """
    # Check if model is CatBoost or XGBoost
    is_catboost = hasattr(model, "evals_result_")

    if is_catboost:
        # CatBoost model
        results = model.evals_result_

        if not results:
            raise ValueError(
                "Model has no evaluation results. Ensure the model was trained with eval_set parameter."
            )

        # CatBoost structure: {'learn': {'Logloss': [...]}, 'validation': {'Logloss': [...]}}
        eval_sets = list(results.keys())
        if not eval_sets:
            raise ValueError("No evaluation sets found in model results.")

        available_metrics = list(results[eval_sets[0]].keys())

        # Auto-detect metric if specified one is not found
        if metric not in results[eval_sets[0]]:
            # Try case-insensitive match for CatBoost
            metric_lower = metric.lower()
            matched_metric = None
            for m in available_metrics:
                if m.lower() == metric_lower:
                    matched_metric = m
                    break

            if matched_metric:
                metric = matched_metric
            elif available_metrics:
                original_metric = metric
                metric = available_metrics[0]
                log.warning(
                    f"Metric '{original_metric}' not found. Using '{metric}' instead. Available metrics: {available_metrics}"
                )
            else:
                raise ValueError("No metrics found in evaluation results.")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot learning curves for each evaluation set
        for eval_set in eval_sets:
            if metric in results[eval_set]:
                # Determine label based on eval_set name
                if eval_set == "learn":
                    label = "Train"
                elif eval_set == "validation":
                    label = "Validation"
                else:
                    label = eval_set.capitalize()

                ax.plot(results[eval_set][metric], label=label, linewidth=2)
    else:
        # XGBoost model
        results = model.evals_result()

        if not results:
            raise ValueError(
                "Model has no evaluation results. Ensure the model was trained with eval_set parameter."
            )

        # Check if metric exists in results
        eval_sets = list(results.keys())
        if not eval_sets:
            raise ValueError("No evaluation sets found in model results.")

        available_metrics = list(results[eval_sets[0]].keys())

        # Auto-detect metric if specified one is not found
        if metric not in results[eval_sets[0]]:
            if available_metrics:
                original_metric = metric
                metric = available_metrics[0]
                log.warning(
                    f"Metric '{original_metric}' not found. Using '{metric}' instead. Available metrics: {available_metrics}"
                )
            else:
                raise ValueError("No metrics found in evaluation results.")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot learning curves for each evaluation set
        for eval_set in eval_sets:
            if metric in results[eval_set]:
                # Determine label based on eval_set name
                if eval_set == "validation_0":
                    label = "Train"
                elif eval_set == "validation_1":
                    label = "Validation"
                else:
                    label = eval_set.replace("validation_", "Set ")

                ax.plot(results[eval_set][metric], label=label, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.capitalize())

    if title is None:
        title = f"Learning Curves - {metric.capitalize()}"
    ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: Optional[Union[List[str], np.ndarray]] = None,
    max_features: int = 20,
    importance_type: str = "weight",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Plot feature importance from a trained model.

    Generates a horizontal bar plot showing the most important features from a trained
    model. Supports XGBoost, LightGBM, CatBoost, and scikit-learn models with
    feature_importances_ attribute.

    Parameters
    ----------
    model : estimator object
        Trained model with feature_importances_ attribute or get_booster() method.
        Supported models:

        - XGBoost: XGBClassifier, XGBRegressor
        - LightGBM: LGBMClassifier, LGBMRegressor
        - CatBoost: CatBoostClassifier, CatBoostRegressor
        - scikit-learn: RandomForestClassifier, GradientBoostingClassifier, etc.

    feature_names : list of str or array-like, optional
        Names of features corresponding to model features. If None:

        - For models with feature_names_in_ attribute, uses those names
        - For XGBoost, tries to extract from model or uses f0, f1, ...
        - Otherwise, uses generic names like 'Feature 0', 'Feature 1', ...

    max_features : int, default=20
        Maximum number of top features to display in the plot.
    importance_type : str, default='weight'
        Type of feature importance for XGBoost models:

        - 'weight': Number of times a feature appears in trees
        - 'gain': Average gain when feature is used for splitting
        - 'cover': Average coverage of feature when it is used
        - 'total_gain': Total gain of splits which use the feature
        - 'total_cover': Total coverage of splits which use the feature

        For non-XGBoost models, this parameter is ignored and the model's
        default feature_importances_ is used.
    title : str, optional
        Custom title for the plot. If None, generates a default title showing
        the number of features displayed.
    figsize : tuple of float, default=(10, 8)
        Figure size as (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Raises
    ------
    AttributeError
        If the model doesn't have feature importance capabilities (no
        get_booster() method and no feature_importances_ attribute).

    Notes
    -----
    The function automatically detects the model type and extracts feature
    importances accordingly. For XGBoost models, different importance types
    can provide different insights:

    - Use 'gain' to find features that provide the most improvement
    - Use 'weight' to find features used most frequently
    - Use 'cover' to find features that affect the most samples

    Examples
    --------
    >>> from evaluation_utils import plot_feature_importance
    >>> from xgboost import XGBClassifier
    >>>
    >>> # For XGBoost model with gain importance
    >>> model = XGBClassifier()
    >>> model.fit(X_train, y_train)
    >>> fig = plot_feature_importance(model, feature_names=X_train.columns,
    ...                                importance_type='gain')
    >>> fig.savefig("feature_importance.png")
    >>> plt.close(fig)

    >>> # For scikit-learn model
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf_model = RandomForestClassifier()
    >>> rf_model.fit(X_train, y_train)
    >>> fig = plot_feature_importance(rf_model, feature_names=feature_list,
    ...                                max_features=15)
    >>> plt.show()

    >>> # Show top 30 features with custom title
    >>> fig = plot_feature_importance(model, feature_names=X.columns,
    ...                                max_features=30,
    ...                                title='Top 30 Most Important Features')
    """
    import pandas as pd

    # Try to get feature importances from different model types
    try:
        # Check if it's an XGBoost model with get_booster method
        if hasattr(model, "get_booster"):
            # XGBoost model
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)

            # If feature_names not provided, try to get from model
            if feature_names is None:
                if hasattr(model, "feature_names_in_"):
                    feature_names = model.feature_names_in_
                else:
                    # Use feature indices from importance dict
                    feature_names = list(importance_dict.keys())

            # Create a mapping if feature_names is provided
            if feature_names is not None and len(feature_names) > 0:
                # Map f0, f1, ... to actual feature names
                name_map = {f"f{i}": name for i, name in enumerate(feature_names)}
                importance_dict = {
                    name_map.get(k, k): v for k, v in importance_dict.items()
                }

            # Convert to pandas Series
            importances = pd.Series(importance_dict)

        elif hasattr(model, "feature_importances_"):
            # Scikit-learn style models (RandomForest, GradientBoosting, etc.)
            importances = model.feature_importances_

            # Get feature names
            if feature_names is None:
                if hasattr(model, "feature_names_in_"):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = [f"Feature {i}" for i in range(len(importances))]

            # Convert to pandas Series
            importances = pd.Series(importances, index=feature_names)

        else:
            raise AttributeError("Model does not have feature importance capabilities.")

        # Sort by importance and get top features
        importances = importances.sort_values(ascending=True).tail(max_features)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar plot
        importances.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")

        ax.set_xlabel(f"Importance ({importance_type})")
        ax.set_ylabel("Features")

        if title is None:
            title = f"Top {len(importances)} Feature Importance"
        ax.set_title(title)

        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        return fig

    except Exception as e:
        log.error(f"Error plotting feature importance: {e}")
        raise