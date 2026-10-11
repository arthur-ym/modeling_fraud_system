"""Optuna search for XGBoost hyperparameters on a prepared fraud workflow.

The search maximizes validation PR-AUC. Each trial still stops on validation
logloss, with the same patience and minimum improvement as ``train_model``.
The test set is not scored during the search. No resampling and no
``scale_pos_weight`` are applied here; pass frames that already have the
natural class rate.

Edit ``TOP_40_FEATURES`` to change the default column list. Each row is
``[rank, feature name, gain]``. Training uses the feature name only.
"""

import logging

import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from src.model_training.training import ModelTrainingWorkflow

log = logging.getLogger(__name__)

# Edit this table to change the default top-40 columns.
# Each row is [rank, feature name, gain]. Only the feature name is trained on.
TOP_40_FEATURES: list[list] = [
    [1, "card1_target_rate", 2898.5080566406],
    [2, "card1_x_product_channel_target_rate", 2619.8493652344],
    [3, "C2", 1788.8270263672],
    [4, "V217", 1463.5167236328],
    [5, "V279", 1325.2569580078],
    [6, "C11", 1288.9261474609],
    [7, "V29", 1281.1726074219],
    [8, "V283", 1202.521484375],
    [9, "V95", 1136.4382324219],
    [10, "V312", 1072.3162841797],
    [11, "C6", 985.2222290039],
    [12, "V282", 894.4429321289],
    [13, "C12", 891.3984375],
    [14, "V258", 871.4403686523],
    [15, "C14", 862.3701782227],
    [16, "card_D10_std", 859.451965332],
    [17, "V313", 833.7211914062],
    [18, "card_C9_mean", 800.0765991211],
    [19, "V57", 793.2509155273],
    [20, "flag_outsider15", 753.5994262695],
    [21, "V69", 727.9138793945],
    [22, "V90", 709.7984008789],
    [23, "D3", 705.2971801758],
    [24, "V53", 701.7498168945],
    [25, "C1", 693.0908813477],
    [26, "card_C13_nunique", 680.6305541992],
    [27, "V265", 663.9107666016],
    [28, "V314", 645.0280761719],
    [29, "amount_cents", 589.6865234375],
    [30, "D2", 581.2379150391],
    [31, "V35", 529.9074707031],
    [32, "card_V309_nunique", 527.7366943359],
    [33, "V285", 516.6747436523],
    [34, "C13", 514.0857543945],
    [35, "V288", 506.8506774902],
    [36, "purchaser_email_domain_target_rate", 505.3368835449],
    [37, "V309", 483.2976989746],
    [38, "V221", 468.3073730469],
    [39, "V284", 459.7435913086],
    [40, "V131", 456.041015625],
]


def feature_names(feature_table: list) -> list[str]:
    """Return the feature-name column from ``[rank, name, gain]`` rows.

    Parameters
    ----------
    feature_table : list
        Rows of ``[rank, feature name, gain]``. Rank and gain are ignored.

    Returns
    -------
    list of str
        Feature names in table order.

    Raises
    ------
    ValueError
        If the table is empty, a row has no name, or a name is repeated.
    """
    if not feature_table:
        raise ValueError("Feature table is empty. Add at least one [rank, name, gain] row.")
    names: list[str] = []
    for row in feature_table:
        if len(row) < 2 or not isinstance(row[1], str) or not row[1].strip():
            raise ValueError(
                "Each feature row must be [rank, feature name, gain]. "
                f"Got {row!r}."
            )
        names.append(row[1])
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate feature names: {sorted(duplicates)}.")
    return names


def _suggest_xgboost_params(trial: optuna.Trial, fixed: dict) -> dict:
    """Sample XGBoost parameters, leaving keys in ``fixed`` untouched."""
    suggested: dict = {}
    if "max_depth" not in fixed:
        suggested["max_depth"] = trial.suggest_int("max_depth", 3, 8)
    if "learning_rate" not in fixed:
        suggested["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    if "min_child_weight" not in fixed:
        suggested["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 50.0, log=True)
    if "subsample" not in fixed:
        suggested["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    if "colsample_bytree" not in fixed:
        suggested["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    if "gamma" not in fixed:
        suggested["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
    if "reg_alpha" not in fixed:
        suggested["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    if "reg_lambda" not in fixed:
        suggested["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    return {**suggested, **fixed}


def tune_xgboost(
    workflow: ModelTrainingWorkflow,
    *,
    n_trials: int = 30,
    seed: int = 42,
    early_stopping_rounds: int | None = 50,
    early_stopping_min_delta: float = 0.01,
    fixed_hyperparameters: dict | None = None,
    show_progress: bool = True,
) -> optuna.Study:
    """Search XGBoost hyperparameters on the workflow's preprocessed frames.

    The workflow must already hold numeric train, validation, and test frames
    (see ``generate_preprocessed_data``). Validation PR-AUC is maximized.
    ``n_estimators`` stays in ``fixed_hyperparameters`` and is the tree cap;
    early stopping chooses the round. ``scale_pos_weight`` is not searched.

    The best trial stores the full parameter dict, including fixed keys, on
    ``study.best_trial.user_attrs["hyperparameters"]``.

    Parameters
    ----------
    workflow : ModelTrainingWorkflow
        Workflow with ``x_train_preproc``, ``x_validation_preproc``,
        ``x_test_preproc``, ``y_train``, and ``y_validation`` already set.
    n_trials : int, default=30
        Number of Optuna trials.
    seed : int, default=42
        Seed for the sampler and for each XGBoost model.
    early_stopping_rounds : int or None, default=50
        Patience on validation logloss. ``None`` trains for every tree.
    early_stopping_min_delta : float, default=0.01
        Minimum absolute logloss improvement versus the best validation score.
    fixed_hyperparameters : dict, optional
        Parameters kept constant. Defaults to 200 trees and 4 threads.
        Keys listed here are not sampled.
    show_progress : bool, default=True
        Show the Optuna progress bar.

    Returns
    -------
    optuna.Study
        Completed study. ``study.best_value`` is the best validation PR-AUC.

    Raises
    ------
    ValueError
        If ``n_trials`` is not a positive integer, or the preprocessed frames
        are missing.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials!r}.")
    if (
        workflow.x_train_preproc is None
        or workflow.x_validation_preproc is None
        or workflow.x_test_preproc is None
        or workflow.y_train is None
        or workflow.y_validation is None
    ):
        raise ValueError(
            "Preprocessed train, validation, and test frames are required before tuning."
        )

    fixed = {"n_estimators": 200, "n_jobs": 4}
    if fixed_hyperparameters:
        fixed.update(fixed_hyperparameters)

    x_train, x_validation, _, _ = workflow._frames_for_training()
    y_validation = np.asarray(workflow.y_validation)
    log.info(
        "Starting XGBoost Optuna search: %s trials, %s rows, %s features.",
        n_trials,
        len(y_validation),
        x_validation.shape[1],
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgboost_params(trial, fixed)
        workflow.model, _ = workflow._instantiate_model(
            "xgboost",
            params,
            None,
            seed,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        workflow._fit_model(
            "xgboost",
            x_train,
            x_validation,
            None,
            False,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        best_iteration = (
            None
            if early_stopping_rounds is None
            else workflow._best_iteration("xgboost")
        )
        proba = workflow.model.predict_proba(x_validation)[:, 1]
        score = float(average_precision_score(y_validation, proba))
        trial.set_user_attr("hyperparameters", params)
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("val_pr_auc", score)
        log.info("Trial %s validation PR-AUC %.4f best_iteration %s", trial.number, score, best_iteration)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="xgboost_no_balancing",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    log.info(
        "Optuna search finished. Best validation PR-AUC %.4f with %s.",
        study.best_value,
        study.best_trial.user_attrs.get("hyperparameters"),
    )
    return study
