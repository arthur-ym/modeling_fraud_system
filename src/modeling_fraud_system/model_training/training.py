"""Workflow class for training a fraud prediction model using XGBoost or CatBoost."""

# Standard library imports
import logging
import os
import pickle  # nosec B403
from datetime import datetime
from pathlib import Path

# Third party imports
import mlflow
import pandas as pd
import yaml
from catboost import CatBoostClassifier

# First party imports
from mlflow.models import infer_signature
from xgboost import XGBClassifier

from modeling_fraud_system.evaluation import (
    evaluate_model,
    get_max_f1_threshold,
    plot_calibration_bins,
    plot_feature_importance,
    plot_ks_curve,
    plot_learning_curves,
    plot_pr_curve,
    plot_probability_histogram,
)
from modeling_fraud_system.balancing_techniques import balance_data
from modeling_fraud_system.run_time_configuration import BaseConfigParams,last_day_of_month

mlflow.autolog(disable=True)
# Configure logging to ensure it prints
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Ensures output to console
        logging.FileHandler("training_workflow.log"),  # Optional: also log to file
    ],
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# pylint: disable=too-many-instance-attributes
class ModelTrainingWorkflow():
    """Workflow class for training fraud prediction models.

    This workflow handles the machine learning pipeline for fraud prediction:
    - Loading tabular data from parquet
    - Train/validation/test splitting with a temporal test window and a random
      stratified split of the training period
    - Optional class balancing on the training set
    - Feature selection and numeric filtering
    - Model training with XGBoost or CatBoost using fixed hyperparameters
    - Evaluation (AUC, PR-AUC, KS, calibration) and optional MLflow logging
    """

    def __init__(self, base_config_params: BaseConfigParams):
        """Initialize the workflow with runtime configuration.

        Parameters
        ----------
        base_config_params : BaseConfigParams
            Runtime configuration, including training/test date ranges and MLflow
            experiment settings (experiment_name, mlflow_run_id, mlflow_run_name).

        Examples
        --------
        >>> runtime_params = BaseConfigParams(
        ...     training_start_date='2023-01-01',
        ...     training_end_date='2023-06-30',
        ...     test_start_date='2023-07-01',
        ...     test_end_date='2023-09-30'
        ... )
        >>> workflow = ModelTrainingWorkflow(runtime_params)
        """
        super().__init__()
        self.run_time_config = base_config_params
        log.info("Initialized the SolvencyNewCustomersModelTraining workflow.")
        self._spark = None

        self.df_train = None
        self.df_test = None
        self.df_validation = None

        self.pipeline = None
        self.feature_baseline = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_validation = None
        self.y_validation = None

        self.x_train_preproc = None
        self.x_test_preproc = None
        self.x_validation_preproc = None
        self.model = None


    def load_data(
        self,
        path_external_data: str,
        sample: bool = False,
        columns: list = [],

    ):
        """Load a parquet dataset into ``self.data_set``.

        Parameters
        ----------
        path_external_data : str
            Path to a parquet file readable by pyarrow.
        sample : bool or float, default=False
            If a positive float, subsample that fraction of rows (``frac=sample``)
            with ``random_state=42``. If False, load the full file.
        columns : list, default=[]
            Column names passed to ``pandas.read_parquet``. An empty list is
            forwarded as-is (pandas uses ``None`` for all columns).

        Returns
        -------
        None
            Sets ``self.data_set`` to the loaded (and optionally sampled) DataFrame.
        """

        #load parquet file
        self.data_set = pd.read_parquet(path_external_data, engine="pyarrow",columns=columns)
        if sample:
            self.data_set = self.data_set.sample(frac=sample,random_state=42)

    def _split_train_validation_test(
        self,
        target_column: str,
        start_training_date: str,
        end_training_date: str,
        start_test_date: str,
        end_test_date: str,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        """Prepare data splits: random train/validation split on same period, separate test set.

        Creates a temporal split where:
        - Training and validation data come from the same time period (randomly split)
        - Test data comes from a separate, later time period
        - Validation split is stratified to maintain class distribution

        This approach prevents data leakage while allowing for robust validation.

        Parameters
        ----------
        target_column : str
            The name of the target column to separate from features.
        start_training_date : str
            Start date for training+validation period (format: 'YYYY-MM-DD').
        end_training_date : str
            End date for training+validation period (format: 'YYYY-MM-DD').
        start_test_date : str
            Start date for test data (format: 'YYYY-MM-DD').
        end_test_date : str
            End date for test data (format: 'YYYY-MM-DD').
        validation_split : float, default=0.2
            Proportion of training+validation data to use for validation (0.0 to 1.0).
        random_state : int, default=42
            Random seed for reproducible train/validation split.

        Returns
        -------
        dict
            Dictionary containing the split dates and validation split ratio:
            - 'training_validation_start': Training/validation period start
            - 'training_validation_end': Training/validation period end
            - 'validation_split': Validation proportion
            - 'test_start': Test period start
            - 'test_end': Test period end

        Notes
        -----
        - End dates are extended to 23:59:59 to include the entire last day
        - Uses last_day_of_month() to ensure complete month coverage
        - Stratified splitting maintains target class distribution
        """
        from datetime import timedelta

        from sklearn.model_selection import train_test_split

        # Convert all dates to datetime with time components
        training_start = pd.to_datetime(start_training_date, utc=True)
        # Add 23:59:59 to the last day of the month for training end
        training_end = pd.to_datetime(
            last_day_of_month(end_training_date), utc=True
        ) + timedelta(hours=23, minutes=59, seconds=59)
        test_start = pd.to_datetime(start_test_date, utc=True)
        # Add 23:59:59 to the last day of the month for test end
        test_end = pd.to_datetime(
            last_day_of_month(end_test_date), utc=True
        ) + timedelta(hours=23, minutes=59, seconds=59)

        split_dates = {
            "training_validation_start": training_start,
            "training_validation_end": training_end,
            "validation_split": validation_split,
            "test_start": test_start,
            "test_end": test_end,
        }

        log.info(
            f"Splitting data - Training+Validation period: {training_start} to {training_end} "
            f"(random split with {validation_split * 100:.0f}% validation), "
            f"Test period: {test_start} to {test_end}"
        )

        # Split training+validation data from the same period (inclusive of end time)
        train_val_mask = (self.data_set["RequestDateTime"] >= training_start) & (
            self.data_set["RequestDateTime"] <= training_end
        )
        df_train_val = self.data_set[train_val_mask].copy()

        # Split test data (inclusive of end time)
        test_mask = (self.data_set["RequestDateTime"] >= test_start) & (
            self.data_set["RequestDateTime"] <= test_end
        )
        self.df_test = self.data_set[test_mask].copy()

        # Perform random split on training+validation data
        self.df_train, self.df_validation = train_test_split(
            df_train_val,
            test_size=validation_split,
            random_state=random_state,
            stratify=df_train_val[
                target_column
            ],  # Stratify to maintain class distribution
        )

        self.x_validation = self.df_validation.drop(
            columns=[target_column, "RequestDateTime"]
        )
        self.y_validation = self.df_validation[target_column]

        log.info(
            f"Data split sizes - Training: {len(self.df_train)}, "
            f"Validation: {len(self.df_validation)}, Testing: {len(self.df_test)}"
        )

        # Split features and target for training and test sets
        self.x_train = self.df_train.drop(columns=[target_column, "RequestDateTime"])
        self.y_train = self.df_train[target_column]

        self.x_test = self.df_test.drop(columns=[target_column, "RequestDateTime"])
        self.y_test = self.df_test[target_column]

        # Calculate and log target rates for each split
        train_target_rate = self.y_train.mean() * 100
        test_target_rate = self.y_test.mean() * 100

        log.info("\nTarget Rate Distribution:")
        log.info(f"  Training set: {train_target_rate:.2f}% (positive class)")
        log.info(f"  Test set: {test_target_rate:.2f}% (positive class)")

        validation_target_rate = self.y_validation.mean() * 100
        log.info(f"  Validation set: {validation_target_rate:.2f}% (positive class)")

        log.info(
            "Data split completed: random train/validation split on same period, separate test set."
        )

        return split_dates

    def split_data(
        self,
        target_column: str = "Flag_NotPaid_Within_90_Days_After_DueDate",
        validation_split: float = 0.2,
    ):
        """Split the dataset into training, validation, and test sets based on
        configured dates.

        Creates temporal splits of the data using date ranges from runtime configuration.
        Training and validation data come from the same period (randomly split), while
        test data comes from a separate later period to evaluate temporal generalization.

        Parameters
        ----------
        target_column : str, default="Flag_NotPaid_Within_90_Days_After_DueDate"
            Name of the target column to predict. This column indicates whether
            a customer failed to pay within 90 days after the due date.
        validation_split : float, default=0.2
            Proportion of the training period data to use for validation (0.0 to 1.0).
            The split is stratified to maintain class distribution.

        Returns
        -------
        None
            Sets the following instance attributes:
            - self.df_train: Training DataFrame with RequestDateTime column
            - self.df_validation: Validation DataFrame with RequestDateTime column
            - self.df_test: Test DataFrame with RequestDateTime column
            - self.x_train: Training features (without target and RequestDateTime)
            - self.y_train: Training target values
            - self.x_validation: Validation features (without target and RequestDateTime)
            - self.y_validation: Validation target values
            - self.x_test: Test features (without target and RequestDateTime)
            - self.y_test: Test target values

        Notes
        -----
        Date ranges are read from self.run_time_config:
        - training_start_date and training_end_date: Define training+validation period
        - test_start_date and test_end_date: Define test period

        The method automatically:
        - Converts dates to UTC timezone
        - Extends end dates to include the entire last day (23:59:59)
        - Logs dataset sizes and target rates for each split
        - Ensures stratified splitting to maintain class balance

        Examples
        --------
        >>> # Split with default 20% validation
        >>> workflow.split_data()

        >>> # Split with 30% validation
        >>> workflow.split_data(validation_split=0.3)

        >>> # Split with custom target column
        >>> workflow.split_data(target_column="custom_target", validation_split=0.25)
        """
        self.data_set["RequestDateTime"] = pd.to_datetime(
            self.data_set["RequestDateTime"], utc=True
        )

        self._split_train_validation_test(
            target_column=target_column,
            start_training_date=self.run_time_config.training_start_date,
            end_training_date=self.run_time_config.training_end_date,
            start_test_date=self.run_time_config.test_start_date,
            end_test_date=self.run_time_config.test_end_date,
            validation_split=validation_split,
        )

    def balance_training_data(
        self,
        target_column: str = "Flag_NotPaid_Within_90_Days_After_DueDate",
        method: str = "smotenc",
        k_neighbors: int = 5,
        show_plot: bool = True,
        numeric_only: bool = False,
        no_nan_columns: bool = False,
    ):
        """Balance the training data using the specified resampling method.

        This method applies the balance_data function to the training set only,
        leaving validation and test sets unchanged.

        Parameters
        ----------
        target_column : str, default="Flag_NotPaid_Within_90_Days_After_DueDate"
            Name of the target column.
        method : str, default="smotenc"
            Resampling method to use. Options: "undersample", "oversample", "smote", "smotenc", "adasyn".
        k_neighbors : int, default=5
            Number of neighbors for SMOTE/SMOTENC/ADASYN.
        show_plot : bool, default=True
            Whether to display the class distribution plot.
        numeric_only : bool, default=False
            If True, only numeric columns will be used for SMOTE and ADASYN methods.
        no_nan_columns : bool, default=False
            If True, only columns with no NaN values will be used.

        Returns
        -------
        None
            Updates ``self.df_train`` and ``self.y_train`` in place. Sets
            ``self.x_train`` to the full balanced DataFrame (including the
            target column). Validation and test frames are not modified.

        Examples
        --------
        >>> # Balance training data using SMOTENC
        >>> workflow.balance_training_data(method="smotenc")

        >>> # Balance using SMOTE with numeric columns only
        >>> workflow.balance_training_data(method="smote", numeric_only=True)

        >>> # Balance using oversampling without showing plot
        >>> workflow.balance_training_data(method="oversample", show_plot=False)
        """
        if self.df_train is None:
            raise ValueError("Training data not loaded. Please run split_data() first.")

        log.info(f"Balancing training data using method: {method}")
        log.info(f"Original training data shape: {self.df_train.shape}")

        # Get original class distribution
        original_counts = self.df_train[target_column].value_counts().sort_index()
        log.info(f"Original class distribution:\n{original_counts}")

        # Apply balancing to training data
        self.df_train = balance_data(
            df=self.df_train,
            target_column=target_column,
            method=method,
            k_neighbors=k_neighbors,
            show_plot=show_plot,
            numeric_only=numeric_only,
            no_nan_columns=no_nan_columns,
        )

        # Update x_train and y_train
        self.x_train = self.df_train
        self.y_train = self.df_train[target_column]

        # Get new class distribution
        new_counts = self.df_train[target_column].value_counts().sort_index()
        log.info(f"New training data shape: {self.df_train.shape}")
        log.info(f"New class distribution:\n{new_counts}")
        log.info("Training data balancing completed successfully.")

    def generate_preprocessed_data(
        self, numeric_only: bool = True, selected_features: list = None
    ):
        """Generate preprocessed data by copying train, test, and validation sets.

        This method creates preprocessed versions of the data by optionally filtering
        to selected features first, then optionally filtering to numeric columns only.

        Parameters
        ----------
        numeric_only : bool, default=True
            If True, only numeric columns will be selected for preprocessing.
            If False, all columns (or selected_features if provided) will be used.
        selected_features : list, optional
            List of feature names to select before applying numeric_only filter.
            If provided, only these features will be considered. If None, all features
            from the original data are used.

        Returns
        -------
        None
            Updates self.x_train_preproc, self.x_test_preproc, and self.x_validation_preproc in place.

        Raises
        ------
        ValueError
            If the data has not been split yet (x_train, x_test, x_validation are None).
            If selected_features contains columns not present in the data.

        Examples
        --------
        >>> # After splitting data, preprocess with numeric columns only
        >>> workflow.split_data()
        >>> workflow.generate_preprocessed_data(numeric_only=True)
        >>> print(workflow.x_train_preproc.shape)

        >>> # Preprocess with selected features only
        >>> workflow.generate_preprocessed_data(selected_features=['feature1', 'feature2', 'feature3'])

        >>> # Preprocess with selected features, then filter to numeric
        >>> workflow.generate_preprocessed_data(selected_features=['feature1', 'feature2'], numeric_only=True)

        >>> # Preprocess with all columns
        >>> workflow.generate_preprocessed_data(numeric_only=False)
        """
        if self.x_train is None:
            raise ValueError("Training data not loaded. Please run split_data() first.")

        if self.x_test is None:
            raise ValueError("Test data not loaded. Please run split_data() first.")

        log.info("Generating preprocessed data by copying original data.")

        # Start with all columns or selected features
        if selected_features is not None:
            # Validate that selected features exist in the training data
            missing_features = set(selected_features) - set(self.x_train.columns)
            if missing_features:
                raise ValueError(
                    f"Selected features not found in training data: {missing_features}"
                )

            log.info(
                f"Using selected features: {len(selected_features)} features specified"
            )
            log.info(f"Selected features: {selected_features}")

            # Filter to selected features
            x_train_filtered = self.x_train[selected_features].copy()
            x_test_filtered = self.x_test[selected_features].copy()

            if self.x_validation is not None:
                x_validation_filtered = self.x_validation[selected_features].copy()
            else:
                x_validation_filtered = None
        else:
            # Use all columns
            x_train_filtered = self.x_train.copy()
            x_test_filtered = self.x_test.copy()
            x_validation_filtered = (
                self.x_validation.copy() if self.x_validation is not None else None
            )

        # Apply numeric filtering if requested
        if numeric_only:
            # Get only numeric columns from filtered training data
            numeric_columns = x_train_filtered.select_dtypes(
                include=["number"]
            ).columns.tolist()

            log.info(
                f"Filtering to numeric columns: {len(numeric_columns)} out of {len(x_train_filtered.columns)} total columns"
            )
            log.info(f"Selected numeric columns: {numeric_columns}")

            # Copy training data with numeric columns only
            self.x_train_preproc = x_train_filtered[numeric_columns].copy()
            log.info(f"Training preprocessed data shape: {self.x_train_preproc.shape}")

            # Copy test data with numeric columns only
            self.x_test_preproc = x_test_filtered[numeric_columns].copy()
            log.info(f"Test preprocessed data shape: {self.x_test_preproc.shape}")

            # Copy validation data with numeric columns only if it exists
            if x_validation_filtered is not None:
                self.x_validation_preproc = x_validation_filtered[
                    numeric_columns
                ].copy()
                log.info(
                    f"Validation preprocessed data shape: {self.x_validation_preproc.shape}"
                )
            else:
                log.warning(
                    "Validation data not found. Skipping validation preprocessing."
                )
                self.x_validation_preproc = None
        else:
            # Copy all columns without numeric filtering
            self.x_train_preproc = x_train_filtered.copy()
            log.info(f"Training preprocessed data shape: {self.x_train_preproc.shape}")

            self.x_test_preproc = x_test_filtered.copy()
            log.info(f"Test preprocessed data shape: {self.x_test_preproc.shape}")

            if x_validation_filtered is not None:
                self.x_validation_preproc = x_validation_filtered.copy()
                log.info(
                    f"Validation preprocessed data shape: {self.x_validation_preproc.shape}"
                )
            else:
                log.warning(
                    "Validation data not found. Skipping validation preprocessing."
                )
                self.x_validation_preproc = None

        log.info("Preprocessed data generation completed successfully.")

    def build_pipeline(self):
        """Apply a fitted preprocessing pipeline to train, test, and validation features.

        Requires ``self.pipeline`` to be a fitted transformer, and ``self.x_train``,
        ``self.x_test``, and ``self.x_val`` to already be set.

        Returns
        -------
        None
            Sets ``self.x_train_preproc``, ``self.x_test_preproc``, and
            ``self.x_val_preproc``.

        Raises
        ------
        AttributeError
            If ``self.pipeline`` is None or not set.

        Notes
        -----
        This method assumes the pipeline is already fitted (or loaded from disk).
        It does not fit the pipeline.

        Examples
        --------
        >>> workflow.pipeline = some_fitted_pipeline
        >>> workflow.build_pipeline()
        """
        log.info("Applying the preprocessing pipeline to the data.")
        self.x_train_preproc = self.pipeline.transform(self.x_train)
        self.x_test_preproc = self.pipeline.transform(self.x_test)
        self.x_val_preproc = self.pipeline.transform(self.x_val)
        log.info("Preprocessing pipeline applied successfully.")


    def _instantiate_model(
        self,
        model_type: str,
        hyperparameters: dict | None,
        categorical_features: list | None,
        seed: int,
    ):
        """Build an unfitted XGBoost or CatBoost classifier.

        Default hyperparameters are applied first, then overridden by ``hyperparameters``.
        For CatBoost, ``categorical_features`` is stored as ``cat_features`` when given.

        Parameters
        ----------
        model_type : str
            ``"xgboost"`` or any other value (treated as CatBoost).
        hyperparameters : dict or None
            Extra constructor kwargs. Keys ``tree_method`` and ``objective`` are
            popped for XGBoost before merging remaining overrides.
        categorical_features : list or None
            CatBoost categorical feature names or indices. Ignored for XGBoost.
        seed : int
            Random seed (``seed`` for XGBoost, ``random_seed`` for CatBoost).

        Returns
        -------
        tuple
            ``(model, model_params)`` where ``model`` is the unfitted classifier
            and ``model_params`` is the dict passed to its constructor.
        """
        hyperparameters = dict(hyperparameters or {})
        if model_type == "xgboost":
            tree_method = hyperparameters.pop("tree_method", "hist")
            objective = hyperparameters.pop("objective", "binary:logistic")
            model_params = {
                "use_label_encoder": False,
                "objective": objective,
                "seed": seed,
                "n_jobs": -1,
                "tree_method": tree_method,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
            }
            model_params.update(hyperparameters)
            model = XGBClassifier(**model_params)
        else:
            model_params = {
                "random_seed": seed,
                "thread_count": -1,
                "depth": 6,
                "learning_rate": 0.1,
                "iterations": 100,
                "subsample": 0.8,
                "rsm": 0.8,
            }
            model_params.update(hyperparameters)
            if categorical_features is not None:
                model_params["cat_features"] = categorical_features
            model = CatBoostClassifier(**model_params)
        return model, model_params

    def _frames_for_training(self):
        """Copy preprocessed frames and drop ``contrafactual_flag`` if present.

        Returns
        -------
        tuple
            ``(x_train, x_validation, x_test, dropped)`` where the first three
            are copies of the preprocessed feature frames and ``dropped`` is True
            if ``contrafactual_flag`` was removed from all three.
        """
        x_train = self.x_train_preproc.copy()
        x_validation = self.x_validation_preproc.copy()
        x_test = self.x_test_preproc.copy()
        dropped = "contrafactual_flag" in x_train.columns
        if dropped:
            log.info(
                "Dropping 'contrafactual_flag' column from training, validation, and test data."
            )
            x_train = x_train.drop(columns=["contrafactual_flag"])
            x_validation = x_validation.drop(columns=["contrafactual_flag"])
            x_test = x_test.drop(columns=["contrafactual_flag"])
        return x_train, x_validation, x_test, dropped

    def _fit_model(
        self,
        model_type: str,
        x_train,
        x_validation,
        categorical_features: list | None,
        verbose: bool,
    ):
        """Fit ``self.model`` on training data with a validation eval set.

        XGBoost uses ``eval_set`` on the raw frames. CatBoost wraps frames in
        ``Pool`` objects so categorical columns can be passed through.

        Parameters
        ----------
        model_type : str
            ``"xgboost"`` or CatBoost (any other value).
        x_train : pd.DataFrame
            Training features (without ``contrafactual_flag`` if it was dropped).
        x_validation : pd.DataFrame
            Validation features used as the eval set.
        categorical_features : list or None
            CatBoost categorical feature names or indices. Ignored for XGBoost.
        verbose : bool
            Forwarded to the estimator's ``fit`` method.

        Returns
        -------
        None
            Updates ``self.model`` in place.
        """
        if model_type == "xgboost":
            self.model.fit(
                x_train,
                self.y_train,
                eval_set=[
                    (x_train, self.y_train),
                    (x_validation, self.y_validation),
                ],
                verbose=verbose,
            )
            return

        from catboost import Pool

        train_pool = Pool(
            x_train, self.y_train, cat_features=categorical_features
        )
        eval_pool = Pool(
            x_validation, self.y_validation, cat_features=categorical_features
        )
        self.model.fit(train_pool, eval_set=eval_pool, verbose=verbose)

    def _evaluate_splits(
        self,
        x_train,
        x_validation,
        x_test,
        threshold: float,
        contrafactual_flag_exists: bool,
    ):
        """Score train, validation, and test splits and collect metrics.

        When ``contrafactual_flag_exists`` is True, also evaluates the test
        subset where ``self.x_test_preproc["contrafactual_flag"] == 1``.

        Parameters
        ----------
        x_train, x_validation, x_test : pd.DataFrame
            Feature frames used for ``predict_proba`` (flag column already dropped).
        threshold : float
            Probability cutoff for converting scores to binary labels.
        contrafactual_flag_exists : bool
            Whether the original preprocessed test frame still has
            ``contrafactual_flag`` (True if it was dropped from the training frames).

        Returns
        -------
        dict
            Keys ``train``, ``validation``, and ``test`` each map to
            ``{"metrics", "y_pred_proba"}``. ``test_contrafactual`` is either
            None or ``{"metrics", "y_true", "y_pred_proba", "size"}``.
        """
        y_train_pred_proba = self.model.predict_proba(x_train)[:, 1]
        y_val_pred_proba = self.model.predict_proba(x_validation)[:, 1]
        y_test_pred_proba = self.model.predict_proba(x_test)[:, 1]
        y_train_pred = (y_train_pred_proba >= threshold).astype(int)
        y_val_pred = (y_val_pred_proba >= threshold).astype(int)
        y_test_pred = (y_test_pred_proba >= threshold).astype(int)

        evaluation = {
            "train": {
                "metrics": evaluate_model(
                    self.y_train, y_train_pred, y_train_pred_proba, "training"
                ),
                "y_pred_proba": y_train_pred_proba,
            },
            "validation": {
                "metrics": evaluate_model(
                    self.y_validation, y_val_pred, y_val_pred_proba, "validation"
                ),
                "y_pred_proba": y_val_pred_proba,
            },
            "test": {
                "metrics": evaluate_model(
                    self.y_test, y_test_pred, y_test_pred_proba, "test"
                ),
                "y_pred_proba": y_test_pred_proba,
            },
            "test_contrafactual": None,
        }

        if not contrafactual_flag_exists:
            return evaluation

        log.info("Evaluating model on contrafactual subset (contrafactual_flag == 1).")
        mask = self.x_test_preproc["contrafactual_flag"] == 1
        if mask.sum() == 0:
            log.warning("No samples found with contrafactual_flag == 1 in test set.")
            return evaluation

        y_cf = self.y_test[mask]
        y_cf_proba = self.model.predict_proba(x_test[mask])[:, 1]
        y_cf_pred = (y_cf_proba >= threshold).astype(int)
        evaluation["test_contrafactual"] = {
            "metrics": evaluate_model(
                y_cf, y_cf_pred, y_cf_proba, "test_contrafactual"
            ),
            "y_true": y_cf,
            "y_pred_proba": y_cf_proba,
            "size": int(mask.sum()),
        }
        log.info(f"Contrafactual subset size: {mask.sum()} samples")
        return evaluation

    def _make_training_figures(self, evaluation: dict, x_train):
        """Build PR, KS, histogram, calibration, learning-curve, and importance plots.

        Parameters
        ----------
        evaluation : dict
            Output of ``_evaluate_splits``.
        x_train : pd.DataFrame
            Training features used for feature-importance labels.

        Returns
        -------
        dict
            Matplotlib figures keyed by plot name (e.g. ``ks_curve_test``).
            Includes contrafactual plots when that subset was evaluated.
        """
        figures = {}
        split_truth = (
            ("training", "train", self.y_train),
            ("validation", "validation", self.y_validation),
            ("test", "test", self.y_test),
        )
        for set_name, key, y_true in split_truth:
            payload = evaluation[key]
            figures[f"precision_recall_curve_{set_name}"] = plot_pr_curve(
                payload["metrics"], set_name=set_name
            )
            figures[f"ks_curve_{set_name}"] = plot_ks_curve(
                y_true, payload["y_pred_proba"], set_name=set_name
            )
            figures[f"probability_histogram_{set_name}"] = plot_probability_histogram(
                payload["y_pred_proba"], y_true, set_name=set_name
            )
            figures[f"calibration_bins_{set_name}"] = plot_calibration_bins(
                y_true, payload["y_pred_proba"], set_name=set_name
            )

        contrafactual = evaluation["test_contrafactual"]
        if contrafactual is not None:
            set_name = "test_contrafactual"
            figures[f"precision_recall_curve_{set_name}"] = plot_pr_curve(
                contrafactual["metrics"], set_name=set_name
            )
            figures[f"ks_curve_{set_name}"] = plot_ks_curve(
                contrafactual["y_true"],
                contrafactual["y_pred_proba"],
                set_name=set_name,
            )
            figures[f"probability_histogram_{set_name}"] = plot_probability_histogram(
                contrafactual["y_pred_proba"],
                contrafactual["y_true"],
                set_name=set_name,
            )
            figures[f"calibration_bins_{set_name}"] = plot_calibration_bins(
                contrafactual["y_true"],
                contrafactual["y_pred_proba"],
                set_name=set_name,
            )

        figures["learning_curves"] = plot_learning_curves(self.model, metric="logloss")
        figures["feature_importance"] = plot_feature_importance(
            self.model,
            feature_names=x_train.columns,
            max_features=20,
            importance_type="gain",
        )
        return figures

    def _log_training_to_mlflow(
        self,
        *,
        model_type: str,
        model_params: dict,
        threshold: float,
        contrafactual_flag_exists: bool,
        x_train,
        x_validation,
        x_test,
        evaluation: dict,
        figures: dict,
    ):
        """Log params, metrics, figures, and the fitted model to the active MLflow run.

        Must be called inside an active MLflow run. Skips CatBoost
        ``cat_features`` in the param dict unless it is a non-None list, in
        which case it is logged as a comma-separated string.

        Parameters
        ----------
        model_type : str
            ``"xgboost"`` or ``"catboost"`` (MLflow flavor and logged param).
        model_params : dict
            Constructor kwargs from ``_instantiate_model``.
        threshold : float
            Classification threshold used at evaluation.
        contrafactual_flag_exists : bool
            Logged as ``contrafactual_flag_dropped``.
        x_train, x_validation, x_test : pd.DataFrame
            Frames used for sizes, feature count, and model signature.
        evaluation : dict
            Output of ``_evaluate_splits``.
        figures : dict
            Output of ``_make_training_figures``, stored under artifact path
            ``plots``.

        Returns
        -------
        None
        """
        from modeling_fraud_system.ml_flow import (
            log_figures_to_mlflow,
            log_metrics_to_mlflow,
            log_params_to_mlflow,
            save_model_into_mlflow,
        )

        params = {
            **{key: value for key, value in model_params.items() if key != "cat_features"},
            "model_type": model_type,
            "threshold": threshold,
            "contrafactual_flag_dropped": contrafactual_flag_exists,
            "training_start_date": str(self.run_time_config.training_start_date),
            "training_end_date": str(self.run_time_config.training_end_date),
            "validation_start_date": str(self.run_time_config.validation_start_date),
            "validation_end_date": str(self.run_time_config.validation_end_date),
            "test_start_date": str(self.run_time_config.test_start_date),
            "test_end_date": str(self.run_time_config.test_end_date),
            "train_size": len(x_train),
            "validation_size": len(x_validation),
            "test_size": len(x_test),
            "n_features": x_train.shape[1],
            "train_positive_rate": f"{self.y_train.mean():.4f}",
            "validation_positive_rate": f"{self.y_validation.mean():.4f}",
            "test_positive_rate": f"{self.y_test.mean():.4f}",
        }
        if "cat_features" in model_params and model_params["cat_features"] is not None:
            params["cat_features"] = ",".join(map(str, model_params["cat_features"]))

        log_params_to_mlflow(params)

        metric_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "pr_auc",
            "ks_statistic",
        ]
        metrics = {}
        for split, prefix in (
            ("train", "train"),
            ("validation", "val"),
            ("test", "test"),
        ):
            split_metrics = evaluation[split]["metrics"]
            for key in metric_keys:
                metrics[f"{prefix}_{key}"] = split_metrics[key]

        contrafactual = evaluation["test_contrafactual"]
        if contrafactual is not None:
            for key in metric_keys:
                metrics[f"test_contrafactual_{key}"] = contrafactual["metrics"][key]
            log_params_to_mlflow({"test_contrafactual_size": contrafactual["size"]})

        log_metrics_to_mlflow(metrics)
        log_figures_to_mlflow(figures, artifact_path="plots")

        signature = infer_signature(x_train, self.model.predict_proba(x_train))
        save_model_into_mlflow(self.model, flavor=model_type, signature=signature)

    def train_model(
        self,
        hyperparameters: dict = None,
        threshold: float = None,
        run_name: str = None,
        model_type: str = "xgboost",
        categorical_features: list = None,
        log_into_mlflow: bool = False,
        show_plots: bool = True,
        verbose: bool = False,
        seed: int = 42,
    ):
        """Train an XGBoost or CatBoost model and evaluate train/validation/test.

        Uses fixed hyperparameters (no search). If ``threshold`` is omitted, the
        F1-maximizing cutoff is taken from the validation set. Plots are always
        generated; they are shown only when ``show_plots`` is True.

        When ``log_into_mlflow`` is True, params, metrics, figures, and the model
        are written to the MLflow experiment on ``self.run_time_config``.

        Parameters
        ----------
        hyperparameters : dict, optional
            Overrides merged on top of model defaults.
        threshold : float, optional
            Probability cutoff. If None, computed with ``get_max_f1_threshold``
            on the validation set.
        run_name : str, optional
            MLflow run name. Falls back to ``mlflow_run_name`` on the runtime
            config, then ``train_{model_type}``.
        model_type : str, default="xgboost"
            ``"xgboost"`` or ``"catboost"``.
        categorical_features : list, optional
            CatBoost categorical feature names or indices.
        log_into_mlflow : bool, default=False
            If True, log results and persist the model in MLflow.
        show_plots : bool, default=True
            If True, display generated figures before closing them.
        verbose : bool, default=False
            Training verbosity forwarded to the estimator.
        seed : int, default=42
            Random seed for the classifier.

        Returns
        -------
        dict
            ``threshold``, ``train_metrics``, ``val_metrics``, ``test_metrics``,
            and ``test_contrafactual_metrics`` (None if that subset is absent).

        Raises
        ------
        ValueError
            If ``model_type`` is not ``xgboost`` or ``catboost``, or if
            preprocessed train/validation/test frames are missing.

        Examples
        --------
        >>> results = workflow.train_model(model_type="xgboost", show_plots=False)
        >>> results = workflow.train_model(
        ...     model_type="catboost",
        ...     categorical_features=["country"],
        ...     log_into_mlflow=True,
        ... )
        """
        import matplotlib.pyplot as plt

        log.info(
            f"Starting model training without hyperparameter tuning using {model_type}."
        )
        if model_type not in ["xgboost", "catboost"]:
            raise ValueError(
                f"model_type must be 'xgboost' or 'catboost', got '{model_type}'"
            )
        if (
            self.x_train_preproc is None
            or self.x_validation_preproc is None
            or self.x_test_preproc is None
        ):
            raise ValueError(
                "Training/validation/test data has not been split and preprocessed yet."
            )

        self.model, model_params = self._instantiate_model(
            model_type, hyperparameters, categorical_features, seed
        )
        x_train, x_validation, x_test, contrafactual_flag_exists = (
            self._frames_for_training()
        )

        log.info(f"Training {model_type} model with fixed hyperparameters.")
        self._fit_model(
            model_type, x_train, x_validation, categorical_features, verbose
        )

        if threshold is None:
            threshold = get_max_f1_threshold(
                self.model, x_validation, self.y_validation
            )
            log.info(
                f"Calculated optimal threshold (F1-based) on validation set: {threshold:.4f}"
            )
        else:
            log.info(f"Using provided threshold: {threshold:.4f}")

        evaluation = self._evaluate_splits(
            x_train, x_validation, x_test, threshold, contrafactual_flag_exists
        )
        train_metrics = evaluation["train"]["metrics"]
        val_metrics = evaluation["validation"]["metrics"]
        test_metrics = evaluation["test"]["metrics"]
        contrafactual = evaluation["test_contrafactual"]

        log.info(
            f"Training - Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}"
        )
        log.info(
            f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}"
        )
        log.info(
            f"Test - Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}"
        )
        if contrafactual is not None:
            metrics = contrafactual["metrics"]
            log.info(
                f"Test Contrafactual - Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            )

        figures = self._make_training_figures(evaluation, x_train)
        self.threshold = threshold
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

        if log_into_mlflow:
            from modeling_fraud_system.ml_flow import start_mlflow_run

            with start_mlflow_run(
                experiment_name=self.run_time_config.experiment_name,
                run_id=self.run_time_config.mlflow_run_id,
                run_name=(
                    run_name
                    or self.run_time_config.mlflow_run_name
                    or f"train_{model_type}"
                ),
            ) as active_run:
                self.run_time_config.mlflow_run_id = active_run.info.run_id
                log.info(
                    f"Logging training results to MLflow run {active_run.info.run_id}."
                )
                self._log_training_to_mlflow(
                    model_type=model_type,
                    model_params=model_params,
                    threshold=threshold,
                    contrafactual_flag_exists=contrafactual_flag_exists,
                    x_train=x_train,
                    x_validation=x_validation,
                    x_test=x_test,
                    evaluation=evaluation,
                    figures=figures,
                )
            log.info(
                f"Model training completed successfully using {model_type} (logged to MLflow)."
            )
        else:
            log.info(
                f"Model training completed successfully using {model_type} (MLflow logging disabled)."
            )

        if show_plots:
            plt.show()
        for figure in figures.values():
            plt.close(figure)

        return {
            "threshold": threshold,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_contrafactual_metrics": (
                None if contrafactual is None else contrafactual["metrics"]
            ),
        }

    def save_model(self):
        """Save configuration YAML and, if set, the MLflow model as a pickle.

        Writes a timestamped config file under ``self.config.model_output_path``.
        If ``self.config.best_model_uri`` is set, loads that model with
        ``mlflow.sklearn.load_model`` and pickles it next to the config.

        Returns
        -------
        None
            Writes ``config_{timestamp}.yaml`` and, when a URI is present,
            ``best_model_{timestamp}.pkl``.

        Notes
        -----
        Requires ``self.config.model_output_path``. If ``best_model_uri`` is
        missing, the config is still saved and the method returns without a
        pickle. The output directory is created if it does not exist.

        Examples
        --------
        >>> workflow.config.model_output_path = '/path/to/output'
        >>> workflow.config.best_model_uri = 'runs:/abc123/model'
        >>> workflow.save_model()
        """
        # this assumes loading from MLflow by a static run ID
        # create output directory if it doesn't exist
        log.info("Starting model saving process.")
        timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        output_path = Path(self.config.model_output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save config YAML
        config_path = output_path / f"config_{timestamp}.yaml"
        config_dict = self.config.model_dump(by_alias=True, exclude_none=True)
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        log.info(f"Configuration saved at: {config_path}")
        # Save the best model
        best_model_uri = self.config.best_model_uri
        if not best_model_uri:
            log.info("Best model URI is not set in the configuration, skipping save.")
            return  # Stop the save process if no model URI is provided

        log.info(f"Loading best model from: {best_model_uri}")
        model = mlflow.sklearn.load_model(best_model_uri)

        file_path = output_path / f"best_model_{timestamp}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        log.info(f"Model saved at: {file_path}")

    @staticmethod
    def evaluate_predictions(
        df: pd.DataFrame,
        target_column: str,
        probability_column: str,
        threshold: float = None,
        set_name: str = "evaluation",
        use_mlflow: bool = False,
        mlflow_run_name: str = None,
        mlflow_experiment_path: str = None,
        show_plots: bool = True,
    ):
        """Evaluate predictions from a DataFrame of labels and probabilities.

        Computes classification metrics, builds PR/KS/histogram/calibration
        plots, and optionally logs them to MLflow. If ``threshold`` is omitted,
        the F1-maximizing cutoff on this DataFrame is used.

        Parameters
        ----------
        df : pd.DataFrame
            Frame with the target column and predicted probabilities.
        target_column : str
            Column of true binary labels (0 or 1).
        probability_column : str
            Column of predicted positive-class probabilities (0.0 to 1.0).
        threshold : float, optional
            Cutoff for converting probabilities to labels. If None, chosen to
            maximize F1 on ``df``.
        set_name : str, default="evaluation"
            Label used in logs, plot titles, and MLflow metric prefixes.
        use_mlflow : bool, default=False
            If True, log params, metrics, and figures to a new MLflow run.
        mlflow_run_name : str, optional
            MLflow run name. Defaults to ``"{set_name}_evaluation"``.
        mlflow_experiment_path : str, optional
            MLflow experiment name/path. Required when ``use_mlflow=True``.
        show_plots : bool, default=True
            If True and ``use_mlflow`` is False, display the evaluation plots.

        Returns
        -------
        dict
            ``metrics`` (output of ``evaluate_model``) and ``threshold``.

        Raises
        ------
        ValueError
            If ``target_column`` or ``probability_column`` is missing, or if
            ``use_mlflow=True`` without ``mlflow_experiment_path``.
        RuntimeError
            If ``use_mlflow=True`` but the MLflow experiment does not exist.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'target': [0, 1, 0, 1, 1],
        ...     'probability': [0.2, 0.8, 0.3, 0.9, 0.7]
        ... })
        >>> results = ModelTrainingWorkflow.evaluate_predictions(
        ...     df=df,
        ...     target_column='target',
        ...     probability_column='probability',
        ...     threshold=0.5,
        ... )

        >>> results = ModelTrainingWorkflow.evaluate_predictions(
        ...     df=df,
        ...     target_column='target',
        ...     probability_column='probability',
        ...     use_mlflow=True,
        ...     mlflow_experiment_path='/experiments/my_model',
        ...     mlflow_run_name='evaluation_run_1',
        ... )
        """
        import matplotlib.pyplot as plt

        # Validate input
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        if probability_column not in df.columns:
            raise ValueError(
                f"Probability column '{probability_column}' not found in DataFrame."
            )

        log.info(f"Starting evaluation for {set_name} set with {len(df)} samples.")

        # Extract target and probabilities
        y_true = df[target_column].values
        y_pred_proba = df[probability_column].values

        # Calculate optimal threshold if not provided
        if threshold is None:
            from sklearn.metrics import f1_score

            thresholds = [i / 100 for i in range(1, 100)]
            f1_scores = [
                f1_score(y_true, (y_pred_proba >= t).astype(int)) for t in thresholds
            ]
            optimal_idx = f1_scores.index(max(f1_scores))
            threshold = thresholds[optimal_idx]
            log.info(f"Calculated optimal threshold (F1-based): {threshold:.4f}")
        else:
            log.info(f"Using provided threshold: {threshold:.4f}")

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Evaluate on full dataset
        metrics = evaluate_model(y_true, y_pred, y_pred_proba, set_name)

        # Log metrics summary
        log.info(
            f"{set_name} - Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
        )
        log.info(
            f"{set_name} - AUC: {metrics['auc']:.4f}, "
            f"PR-AUC: {metrics['pr_auc']:.4f}, KS: {metrics['ks_statistic']:.4f}"
        )

        if use_mlflow:
            # Configure MLflow
            if mlflow_experiment_path is None:
                raise ValueError(
                    "mlflow_experiment_path is required when use_mlflow=True"
                )

            mlflow.set_experiment(mlflow_experiment_path)
            log.info(f"Set MLflow experiment to: {mlflow_experiment_path}")

            # Get experiment
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_path)
            if experiment is None:
                raise RuntimeError(f"Experiment {mlflow_experiment_path} not found.")

            # Determine run name
            if mlflow_run_name is None:
                mlflow_run_name = f"{set_name}_evaluation"

            # Start MLflow run
            with mlflow.start_run(
                run_name=mlflow_run_name, experiment_id=experiment.experiment_id
            ):
                # Log parameters
                mlflow.log_param("threshold", threshold)
                mlflow.log_param("set_name", set_name)
                mlflow.log_param("dataset_size", len(df))
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("probability_column", probability_column)
                mlflow.log_param("positive_rate", f"{y_true.mean():.4f}")

                # Log main metrics
                mlflow.log_metrics(
                    {
                        f"{set_name}_accuracy": metrics["accuracy"],
                        f"{set_name}_precision": metrics["precision"],
                        f"{set_name}_recall": metrics["recall"],
                        f"{set_name}_f1": metrics["f1"],
                        f"{set_name}_auc": metrics["auc"],
                        f"{set_name}_pr_auc": metrics["pr_auc"],
                        f"{set_name}_ks_statistic": metrics["ks_statistic"],
                    }
                )

                # Create and log plots
                # PR curve
                fig_pr = plot_pr_curve(metrics, set_name=set_name)
                mlflow.log_figure(fig_pr, f"precision_recall_curve_{set_name}.png")
                plt.close(fig_pr)

                # KS curve
                fig_ks = plot_ks_curve(y_true, y_pred_proba, set_name=set_name)
                mlflow.log_figure(fig_ks, f"ks_curve_{set_name}.png")
                plt.close(fig_ks)

                # Probability histogram
                fig_prob = plot_probability_histogram(
                    y_pred_proba, y_true, set_name=set_name
                )
                mlflow.log_figure(fig_prob, f"probability_histogram_{set_name}.png")
                plt.close(fig_prob)

                # Calibration plot
                fig_cal = plot_calibration_bins(y_true, y_pred_proba, set_name=set_name)
                mlflow.log_figure(fig_cal, f"calibration_bins_{set_name}.png")

                log.info(f"Evaluation results logged to MLflow for {set_name}.")

        elif show_plots:
            # Display plots without MLflow
            fig_pr = plot_pr_curve(metrics, set_name=set_name)
            fig_ks = plot_ks_curve(y_true, y_pred_proba, set_name=set_name)
            fig_prob = plot_probability_histogram(
                y_pred_proba, y_true, set_name=set_name
            )
            fig_cal = plot_calibration_bins(y_true, y_pred_proba, set_name=set_name)

            plt.show()

        # Return results
        results = {
            "metrics": metrics,
            "threshold": threshold,
        }

        log.info(f"Evaluation completed for {set_name}.")
        return results

