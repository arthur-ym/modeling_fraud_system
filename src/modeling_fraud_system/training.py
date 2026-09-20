"""Workflow class for training a solvency prediction model using XGBoost or CatBoost."""

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
    """Workflow class for training solvency prediction models on new customer data.

    This workflow handles the complete machine learning pipeline for solvency prediction:
    - Data loading and feature engineering
    - Train/validation/test splitting with temporal and random strategies
    - Data preprocessing and balancing
    - Model training with hyperparameter tuning or fixed parameters
    - Comprehensive model evaluation and visualization
    - MLflow experiment tracking and model persistence

    Supports both XGBoost and CatBoost classifiers with extensive evaluation metrics
    including AUC, PR-AUC, KS statistic, calibration analysis, and SHAP explanations.
    """

    def __init__(self, base_config_params: BaseConfigParams):
        """Initialize the workflow with the provided configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing model training parameters including:
            - Model hyperparameters (learning_rate, max_depth, etc.)
            - MLflow experiment settings
            - Data balancing options
            - Evaluation thresholds
        BaseConfigParams : BaseConfigParams
            Runtime configuration parameters including:
            - training_start_date: Start date for training data
            - training_end_date: End date for training data
            - test_start_date: Start date for test data
            - test_end_date: End date for test data

        Examples
        --------
        >>> config = {
        ...     'model_seed': 42,
        ...     'max_evals': 50,
        ...     'mlflow_experiment_path': '/experiments/solvency'
        ... }
        >>> runtime_params = BaseConfigParams(
        ...     training_start_date='2023-01-01',
        ...     training_end_date='2023-06-30',
        ...     test_start_date='2023-07-01',
        ...     test_end_date='2023-09-30'
        ... )
        >>> workflow = SolvencyNewCustomersModelTraining(config, runtime_params)
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
        """Load base data from external  source and generate features.

        self.data_set = pandas_df

        log.info("Data loading completed successfully.")"""

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
            Updates self.df_train, self.x_train, and self.y_train in place.

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
        """Apply preprocessing pipeline to training, test, and validation data.

        Applies a pre-loaded scikit-learn pipeline to transform the feature sets.
        The pipeline should be loaded before calling this method.

        Returns
        -------
        None
            Sets self.x_train_preproc, self.x_test_preproc, self.x_val_preproc.

        Raises
        ------
        AttributeError
            If self.pipeline is None or not set.

        Notes
        -----
        This method assumes that the pipeline has already been fitted on training data
        or is a fitted pipeline loaded from disk.

        Examples
        --------
        >>> # Load or create pipeline first
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
        from modeling_fraud_system.ml_flow import (
            log_artifacts_to_mlflow,
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

        log_params_to_mlflow(params, self.run_time_config)

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
            log_params_to_mlflow(
                {"test_contrafactual_size": contrafactual["size"]},
                self.run_time_config,
            )

        mlflow.log_metrics(metrics)

        for name, figure in figures.items():
            log_artifacts_to_mlflow(figure, format="plt", name=name, path="plots")

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

        Training, threshold selection, metrics, and plots always run. When
        ``log_into_mlflow`` is True, the same results are written to the MLflow
        experiment on ``self.run_time_config``; when False they stay local.
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
            experiment_name = self.run_time_config.experiment_name
            mlflow.set_experiment(experiment_name)
            log.info(f"Set MLflow experiment to: {experiment_name}")
            start_kwargs = {}
            if self.run_time_config.mlflow_run_id:
                start_kwargs["run_id"] = self.run_time_config.mlflow_run_id
            else:
                start_kwargs["run_name"] = (
                    run_name
                    or self.run_time_config.mlflow_run_name
                    or f"train_{model_type}"
                )
            with mlflow.start_run(**start_kwargs) as active_run:
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
        """Save the trained model and configuration to disk.

        Loads the best model from MLflow using the URI specified in configuration and
        saves it as a pickle file along with the configuration in YAML format.
        Creates timestamped files in the specified output directory.

        Returns
        -------
        None
            Saves two files to disk:
            - config_{timestamp}.yaml: Model configuration
            - best_model_{timestamp}.pkl: Trained model pickle file

        Notes
        -----
        Requires that self.config.best_model_uri and self.config.model_output_path
        are set. If best_model_uri is None, the method returns without saving.

        The output directory is created if it doesn't exist.

        Examples
        --------
        >>> # Set output path and model URI in config first
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
        """Evaluate model predictions given a DataFrame with target and probability
        columns.

        This static method performs comprehensive evaluation of model predictions including
        metrics calculation, plot generation, and optional MLflow logging. It evaluates
        both the full dataset and optionally a contrafactual subset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the target column and probability predictions.
        target_column : str
            Name of the column containing true target values (0 or 1).
        probability_column : str
            Name of the column containing predicted probabilities (0.0 to 1.0).
        threshold : float, optional
            Classification threshold for converting probabilities to binary predictions.
            If None (default), automatically calculates optimal threshold using F1 score
            maximization on the full dataset.
        set_name : str, default="evaluation"
            Name identifier for this evaluation (used in plot titles and logs).
        use_mlflow : bool, default=False
            Whether to log metrics and plots to MLflow.
        mlflow_run_name : str, optional
            Custom name for the MLflow run. If None and use_mlflow=True, uses set_name.
        mlflow_experiment_path : str, optional
            MLflow experiment path. Required if use_mlflow=True.
        show_plots : bool, default=True
            Whether to display plots. Only applicable when use_mlflow=False.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics with keys:
            - 'metrics': Main evaluation metrics for the full dataset
            - 'contrafactual_metrics': Metrics for contrafactual subset (if applicable)
            - 'threshold': Threshold used for binary classification
            - 'contrafactual_count': Number of contrafactual samples (if applicable)

        Raises
        ------
        ValueError
            If required columns are missing from the DataFrame.
        RuntimeError
            If use_mlflow=True but MLflow experiment is not found.

        Examples
        --------
        >>> # Basic evaluation without MLflow
        >>> df = pd.DataFrame({
        ...     'target': [0, 1, 0, 1, 1],
        ...     'probability': [0.2, 0.8, 0.3, 0.9, 0.7]
        ... })
        >>> results = SolvencyNewCustomersModelTraining.evaluate_predictions(
        ...     df=df,
        ...     target_column='target',
        ...     probability_column='probability',
        ...     threshold=0.5
        ... )

        >>> # Evaluation with contrafactual subset
        >>> df['contrafactual_flag'] = [0, 1, 0, 1, 0]
        >>> results = SolvencyNewCustomersModelTraining.evaluate_predictions(
        ...     df=df,
        ...     target_column='target',
        ...     probability_column='probability',
        ...     contrafactual_flag_column='contrafactual_flag'
        ... )

        >>> # Evaluation with MLflow logging
        >>> results = SolvencyNewCustomersModelTraining.evaluate_predictions(
        ...     df=df,
        ...     target_column='target',
        ...     probability_column='probability',
        ...     use_mlflow=True,
        ...     mlflow_experiment_path='/experiments/my_model',
        ...     mlflow_run_name='evaluation_run_1'
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

