from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator


def first_day_of_month(date_str: str) -> str:
    """Given a date string in 'yyyy-mm-dd' format, return the first day of that month as
    a string in the same format.

    Args:
        date_str (str): Date string in 'yyyy-mm-dd' format.

    Returns:
        str: Date string for the first day of the month in 'yyyy-mm-01' format.

    Raises:
        ValueError: If the input date string is not in the correct format.

    Example:
        >>> first_day_of_month('2025-02-15')
        '2025-02-01'
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        first_day = date_obj.replace(day=1)
        return first_day.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Invalid date format for first_day_of_month: {e}")


def last_day_of_month(date_str: str) -> str:
    """Given a date string in 'yyyy-mm-dd' format, return the last day of that month as
    a string in the same format.

    Args:
        date_str (str): Date string in 'yyyy-mm-dd' format.

    Returns:
        str: Date string for the last day of the month in 'yyyy-mm-dd' format.

    Raises:
        ValueError: If the input date string is not in the correct format.

    Example:
        >>> last_day_of_month('2025-02-10')
        '2025-02-28'
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # Move to the first day of the next month, then subtract one day
        if date_obj.month == 12:
            next_month = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
        else:
            next_month = date_obj.replace(month=date_obj.month + 1, day=1)
        last_day = next_month - timedelta(days=1)
        return last_day.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Invalid date format for last_day_of_month: {e}")


class BaseConfigParams(BaseModel):
    """Configuration parameters for MLflow experiments and runtime.

    Attributes:
        country_code (str): Identifier for the country (default: "DE").
        product (str): Name of the product (default: "AP").
        experiment_name (str): Name of the experiment (default: "rp_solvency_newcustomers_4_0_0").
        is_dev (Optional[bool]): Flag for development environment (default: True).
        mlflow_run_id (Optional[str]): MLflow run ID.
        mlflow_run_name (Optional[str]): MLflow run name.
        training_start_date (str): Training period start date ('yyyy-mm-dd', default: "2024-01-01").
        training_end_date (str): Training period end date ('yyyy-mm-dd', default: "2025-06-3").
        test_start_date (Optional[str]): Validation period start date ('yyyy-mm-dd').
        test_end_date (Optional[str]): Validation period end date ('yyyy-mm-dd').

    Example:
        >>> params = BaseConfigParams()
        >>> print(params.training_start_date)
        '2024-01-01'
    """

    experiment_name: str = "rp_solvency_newcustomers_4_0_0"
    is_dev: Optional[bool] = True
    mlflow_run_id: Optional[str] = None
    mlflow_run_name: Optional[str] = ""
    training_start_date: str = "2024-01-01"
    training_end_date: str = "2025-06-30"
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None
    validation_start_date: Optional[str] = None
    validation_end_date: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # If validation dates are not provided, set them to two months after training_end_date
        self.adjust_test_dates(init=True)
        if self.mlflow_run_name == "" or self.mlflow_run_name is None:
            self.mlflow_run_name = (
                self.training_start_date
                + "_"
                + self.training_end_date
                + "_"
                + self.test_start_date
                + "_"
                + self.test_end_date
            )

    def adjust_test_dates(self, init=False):
        """Adjust validation date fields if not provided, based on training_end_date.

        If validation dates are missing, sets test_start_date to the first day of the month after training_end_date,
        and test_end_date to the first day of the following month.

        Args:
            init (bool): If True, only set if validation dates are missing. If False, always recalculate.

        Raises:
            ValueError: If training_end_date is not in the correct format.

        Example:
            >>> params = BaseConfigParams(training_end_date="2025-06-30")
            >>> params.adjust_test_dates()
            >>> print(params.test_start_date)
            '2025-07-01'
        """
        if self.test_start_date is None or self.test_end_date is None or not init:
            try:
                end_date = datetime.strptime(self.training_end_date, "%Y-%m-%d")
                val_start = (
                    end_date.replace(day=1) + pd.DateOffset(months=1)
                ).strftime("%Y-%m-%d")
                val_end = last_day_of_month(
                    (end_date.replace(day=1) + pd.DateOffset(months=2)).strftime(
                        "%Y-%m-%d"
                    )
                )
                self.test_start_date = val_start
                self.test_end_date = val_end
            except Exception as e:
                raise ValueError(
                    f"Could not calculate test_start_date and test_end_date from training_end_date: {e}"
                )

    @field_validator("training_start_date")
    def validate_start_training_date(cls, v: str) -> str:
        """Validate and normalize the training_start_date field.

        Converts the input date string to the first day of the month in 'yyyy-mm-01' format.

        Args:
            v (str): Date string in 'yyyy-mm-dd' format.

        Returns:
            str: Date string for the first day of the month in 'yyyy-mm-01' format.

        Raises:
            ValueError: If the input date string is not in the correct format.

        Example:
            >>> BaseConfigParams.validate_start_training_date('2025-06-15')
            '2025-06-01'
        """
        try:
            date_obj = datetime.strptime(v, "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-01")
        except ValueError:
            raise ValueError("training_start_date must be in the format yyyy-mm-dd")

    @field_validator("training_end_date")
    def validate_training_end_date(cls, v: str) -> str:
        """Validate and normalize the training_end_date field.

        Converts the input date string to the last day of the month in 'yyyy-mm-dd' format.

        Args:
            v (str): Date string in 'yyyy-mm-dd' format.

        Returns:
            str: Date string for the last day of the month in 'yyyy-mm-dd' format.

        Raises:
            ValueError: If the input date string is not in the correct format.

        Example:
            >>> BaseConfigParams.validate_training_end_date('2025-06-15')
            '2025-06-30'
        """
        try:
            return last_day_of_month(v)
        except Exception:
            raise ValueError("training_end_date must be in the format yyyy-mm-dd")