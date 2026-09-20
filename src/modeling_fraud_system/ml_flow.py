# from modelling.rp_de_solvency_newcustomers_4_0_0.utils.configs import BaseConfigParams
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
from pandas import DataFrame
from modeling_fraud_system.run_time_configuration import BaseConfigParams


def get_mlflow_run(base_config_params: BaseConfigParams) -> str:
    """Get the latest MLflow run ID for a given experiment run name.

    This function retrieves the latest MLflow run ID for a specified experiment based on the provided
    configuration parameters experiment name and run name.
    If no run is found that matches the given tags (country, prediction_date, product),
    a new run is started with the provided tags.
    If no experimnt if found, it will create a new experiment and start a new run with the provided tags.

        Note:
        If this project is not a solvency project, change the `solvency_mlflow_exp_path` to the appropriate location.

    Args:
        self (object): The object that contains the base configuration parameters.

    Returns:
        str: MLflow run ID.
    """

    solvency_mlflow_exp_path = "/dna-data-scientists/solvency"
    solvency_mlflow_exp_location = (
        f"{solvency_mlflow_exp_path}/{base_config_params.experiment_name}"
    )

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(solvency_mlflow_exp_location)

    if experiment:
        print("experiment found")
        experiment_id = experiment.experiment_id
        filter_string = (
            f"tags.`country` = '{base_config_params.country_code}' and "
            f"tags.`training_start_date` = '{base_config_params.training_start_date}' and "
            f"tags.`training_end_date` = '{base_config_params.training_end_date}' and "
            f"tags.`product` = '{base_config_params.product}' and "
            f"tags.`mlflow.runName` = '{base_config_params.mlflow_run_name}' "
        )
        runs = client.search_runs(
            experiment_ids=[experiment_id], filter_string=filter_string, max_results=1
        )
        if len(runs) > 0:
            mlflow_run_id = runs[0].info.run_id
            print(mlflow_run_id)
            return mlflow_run_id

    else:
        print("create new experiment")
        mlflow.create_experiment(solvency_mlflow_exp_location)

    print("creating new run")
    mlflow.set_experiment(solvency_mlflow_exp_location)
    mlflow.start_run(
        run_name=base_config_params.mlflow_run_name,
        tags={
            "country": base_config_params.country_code,
            "training_start_date": base_config_params.training_start_date,
            "training_end_date": base_config_params.training_end_date,
            "product": base_config_params.product,
        },
    )
    mlflow_run_id = mlflow.active_run().info.run_id
    print(mlflow_run_id)
    mlflow.end_run()
    return mlflow_run_id


def get_custom_mlflow_run_id(
    country_code: str,
    training_start_date: str,
    training_end_date: str,
    partner: str,
    experiment_name: str,
) -> str:
    """Retrieves the MLflow run ID for a given set of parameters.

    Args:
        country_code (str): The country code tag.
        prediction_date (str): The prediction date tag.
        partner (str): The partner tag.
        experiment_name (str): The name of the MLflow experiment.

    Returns:
        str: The MLflow run ID if found, else None.
    """

    mlflow_run_id = None
    solvency_mlflow_exp_path = (
        "/Shared"  # change this for projects tht are not solvency.
    )
    solvency_mlflow_exp_path = f"{solvency_mlflow_exp_path}/{experiment_name}"

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(solvency_mlflow_exp_path)
    if experiment:
        experiment_id = experiment.experiment_id
        filter_string = (
            f"tags.`country` = '{country_code}' and "
            f"tags.`training_start_date` = '{training_start_date}' and "
            f"tags.`training_end_date` = '{training_end_date}' and "
            f"tags.`partner` = '{partner}' "
        )
        runs = client.search_runs(
            experiment_ids=[experiment_id], filter_string=filter_string, max_results=1
        )
        if len(runs) > 0:
            mlflow_run_id = runs[0].info.run_id

    return mlflow_run_id


def get_mlflow_run_params(run_id: str, parameter_name: str) -> str:
    """Retrieve a specific parameter from an MLflow run.

    Args:
        run_id (str): The ID of the MLflow run.
        parameter_name (str): The name of the parameter to retrieve.

    Returns:
        str: The value of the specified parameter if it exists, otherwise None.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return run.data.params.get(parameter_name, None)
    except mlflow.exceptions.MlflowException as e:
        print(f"MLflowException: {e}")
        return None


def log_table_to_mlflow(
    base_config_params: BaseConfigParams,
    dataframe: pd.DataFrame,
    format: str,
    file_name: str,
    path: str = "datafiles",
) -> None:
    """Save data as an artifact in MLflow.

    Args:
        dataframe (pd.DataFrame): The data to save.
        format (str): The format of the file.
        path (str): The path to save the file the MLflow.
        file_name (str): The name of the file.
        base_config_params (BaseConfigParams): The base configuration parameters.

    Returns:
        None
    """

    with mlflow.start_run(run_id=base_config_params.mlflow_run_id):
        if format not in ["csv", "parquet"]:
            raise ValueError(
                f"Format {format} not supported. Please use 'csv' or 'parquet'."
            )
        temp_file_path = os.path.join("/tmp", f"{file_name}.{format}")
        if not isinstance(dataframe, DataFrame):
            print("Converting Spark DataFrame to Pandas DataFrame for logging.")
            dataframe = dataframe.toPandas()
        if format == "parquet":
            dataframe.to_parquet(temp_file_path, index=False)
        else:
            dataframe.to_csv(temp_file_path, index=False)

        mlflow.log_artifact(temp_file_path, artifact_path=path)
        os.remove(temp_file_path)  # Clean up the temporary file

    mlflow.end_run()


def log_artifacts_to_mlflow(
    artifact,
    format: str,
    name: str,
    path: str = "artifacts",
) -> None:
    """Log artifacts to MLflow. Formats supported: html, dict, png.

    Args:
        artifact (object): The artifact to log.
        format (str): The format of the artifact.
        path (str): The path to save the artifact.
        name (str): The name of the artifact.

    Returns:
        None

    Example:

        htmlArtifact = "<h1>Hello World</h1>"

        dictArtifact = {"a": 1, "b": 2}

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        plt.title("My Plot")
        plt.tight_layout()
        pngArtifact = fig
        plt.close(fig)
    log_artifacts_to_mlflow(base_config_params = base_config_params,
                            artifact = htmlArtifact,
                            format = "html",
                            name = "htmlArtifact",
                            path = "datafiles1123")
    log_artifacts_to_mlflow(base_config_params = base_config_params,
                            artifact = dictArtifact,
                            format = "dict",
                            name = "dictArtifact")
    log_artifacts_to_mlflow(base_config_params = base_config_params,
                            artifact = pngArtifact,
                            format = "plt",
                            name = "pngArtifact")
    """
    supported_formats = ["html", "dict", "plt"]

    if format not in supported_formats:
        raise ValueError(
            f"Format {format} not supported. Supported formats: {supported_formats}"
        )

    if mlflow.active_run() is None:
        raise Exception(
            "No active MLflow run found. Please start an MLflow run before calling this function."
        )

    try:
        if format == "html":
            mlflow.log_text(str(artifact), f"{path}/{name}.html")
        elif format == "dict":
            mlflow.log_dict(artifact, f"{path}/{name}.json")
        elif format == "plt":
            mlflow.log_figure(artifact, f"{path}/{name}.png")
    except Exception as e:
        raise ValueError(f"Not able to log the artifact {name} to MLflow, error: {e}")


def log_params_to_mlflow(params: dict, base_config_params: BaseConfigParams) -> None:
    """Log parameters to MLflow.

    Args:
        params (dict): The parameters to log.
        base_config_params (BaseConfigParams): The base configuration parameters.
    Returns:


    Example:
        base_config_params = BaseConfigParams(mlflow_run_id="your_mlflow_run_id")
        params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "num_epochs": 10,
            "type": "classification",
        }
        log_params_to_mlflow(params, base_config_params)
    """

    if mlflow.active_run() is None:
        raise Exception(
            "No active MLflow run found. Please start an MLflow run before calling this function."
        )
    try:
        for key, value in params.items():
            mlflow.log_param(key, value)
    except Exception as e:
        raise ValueError(f"Not able to log the parameters to MLflow, error: {e}")


def save_model_into_mlflow(
    model,
    flavor: str,
    artifact_path: str = "model",
    signature=None,
) -> None:
    """Save a trained model into MLflow. The supported flavors are 'sklearn', 'pytorch',
    'catboost', 'pyfunc', 'tensorflow', 'xgboost', 'lightgbm', and 'keras'.

    Note: use pyfunc as a flavor if you want to use the model in a different environment.

    Args:
        model: The trained model to be saved.
        flavor (str): The flavor of the model. Supported flavors include 'sklearn', 'pytorch', 'catboost', 'pyfunc', 'tensorflow', 'xgboost', 'lightgbm', and 'keras'.
        base_config_params (BaseConfigParams): The base configuration parameters.
        signature: The signature of the model (optional).

    Returns:
        None
    """
    if model is None:
        print("Model not trained yet. Please call 'train()' to do it.")
        return None

    if mlflow.active_run() is None:
        raise Exception(
            "No active MLflow run found. Please start an MLflow run before calling this function."
        )

    try:
        if flavor == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path=artifact_path, signature=signature
            )
        elif flavor == "pytorch":
            mlflow.pytorch.log_model(
                pytorch_model=model, artifact_path=artifact_path, signature=signature
            )
        elif flavor == "catboost":
            mlflow.catboost.log_model(
                cb_model=model, artifact_path=artifact_path, signature=signature
            )
        elif flavor == "pyfunc":
            mlflow.pyfunc.log_model(artifact_path=model, signature=signature)
        elif flavor == "tensorflow":
            mlflow.tensorflow.log_model(
                model=model, artifact_path=artifact_path, signature=signature
            )
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(
                xgb_model=model, artifact_path=artifact_path, signature=signature
            )
        elif flavor == "lightgbm":
            mlflow.lightgbm.log_model(
                lgb_model=model, artifact_path=artifact_path, signature=signature
            )
        elif flavor == "keras":
            mlflow.keras.log_model(
                model=model, artifact_path=artifact_path, signature=signature
            )
        else:
            raise ValueError(
                f"Flavor {flavor} not supported. Please use 'sklearn', 'pytorch','catboost', 'tensorflow', 'xgboost', 'lightgbm', 'keras' or 'pyfunc'."
            )
    except Exception as e:
        raise ValueError(f"Not able to save the model to MLflow, error: {e}")


def load_model_from_mlflow(
    flavor: str,
    base_config_params: BaseConfigParams,
    model_path: str = "model",
) -> None:
    """Load a model from MLflow using the run ID and flavor.

    Args:
        flavor (str): The flavor of the model. Supported flavors include 'sklearn', 'pytorch', 'catboost', 'pyfunc', 'tensorflow', 'xgboost', 'lightgbm', and 'keras'.
        base_config_params (BaseConfigParams): The base configuration parameters containing the MLflow run ID.
        model_path (str): The path of the model within the run. Default is "model".

    Returns:
        The loaded model.
    """

    model_uri = f"runs:/{base_config_params.mlflow_run_id}/{model_path}"
    if mlflow.active_run() is None:
        raise Exception(
            "No active MLflow run found. Please start an MLflow run before calling this function."
        )

    try:
        if flavor == "sklearn":
            model = mlflow.sklearn.load_model(model_uri)
        elif flavor == "pytorch":
            model = mlflow.pytorch.load_model(model_uri)
        elif flavor == "catboost":
            model = mlflow.catboost.load_model(model_uri)
        elif flavor == "pyfunc":
            model = mlflow.pyfunc.load_model(model_uri)
        elif flavor == "tensorflow":
            model = mlflow.tensorflow.load_model(model_uri)
        elif flavor == "xgboost":
            model = mlflow.xgboost.load_model(model_uri)
        elif flavor == "lightgbm":
            model = mlflow.lightgbm.load_model(model_uri)
        elif flavor == "keras":
            model = mlflow.keras.load_model(model_uri)
        else:
            raise ValueError(
                f"Flavor {flavor} not supported. Please use 'sklearn', 'pytorch','catboost', 'tensorflow', 'xgboost', 'lightgbm', 'keras' or 'pyfunc'."
            )
    except Exception as e:
        raise ValueError(f"Not able to load the model from MLflow, error: {e}")
    return model