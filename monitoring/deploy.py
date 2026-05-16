import sys
import mlflow


def deploy_model(mlflow_uri=None, run_id=None, model_name=None):
    if mlflow_uri is None:
        sys.exit("MLflow URI must be provided as an argument")
    if run_id is None:
        sys.exit("Run ID must be provided as an argument")
    if model_name is None:
        sys.exit("Model name must be provided as an argument")
    

    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.MlflowClient()
    run = mlflow.get_run(run_id)

    # Register model
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
    )

    # Assign alias
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=result.version
    )