import sys
import os
import mlflow
import pickle


def save_model_recommendation_data(model_version=None, features_pos_effects=None, dependencies=None, recommendation_msgs=None):
    if model_version is None:
        sys.exit("Model version must be provided as an argument")

    # Load dependencies and recommendation messages from previous version if not provided
    if dependencies is None:
        with open(f"../recommendation_data/model_v{int(model_version) - 1}/dependencies.pkl", "rb") as f:
            dependencies = pickle.load(f)
    if recommendation_msgs is None:
        with open(f"../recommendation_data/model_v{int(model_version) - 1}/recommendation_msgs.pkl", "rb") as f:
            recommendation_msgs = pickle.load(f)
    if features_pos_effects is None:
        with open(f"../recommendation_data/model_v{int(model_version) - 1}/features_pos_effects.pkl", "rb") as f:
            features_pos_effects = pickle.load(f)
    

    os.makedirs(f"../recommendation_data/model_v{model_version}", exist_ok=True)
    
    with open(f"../recommendation_data/model_v{model_version}/features_pos_effects.pkl", "wb") as f:
        pickle.dump(features_pos_effects, f)

    with open(f"../recommendation_data/model_v{model_version}/dependencies.pkl", "wb") as f:
        pickle.dump(dependencies, f)

    with open(f"../recommendation_data/model_v{model_version}/recommendation_msgs.pkl", "wb") as f:
        pickle.dump(recommendation_msgs, f)

 



def deploy_model(mlflow_uri=None, run_id=None, model_name=None):
    if mlflow_uri is None:
        sys.exit("MLflow URI must be provided as an argument")
    if run_id is None:
        sys.exit("Run ID must be provided as an argument")
    if model_name is None:
        sys.exit("Model name must be provided as an argument")
    

    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.MlflowClient()

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