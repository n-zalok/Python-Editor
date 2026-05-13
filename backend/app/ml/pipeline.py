import mlflow
import pandas as pd
import pickle
from python_editor.generate_recommendations import get_recommendations, construct_recommendations
from python_editor.feature_generation_v2 import generate_transformed_features, TRANSFORMED_FEATURES, LOG_FEATURES
from python_editor.data_processing import split_by_developer, get_vectorized_features_and_label



mlflow.set_tracking_uri("sqlite:////mnt/ssd/ME/ML_files/python-editor/Python-Editor/notebooks/models/mlflow/mlflow.db")
model_uri = f"models:/recommendation_model@production"


model_info = mlflow.models.get_model_info(model_uri)
run_id = model_info.run_id
run = mlflow.get_run(run_id)

df = pd.read_pickle(f"/app/app/ml/data/{run.data.tags['dataset']}")
df = df.drop(columns=["embedding"])  ## extra
train, _ = split_by_developer(df, test_size=float(run.data.tags["test_size"]), random_state=int(run.data.tags["random_state"]))
X_train, _ = get_vectorized_features_and_label(train, TRANSFORMED_FEATURES)


client = mlflow.MlflowClient()
version = client.get_model_version_by_alias(
    name="recommendation_model",
    alias="production"
)

features_pos_effects = pickle.load(open(f"/app/app/ml/recommendation_data/model_v{version.version}/features_pos_effects.pkl", "rb"))
dependencies = pickle.load(open(f"/app/app/ml/recommendation_data/model_v{version.version}/dependencies.pkl", "rb"))
recommendation_msgs = pickle.load(open(f"/app/app/ml/recommendation_data/model_v{version.version}/recommendation_msgs.pkl", "rb"))


model = mlflow.sklearn.load_model(model_uri)


def pipeline(code):
    row = pd.Series({"text": code})

    features, score, recommendations = get_recommendations(
        row,
        generate_transformed_features,
        model,
        X_train,
        features_pos_effects,
        dependencies,
        LOG_FEATURES
    )

    message = construct_recommendations(score, recommendations, recommendation_msgs)

    return features, score, message, version

