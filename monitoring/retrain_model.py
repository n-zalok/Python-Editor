from pathlib import Path
import sys
import mlflow
import pandas as pd
from datetime import datetime, timezone
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from python_editor.data_processing import split_by_developer, get_vectorized_features_and_label
from python_editor.feature_generation_v2 import TRANSFORMED_FEATURES
from python_editor.model_evaluation import get_metrics
from monitoring.monitor_performance import get_df, prepare_df


ROOT_DIR = Path(__file__).resolve().parent.parent


def get_model_and_its_df(mlflow_uri, model_version):
    print(f"Connecting to MLflow at {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    model_uri = f"models:/recommendation_model/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    model_info = mlflow.models.get_model_info(model_uri)
    run_id = model_info.run_id
    run = mlflow.get_run(run_id)
    df = pd.read_pickle(f"{ROOT_DIR}/data/{run.data.tags['dataset']}")

    if "created_at" not in df.columns:
        end_time_ms = run.info.end_time
        end_datetime = datetime.fromtimestamp(end_time_ms / 1000, tz=timezone.utc)
        df["created_at"] = end_datetime
        
    return model, df


def create_and_save_training_df(old_df, new_df, model_version, timestamp):
    combined_df = pd.concat([old_df, new_df], join="inner", ignore_index=True)
    new_df_start_index = len(old_df)

    df_name = f"combined_features_{model_version}_{timestamp}.pkl"
    df_path = f"{ROOT_DIR}/data/{df_name}"
    combined_df.to_pickle(df_path)

    return df_name, new_df_start_index


def time_decay_weights(df, timestamp, half_life_days=30):
    if df["created_at"].max() > datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S_UTC").replace(tzinfo=timezone.utc):
        raise ValueError("Some rows in the dataframe have created_at timestamps after the provided timestamp.")
    
    decay_rate = np.log(2) / half_life_days
    timestamp_dt = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S_UTC").replace(tzinfo=timezone.utc)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    time_diff = (timestamp_dt - df["created_at"]).dt.total_seconds() / (3600 * 24)  # Convert to days
    weights = np.exp(-decay_rate * time_diff)

    return weights


def train_new_model(
                    mlflow_uri, run_name, dataset, X_train, y_train, X_test, y_test,
                    test_size=0.3,
                    random_state=0,
                    sample_weight=None,
                    half_life_days=None
                ):

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("random_forest_regressor")

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "oob_score": True,
            "random_state": 0,
            "n_jobs": -1        
        }

        model = RandomForestRegressor(**params)

        cv = KFold(n_splits=5, shuffle=True, random_state=0)

        # negative RMSE because sklearn maximizes score
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            params={"sample_weight": sample_weight}
        )

        rmse = -scores.mean()

        # Nested MLflow run for each trial
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse", rmse)

        return rmse


    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "dataset": dataset,
                "processing": "data_processing.py",
                "feature_generation": "feature_generation_v2.py",
                "test_size": test_size,
                "random_state": random_state,
                "half_life_days": half_life_days
            }
        )

        study = optuna.create_study(direction="minimize")

        study.optimize(objective, n_trials=50, show_progress_bar=True)

        best_params = study.best_params

        # Final model
        best_model = RandomForestRegressor(
            **best_params,
            oob_score=True,
            random_state=0,
            n_jobs=-1
        )

        best_model.fit(X_train, y_train, sample_weight=sample_weight)

        preds = best_model.predict(X_test)
        metrics = get_metrics(y_test, preds)

        # Log best results
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_rmse", study.best_value)
        mlflow.log_metric("test_mae", metrics["MAE"])
        mlflow.log_metric("test_rmse", metrics["RMSE"])
        mlflow.log_metric("test_r2", metrics["R2"])

        mlflow.sklearn.log_model(best_model, "model")
    

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("random_forest_regressor")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        max_results=1
    )

    parent_run = runs[0]
    run_id = parent_run.info.run_id

    return run_id, metrics


def retrain_model(
                    mlflow_uri=None, db_uri=None, model_version=None, timestamp=None,
                    test_size=0.3,
                    random_state=0
                ):
    
    if mlflow_uri is None:
        sys.exit("MLflow URI must be provided as an argument")
    if db_uri is None:
        sys.exit("Database URI must be provided as an argument")
    if model_version is None:
        sys.exit("Model version must be provided as an argument")
    if timestamp is None:
        sys.exit("Timestamp must be provided as an argument in the format YYYY-MM-DD_HH:MM:SS")
    

    old_model, old_df = get_model_and_its_df(mlflow_uri, model_version)
    print(f"Evaluating model version {model_version}")


    new_df, _ = get_df(model_version, db_uri, timestamp)
    print(f"Loaded {len(new_df)} rows from database")


    new_df = prepare_df(new_df)
    print(f"Prepared {len(new_df)} rows")


    df_name, new_df_start_index = create_and_save_training_df(old_df, new_df, model_version, timestamp)
    print(f"Created combined dataset at {df_name}")


    df = pd.read_pickle(f"{ROOT_DIR}/data/{df_name}")
    old_df = df.iloc[:new_df_start_index]
    new_df = df.iloc[new_df_start_index:]
    print(f"Split combined dataset into old ({len(old_df)}) and new ({len(new_df)})")


    train, test = split_by_developer(new_df, test_size, random_state)
    train = pd.concat([old_df, train], ignore_index=True)
    X_train, y_train = get_vectorized_features_and_label(train, TRANSFORMED_FEATURES)
    X_test, y_test = get_vectorized_features_and_label(test, TRANSFORMED_FEATURES)
    print(f"Split new dataset into train ({len(train)}) and test ({len(test)})")


    half_life_days = 30
    sample_weight = time_decay_weights(train, timestamp, half_life_days=half_life_days)
    print("Calculated time decay sample weights for training data")
    print(f"Sample weights summary: min={sample_weight.min()}, max={sample_weight.max()}, mean={sample_weight.mean()}")


    run_name = f"retrain_{model_version}_{timestamp}"
    run_id, new_metrics = train_new_model(
        mlflow_uri, run_name, df_name, X_train, y_train, X_test, y_test,
        test_size=test_size,
        random_state=random_state,
        sample_weight=sample_weight,
        half_life_days=half_life_days
    )
    print(f"Trained new model and logged to MLflow with run ID {run_id}")


    old_y_preds = old_model.predict(X_test)
    old_metrics = get_metrics(y_test, old_y_preds)

    return run_id, old_metrics, new_metrics