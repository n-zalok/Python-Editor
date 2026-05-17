from pathlib import Path
import sys
import mlflow
from sqlalchemy import create_engine
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from datetime import datetime, timezone
from python_editor.data_processing import split_by_developer, get_pylint_text, get_pylint_score
from python_editor.feature_generation_v2 import TRANSFORMED_FEATURES
from evidently import Report
from evidently.presets import DataDriftPreset
from jinja2 import Environment, FileSystemLoader
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
DECLINE_THRESHOLD = 0.25
DRIFT_THRESHOLD = 0.2


def get_production_model_version(mlflow_uri):
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.MlflowClient()

    print("Retrieving model version from MLflow")
    version = client.get_model_version_by_alias(
        name="recommendation_model",
        alias="production"
    )
    version = version.version

    return version

def get_model_info(mlflow_uri, model_version):
    print(f"Connecting to MLflow at {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    model_uri = f"models:/recommendation_model/{model_version}"
    model_info = mlflow.models.get_model_info(model_uri)
    run_id = model_info.run_id
    run = mlflow.get_run(run_id)

    print("Getting model test performance and training data from MLflow")
    rmse = run.data.metrics["test_rmse"]

    df = pd.read_pickle(f"{ROOT_DIR}/data/{run.data.tags['dataset']}")
    train, _ = split_by_developer(df, test_size=float(run.data.tags["test_size"]), random_state=int(run.data.tags["random_state"]))
        
    return rmse, train


def get_df(model_version, db_uri, max_time=None, limit=None):
    print(f"Connecting to database at {db_uri}")
    engine = create_engine(db_uri)
    
    if max_time:
        max_time_dt = datetime.strptime(max_time, "%Y-%m-%d_%H-%M-%S_UTC").replace(tzinfo=timezone.utc)
        query = f"SELECT * FROM code_submissions WHERE model_version = '{model_version}' AND created_at <= '{max_time_dt}'"
        df = pd.read_sql(query, engine)

        if limit:
            df = df.head(limit)

        return df, max_time
    else:
        query = f"SELECT * FROM code_submissions WHERE model_version = '{model_version}'"
        df = pd.read_sql(query, engine)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
        print(f"Query timestamp: {timestamp}")

        if limit:
            df = df.head(limit)
        
        return df, timestamp


def prepare_df(df):
    df = df[df["prediction_output"] != -1]

    df_expanded = pd.json_normalize(df['extracted_features'])
    df = df.join(df_expanded).drop(columns=['extracted_features'])

    df.rename(columns={'raw_code': 'text', 'user_id': 'repo_name'}, inplace=True)
    df["repo_name"] = df["repo_name"].astype(str)

    print("Calculating pylint scores")
    df["pylint_text"] = df.progress_apply(get_pylint_text, axis=1)
    df["pylint_score"] = df.progress_apply(get_pylint_score, axis=1)

    df.drop(columns=["pylint_text"], inplace=True)
    df = df[df["pylint_score"] != -1]

    df = df.reset_index(drop=True)

    return df


def evaluate_distribution(df, train_df):
    report = Report(metrics=[DataDriftPreset()])
    report_results = report.run(reference_data=train_df[TRANSFORMED_FEATURES], current_data=df[TRANSFORMED_FEATURES])

    report_dict = report_results.dict()
    share_drifted_features = report_dict["metrics"][0]["value"]["share"]
    
    drifts = {}
    for col in report_dict["metrics"][1:]:
        if col["value"] > col["config"]["threshold"]:
            drifts[col["config"]["column"]] = col["value"]

    return report_results, share_drifted_features, drifts


def generate_report(mlflow_uri=None, db_uri=None, model_version=None):
    if mlflow_uri is None:
        sys.exit("MLflow URI must be provided as an argument")
    if db_uri is None:
        sys.exit("Database URI must be provided as an argument")
    if model_version is None:
        model_version = get_production_model_version(mlflow_uri)

    old_rmse, train_df = get_model_info(mlflow_uri, model_version)
    print(f"Evaluating model version {model_version}")

    df, timestamp = get_df(model_version, db_uri)
    print(f"Loaded {len(df)} rows from database")

    df = prepare_df(df)
    print(f"Prepared {len(df)} rows")

    new_rmse = ((df["pylint_score"] - df["prediction_output"]) ** 2).mean() ** 0.5
    rmse_diff = new_rmse - old_rmse
    if rmse_diff > (DECLINE_THRESHOLD * old_rmse):
        rmse_msg = "exceeded"
    else:
        rmse_msg = "did not exceed"
    print("Calculated new rmse")


    report, share_drifted_features, drifts = evaluate_distribution(df, train_df)
    if share_drifted_features > DRIFT_THRESHOLD:
        drift_msg = "exceeded"
    else:
        drift_msg = "did not exceed"
    print("Calculated drift")

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("report_template.html")
    print("Loaded template")

    report.save_html(f"temp_report_{timestamp}.html")
    with open(f"temp_report_{timestamp}.html", "r", encoding="utf-8") as f:
        plot_html = f.read()

    html = template.render(
        old_rmse=old_rmse,
        new_rmse=new_rmse,
        rmse_diff=rmse_diff,
        rmse_msg=rmse_msg,
        share_drifted_features=share_drifted_features,
        drift_msg=drift_msg,
        drifts=drifts,
        plot=plot_html
    )

    os.remove(f"temp_report_{timestamp}.html")

    os.makedirs("reports", exist_ok=True)
    with open(f"reports/report_{model_version}_{timestamp}.html", "w") as f:
        f.write(html)
    
    print("Saved report")