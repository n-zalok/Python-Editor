from monitoring.monitor_performance import get_production_model_version, get_model_info, get_df, prepare_df
from monitoring.retrain_model import get_model_and_its_df, time_decay_weights
from python_editor.feature_generation_v2 import TRANSFORMED_FEATURES
import pandas as pd
from datetime import datetime, timezone
import pytest
import subprocess


MLFLOW_URI = "sqlite:////mnt/ssd/ME/ML_files/python-editor/Python-Editor/notebooks/models/mlflow/mlflow.db"
DB_URI = "postgresql+psycopg2://postgres:postgres@localhost:5432/python_editor"

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "up", "db", "-d"],
        check=True
    )

    # wait until DB is ready here

    yield

    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.yml", "down"],
        check=True
    )

def test_get_production_model_version():
    version = get_production_model_version(MLFLOW_URI)

    assert version is not None


def test_get_model_info():
    version = "1"
    rmse, train_df = get_model_info(MLFLOW_URI, version)

    assert rmse is not None
    assert isinstance(train_df, pd.DataFrame)


def test_get_df():
    version = "1"
    df, timestamp = get_df(version, DB_URI, limit=10)

    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 10
    assert isinstance(timestamp, str)
    # Check if timestamp is in the correct format
    try:
        datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S_UTC")
    except ValueError:
        assert False, "Timestamp is not in the correct format"


def test_prepare_df():
    version = "1"
    df, _ = get_df(version, DB_URI, limit=10)
    prepared_df = prepare_df(df)

    assert isinstance(prepared_df, pd.DataFrame)
    assert "prediction_output" in prepared_df.columns
    assert "pylint_score" in prepared_df.columns
    assert "text" in prepared_df.columns
    assert "repo_name" in prepared_df.columns
    for col in TRANSFORMED_FEATURES:
        assert col in prepared_df.columns
    
    assert (prepared_df["prediction_output"] >= 0).all()
    assert (prepared_df["prediction_output"] <= 10).all()


def test_get_model_and_its_df():
    version = "1"
    model, df = get_model_and_its_df(MLFLOW_URI, version)

    assert model is not None
    assert isinstance(df, pd.DataFrame)
    assert "created_at" in df.columns


def test_time_decay_weights():
    version = "1"
    _, df = get_model_and_its_df(MLFLOW_URI, version)

    try:
        time_decay_weights(df, "2000-06-01_00-00-00_UTC")
        assert False, "Expected ValueError for timestamps after the provided timestamp"
    except ValueError:
        assert True

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    sample_weight = time_decay_weights(df, timestamp)

    assert isinstance(sample_weight, pd.Series)
    assert len(sample_weight) == len(df)
    assert (sample_weight >= 0).all()
    assert (sample_weight <= 1).all()
