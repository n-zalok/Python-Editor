import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import shap
import time
from tqdm import tqdm
from python_editor.model_prediction import get_model_prediction_from_text


def get_metrics(y_true, y_predicted):  
    mae = mean_absolute_error(y_true, y_predicted)
    rmse = root_mean_squared_error(y_true, y_predicted)
    r_squared = r2_score(y_true, y_predicted)

    return {"MAE": f"{mae:.3f}", "RMSE": f"{rmse:.3f}", "R2": f"{r_squared:.3f}"}


def get_top_k(df: pd.DataFrame, metric_col: str, k: int) -> tuple:
    df_copy = df.copy()
    df_copy["abs_error"] = df[metric_col].abs()

    top_performing = df_copy.nsmallest(k, "abs_error")
    most_over_estimated = df_copy.nsmallest(k, metric_col)
    most_under_estimated = df_copy.nlargest(k, metric_col)

    return (
        top_performing,
        most_over_estimated,
        most_under_estimated
    )


def get_feature_importance(model, features: list, embedding_dim: int = 0) -> pd.DataFrame:
    importances = model.feature_importances_
    feature_names = list(str(x) for x in range(embedding_dim)) + features

    # make it readable
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    return importance_df


def get_shap_df(model, features, X_train, X_test, embedding_dim: int = 0):
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test)

    shap_values.feature_names = list(str(x) for x in range(embedding_dim)) + features
    return shap_values


def compare_time(
                    pylint_func,
                    generate_features_func, 
                    model, 
                    test_texts: pd.Series, 
                    embedding_dim: int = 0
                ):
    model_time = 0
    pylint_time = 0

    for text in tqdm(test_texts):
        row = pd.Series({"text": text})

        start_time = time.time()
        _, _ = get_model_prediction_from_text(row, generate_features_func, model, embedding_dim)
        end_time = time.time()
        model_time += (end_time - start_time)

        start_time = time.time()
        _ = pylint_func(row)
        end_time = time.time()
        pylint_time += (end_time - start_time)

    return {
        "model_time": model_time / len(test_texts),
        "pylint_time": pylint_time / len(test_texts)
    }