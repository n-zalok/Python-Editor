import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import shap


def get_metrics(y_true, y_predicted):  
    mae = mean_absolute_error(y_true, y_predicted)
    rmse = root_mean_squared_error(y_true, y_predicted)
    r_squared = r2_score(y_true, y_predicted)

    return {"MAE": mae, "RMSE": rmse, "R2": r_squared}


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


def get_feature_importance(model, embedding_dim: int, features: list) -> pd.DataFrame:
    importances = model.feature_importances_
    feature_names = list(str(x) for x in range(embedding_dim)) + features

    # make it readable
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    return importance_df


def get_shap_df(model, embed_dim, features, X_train, X_test):
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test)

    shap_values.feature_names = list(str(x) for x in range(embed_dim)) + features
    return shap_values