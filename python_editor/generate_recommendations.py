import numpy as np
import pandas as pd
from python_editor.model_prediction import get_model_prediction_from_text
from python_editor.model_evaluation import get_shap_df


def get_widest_pos_range(shap_values, features=None, window=5):
    shap_arr = shap_values.values
    data_arr = shap_values.data
    feature_names = shap_values.feature_names

    if features is None:
        features = feature_names

    results = {}

    for j in range(len(feature_names)):
        feature_name = feature_names[j]
        if feature_name not in features:
            continue

        # values of this feature
        x = data_arr[:, j]

        # shap effect of this feature
        s = shap_arr[:, j]

        # sort by feature value
        idx = np.argsort(x)
        x_sorted = x[idx]
        s_sorted = s[idx]

        s_smooth = np.convolve(s_sorted, np.ones(window)/window, mode='same')
        mask = s_smooth >= 0

        ranges = []
        start = None

        for i in range(len(mask)):
            if mask[i] and start is None:
                start = i

            if start is not None:
                is_last = i == len(mask) - 1
                next_negative = not is_last and not mask[i + 1]

                if is_last or next_negative:
                    end = i
                    ranges.append(
                        (x_sorted[start], x_sorted[end])
                    )
                    start = None

        results[feature_name] = ranges
    
    widest = {
    k: max(v, key=lambda r: r[1]-r[0]) if v else "never"
    for k, v in results.items()
    }

    return widest


def get_recommendations(
                        row: pd.Series,
                        generate_features_func,
                        model,
                        X_train: np.ndarray,
                        features_pos_effects: dict[str, str | tuple[float, float]],
                        dependencies: dict[str, dict] | None = None,
                        log_features: list[str] | None = None,
                        embedding_dim: int = 0
                    ):
    
    features, score = get_model_prediction_from_text(row, generate_features_func, model, embedding_dim)
    if score == -1 or score == 10:
        return features, score, {}
    
    vectorized_features = np.array(list(features.values())).reshape(1, -1)

    shap_values = get_shap_df(model, list(features.keys()), X_train, vectorized_features, embedding_dim)

    lines = row["text"].splitlines()
    effective_lines = len([line for line in lines if line.strip()])
    features["effective_lines"] = effective_lines


    if dependencies is None:
        dependencies = {}

    if log_features is None:
        log_features = []
    for feat in log_features:
        mask = np.array(shap_values.feature_names) == feat
        shap_values.data[:, mask] = np.expm1(shap_values.data[:, mask])
    
    feats_from_neg_to_pos_effect = shap_values.values.argsort()[0]
    recommendations = {}

    for feat in feats_from_neg_to_pos_effect:
        feat_name = shap_values.feature_names[feat]
        if feat_name in recommendations.keys():
            continue
        
        feat_value = shap_values.data[0, feat]
        pos_effect = features_pos_effects[feat_name]

        if pos_effect == "always":
            if feat_value == 0:
                recommendations[feat_name] = "increase"
        elif pos_effect == "never":
            if feat_value > 0:
                recommendations[feat_name] = "decrease"
        else:
            if feat_value < pos_effect[0]:
                if feat_name in dependencies.keys():
                    for dependency, value in dependencies[feat_name].items():
                        if features[dependency] > value:
                            recommendations[feat_name] = "increase"
                        else:
                            recommendations[dependency] = "increase"
                else:
                    recommendations[feat_name] = "increase"
            elif feat_value > pos_effect[1]:
                if feat_name in dependencies.keys():
                    for dependency, value in dependencies[feat_name].items():
                        if features[dependency] > value:
                            recommendations[feat_name] = "decrease"
                        else:
                            recommendations[dependency] = "increase"
                else:
                    recommendations[feat_name] = "decrease"
        
        if "characters" in recommendations and "effective_lines" in recommendations:
            recommendations.pop("characters")
        
        if len(recommendations) >= 3:
            break
    
    return features, score, recommendations


def construct_recommendations(score, recommendations, recommendation_msgs):
    message = f"Estimated score: {score:.2f}\n\n"

    if score == -1:
        return "Not a valid python code or code contains no logic\n\n"
    elif score == 10:
        return message + "Code is perfect! No suggestions\n\n"
    else:
        message +="Suggested improvements:\n\n"

        for feat, rec in recommendations.items():
            message += recommendation_msgs[feat][f"{rec}_msg"]
            message += "\n\n"
            
        return message