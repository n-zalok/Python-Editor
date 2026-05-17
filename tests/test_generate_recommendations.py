import pandas as pd
import numpy as np
import pickle
from python_editor.data_processing import split_by_developer, get_vectorized_features_and_label
from python_editor.model_prediction import get_model_prediction_from_text
from python_editor.feature_generation_v2 import generate_transformed_features, TRANSFORMED_FEATURES, LOG_FEATURES
from python_editor.model_evaluation import get_shap_df
from python_editor.generate_recommendations import get_widest_pos_range, get_recommendations


df = pd.read_csv("data/test_sample.csv")
df["pylint_score"] = np.full(len(df), 0.0)

features = df.progress_apply(generate_transformed_features, axis=1, result_type="expand")
df_transformed_features = df.join(features)

train, test = split_by_developer(df_transformed_features, test_size=0.3, random_state=0)
X_train, _ = get_vectorized_features_and_label(train, TRANSFORMED_FEATURES)
X_test, _ = get_vectorized_features_and_label(test, TRANSFORMED_FEATURES)


model = pickle.load(open("notebooks/models/model_v3.pkl", "rb"))
shap_values = get_shap_df(model, TRANSFORMED_FEATURES, X_train, X_test)


def test_get_model_prediction_from_text():
    df['score'] = df.apply(lambda row: get_model_prediction_from_text(
                                                                        row, 
                                                                        generate_transformed_features, 
                                                                        model
                                                                    )[1], axis=1)

    assert df['score'].isna().sum() == 0
    assert df['score'].between(0, 10).all()


def test_get_shap_df():
    assert isinstance(shap_values.values, np.ndarray)
    assert shap_values.values.shape[0] == len(X_test)
    assert shap_values.values.shape[1] == len(TRANSFORMED_FEATURES)


def test_get_widest_pos_range():
    ranges = get_widest_pos_range(shap_values, TRANSFORMED_FEATURES)

    assert isinstance(ranges, dict)
    assert len(ranges) == len(TRANSFORMED_FEATURES)

    for key in ranges:
        assert isinstance(ranges[key], tuple | str)
    for key in ranges:
        if isinstance(ranges[key], tuple):
            assert len(ranges[key]) == 2
            assert ranges[key][0] <= ranges[key][1]
        else:
            assert ranges[key] == "never"


def test_get_recommendations():
    ranges = get_widest_pos_range(shap_values, TRANSFORMED_FEATURES)
    _, _, recommendations = get_recommendations(
                                            test.iloc[0],
                                            generate_transformed_features,
                                            model,
                                            X_train,
                                            ranges,
                                            log_features=LOG_FEATURES
                                        )

    assert isinstance(recommendations, dict)

    for key in recommendations:
        assert key in TRANSFORMED_FEATURES
        assert recommendations[key] in ["increase", "decrease"]



