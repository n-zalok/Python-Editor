import pickle
import pandas as pd
import numpy as np
from python_editor.data_processing import has_executable_code
from python_editor.feature_generation_v2 import generate_transformed_features


with open("../models/model_v3.pkl", "rb") as f:
    MODEL = pickle.load(f)


def get_model_prediction_from_text(row: pd.Series)-> float:
    if has_executable_code(row):
        features = generate_transformed_features(row)
        vectorized_features = np.array(list(features.values())).reshape(1, -1)

        prediction = MODEL.predict(vectorized_features)

        return features, prediction[0].item()
    else:
        return {}, -1