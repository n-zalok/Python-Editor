import pickle
import pandas as pd
import numpy as np
from python_editor.data_processing import has_executable_code, vectorize_code
from python_editor.feature_generation import generate_features


with open("../models/model_v1.pkl", "rb") as f:
    MODEL = pickle.load(f)


def get_model_prediction_from_text(row: pd.Series)-> float:
    if has_executable_code(row):
        embeddings = vectorize_code(row)
        embeddings_array = embeddings.reshape(1, -1)

        features = generate_features(row)
        features_array = np.array(list(features.values())).reshape(1, -1)

        vectorized_features = np.concatenate(
        [
            embeddings_array,
            features_array
        ],
        axis=1
    )
        prediction = MODEL.predict(vectorized_features)

        return prediction[0].item()
    else:
        return -1