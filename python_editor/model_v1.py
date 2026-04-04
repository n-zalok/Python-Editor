import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from python_editor.data_processing import has_executable_code, vectorize_code
from python_editor.feature_generation import generate_features


curr_path = Path(os.path.dirname(__file__))

model_path = Path("../models/model_v1.pkl")
scaler_path = Path("../models/scaler_v1.pkl")
SCALER = joblib.load(curr_path / scaler_path)
MODEL = joblib.load(curr_path / model_path)

NUM_FEATURES = [
                "characters",
                "code_compactness",
                "chars_per_line",
                "comment_ratio",
                "variable_ratio",
                "avg_var_name",
                "num_funcs_and_classes",
                "avg_func_class_name",
                "avg_func_class_chars",
                "avg_func_class_args",
                "func_class_docstring_ratio"
            ]

def get_model_prediction_from_text(row: pd.Series)-> float:
    if has_executable_code(row):
        embeddings = vectorize_code(row)
        features = generate_features(row)
        df = pd.DataFrame([features]).astype(float)
        df.loc[:, NUM_FEATURES] = SCALER.transform(df[NUM_FEATURES])

        vectorized_features = np.concatenate(
        [
            embeddings,
            df.values
        ],
        axis=1
    )
        prediction = MODEL.predict(vectorized_features)

        return prediction[0].item()
    else:
        return -1