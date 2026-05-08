import pandas as pd
import numpy as np
from python_editor.data_processing import has_executable_code, vectorize_code


def get_model_prediction_from_text(
                                    row: pd.Series,
                                    generate_features_func,
                                    model,
                                    embedding_dim: int = 0
                                ):
    
    if has_executable_code(row):
        features = generate_features_func(row)
        vectorized_features = np.array(list(features.values())).reshape(1, -1)

        if embedding_dim:
            embeddings = vectorize_code(row)
            embeddings_array = embeddings.reshape(1, -1)

            vectorized_features = np.concatenate(
                [
                    embeddings_array,
                    vectorized_features
                ],
                axis=1
            )
        
        prediction = model.predict(vectorized_features)

        return features, prediction[0].item()
    else:
        return {}, -1