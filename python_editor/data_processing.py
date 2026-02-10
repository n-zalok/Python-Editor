import pandas as pd
import subprocess
import tempfile
import json
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm


def get_pylint_text(row: pd.Series) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False
    ) as f:
        f.write(row["text"])
        filename = f.name

    result = subprocess.run(
        ["pylint", filename, "-r", "n", "--output-format=json2"],
        capture_output=True,
        text=True
    )

    return result.stdout


def get_pylint_score(row: pd.Series) -> float:
    pylint_json = json.loads(row["pylint_text"])
    fatal = pylint_json["statistics"]["messageTypeCount"]["fatal"]
    error = pylint_json["statistics"]["messageTypeCount"]["error"]

    if fatal or error:
        return -1
    else:
        return pylint_json["statistics"]["score"]


def split_by_developer(df: pd.DataFrame, test_size: float, random_state: int) -> pd.DataFrame:
    df["developer"] = df.apply(lambda row: row["repo_name"].split("/")[0], axis=1)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    splits = splitter.split(df, groups=df["developer"])
    train_idx, test_idx = next(splits)

    return df.iloc[train_idx, :], df.iloc[test_idx, :]

def vectorize_code(code_snippets: pd.Series) -> float:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    embeddings = []
    for snippet in tqdm(code_snippets):
        tokens = tokenizer(
                        snippet,
                        max_length=512,
                        truncation=True,
                        padding=True,
                        return_overflowing_tokens=True,
                        stride=128,
                        return_tensors="pt"
                        )
        
        del tokens["overflow_to_sample_mapping"]
        with torch.no_grad():
            output = model(**tokens)
        
        chunck_emb = output.last_hidden_state.mean(dim=1)
        file_emb = chunck_emb.mean(dim=0)

        embeddings.append(file_emb)

    return np.array(embeddings)
