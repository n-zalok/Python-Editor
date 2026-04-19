import pandas as pd
import subprocess
import tempfile
import json
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import ast


def remove_skipped_files(df: pd.DataFrame, skip_strings: list) -> pd.DataFrame:
    for skip_str in skip_strings:
        df = df[~df["text"].str.contains(skip_str)]
    
    return df


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


def vectorize_code(row: pd.Series) -> float:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    tokens = tokenizer(
                    row["text"],
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

    return np.array(file_emb)


def has_executable_code(row: pd.Series) -> bool:
    # BOM (Byte Order Mark) issue.
    try:
        text = row["text"].lstrip("\uefff")
        text = text.lstrip("\uffef")
        text = text.lstrip("\uefbb")
        tree = ast.parse(text)
    except SyntaxError:
        return False
    
    # Recursively check for meaningful statements
    for child in ast.walk(tree):
        # Skip docstrings
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
            continue
        
        # Skip pass statements
        if isinstance(child, ast.Pass):
            continue
        
        # Skip type-only function/class definitions
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        
        # If we find assignment, control flow, return, etc.
        if isinstance(child, (
            ast.Assign, ast.AugAssign, ast.AnnAssign,
            ast.Return, ast.If, ast.For, ast.While,
            ast.Try, ast.With, ast.Call, ast.Raise,
            ast.Assert, ast.Yield
        )):
            return True

    return False


def get_vectorized_features_and_label(df: pd.DataFrame, features: list):
    if  "embedding" not in df.columns:
        vectorized_features = df[features].values
    else:
        vectorized_features = np.concatenate(
            [
                np.vstack(df["embedding"]),
                df[features].values
            ],
            axis=1
        )

    label = df["pylint_score"].values
    
    return vectorized_features, label
