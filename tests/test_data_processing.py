import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from python_editor.data_processing import  split_by_developer, has_executable_code, vectorize_code


df = pd.read_csv("../data/test_sample.csv")


def test_split_by_developer():
    train, test = split_by_developer(df, test_size=0.3)
    total = len(train) + len(test)
    test_ratio = len(test) / total
    assert total == len(df)
    assert abs(test_ratio - 0.3) < 0.05

    train_developers =  set(train["developer"])
    test_developers  =  set(test["developer"])
    assert len(train_developers.intersection(test_developers)) == 0


def test_has_executable_code():
    assert has_executable_code(pd.Series({"text": "print('Hello World')"}))
    assert not has_executable_code(pd.Series({"text": "import sys"}))
    assert not has_executable_code(pd.Series({"text": ""}))
    assert not has_executable_code(pd.Series({"text": "# just a comment"}))


def test_vectorize_code():
    embeddings_1 = vectorize_code(pd.Series({"text": "print('Hello World')"}))
    assert len(embeddings_1) == 768
    assert not np.isnan(embeddings_1).any()
    assert np.isfinite(embeddings_1).all()

    embeddings_2 = vectorize_code(pd.Series({"text": "print('Hello World')"}))
    assert (embeddings_1 == embeddings_2).all()