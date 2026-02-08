import pandas as pd

SAFE_LICENSES = [
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "cc0-1.0"
]


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["language", "score"], axis=1)
    df = df.rename(columns={"size": "NUM_CHARS"})
    df["NUM_CHARS"] = df["NUM_CHARS"].astype(int)

    return df


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["license"].isin(SAFE_LICENSES)]
    df = df[df["NUM_CHARS"] < df["NUM_CHARS"].quantile(0.99)]
    df = df.reset_index(drop=True)

    return df