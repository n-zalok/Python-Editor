import pandas as pd
import subprocess
import tempfile
import json


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