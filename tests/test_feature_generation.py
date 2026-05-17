import pandas as pd
from python_editor.feature_generation_v2 import analyze_code, get_cyclomatic_complexity, generate_features


empty_code = ""

general_code = """
''' create a user '''

import datetime
import os

class User:
    def __init__(self, name):
        self.name = name

    def get_birthday(self, birthday):
        self.birthday = datetime.datetime.strptime(birthday, "%m/%d/%Y").date()

    def days_till_birthday(self):
        today = datetime.datetime.today()

        next_birthday = datetime.datetime(today.year, self.birthday.month, self.birthday.day)
        if next_birthday < today:
            next_birthday = datetime.datetime(today.year + 1, self.birthday.month, self.birthday.day)

        return (next_birthday - today).days + 1


# get user's data   
name = input("Enter your name: ")
BirthDay = input("Enter your birthday (mm/dd/yyyy): ")

# create a user
user = User(name)
user.get_birthday(BirthDay)

# calculate days left until birthday
days_left = user.days_till_birthday()
print(f"Hello, {user.name}! You were born on {user.birthday}. You have {days_left} days left until your birthday.")
"""

df = pd.read_csv("data/test_sample.csv")

def test_analyze_code():
    empty_features = analyze_code(empty_code)

    assert empty_features["lines"] == 0
    assert empty_features["empty_lines"] == 0
    assert empty_features["comment_lines"] == 0
    assert empty_features["has_docstring"] == 0
    assert len(empty_features["functions"]) == 0
    assert len(empty_features["classes"]) == 0
    assert len(empty_features["variables"]) == 0
    assert empty_features["long_lines"] == 0
    assert len(empty_features["imports"]) == 0
    assert len(empty_features["used_names"]) == 0
    assert empty_features["total_names"] == 0
    assert empty_features["bad_names"] == 0
    assert empty_features["too_many_args"] == 0

    general_features = analyze_code(general_code)

    assert general_features["lines"] == 34
    assert general_features["empty_lines"] == 11
    assert general_features["comment_lines"] == 3
    assert general_features["has_docstring"] == 1
    assert len(general_features["functions"]) == 3
    assert len(general_features["classes"]) == 1
    assert len(general_features["variables"]) == 6
    assert general_features["long_lines"] == 2
    assert len(general_features["imports"]) == 2
    assert len(general_features["used_names"]) == 12
    assert general_features["total_names"] == 16
    assert general_features["bad_names"] == 1
    assert general_features["too_many_args"] == 0


def test_get_cyclomatic_complexity():
    assert get_cyclomatic_complexity(empty_code) == 1
    assert get_cyclomatic_complexity(general_code) == 2


def test_generate_features():
    features = df.progress_apply(generate_features, axis=1, result_type="expand")

    assert features["characters"].isna().sum() == 0
    assert features["characters"].min() > 0
    assert features["code_compactness"].isna().sum() == 0
    assert features["code_compactness"].between(0, 1).all()
    assert features["line_length_std"].isna().sum() == 0
    assert features["line_length_std"].min() >= 0
    assert features["cyclomatic_complexity"].isna().sum() == 0
    assert features["cyclomatic_complexity"].min() >= 1
    assert features["long_line_ratio"].isna().sum() == 0
    assert features["long_line_ratio"].between(0, 1).all()
    assert features["bad_name_ratio"].isna().sum() == 0
    assert features["bad_name_ratio"].between(0, 1).all()
    assert features["comment_ratio"].isna().sum() == 0
    assert features["comment_ratio"].between(0, 1).all()
    assert features["has_docstring"].isna().sum() == 0
    assert features["has_docstring"].isin([0, 1]).all()
    assert features["variable_density"].isna().sum() == 0
    assert features["variable_density"].min() >= 0
    assert features["func_density"].isna().sum() == 0
    assert features["func_density"].min() >= 0
    assert features["too_many_args_ratio"].isna().sum() == 0
    assert features["too_many_args_ratio"].between(0, 1).all()
    assert features["class_density"].isna().sum() == 0
    assert features["class_density"].min() >= 0
    assert features["avg_class_methods"].isna().sum() == 0
    assert features["avg_class_methods"].min() >= 0
    assert features["func_class_docstring_ratio"].isna().sum() == 0
    assert features["func_class_docstring_ratio"].between(0, 1).all()
    assert features["unused_imports_ratio"].isna().sum() == 0
    assert features["unused_imports_ratio"].between(0, 1).all()
