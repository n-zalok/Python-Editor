import ast
import tokenize
from io import BytesIO
import numpy as np
import pandas as pd
from python_editor.data_processing import split_by_developer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
tqdm.pandas()

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines()

        self.empty_lines = 0
        for line in source.splitlines():
            if not line.strip():
                self.empty_lines += 1

        self.functions = []
        self.classes = []
        self.global_vars = set()

    # ---------- Helpers ----------

    def _get_text(self, node):
        if not hasattr(node, "lineno"):
            return ""

        start = node.lineno - 1
        end = node.end_lineno

        return "\n".join(self.lines[start:end])

    def _count_chars(self, node):
        return len(self._get_text(node))

    def _count_lines(self, node):
        return node.end_lineno - node.lineno + 1
    

    # ---------- Variables ----------

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.global_vars.add(t.id)

        self.generic_visit(node)

    # ---------- Functions ----------

    def visit_FunctionDef(self, node):
        args = node.args
        arg_names = [a.arg for a in args.args]

        doc = ast.get_docstring(node)
        has_docstring = 1 if doc else 0

        func_info = {
            "name": node.name,
            "lines": self._count_lines(node),
            "chars": self._count_chars(node),
            "arg_names": arg_names,
            "has_docstring": has_docstring
        }

        self.functions.append(func_info)

        self.generic_visit(node)

    # ---------- Classes ----------

    def visit_ClassDef(self, node):
        methods = []
        init_args = []

        doc = ast.get_docstring(node)
        has_docstring = 1 if doc else 0

        for item in node.body:
            if isinstance(item, ast.FunctionDef):

                methods.append(item.name)

                if item.name == "__init__":
                    init_args = [
                        a.arg for a in item.args.args
                        if a.arg != "self"
                    ]

        class_info = {
            "name": node.name,
            "lines": self._count_lines(node),
            "chars": self._count_chars(node),
            "methods": methods,
            "init_args": init_args,
            "has_docstring": has_docstring
        }

        self.classes.append(class_info)

        self.generic_visit(node)


def extract_comments(code: str) -> int:
    comment_chars = 0

    tokens = tokenize.tokenize(
        BytesIO(code.encode()).readline
    )

    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            comment_chars += len(tok.string)

    return comment_chars


def analyze_code(code: str) -> dict:
    comment_chars = extract_comments(code)

    stats = CodeAnalyzer(code)
    tree = ast.parse(code)
    stats.visit(tree)

    module_doc = ast.get_docstring(tree)
    has_docstring = 1 if module_doc else 0

    return {
        "lines": len(stats.lines),
        "empty_lines": stats.empty_lines,
        "characters": len(code),
        "comment_chars": comment_chars,
        "has_docstring": has_docstring,
        "functions": stats.functions,
        "classes": stats.classes,
        "variables": stats.global_vars
    }


def generate_features(row: pd.Series) -> dict:
    stats = analyze_code(row["text"])

    # Whole file features
    characters = stats["characters"]
    code_compactness = (stats["lines"] - stats["empty_lines"]) / stats["lines"]
    chars_per_line = stats["characters"] / stats["lines"]
    comment_ratio = stats["comment_chars"] / stats["characters"]
    has_docstring = stats["has_docstring"]

    # Variables features
    num_vars = len(stats["variables"])

    variable_ratio = len(stats["variables"]) / stats["characters"]
    avg_var_name = (sum(len(var) for var in stats["variables"]) / 
                    num_vars) if num_vars else 0

    # Functions and classes features
    num_funcs_and_classes = len(stats["functions"]) + len(stats["classes"])

    avg_func_class_name = ( (sum(len(func["name"]) for func in stats["functions"]) +
                             sum(len(clss["name"]) for clss in stats["classes"])) /
                             num_funcs_and_classes ) if num_funcs_and_classes else 0
    
    avg_func_class_chars = ( (sum(func["chars"] for func in stats["functions"]) +
                              sum(clss["chars"] for clss in stats["classes"])) /
                              num_funcs_and_classes ) if num_funcs_and_classes else 0
    
    avg_func_class_args = ( (sum(len(func["arg_names"]) for func in stats["functions"]) +
                             sum(len(clss["init_args"]) for clss in stats["classes"])) /
                             num_funcs_and_classes ) if num_funcs_and_classes else 0
    
    func_class_docstring_ratio = ( (sum(func["has_docstring"] for func in stats["functions"]) +
                                    sum(clss["has_docstring"] for clss in stats["classes"])) /
                                    num_funcs_and_classes ) if num_funcs_and_classes else 0
    
    return {
        "characters": characters,
        "code_compactness": code_compactness,
        "chars_per_line": chars_per_line,
        "comment_ratio": comment_ratio,
        "has_docstring": has_docstring,
        "variable_ratio": variable_ratio,
        "avg_var_name": avg_var_name,
        "num_funcs_and_classes": num_funcs_and_classes,
        "avg_func_class_name": avg_func_class_name,
        "avg_func_class_chars": avg_func_class_chars,
        "avg_func_class_args": avg_func_class_args,
        "func_class_docstring_ratio": func_class_docstring_ratio
    }


def split_and_normalize_features(df: pd.DataFrame,
                                 num_cols: list,
                                 test_size: float,
                                 random_state: int
                                ):
    
    train, test = split_by_developer(df, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()

    train.loc[:, num_cols] = scaler.fit_transform(train[num_cols])
    test.loc[:, num_cols] = scaler.transform(test[num_cols])

    return train, test, scaler


def get_vectorized_features_and_label(df: pd.DataFrame, features: list):
    vectorized_features = np.concatenate(
        [
            np.vstack(df["embedding"]),
            df[features].values
        ],
        axis=1
    )

    label = df["pylint_score"].values
    
    return vectorized_features, label