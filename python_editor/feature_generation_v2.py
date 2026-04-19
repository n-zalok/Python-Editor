import ast
import re
import tokenize
from io import StringIO
import numpy as np
import pandas as pd
from python_editor.data_processing import split_by_developer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
tqdm.pandas()


FEATURES = [
        "characters",
        "code_compactness",
        "line_length_std",
        "long_line_ratio",
        "trailing_ws_ratio",
        "bad_name_ratio",
        "bare_except_ratio",
        "comment_ratio",
        "has_docstring",
        "variable_density",
        "func_density",
        "avg_func_lines",
        "avg_func_args",
        "too_many_args_ratio",
        "class_density",
        "avg_class_lines",
        "avg_class_methods",
        "func_class_docstring_ratio",
        "unused_imports_ratio"
    ]

LOG_FEATURES = [
    "avg_func_lines",
    "line_length_std",
    "avg_class_lines",
    "characters"
]

BINARY_FEATURES = [
    "too_many_args",
    "bare_except",
    "trailing_ws",
    "long_line",
    "bad_name",
    "unused_imports"
]

TRANSFORMED_FEATURES = [
        "characters",
        "code_compactness",
        "line_length_std",
        "long_line",
        "trailing_ws",
        "bad_name",
        "bare_except",
        "comment_ratio",
        "has_docstring",
        "variable_density",
        "func_density",
        "avg_func_lines",
        "avg_func_args",
        "too_many_args",
        "class_density",
        "avg_class_lines",
        "avg_class_methods",
        "func_class_docstring_ratio",
        "unused_imports"
    ]

SNAKE_CASE = re.compile(r'^[a-z_][a-z0-9_]*$')

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, source):
        self.source = source
        self.lines = source.splitlines()

        self.empty_lines = 0
        for line in source.splitlines():
            if not line.strip():
                self.empty_lines += 1

        self.long_lines = sum(1 for l in self.lines if len(l) > 100)
        self.trailing_ws = sum(1 for l in self.lines if l.rstrip() != l)

        # Import tracking
        self.imports = set()
        self.used_names = set()

        # Naming tracking
        self.total_names = 0
        self.bad_names = 0

        # Structure counts
        self.functions = []
        self.classes = []
        self.too_many_args = 0
        self.global_vars = set()

        # Try-Except
        self.total_try = 0
        self.bare_except = 0


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
    
    def __check_name(self, name):
        self.total_names += 1
        if not SNAKE_CASE.match(name):
            self.bad_names += 1
    

    # ---------- Variables ----------

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.global_vars.add(t.id)
                self.__check_name(t.id)

        self.generic_visit(node)

    # ---------- Functions ----------

    def visit_FunctionDef(self, node):
        self.__check_name(node.name)

        args = node.args
        arg_names = [a.arg for a in args.args]

        # args naming
        for arg in arg_names:
            self.__check_name(arg)

        # too many args
        if len(arg_names) > 5:
            self.too_many_args += 1

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

        doc = ast.get_docstring(node)
        has_docstring = 1 if doc else 0

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)

        class_info = {
            "name": node.name,
            "lines": self._count_lines(node),
            "chars": self._count_chars(node),
            "methods": methods,
            "has_docstring": has_docstring
        }

        self.classes.append(class_info)

        self.generic_visit(node)

    
    # ---------- IMPORTS ----------

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.imports.add(name)

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.imports.add(name)

        self.generic_visit(node)

    def visit_Name(self, node):
        self.used_names.add(node.id)

        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)

        self.generic_visit(node)


    # ---------- ERROR PATTERNS ----------

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.bare_except += 1

        self.generic_visit(node)

    def visit_Try(self, node):
        self.total_try += 1
        
        self.generic_visit(node)


def extract_comments(code: str):
    comment_lines = 0
    tokens = tokenize.generate_tokens(StringIO(code).readline)

    # tok_type(int), tok_string(str), start(tuple(line, char)), end(tuple(line, char))
    for tok_type, tok_string, start, end, _ in tokens:
        if tok_type == tokenize.COMMENT:
            comment_lines += 1

    return comment_lines


def analyze_code(code: str) -> dict:
    comment_lines = extract_comments(code)

    stats = CodeAnalyzer(code)
    tree = ast.parse(code)
    stats.visit(tree)

    module_doc = ast.get_docstring(tree)
    has_docstring = 1 if module_doc else 0

    return {
        "lines": len(stats.lines),
        "empty_lines": stats.empty_lines,
        "characters": len(code),
        "comment_lines": comment_lines,
        "has_docstring": has_docstring,
        "functions": stats.functions,
        "classes": stats.classes,
        "variables": stats.global_vars,
        "long_lines": stats.long_lines,
        "trailing_ws": stats.trailing_ws,
        "imports": stats.imports,
        "used_names": stats.used_names,
        "total_names": stats.total_names,
        "bad_names": stats.bad_names,
        "too_many_args": stats.too_many_args,
        "total_try": stats.total_try,
        "bare_except": stats.bare_except
    }


def get_line_length_std(code: str):
    lines = code.splitlines()
    
    lengths = [len(line) for line in lines if line.strip()]

    if not lengths:
        return 0

    return np.std(lengths).item()



def generate_features(row: pd.Series) -> dict:
    stats = analyze_code(row["text"])
    effective_lines = stats["lines"] - stats["empty_lines"]
    
    # Whole file features
    characters = stats["characters"]
    code_compactness = effective_lines / stats["lines"]
    line_length_std = get_line_length_std(row["text"])
    long_line_ratio = stats["long_lines"] / effective_lines
    trailing_ws_ratio = stats["trailing_ws"] / effective_lines
    bad_name_ratio = stats["bad_names"] / stats["total_names"] if stats["total_names"] else 0
    bare_except_ratio = stats["bare_except"] / stats["total_try"] if stats["total_try"] else 0

    # Documentation features
    comment_ratio = stats["comment_lines"] / effective_lines
    has_docstring = stats["has_docstring"]

    # Variables features
    variable_density = len(stats["variables"]) / effective_lines

    # Functions features
    num_funcs = len(stats["functions"])
    func_density = num_funcs / effective_lines
    avg_func_lines = sum(func["lines"] for func in stats["functions"]) / num_funcs if num_funcs else 0
    avg_func_args = sum(len(func["arg_names"]) for func in stats["functions"]) / num_funcs if num_funcs else 0
    too_many_args_ratio = stats["too_many_args"] / num_funcs if num_funcs else 0

    # Classes features
    num_classes = len(stats["classes"])
    class_density = num_classes / effective_lines
    avg_class_lines = sum(clss["lines"] for clss in stats["classes"]) / num_classes if num_classes else 0
    avg_class_methods = sum(len(clss["methods"]) for clss in stats["classes"]) / num_classes if num_classes else 0
    
    # Combined functions and classes features
    num_funcs_and_classes = num_funcs + num_classes

    func_class_docstring_ratio = ( (sum(func["has_docstring"] for func in stats["functions"]) +
                                    sum(clss["has_docstring"] for clss in stats["classes"])) /
                                    num_funcs_and_classes ) if num_funcs_and_classes else 0
        
    # Imports features
    num_imports = len(stats["imports"])

    unused_imports = 0
    for name in stats["imports"]:
        if name not in stats["used_names"]:
            unused_imports += 1
    
    unused_imports_ratio = unused_imports / num_imports if num_imports else 0
    

    return {
        "characters": characters,
        "code_compactness": code_compactness,
        "line_length_std": line_length_std,
        "long_line_ratio": long_line_ratio,
        "trailing_ws_ratio": trailing_ws_ratio,
        "bad_name_ratio": bad_name_ratio,
        "bare_except_ratio": bare_except_ratio,
        "comment_ratio": comment_ratio,
        "has_docstring": has_docstring,
        "variable_density": variable_density,
        "func_density": func_density,
        "avg_func_lines": avg_func_lines,
        "avg_func_args": avg_func_args,
        "too_many_args_ratio": too_many_args_ratio,
        "class_density": class_density,
        "avg_class_lines": avg_class_lines,
        "avg_class_methods": avg_class_methods,
        "func_class_docstring_ratio": func_class_docstring_ratio,
        "unused_imports_ratio": unused_imports_ratio
    }