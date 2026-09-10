"""Offline regressions exercising main.py's actual pure condition functions."""
import ast
from pathlib import Path
import re
import math
import regex
from types import SimpleNamespace
import unittest


def load_conditions():
    path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(path.read_text())
    names = {"extract_value", "value_exists", "check_condition"}
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"re": re, "math": math, "regex": regex}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["check_condition"]


check_condition = load_conditions()


def evaluate(details, key, kind="exists", threshold=None):
    return check_condition(SimpleNamespace(details=details),
                           SimpleNamespace(key=key, condition_type=kind, threshold=threshold))


class ConditionTests(unittest.TestCase):
    def test_missing_required_field_denies(self):
        self.assertFalse(evaluate({}, "required"))

    def test_missing_nested_field_denies(self):
        self.assertFalse(evaluate({"policy": {}}, "policy.approved"))

    def test_explicit_null_denies(self):
        self.assertFalse(evaluate({"approved": None}, "approved"))

    def test_present_falsy_values_exist(self):
        for value in [False, 0, "", [], {}]:
            with self.subTest(value=value):
                self.assertTrue(evaluate({"value": value}, "value"))

    def test_present_nested_array_value(self):
        self.assertTrue(evaluate({"items": [{"id": 0}]}, "items[0].id"))
        self.assertTrue(evaluate({"items": [False]}, "items[-1]"))

    def test_missing_array_element_denies(self):
        for key in ["items[0]", "items[-1]", "absent[0]", "items[999999999999999999999999]"]:
            with self.subTest(key=key):
                self.assertFalse(evaluate({"items": []}, key))

    def test_terminal_wildcard_requires_actual_array(self):
        self.assertTrue(evaluate({"items": []}, "items[*]"))
        self.assertTrue(evaluate({"items": [None]}, "items[*]"))
        for details in [{}, {"items": None}, {"items": {}}]:
            self.assertFalse(evaluate(details, "items[*]"))

    def test_invalid_paths_deny(self):
        for key in ["", ".", "items[", "items[x]", "items[0][1]", "items[*].id", "items..id"]:
            with self.subTest(key=key):
                self.assertFalse(evaluate({"items": [{"id": 1}]}, key))

    def test_traversal_through_scalar_denies(self):
        for value in [None, False, 1, "text", []]:
            self.assertFalse(evaluate({"value": value}, "value.child"))

    def test_other_conditions_preserved(self):
        self.assertTrue(evaluate({"score": 0.8}, "score", "greater", 0.5))
        self.assertFalse(evaluate({"score": 0.2}, "score", "greater", 0.5))
        self.assertTrue(evaluate({"label": "safe"}, "label", "equal", "safe"))
        self.assertTrue(evaluate({"tags": ["a"]}, "tags", "length_equal", 1))


if __name__ == "__main__":
    unittest.main()
