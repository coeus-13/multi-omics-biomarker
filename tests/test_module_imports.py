"""
test_module_imports.py
=======================
Import smoke test across every production module. Originally
written as a placeholder to unblock pytest's exit-code-5 failure
before any real tests existed; kept now that test_preprocessing.py
and test_feature_selection.py provide real behavioral coverage,
because it's still the only check touching data_ingestion.py,
ensemble_model.py, shap_explainer.py, and api/main.py at all.

Broad-but-shallow (does it import?) here; narrow-but-deep (does it
behave correctly?) in the transformer-specific test files. Neither
replaces the other.

Run:
    pytest tests/test_module_imports.py -v
"""

import importlib

import pytest

MODULES_UNDER_TEST = [
    "src.data_ingestion",
    "src.preprocessing",
    "src.feature_selection",
    "src.models.ensemble_model",
    "src.explainability.shap_explainer",
    "src.api.main",
]


@pytest.mark.parametrize("module_name", MODULES_UNDER_TEST)
def test_module_imports_without_error(module_name: str) -> None:
    """
    Fail loudly if a production module cannot be imported.

    Catches syntax errors, broken imports, and import-time
    exceptions -- the class of bug that would otherwise surface
    for the first time in someone's IDE rather than in CI.
    """
    importlib.import_module(module_name)
