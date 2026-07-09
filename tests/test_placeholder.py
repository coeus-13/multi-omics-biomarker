"""
test_placeholder.py
===================
Minimal CI smoke test -- keeps the pytest step in ci.yml passing
(exit code 0) until the real test suite from README's "Known
Limitations & Future Work" section is written.

WHY THIS IS AN IMPORT SMOKE TEST, NOT A BARE `assert True`
--------------------------------------------------------------
A single `def test_x(): assert True` would satisfy pytest's
"at least one test collected" requirement and fix exit code 5,
but would verify nothing about the codebase -- CI would report
green even if every module in src/ had a syntax error or a
broken import. This file does barely more work for the same
goal: it imports every production module and asserts each import
succeeds. That is a real, if shallow, regression check.

This is explicitly NOT a substitute for the real test suite --
no fixtures, no assertions about actual pipeline behavior
(leakage guards, shape contracts, dtype enforcement) are
exercised here. Coverage reports from this file will show
nonzero numbers purely from import-time execution of module-
level code (class/function definitions) -- that is import
coverage, not behavior coverage. Do not read it as meaningful
test coverage.

ONE THING TO WATCH FOR
---------------------------
This is the first step in the CI pipeline that actually imports
xgboost and shap rather than just parsing the files (flake8/black
only read source text; they never execute it). If this fails in
CI but the code runs fine locally, that is most likely a real
environment difference between your machine and the ubuntu-latest
runner -- worth knowing about now rather than later, not a false
alarm to silence.

Replace this file once real coverage exists for
src/data_ingestion.py, src/preprocessing.py,
src/feature_selection.py, src/models/ensemble_model.py,
src/explainability/shap_explainer.py, and src/api/main.py.
"""

import importlib

import pytest

# Every production module expected to be import-safe with no
# models/*.joblib artifacts present and no server running.
# src/api/main.py is included deliberately: create_app() only
# constructs the FastAPI app and registers the lifespan callback
# at import time -- artifact loading happens later, only when an
# ASGI server actually starts serving. A broken API import should
# fail CI just as loudly as a broken pipeline module.
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
