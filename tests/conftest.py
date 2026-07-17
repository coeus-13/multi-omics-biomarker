"""
conftest.py
===========
Shared pytest fixtures for the unit test suite. Auto-discovered by
pytest -- no explicit import needed in individual test files.

Fixtures here are intentionally generic (structural validity
checks: TypeError/ValueError guards, fit-before-transform,
column-mismatch). Fixtures needing SPECIFIC numeric properties to
test one transformer's particular logic (e.g. exact log2 outputs,
a hand-calculable MAD value, a precise expressed/unexpressed
pattern) live locally in that transformer's own test file, since
forcing them into this shared file would make them less readable
at the point of use.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_expression_df() -> pd.DataFrame:
    """
    A generic, valid, small synthetic gene-expression matrix:
    6 samples x 4 genes, all non-negative float32 values, no NaN.

    Suitable as a baseline "valid input" for structural tests
    (type checks, fit-before-transform, column-alignment) shared
    across transformer test files. Not tailored to any one
    transformer's specific correctness logic -- see each test
    file's local fixtures for that.
    """
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(0.0, 10.0, size=(6, 4)).astype(np.float32)
    return pd.DataFrame(
        data,
        index=[f"SAMPLE-{i}" for i in range(6)],
        columns=["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
    )
