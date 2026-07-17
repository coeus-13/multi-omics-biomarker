"""
test_feature_selection.py
==========================
Unit tests for MADFilter in src/feature_selection.py.

Named and located to mirror src/feature_selection.py directly,
rather than folding into test_preprocessing.py -- MADFilter has
never lived in preprocessing.py, and a test file that doesn't
mirror its module gets confusing fast as the suite grows.

Run:
    pytest tests/test_feature_selection.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.feature_selection import MADFilter


class TestMADFilter:
    """Unit tests for MADFilter."""

    @pytest.fixture
    def variable_vs_flat_df(self) -> pd.DataFrame:
        """
        3 genes with deliberately distinct spread:

        VARIABLE : [1,2,3,4,5]      -> median=3, MAD=1.0
        FLAT     : [5,5,5,5,5]      -> MAD=0.0 (zero spread)
        OUTLIER  : [5,5,5,5,100]    -> median=5, MAD=0.0

        OUTLIER is the key case: its variance (1444) dwarfs
        VARIABLE's (2.0) because of the single spike, but its MAD
        correctly reflects that 4 of 5 samples don't really vary --
        the concrete, hand-verified version of "MAD is robust to
        single-sample outliers," the stated reason MAD was chosen
        over VarianceThreshold.
        """
        return pd.DataFrame(
            {
                "VARIABLE": [1.0, 2.0, 3.0, 4.0, 5.0],
                "FLAT": [5.0, 5.0, 5.0, 5.0, 5.0],
                "OUTLIER": [5.0, 5.0, 5.0, 5.0, 100.0],
            },
            index=[f"S{i}" for i in range(5)],
        )

    def test_mad_scores_match_hand_calculation(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        filt = MADFilter(n_genes=1).fit(variable_vs_flat_df)
        scores = dict(zip(filt.feature_names_in_, filt.mad_scores_))

        assert scores["VARIABLE"] == pytest.approx(1.0)
        assert scores["FLAT"] == pytest.approx(0.0)

    def test_mad_is_not_fooled_by_a_single_outlier(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        filt = MADFilter(n_genes=1).fit(variable_vs_flat_df)
        scores = dict(zip(filt.feature_names_in_, filt.mad_scores_))

        assert scores["VARIABLE"] > scores["OUTLIER"]

    def test_top_gene_selected_by_mad(self, variable_vs_flat_df: pd.DataFrame) -> None:
        filt = MADFilter(n_genes=1).fit(variable_vs_flat_df)
        assert list(filt.get_feature_names_out()) == ["VARIABLE"]

    def test_n_genes_removed_is_correct(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        filt = MADFilter(n_genes=1).fit(variable_vs_flat_df)
        assert filt.n_genes_removed_ == 2

    def test_transform_returns_only_selected_column(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        filt = MADFilter(n_genes=1).fit(variable_vs_flat_df)
        result = filt.transform(variable_vs_flat_df)
        assert list(result.columns) == ["VARIABLE"]

    def test_fit_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            MADFilter(n_genes=1).fit(np.array([[1.0, 2.0]]))

    def test_fit_rejects_n_genes_too_large(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="n_genes"):
            MADFilter(n_genes=10).fit(variable_vs_flat_df)

    def test_transform_rejects_array_input_with_no_coercion(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        """
        Unlike LowExpressionFilter, MADFilter does NOT implement an
        array-like coercion fallback in transform() -- it stays
        strict. A real, deliberate asymmetry between the two
        classes, not an oversight; this pins the behavior down so
        a future "helpful" edit doesn't silently unify them.
        """
        filt = MADFilter(n_genes=1).fit(variable_vs_flat_df)
        with pytest.raises(TypeError):
            filt.transform(variable_vs_flat_df.to_numpy())

    def test_transform_rejects_mismatched_columns(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        filt = MADFilter(n_genes=1).fit(variable_vs_flat_df)
        wrong = variable_vs_flat_df.rename(columns={"FLAT": "X"})
        with pytest.raises(ValueError, match="Column mismatch"):
            filt.transform(wrong)

    def test_transform_before_fit_raises(
        self, variable_vs_flat_df: pd.DataFrame
    ) -> None:
        with pytest.raises(NotFittedError):
            MADFilter(n_genes=1).transform(variable_vs_flat_df)
