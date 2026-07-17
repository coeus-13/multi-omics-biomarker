"""
test_preprocessing.py
=====================
Unit tests for the custom transformers in src/preprocessing.py:
Log2FPKMTransformer and LowExpressionFilter.

All fixtures are small, hand-constructed synthetic DataFrames --
no TCGA data, no disk I/O, no network calls. The full suite runs
in well under a second, which is the point: these run on every
commit in CI, not just before a release.

Run:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.preprocessing import Log2FPKMTransformer, LowExpressionFilter

# ── Log2FPKMTransformer ──


class TestLog2FPKMTransformer:
    """Unit tests for Log2FPKMTransformer."""

    @pytest.fixture
    def known_values_df(self) -> pd.DataFrame:
        """
        Values chosen so log2(x + 1) lands on an exact integer --
        avoids floating-point tolerance juggling in assertions.

        3.0 -> log2(4.0) = 2.0    7.0 -> log2(8.0) = 3.0
        0.0 -> log2(1.0) = 0.0
        """
        return pd.DataFrame(
            {"GENE_A": [3.0, 7.0], "GENE_B": [0.0, 3.0]},
            index=["S1", "S2"],
        )

    def test_transform_computes_exact_log2(self, known_values_df: pd.DataFrame) -> None:
        transformer = Log2FPKMTransformer(pseudocount=1.0)
        transformer.fit(known_values_df)
        result = transformer.transform(known_values_df)

        expected = pd.DataFrame(
            {"GENE_A": [2.0, 3.0], "GENE_B": [0.0, 2.0]},
            index=["S1", "S2"],
        ).astype(np.float32)
        pd.testing.assert_frame_equal(result, expected)

    def test_output_dtype_is_float32(self, small_expression_df: pd.DataFrame) -> None:
        transformer = Log2FPKMTransformer().fit(small_expression_df)
        result = transformer.transform(small_expression_df)
        assert (result.dtypes == np.float32).all()

    def test_index_and_columns_preserved(
        self, small_expression_df: pd.DataFrame
    ) -> None:
        transformer = Log2FPKMTransformer().fit(small_expression_df)
        result = transformer.transform(small_expression_df)
        assert list(result.index) == list(small_expression_df.index)
        assert list(result.columns) == list(small_expression_df.columns)

    def test_get_feature_names_out_matches_input_columns(
        self, small_expression_df: pd.DataFrame
    ) -> None:
        transformer = Log2FPKMTransformer().fit(small_expression_df)
        np.testing.assert_array_equal(
            transformer.get_feature_names_out(),
            small_expression_df.columns,
        )

    def test_fit_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            Log2FPKMTransformer().fit(np.array([[1.0, 2.0]]))

    def test_fit_rejects_nan_values(self) -> None:
        df = pd.DataFrame({"GENE_A": [1.0, np.nan]})
        with pytest.raises(ValueError, match="NaN"):
            Log2FPKMTransformer().fit(df)

    def test_fit_rejects_negative_values(self) -> None:
        df = pd.DataFrame({"GENE_A": [1.0, -0.5]})
        with pytest.raises(ValueError, match="negative"):
            Log2FPKMTransformer().fit(df)

    def test_transform_rejects_mismatched_columns(
        self, small_expression_df: pd.DataFrame
    ) -> None:
        transformer = Log2FPKMTransformer().fit(small_expression_df)
        wrong = small_expression_df.rename(columns={"GENE_A": "X"})
        with pytest.raises(ValueError, match="Column mismatch"):
            transformer.transform(wrong)

    def test_transform_before_fit_raises(
        self, small_expression_df: pd.DataFrame
    ) -> None:
        with pytest.raises(NotFittedError):
            Log2FPKMTransformer().transform(small_expression_df)


# ── LowExpressionFilter ──


class TestLowExpressionFilter:
    """Unit tests for LowExpressionFilter."""

    @pytest.fixture
    def mixed_expression_df(self) -> pd.DataFrame:
        """
        5 samples x 3 genes, engineered so each gene lands in a
        precisely known bucket relative to the default thresholds
        (min_expression=1.0, min_sample_fraction=0.2):

        ALWAYS_ON : 2.0 in all 5 samples -> fraction=1.0, survives.
        NEVER_ON  : 0.0 in all 5 samples -> fraction=0.0, dropped.
        BOUNDARY  : 2.0 in exactly 1 of 5 -> fraction=0.2, exactly
                    equal to the threshold -- tests that >= is
                    inclusive, not exclusive.
        """
        return pd.DataFrame(
            {
                "ALWAYS_ON": [2.0, 2.0, 2.0, 2.0, 2.0],
                "NEVER_ON": [0.0, 0.0, 0.0, 0.0, 0.0],
                "BOUNDARY": [2.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=[f"S{i}" for i in range(5)],
        )

    def test_correct_genes_survive_the_threshold(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        filt = LowExpressionFilter(min_expression=1.0, min_sample_fraction=0.2)
        filt.fit(mixed_expression_df)

        assert set(filt.get_feature_names_out()) == {
            "ALWAYS_ON",
            "BOUNDARY",
        }
        assert filt.n_genes_removed_ == 1

    def test_transform_returns_only_surviving_columns(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        filt = LowExpressionFilter().fit(mixed_expression_df)
        result = filt.transform(mixed_expression_df)
        assert list(result.columns) == ["ALWAYS_ON", "BOUNDARY"]
        assert result.shape[0] == mixed_expression_df.shape[0]

    def test_fit_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            LowExpressionFilter().fit(np.array([[1.0, 2.0]]))

    def test_fit_rejects_out_of_range_sample_fraction(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        filt = LowExpressionFilter(min_sample_fraction=1.5)
        with pytest.raises(ValueError, match="min_sample_fraction"):
            filt.fit(mixed_expression_df)

    def test_fit_raises_when_all_genes_removed(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        filt = LowExpressionFilter(min_expression=1000.0)
        with pytest.raises(ValueError, match="removed all"):
            filt.fit(mixed_expression_df)

    def test_transform_rejects_mismatched_columns(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        filt = LowExpressionFilter().fit(mixed_expression_df)
        wrong = mixed_expression_df.rename(columns={"ALWAYS_ON": "X"})
        with pytest.raises(ValueError, match="Column mismatch"):
            filt.transform(wrong)

    def test_transform_coerces_array_like_input(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        """
        Regression test for a real bug from earlier in this
        project: an earlier revision of transform() had two
        unconditional, unguarded `raise` statements firing on
        every call regardless of input type. The fix added a
        best-effort array-like -> DataFrame coercion specifically
        to transform() (fit() stays strict on purpose -- see the
        class docstring's "Type Handling" section). This exercises
        exactly the coercion path that fix added.
        """
        filt = LowExpressionFilter().fit(mixed_expression_df)
        result = filt.transform(mixed_expression_df.to_numpy())

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["ALWAYS_ON", "BOUNDARY"]

    def test_transform_coercion_with_wrong_shape_still_raises(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        """
        The coercion fallback is best-effort, not a silent pass-
        through: if the array's column count doesn't match fit-
        time, gene names can't be recovered, and the column-
        alignment check must still raise rather than mislabel
        genes silently.
        """
        filt = LowExpressionFilter().fit(mixed_expression_df)
        wrong_shape = mixed_expression_df.to_numpy()[:, :2]

        with pytest.raises(ValueError, match="Column mismatch"):
            filt.transform(wrong_shape)

    def test_transform_before_fit_raises(
        self, mixed_expression_df: pd.DataFrame
    ) -> None:
        with pytest.raises(NotFittedError):
            LowExpressionFilter().transform(mixed_expression_df)
