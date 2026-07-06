"""
feature_selection.py
====================
Leakage-free feature selection reducing the ~5,658 preprocessed
genes to a compact, clinically-plausible biomarker panel for binary
vital-status (Alive vs Dead) classification.

Implements a 2-stage sklearn Pipeline:
    Stage 1 -- MADFilter   : Unsupervised, robust pre-filter using
                             Median Absolute Deviation. Reduces
                             ~5,658 genes -> n_mad_genes (default
                             2,000).
    Stage 2 -- SelectKBest : Supervised ANOVA F-statistic filter.
                             Reduces n_mad_genes -> n_kbest_genes
                             (default 100).

SCORING FUNCTION CHOICE: f_classif over mutual_info_classif
------------------------------------------------------------
For a two-group comparison (Alive vs Dead) on log-transformed,
scaled expression data, the ANOVA F-test is mathematically
equivalent to a two-sample t-test squared -- the textbook-standard
approach for comparing gene expression between two conditions. It
is deterministic (reproducible gene rankings for a portfolio
README) and orders of magnitude faster than mutual_info_classif's
k-NN density estimation at this scale. mutual_info_classif remains
available via config.scoring_func for non-linear or non-Gaussian
inputs.

WHY THE MAD PRE-FILTER STILL EARNS ITS PLACE
------------------------------------------------
At 5,658 genes, SelectKBest alone would run in well under a second
-- the MAD stage is no longer here for compute-budget reasons the
way it was at 60,000 genes. It is retained because MAD (median-
based) is robust to the single-outlier-sample distortions that
inflate ordinary variance, so it still improves signal quality
feeding into Stage 2, even though it is no longer load-bearing for
speed.

BUG FIX -- PERMANENT PICKLE-TRAP PREVENTION
------------------------------------------------
joblib/pickle stores a class reference as (module_name, qualname)
and reconstructs it by importing module_name and doing
getattr(module, qualname). If this file is ever run directly
(python src/feature_selection.py), Python sets this module's
__name__ to '__main__' for that execution, so MADFilter gets
pickled under module '__main__' -- and any other script that later
tries to unpickle that artifact as its own __main__ will raise
AttributeError, since it isn't the module MADFilter actually lives
in.

MADFilter.__module__ is explicitly forced to 'src.feature_selection'
immediately after the class body, overriding whatever __name__
happened to be at definition time. This makes every future pickle
of MADFilter correct regardless of how this file is executed --
imported normally or run directly -- closing the issue at the root
rather than relying on a reactive try/except recovery at load time.

Integration Contract:
    Consumes : Output of GenomicsPreprocessor.transform() -- a
               float32 DataFrame of shape (n_samples, ~5,658 genes).
    Produces : float32 DataFrame of shape (n_samples,
               n_kbest_genes=100) with gene-symbol columns
               preserved end-to-end.
    Passes to: src/models/ensemble_model.py and
               src/explainability/shap_explainer.py

Usage:
    >>> from src.feature_selection import (
    ...     GenomicsFeatureSelector, FeatureSelectionConfig
    ... )
    >>> selector = GenomicsFeatureSelector()
    >>> X_train_sel = selector.fit_transform(
    ...     X_train_processed, y_train_encoded
    ... )
    >>> X_test_sel = selector.transform(X_test_processed)
    >>> genes = selector.get_selected_genes()

Author : [Your Name]
Date   : [Project Date]
"""

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.data_ingestion import MemoryReporter

logger = logging.getLogger(__name__)

_SCORING_REGISTRY: Dict[str, Callable] = {
    "f_classif": f_classif,
    "mutual_info_classif": mutual_info_classif,
}


# ── Configuration Dataclass ──


@dataclass
class FeatureSelectionConfig:
    """
    Typed, injectable configuration for the 2-stage selector.

    Attributes:
        n_mad_genes    : Genes retained after the MAD pre-filter.
                         Must exceed n_kbest_genes.
        n_kbest_genes  : Final biomarker panel size. 100 sits
                         comfortably inside a clinically-plausible
                         range (Oncotype DX = 21 genes, MammaPrint
                         = 70, PAM50 = 50), while leaving room for
                         a richer panel than those.
        scoring_func   : Key into _SCORING_REGISTRY. 'f_classif' is
                         the default -- see module docstring.
        random_state   : Seed for mutual_info_classif's tie-
                         breaking. Ignored for 'f_classif'.
        low_memory_mode: If True, adds extra MemoryReporter calls.
                         Largely cosmetic at this data scale (the
                         full matrix is now under 30 MB), retained
                         for architectural continuity.
    """

    n_mad_genes: int = 2_000
    n_kbest_genes: int = 100
    scoring_func: Literal["f_classif", "mutual_info_classif"] = "f_classif"
    random_state: int = 42
    low_memory_mode: bool = True

    def __post_init__(self) -> None:
        if self.n_mad_genes <= self.n_kbest_genes:
            raise ValueError(
                f"n_mad_genes ({self.n_mad_genes}) must exceed "
                f"n_kbest_genes ({self.n_kbest_genes})."
            )
        if self.scoring_func not in _SCORING_REGISTRY:
            raise ValueError(
                f"scoring_func='{self.scoring_func}' not "
                f"registered. Valid: {list(_SCORING_REGISTRY)}"
            )


# ── Custom Transformer: MADFilter ──


class MADFilter(BaseEstimator, TransformerMixin):
    """
    Unsupervised, robust gene pre-filter based on Median Absolute
    Deviation.

    MAD uses the median rather than the mean, giving it a 50%
    breakdown point -- a single outlier sample cannot distort a
    gene's MAD score the way it can distort ordinary variance,
    which matters for cohorts with batch effects or atypical
    samples.

    Leakage Contract:
        fit() learns the mask from training data only; y is always
        ignored (fully unsupervised). transform() applies the
        stored mask verbatim.

    Attributes:
        support_mask_    : Boolean array, True for retained genes.
        mad_scores_      : float32 array of per-gene MAD values.
        feature_names_in_: Gene identifiers seen at fit time.
        n_features_in_   : Number of genes seen at fit time.
        n_genes_removed_ : Genes dropped by the filter.
    """

    def __init__(self, n_genes: int = 2_000) -> None:
        self.n_genes = n_genes

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MADFilter":
        """
        Compute per-gene MAD on training data; store the top-
        n_genes mask.

        Args:
            X: Preprocessed expression matrix, training fold only.
            y: Ignored.

        Returns:
            self

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If n_genes >= X.shape[1].
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MADFilter.fit() requires a pandas DataFrame; " f"got {type(X)}."
            )
        if self.n_genes >= X.shape[1]:
            raise ValueError(
                f"MADFilter.n_genes ({self.n_genes}) must be less "
                f"than input gene count ({X.shape[1]})."
            )

        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        self.mad_scores_ = self._compute_mad_vectorized(X)

        top_indices = np.argpartition(self.mad_scores_, -self.n_genes)[-self.n_genes :]
        self.support_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_mask_[top_indices] = True
        self.n_genes_removed_ = self.n_features_in_ - self.n_genes

        logger.info(
            "MADFilter fit | Retained %d / %d genes",
            self.n_genes,
            self.n_features_in_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mask learned during fit(). No recomputation.

        Args:
            X: Must have identical gene columns to fit().

        Returns:
            DataFrame of shape (n_samples, n_genes), gene names
            intact.

        Raises:
            NotFittedError: If called before fit().
            ValueError: If X's columns don't match fit()'s genes.
        """
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MADFilter.transform() requires a pandas " f"DataFrame; got {type(X)}."
            )
        if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
            raise ValueError(
                "Column mismatch: X passed to transform() has "
                "different genes than were seen during fit()."
            )
        return X.loc[:, self.support_mask_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return gene names surviving the MAD filter."""
        check_is_fitted(self)
        return self.feature_names_in_[self.support_mask_]

    @staticmethod
    def _compute_mad_vectorized(X: pd.DataFrame) -> np.ndarray:
        """
        Compute per-gene MAD via two vectorized NumPy passes -- no
        Python-level loop over genes.
        """
        X_np = X.to_numpy(dtype=np.float32)
        median_per_gene = np.median(X_np, axis=0)
        mad_per_gene = np.median(np.abs(X_np - median_per_gene), axis=0)
        return mad_per_gene.astype(np.float32)


# Force the correct pickling identity regardless of how this file
# is executed (imported normally vs. run directly as __main__).
# See "BUG FIX -- PERMANENT PICKLE-TRAP PREVENTION" in the module
# docstring above.
MADFilter.__module__ = "src.feature_selection"


# ── Orchestrator: GenomicsFeatureSelector ──


class GenomicsFeatureSelector:
    """
    Orchestrates the 2-stage feature selection pipeline: MADFilter
    then SelectKBest.

    Attributes:
        config    : FeatureSelectionConfig instance.
        pipeline_ : sklearn Pipeline, set after fit_transform().
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None) -> None:
        self.config = config or FeatureSelectionConfig()
        self.reporter = MemoryReporter()
        self.pipeline_: Optional[Pipeline] = None
        logger.info(
            "GenomicsFeatureSelector initialized | MAD: %d -> "
            "KBest: %d | scoring: %s",
            self.config.n_mad_genes,
            self.config.n_kbest_genes,
            self.config.scoring_func,
        )

    def build_pipeline(self) -> Pipeline:
        """Construct (but do not fit) the 2-stage sklearn Pipeline."""
        score_func = self._resolve_scoring_function()
        kbest = SelectKBest(score_func=score_func, k=self.config.n_kbest_genes)
        kbest.set_output(transform="pandas")

        return Pipeline(
            steps=[
                ("mad_filter", MADFilter(n_genes=self.config.n_mad_genes)),
                ("kbest_selector", kbest),
            ]
        )

    def _resolve_scoring_function(self) -> Callable:
        if self.config.scoring_func == "mutual_info_classif":
            return partial(
                mutual_info_classif,
                random_state=self.config.random_state,
            )
        return _SCORING_REGISTRY[self.config.scoring_func]

    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        """
        Fit on training data only, then transform it.

        Args:
            X_train: Preprocessed matrix, training fold only.
            y_train: Integer-encoded vital-status labels
                (0=Alive, 1=Dead under LabelEncoder's default
                alphabetical ordering -- verify via
                label_encoder.classes_).

        Returns:
            float32 DataFrame, shape (n_train, n_kbest_genes).

        Raises:
            TypeError: If X_train is not a pandas DataFrame.
            ValueError: If X_train/y_train sample counts differ.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                f"X_train must be a pandas DataFrame; got " f"{type(X_train)}."
            )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train sample counts differ: "
                f"{len(X_train)} vs {len(y_train)}."
            )

        logger.info(
            "GenomicsFeatureSelector.fit_transform() START | " "Input: %s",
            X_train.shape,
        )
        self.reporter.report(X_train, "Feature Selection Input")

        self.pipeline_ = self.build_pipeline()
        X_selected = self.pipeline_.fit_transform(X_train, y_train)
        X_selected = self._ensure_float32(X_selected)

        self.reporter.report(X_selected, "Feature Selection Output")
        self._log_top_genes(n=10)
        logger.info(
            "GenomicsFeatureSelector.fit_transform() COMPLETE | " "%d -> %d genes",
            X_train.shape[1],
            X_selected.shape[1],
        )
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted pipeline to held-out or inference data.

        Args:
            X: Must have the same ~5,658 gene columns as X_train.

        Returns:
            float32 DataFrame, shape (n_samples, n_kbest_genes).

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "GenomicsFeatureSelector.transform() called " "before fit_transform()."
            )
        X_selected = self.pipeline_.transform(X)
        X_selected = self._ensure_float32(X_selected)
        self.reporter.report(X_selected, "Feature Selection Output (holdout)")
        return X_selected

    def get_selected_genes(self) -> List[str]:
        """
        Return the final gene identifiers surviving both stages.

        Pass this directly as feature_names into SHAP calls so gene
        labels stay in sync with what the model actually saw.

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before get_selected_genes().")
        return list(self.pipeline_.get_feature_names_out())

    def get_mad_scores(self) -> pd.Series:
        """Return per-gene MAD scores from Stage 1, sorted descending."""
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before get_mad_scores().")
        mad_step: MADFilter = self.pipeline_.named_steps["mad_filter"]
        return pd.Series(
            mad_step.mad_scores_,
            index=mad_step.feature_names_in_,
            name="mad_score",
        ).sort_values(ascending=False)

    def get_kbest_scores(self) -> pd.Series:
        """Return per-gene F-scores from Stage 2, sorted descending."""
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before get_kbest_scores().")
        mad_step: MADFilter = self.pipeline_.named_steps["mad_filter"]
        kbest_step: SelectKBest = self.pipeline_.named_steps["kbest_selector"]
        post_mad_genes = mad_step.feature_names_in_[mad_step.support_mask_]
        return pd.Series(
            kbest_step.scores_.astype(np.float32),
            index=post_mad_genes,
            name="f_score",
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """
        Serialize the fitted pipeline to disk via joblib.

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline_, path)
        logger.info("Feature selection pipeline saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "GenomicsFeatureSelector":
        """Reconstruct a fitted GenomicsFeatureSelector from disk."""
        instance = cls()
        instance.pipeline_ = joblib.load(path)
        logger.info("Feature selection pipeline loaded from %s", path)
        return instance

    @staticmethod
    def _ensure_float32(df: pd.DataFrame) -> pd.DataFrame:
        float64_cols = df.select_dtypes(include=["float64"]).columns
        if len(float64_cols) == 0:
            return df
        for col in float64_cols:
            df[col] = df[col].astype(np.float32)
        return df

    def _log_top_genes(self, n: int = 10) -> None:
        try:
            top = self.get_kbest_scores().head(n)
            logger.info("Top %d genes by F-score:\n%s", n, top.to_string())
        except Exception:
            pass


# ── Standalone Entry Point ──

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.data_ingestion import TCGADataIngester, setup_logging
    from src.preprocessing import GenomicsPreprocessor

    setup_logging("INFO")

    X, y, _ = TCGADataIngester(data_dir="data/raw/").run()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Same self-healing pattern established in
    # src/models/ensemble_model.py: Log2FPKMTransformer and
    # LowExpressionFilter are custom classes defined in
    # preprocessing.py, so a prior direct run of that file could
    # have pickled them under module '__main__'.
    preproc_path = "models/preprocessing_pipeline.joblib"
    try:
        preprocessor = GenomicsPreprocessor.load(preproc_path)
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
    except AttributeError as exc:
        logger.warning(
            "Failed to unpickle '%s' (%s). Refitting a fresh "
            "preprocessor and overwriting the artifact.",
            preproc_path,
            exc,
        )
        preprocessor = GenomicsPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        preprocessor.save(preproc_path)

    label_encoder = joblib.load("models/label_encoder.joblib")
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    selector = GenomicsFeatureSelector()
    X_train_sel = selector.fit_transform(X_train_proc, y_train_enc)
    X_test_sel = selector.transform(X_test_proc)
    selector.save("models/feature_selection_pipeline.joblib")

    selected_genes = selector.get_selected_genes()
    sep = "=" * 60
    print(f"\n{sep}")
    print("  FEATURE SELECTION COMPLETE -- SUMMARY")
    print(sep)
    print(f"  Preprocessed genes : {X_train_proc.shape[1]:>7,}")
    print(f"  Post-MAD filter    : {selector.config.n_mad_genes:>7,}")
    print(f"  Post-SelectKBest   : {len(selected_genes):>7,}")
    print(f"  X_train_sel shape  : {X_train_sel.shape}")
    print(f"  Classes            : {list(label_encoder.classes_)}")
    print("\n  Top 5 genes by F-score:")
    for gene, score in selector.get_kbest_scores().head(5).items():
        print(f"    {gene:<20} F = {score:.2f}")
    print(sep)
