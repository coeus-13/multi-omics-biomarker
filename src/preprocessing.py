"""
preprocessing.py
================
Production-ready, memory-optimized genomics preprocessing pipeline for
TCGA-BRCA RNA-Seq expression data.

Implements a 4-stage, leakage-free sklearn Pipeline:
    1. Log2(FPKM + 1) variance-stabilizing transform
    2. Low-expression gene filtering (removes sequencing noise)
    3. VarianceThreshold (drops zero/near-zero-variance genes)
    4. StandardScaler (zero mean, unit variance)

Consumes the (X, y) output of src/data_ingestion.py directly.

Memory Budget (continuing from data_ingestion.py, ~880 training samples):
    Input X_train (880 x ~60,000 genes, float32)      : ~200 MB
    Post low-expression filter (880 x ~20,000 genes)   : ~67 MB
    Post VarianceThreshold (880 x ~15,000 genes)       : ~50 MB
    Final scaled output                                : ~50 MB

Optimization Techniques Used:
    1. Every transform() is a single vectorized NumPy ufunc/reduction call -
       zero Python-level loops over genes or samples.
    2. float32 enforced at every stage via explicit casts, mirroring the
       safety-net pattern in TCGADataIngester._enforce_float32().
    3. Feature masks are boolean NumPy arrays applied via pandas .loc - a
       single vectorized indexing operation, not iterative filtering.
    4. set_output(transform="pandas") is applied surgically only to the two
       sklearn built-in steps that need it (VarianceThreshold, StandardScaler).
       The custom steps already construct DataFrames manually.

Leakage Prevention:
    All stateful steps (LowExpressionFilter, VarianceThreshold, StandardScaler)
    learn their statistics EXCLUSIVELY inside .fit()/.fit_transform(), which
    must only ever be called on the training split. .transform() reuses those
    learned statistics verbatim - see GenomicsPreprocessor docstrings.

Usage:
    >>> from src.data_ingestion import TCGADataIngester
    >>> from src.preprocessing import GenomicsPreprocessor, encode_pam50_labels
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y, _ = TCGADataIngester(data_dir="data/raw/").run()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, stratify=y, random_state=42
    ... )
    >>>
    >>> preprocessor = GenomicsPreprocessor()
    >>> X_train_processed = preprocessor.fit_transform(X_train)
    >>> X_test_processed = preprocessor.transform(X_test)
    >>> selected_genes = preprocessor.get_selected_genes()

Author: [Your Name]
Date  : [Project Date]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data_ingestion import MemoryReporter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PreprocessingConfig:
    """
    Configuration for the genomics preprocessing pipeline.

    Mirrors the IngestionConfig pattern from data_ingestion.py: a typed
    dataclass injected into the orchestrator class, making every threshold
    explicit, IDE-autocompletable, and trivially swappable in unit tests.

    Attributes:
        log_pseudocount: Pseudocount added before log2 transform.
        min_expression: Minimum log2(FPKM+1) value to count a gene as
            "expressed" in a sample. 1.0 (raw FPKM >= 1) is a standard
            cutoff used broadly across TCGA/GTEx differential-expression
            literature.
        min_sample_fraction: Minimum fraction of samples a gene must be
            "expressed" in to survive filtering.
        variance_threshold: Minimum feature variance to retain a gene.
            Kept at 0.1 to match the original project roadmap's Step 3.
        with_mean: Whether StandardScaler centers data to zero mean.
        with_std: Whether StandardScaler scales data to unit variance.
        low_memory_mode: If True, adds extra memory instrumentation logging,
            mirroring IngestionConfig.low_memory_mode.
    """

    log_pseudocount: float = 1.0
    min_expression: float = 1.0
    min_sample_fraction: float = 0.2
    variance_threshold: float = 0.1
    with_mean: bool = True
    with_std: bool = True
    low_memory_mode: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Custom Transformer 1: Log2 Variance Stabilization
# ──────────────────────────────────────────────────────────────────────────────


class Log2FPKMTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a log2(x + pseudocount) variance-stabilizing transform to
    RNA-Seq FPKM expression values.

    RNA-Seq FPKM/FPKM-UQ values are right-skewed with variance that scales
    with the mean. A small number of highly-expressed housekeeping genes
    can dominate Euclidean-distance-based models (SVM, KNN) and violate the
    approximate-Gaussian assumption StandardScaler relies on. log2(x + 1)
    is the standard variance-stabilizing transform used across TCGA/GTEx.

    This transformer is stateless with respect to the data (no per-gene
    statistics are learned), but still implements the full
    BaseEstimator/TransformerMixin contract - including get_feature_names_out
    - so it composes correctly inside a sklearn Pipeline.

    Attributes:
        feature_names_in_: Gene identifiers seen during fit.
        n_features_in_: Number of genes seen during fit.
    """

    def __init__(self, pseudocount: float = 1.0) -> None:
        self.pseudocount = pseudocount

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "Log2FPKMTransformer":
        """
        Validate input and record feature names. No statistics are learned -
        the log transform is a fixed, parameter-free function.

        Args:
            X: Raw (non-negative) FPKM expression values, shape (n_samples, n_genes).
            y: Ignored. Present for sklearn Pipeline API compatibility.

        Returns:
            self

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If X contains negative values or NaN values.
        """
        self._validate_input(X, context="fit")
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log2(X + pseudocount) using a single vectorized NumPy call.

        Performance note: this executes as ONE C-level NumPy ufunc pass
        over the full (n_samples x n_genes) matrix. The anti-pattern
        `X.applymap(lambda v: np.log2(v + 1))` iterates at the Python
        level over every individual cell and is 100-1000x slower on a
        matrix this wide; it is never used here.

        Args:
            X: DataFrame of shape (n_samples, n_genes).

        Returns:
            DataFrame of shape (n_samples, n_genes), dtype float32, with
            the original index and column names preserved.

        Raises:
            NotFittedError: If called before fit().
            ValueError: If X's columns don't match the genes seen during fit.
        """
        check_is_fitted(self)
        self._validate_input(X, context="transform")
        self._validate_feature_alignment(X)

        # Single vectorized ufunc call - see performance note above.
        # Explicit float32 cast guards against silent float64 promotion
        # from the Python-float pseudocount on some NumPy versions.
        log_values = np.log2(
            X.to_numpy(dtype=np.float32) + np.float32(self.pseudocount)
        )

        return pd.DataFrame(
            log_values.astype(np.float32, copy=False),
            index=X.index,
            columns=X.columns,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return gene names unchanged - this step drops no columns."""
        check_is_fitted(self)
        return self.feature_names_in_

    @staticmethod
    def _validate_input(X, context: str) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"Log2FPKMTransformer.{context}() requires a pandas DataFrame "
                f"(to preserve gene name columns through the pipeline); got {type(X)}."
            )
        if X.isnull().values.any():
            raise ValueError(
                f"Input to Log2FPKMTransformer.{context}() contains NaN values. "
                f"Resolve upstream in data_ingestion before preprocessing."
            )
        if (X.to_numpy() < 0).any():
            raise ValueError(
                f"Input to Log2FPKMTransformer.{context}() contains negative "
                f"values, which is invalid for FPKM expression data."
            )

    def _validate_feature_alignment(self, X: pd.DataFrame) -> None:
        if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
            raise ValueError(
                "Column mismatch: X passed to transform() has different genes "
                "than were seen during fit(). This usually means train/test "
                "gene panels are misaligned - verify both came from the same "
                "TCGADataIngester run."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Custom Transformer 2: Low-Expression Gene Filter
# ──────────────────────────────────────────────────────────────────────────────


class LowExpressionFilter(BaseEstimator, TransformerMixin):
    """
    Removes genes that are not meaningfully expressed across the cohort.

    A gene is retained only if its log2(FPKM+1) value reaches at least
    min_expression in at least min_sample_fraction of training samples.
    Genes failing this bar are overwhelmingly sequencing noise / biological
    dropout, and removing them BEFORE VarianceThreshold/StandardScaler
    prevents these near-zero columns from being z-scored into spurious
    unit-variance noise.

    Leakage Prevention:
        The expressed-gene mask is learned EXCLUSIVELY from the data passed
        to .fit() (the training fold). The identical mask - not a
        re-computed one - is applied in .transform(), including to held-out
        test data. This is what makes the transformer safe to use inside
        cross_val_score / GridSearchCV without leaking test-set statistics.

    Design note: this class deliberately does NOT subclass sklearn's
    internal SelectorMixin (sklearn.feature_selection._base.SelectorMixin).
    That class lives in a private module (leading underscore) not covered
    by sklearn's semver compatibility guarantees. get_feature_names_out()
    is implemented explicitly by hand instead, so this transformer is
    resilient to future sklearn internal refactors.

    Attributes:
        support_mask_: Boolean array, True for genes retained after fitting.
        feature_names_in_: Gene identifiers seen during fit (pre-filter set).
        n_features_in_: Number of genes seen during fit (pre-filter).
        n_genes_removed_: Count of genes dropped by the filter.
    """

    def __init__(
        self, min_expression: float = 1.0, min_sample_fraction: float = 0.2
    ) -> None:
        self.min_expression = min_expression
        self.min_sample_fraction = min_sample_fraction

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "LowExpressionFilter":
        """
        Learn the gene-retention mask from training data only.

        Vectorization note: the mask is computed via two NumPy reduction
        passes over the full matrix - a boolean comparison
        (n_samples x n_genes) followed by a single column-wise mean(axis=0)
        - both executed in compiled C code. This avoids the anti-pattern of
        looping `for gene in X.columns: X[gene].mean()`, which would issue
        60,000 separate Python-level pandas calls.

        Args:
            X: Log2-transformed expression values, training fold only.
            y: Ignored.

        Returns:
            self

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If min_sample_fraction is outside [0, 1], or if
                zero genes survive the filter (min_expression too strict).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"LowExpressionFilter.fit() requires a pandas DataFrame; got {type(X)}."
            )
        if not 0.0 <= self.min_sample_fraction <= 1.0:
            raise ValueError(
                f"min_sample_fraction must be in [0, 1]; "
                f"got {self.min_sample_fraction}."
            )

        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]

        # ── Vectorized mask computation (no Python-level loop) ───────────
        values = X.to_numpy(dtype=np.float32)
        expressed_mask = values >= np.float32(
            self.min_expression
        )  # (n_samples, n_genes)
        fraction_expressed = expressed_mask.mean(axis=0)  # (n_genes,)
        self.support_mask_ = fraction_expressed >= self.min_sample_fraction

        n_retained = int(self.support_mask_.sum())
        self.n_genes_removed_ = self.n_features_in_ - n_retained

        if n_retained == 0:
            raise ValueError(
                f"LowExpressionFilter removed all {self.n_features_in_} genes. "
                f"min_expression={self.min_expression} is likely too strict - "
                f"verify X is log2-transformed, and consider lowering "
                f"min_expression or min_sample_fraction."
            )

        logger.info(
            "LowExpressionFilter fit | Retained %d / %d genes (%.1f%%) | "
            "min_expression=%.2f, min_sample_fraction=%.2f",
            n_retained,
            self.n_features_in_,
            100 * n_retained / self.n_features_in_,
            self.min_expression,
            self.min_sample_fraction,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mask learned during fit() - no re-computation on X.

        Args:
            X: Must have the exact same genes (columns) seen during fit().

        Returns:
            DataFrame of shape (n_samples, n_retained_genes).

        Raises:
            NotFittedError: If called before fit().
            ValueError: If X's columns don't match those seen during fit().
        """
        check_is_fitted(self)
        raise TypeError(
            "LowExpressionFilter.transform() requires a pandas DataFrame; "
            f"got {type(X)}."
        )
        raise ValueError(
            "Column mismatch: X passed to transform() has different genes "
            "than were seen during fit(). Train/test gene panels must "
            "match exactly."
        )

        # Vectorized boolean column selection - single pandas operation, no loop.
        return X.loc[:, self.support_mask_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the subset of gene names that survived filtering."""
        check_is_fitted(self)
        return self.feature_names_in_[self.support_mask_]


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator: Assembles and Owns the Full sklearn Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class GenomicsPreprocessor:
    """
    Orchestrates the 4-stage genomics preprocessing pipeline:
        1. Log2(FPKM + 1) variance-stabilizing transform
        2. Low-expression gene filtering (noise removal)
        3. VarianceThreshold (drops zero/near-zero-variance genes)
        4. StandardScaler (zero mean, unit variance)

    Internally builds and owns a single sklearn.pipeline.Pipeline so calling
    code gets sklearn-native behavior (fit/transform, compatibility with
    cross_val_score, GridSearchCV, joblib serialization) "for free," while
    this class adds genomics-specific logging, memory instrumentation, and
    a convenience accessor for surviving gene names (needed downstream for
    SHAP biomarker mapping).

    Why label encoding is NOT part of this class:
        sklearn.pipeline.Pipeline.transform() operates on X only - mixing y
        label-encoding into an X-only Pipeline is a common anti-pattern that
        silently breaks the moment the pipeline is used inside
        cross_val_score or GridSearchCV, both of which assume a Pipeline
        transforms X and nothing else. Label encoding is handled by the
        standalone encode_pam50_labels() function in this module instead.

    Attributes:
        config: PreprocessingConfig instance controlling all thresholds.
        pipeline_: Set after the first call to fit_transform(). Use this
            directly for joblib serialization or sklearn CV utilities if
            you don't need the convenience wrappers below.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self.reporter = MemoryReporter()
        self.pipeline_: Optional[Pipeline] = None

        logger.info(
            "GenomicsPreprocessor initialized | min_expression=%.2f | "
            "min_sample_fraction=%.2f | variance_threshold=%.2f",
            self.config.min_expression,
            self.config.min_sample_fraction,
            self.config.variance_threshold,
        )

    def build_pipeline(self) -> Pipeline:
        """
        Construct (but do not fit) the 4-stage sklearn Pipeline.

        Only the two sklearn built-in steps (VarianceThreshold,
        StandardScaler) need explicit set_output(transform="pandas")
        configuration - they return bare ndarrays by default. Our custom
        transformers above always construct and return a pandas DataFrame
        directly inside their own transform() methods, so no output-
        container configuration is needed for them.

        Returns:
            An unfitted sklearn.pipeline.Pipeline. Every stage emits a
            pandas DataFrame with correct gene-name columns end-to-end.
        """
        variance_filter = VarianceThreshold(threshold=self.config.variance_threshold)
        variance_filter.set_output(transform="pandas")

        scaler = StandardScaler(
            with_mean=self.config.with_mean, with_std=self.config.with_std
        )
        scaler.set_output(transform="pandas")

        return Pipeline(
            steps=[
                (
                    "log2_transform",
                    Log2FPKMTransformer(pseudocount=self.config.log_pseudocount),
                ),
                (
                    "low_expression_filter",
                    LowExpressionFilter(
                        min_expression=self.config.min_expression,
                        min_sample_fraction=self.config.min_sample_fraction,
                    ),
                ),
                ("variance_threshold", variance_filter),
                ("scaler", scaler),
            ]
        )

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit the pipeline on training data ONLY, then transform it.

        This is the single point in the entire workflow where any
        statistic (expression mask, variance mask, scaler mean/std) is
        learned. Call this exactly once, on the training split.

        Args:
            X_train: Raw FPKM expression matrix, training fold only.
            y_train: Ignored. Accepted for sklearn API symmetry; every
                step in this pipeline is unsupervised.

        Returns:
            DataFrame of shape (n_train_samples, n_selected_genes), float32,
            zero-mean/unit-variance, ready for SVM / KNN / Tree-based models.
        """
        logger.info(
            "GenomicsPreprocessor.fit_transform() START | Input: %s", X_train.shape
        )
        self.reporter.report(X_train, "Preprocessing Input (raw FPKM, train fold)")

        self.pipeline_ = self.build_pipeline()
        X_processed = self.pipeline_.fit_transform(X_train)
        X_processed = self._enforce_float32(X_processed)

        self.reporter.report(X_processed, "Preprocessing Output (scaled, train fold)")
        logger.info(
            "GenomicsPreprocessor.fit_transform() COMPLETE | %d -> %d genes retained",
            X_train.shape[1],
            X_processed.shape[1],
        )
        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the already-fitted pipeline to new data (validation/test/inference).

        Reuses every statistic learned in fit_transform() - no re-fitting,
        no re-computation of masks or scaler parameters. This is what
        guarantees zero test-set leakage.

        Args:
            X: Must contain the exact same gene columns as X_train.

        Returns:
            DataFrame of shape (n_samples, n_selected_genes), float32.

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "GenomicsPreprocessor.transform() called before fit_transform(). "
                "Fit on the training split first."
            )
        logger.info("GenomicsPreprocessor.transform() | Input: %s", X.shape)
        X_processed = self.pipeline_.transform(X)
        X_processed = self._enforce_float32(X_processed)
        self.reporter.report(X_processed, "Preprocessing Output (scaled, holdout)")
        return X_processed

    def get_selected_genes(self) -> List[str]:
        """
        Return the final gene names that survived the full pipeline (post
        log-transform, low-expression filter, AND variance threshold).

        This is the exact gene list to pass as feature_names into SHAP's
        summary_plot/beeswarm calls later - keeping SHAP gene labels
        perfectly in sync with what the model actually saw.

        Returns:
            List of surviving gene identifiers.

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before get_selected_genes().")
        return list(self.pipeline_.get_feature_names_out())

    def save(self, path: str) -> None:
        """
        Serialize the fitted pipeline to disk via joblib.

        Use this exact artifact at inference time in deployment/api/main.py
        so the FastAPI service applies IDENTICAL preprocessing to incoming
        samples as was used during training - a common source of
        train/serve skew if reimplemented separately.

        Args:
            path: Destination path, e.g. "models/preprocessing_pipeline.joblib".

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline_, path)
        logger.info("Fitted pipeline saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "GenomicsPreprocessor":
        """
        Reconstruct a GenomicsPreprocessor from a previously saved pipeline.

        Args:
            path: Path to a .joblib file written by save().

        Returns:
            GenomicsPreprocessor with .pipeline_ populated and ready for
            .transform() calls (fit_transform() should NOT be called again).
        """
        instance = cls()
        instance.pipeline_ = joblib.load(path)
        logger.info("Pipeline loaded from %s", path)
        return instance

    def _enforce_float32(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safety-net downcast mirroring TCGADataIngester._enforce_float32().

        sklearn's internal numerical routines (e.g., StandardScaler's
        variance computation) may upcast to float64 for numerical
        stability regardless of input dtype. This guarantees the final
        output honors the project's memory budget on the i7-1360P target.
        """
        float64_cols = df.select_dtypes(include=["float64"]).columns
        if len(float64_cols) == 0:
            return df
        if self.config.low_memory_mode:
            logger.warning(
                "Downcasting %d float64 columns to float32 post-pipeline "
                "(sklearn internals upcast for numerical stability).",
                len(float64_cols),
            )
        for col in float64_cols:
            df[col] = df[col].astype(np.float32)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Utility: Label Encoding (deliberately outside the Pipeline)
# ──────────────────────────────────────────────────────────────────────────────


def encode_pam50_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode PAM50 string labels into integers for XGBoost/sklearn compatibility.

    Deliberately kept OUTSIDE the GenomicsPreprocessor/Pipeline - see the
    "Why label encoding is NOT part of this class" note in
    GenomicsPreprocessor's docstring.

    Args:
        y: PAM50 subtype labels (string or categorical dtype), as produced
            by TCGADataIngester.run().

    Returns:
        Tuple of (y_encoded, encoder):
            y_encoded: Integer-encoded labels, shape (n_samples,).
            encoder: Fitted LabelEncoder - keep this to inverse_transform()
                predictions back to PAM50 subtype names for reporting.

    Raises:
        ValueError: If y contains fewer than 2 unique classes.
    """
    if y.nunique() < 2:
        raise ValueError(f"y must contain at least 2 classes; found {y.nunique()}.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    logger.info(
        "Labels encoded | Classes: %s",
        dict(zip(encoder.classes_, range(len(encoder.classes_)))),
    )
    return y_encoded, encoder


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Entry Point (for manual testing in PyCharm Run configuration)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Configure PyCharm Run/Debug:
        Script: src/preprocessing.py
        Working directory: <project root>

    TCGADataIngester, setup_logging, and train_test_split are imported here
    (not at module level) to keep the importable library surface of this
    module minimal - a deployment script that only needs GenomicsPreprocessor
    shouldn't have to drag in the ingestion module's dependencies.
    """
    from sklearn.model_selection import train_test_split

    from src.data_ingestion import TCGADataIngester, setup_logging

    setup_logging("INFO")

    # ── Stage 1: Ingest raw data (previous pipeline stage) ──────────────
    ingester = TCGADataIngester(data_dir="data/raw/")
    X, y, ingestion_meta = ingester.run()

    # ── Stage 2: Split BEFORE any preprocessing statistic is learned ────
    # Note: although preprocessing appears as Step 3 in the original
    # roadmap (before the Step 5 train/test split), every fit-time
    # statistic (expression mask, variance mask, scaler mean/std) must
    # only ever see training data. We split first here, then fit
    # preprocessing exclusively on X_train.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Stage 3: Preprocess (fit on train, transform both) ──────────────
    preprocessor = GenomicsPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # ── Stage 4: Encode labels (kept separate - see docstring) ──────────
    y_train_encoded, label_encoder = encode_pam50_labels(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # ── Summary ───────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  PREPROCESSING COMPLETE - SUMMARY")
    print(sep)
    print(f"  X_train_processed : {X_train_processed.shape}")
    print(f"  X_test_processed  : {X_test_processed.shape}")
    print(f"  Genes retained    : {len(preprocessor.get_selected_genes())}")
    print(f"  Classes           : {list(label_encoder.classes_)}")
    print(sep)

    # ── Persist for the modeling stage / FastAPI deployment ─────────────
    preprocessor.save("models/preprocessing_pipeline.joblib")
    print("  Pipeline saved to models/preprocessing_pipeline.joblib")
    print("  X_train_processed, X_test_processed ready for src/models/")
    print(sep)
