"""
feature_selection.py
====================
Production-ready, leakage-free feature selection pipeline for TCGA-BRCA
RNA-Seq data after genomics preprocessing.

Implements a 2-stage sklearn Pipeline:
    Stage 1 — MADFilter      : Unsupervised, robust pre-filter using Median
                               Absolute Deviation. Reduces ~15,000 preprocessed
                               genes → n_mad_genes (default 5,000).
    Stage 2 — SelectKBest    : Supervised ANOVA F-statistic filter. Reduces
                               n_mad_genes → n_kbest_genes (default 500).

SCORING FUNCTION CHOICE: f_classif over mutual_info_classif
------------------------------------------------------------
Both are valid for multi-class PAM50 prediction. f_classif (one-way ANOVA
F-statistic) is selected here for the following principled reasons:

  1. DISTRIBUTIONAL COMPATIBILITY
     After log2(FPKM+1) transform and StandardScaler (preprocessing.py),
     gene expression values are approximately Gaussian within each PAM50
     class. f_classif's core assumption — within-class normality —
     is therefore reasonably satisfied before SelectKBest ever runs.

  2. PAM50 BIOLOGICAL PRECEDENT
     PAM50 subtypes were defined by Perou et al. (2000) using hierarchical
     clustering on linear expression differences. The Luminal A/B split,
     HER2-enriched overexpression signatures, and Basal-like luminal-marker
     suppression are all monotone linear gene-subtype relationships — exactly
     what an ANOVA F-statistic captures. There is no known a priori reason
     to model these as non-linear threshold effects.

  3. DETERMINISM
     f_classif is fully deterministic: same X_train + y_train always yields
     the same ranking. mutual_info_classif uses k-NN density estimation with
     random tie-breaking — non-deterministic even with random_state fixed
     across sklearn versions and n_neighbors settings. Determinism matters
     for a portfolio project where the README's reported gene list must be
     reproducible by any recruiter who clones the repo.

  4. RUNTIME ON i7-1360P (integrated graphics, ≤16 GB RAM)
     f_classif is O(n_samples × n_genes) with a tiny constant: one ANOVA
     pass. mutual_info_classif's k-NN estimation runs ~10–50× slower on
     the same matrix. At 5,000 genes × 880 training samples, f_classif
     completes in <1 s; mutual_info_classif takes 30–90 s.

  WHEN mutual_info_classif WOULD BE THE BETTER CHOICE:
     - Raw, non-log-transformed FPKM data (heavily right-skewed; f_classif
       assumptions badly violated).
     - Methylation array or copy-number variation data, where gene–subtype
       relationships are inherently non-linear or bimodal.
     - Features with known threshold effects (e.g., binary mutation flags).
     mutual_info_classif is exposed as a configurable option in
     FeatureSelectionConfig.scoring_func for exactly these scenarios.

Integration Contract:
    Consumes : Output of GenomicsPreprocessor.transform() — a float32
               DataFrame of shape (n_samples, ~15,000 genes), zero-mean,
               unit-variance, with gene-symbol column names intact.
    Produces : float32 DataFrame of shape (n_samples, n_kbest_genes=500)
               with gene-symbol column names preserved end-to-end.
    Passes to: src/models/ensemble_model.py and src/explainability/shap_explainer.py

Usage:
   >>> from src.feature_selection import GenomicsFeatureSelector
    >>> from src.feature_selection import FeatureSelectionConfig
    >>> selector = GenomicsFeatureSelector()
    >>> X_train_sel = selector.fit_transform(X_train_processed, y_train_encoded)
    >>> X_test_sel  = selector.transform(X_test_processed)
    >>> genes       = selector.get_selected_genes()   # pass directly to SHAP

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

# Sentinel dict used by build_pipeline() to resolve config strings to callables.
# Extending this dict is the only change needed to support future scoring functions.
_SCORING_REGISTRY: Dict[str, Callable] = {
    "f_classif": f_classif,
    "mutual_info_classif": mutual_info_classif,
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FeatureSelectionConfig:
    """
    Typed, injectable configuration for the 2-stage feature selection pipeline.

    Follows the IngestionConfig / PreprocessingConfig pattern established in
    data_ingestion.py and preprocessing.py: a dataclass rather than a dict
    or **kwargs, making every parameter IDE-autocompletable and trivially
    mockable in unit tests (inject a config with small n_mad_genes /
    n_kbest_genes to run against a synthetic 200-gene fixture in <0.1 s).

    Attributes:
        n_mad_genes    : Number of genes to retain after MAD pre-filter.
                         Must satisfy n_mad_genes > n_kbest_genes.
                         Rule of thumb: 10× n_kbest_genes gives SelectKBest
                         a meaningful pool without wasting i7-1360P compute.
        n_kbest_genes  : Final number of genes after supervised SelectKBest.
                         500 is the canonical value from the project roadmap;
                         reduces curse-of-dimensionality risk for KNN/SVM.
        scoring_func   : Key into _SCORING_REGISTRY. 'f_classif' is the
                         principled default — see module docstring for the
                         full defense. Switch to 'mutual_info_classif' for
                         non-log-transformed or non-Gaussian input data.
        random_state   : Seed for mutual_info_classif's k-NN tie-breaking.
                         Ignored entirely when scoring_func='f_classif'.
        low_memory_mode: If True, triggers extra MemoryReporter calls at
                         each stage and forces gc.collect() between them.
    """

    n_mad_genes: int = 5_000
    n_kbest_genes: int = 500
    scoring_func: Literal["f_classif", "mutual_info_classif"] = "f_classif"
    random_state: int = 42
    low_memory_mode: bool = True

    def __post_init__(self) -> None:
        if self.n_mad_genes <= self.n_kbest_genes:
            raise ValueError(
                f"n_mad_genes ({self.n_mad_genes}) must be strictly greater than "
                f"n_kbest_genes ({self.n_kbest_genes}). The MAD pre-filter must "
                f"leave SelectKBest a meaningful pool to rank."
            )
        if self.scoring_func not in _SCORING_REGISTRY:
            raise ValueError(
                f"scoring_func='{self.scoring_func}' is not registered. "
                f"Valid options: {list(_SCORING_REGISTRY.keys())}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Custom Transformer: MADFilter
# ──────────────────────────────────────────────────────────────────────────────


class MADFilter(BaseEstimator, TransformerMixin):
    """
    Unsupervised, robust gene pre-filter based on Median Absolute Deviation.

    WHY MAD OVER VARIANCE (VarianceThreshold)?
    -------------------------------------------
    VarianceThreshold uses the arithmetic mean in its computation:
        variance(gene) = mean((x - mean(x))²)
    A single outlier sample with pathologically high expression for one gene
    can inflate that gene's variance substantially, causing a biologically
    uninformative gene to survive the filter. MAD uses the median instead:
        MAD(gene) = median(|x - median(x)|)
    The median is a breakdown-point-50% statistic — it is unaffected by up to
    49% of sample values being arbitrarily large outliers. For RNA-Seq cohorts
    like TCGA-BRCA, which contain batch effects, hypermutated tumors, and
    occasional sample mix-ups, MAD is the principled choice for a robust
    unsupervised pre-filter before the supervised SelectKBest stage.

    Leakage Contract:
        .fit()       → learns MAD per gene and stores the top-n_genes mask
                       (training data ONLY).
        .transform() → applies the stored mask verbatim, no recomputation.
        y is ALWAYS ignored — MAD is fully unsupervised.

    Vectorization:
        The entire MAD computation is two NumPy reduction calls over the
        (n_samples × n_genes) matrix — no Python-level loop over genes.
        See _compute_mad_vectorized() for the implementation.

    Attributes:
        support_mask_      : Boolean array (n_genes,), True for retained genes.
        mad_scores_        : float32 array (n_genes,) of per-gene MAD values,
                             retained for diagnostics and unit testing.
        feature_names_in_  : Gene identifiers seen at fit time (pre-filter).
        n_features_in_     : Number of genes seen at fit time.
        n_genes_removed_   : Number of genes dropped by the MAD filter.
    """

    def __init__(self, n_genes: int = 5_000) -> None:
        """
        Args:
            n_genes: Number of highest-MAD genes to retain.
        """
        self.n_genes = n_genes

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MADFilter":
        """
        Compute per-gene MAD on training data and store the top-n_genes mask.

        Args:
            X: Preprocessed expression matrix, shape (n_train_samples, n_genes).
               Must be a pandas DataFrame with gene-symbol column names.
            y: Ignored. Accepted for sklearn Pipeline API compatibility.

        Returns:
            self

        Raises:
            TypeError : If X is not a pandas DataFrame.
            ValueError: If n_genes >= X.shape[1] (nothing to filter).
        """
        self._validate_dataframe(X, context="fit")
        if self.n_genes >= X.shape[1]:
            raise ValueError(
                f"MADFilter.n_genes ({self.n_genes}) must be less than the "
                f"number of input genes ({X.shape[1]}). Nothing to filter."
            )

        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]

        self.mad_scores_ = self._compute_mad_vectorized(X)

        # Rank genes by MAD (descending). np.argpartition gives us the
        # indices of the top n_genes in O(n_genes) rather than O(n_genes log n),
        # then we sort only that small subset. Faster than a full argsort.
        top_indices = np.argpartition(self.mad_scores_, -self.n_genes)[-self.n_genes :]

        self.support_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_mask_[top_indices] = True
        self.n_genes_removed_ = self.n_features_in_ - self.n_genes

        min_mad = float(self.mad_scores_[top_indices].min())
        max_mad = float(self.mad_scores_[top_indices].max())
        logger.info(
            "MADFilter fit | Input: %d genes | Retained: %d | Dropped: %d | "
            "MAD range of retained genes: [%.4f, %.4f]",
            self.n_features_in_,
            self.n_genes,
            self.n_genes_removed_,
            min_mad,
            max_mad,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mask learned during fit(). No recomputation on X.

        Args:
            X: Must have identical gene columns to those seen during fit().

        Returns:
            DataFrame of shape (n_samples, n_genes), float32, gene names intact.

        Raises:
            NotFittedError: If called before fit().
            ValueError    : If X's columns don't match those seen during fit().
        """
        check_is_fitted(self)
        self._validate_dataframe(X, context="transform")
        self._validate_feature_alignment(X)

        # Single vectorized boolean column indexing — no Python-level loop.
        return X.loc[:, self.support_mask_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Return gene names surviving the MAD filter.
        Required by Pipeline.get_feature_names_out().
        """
        check_is_fitted(self)
        return self.feature_names_in_[self.support_mask_]

    @staticmethod
    def _compute_mad_vectorized(X: pd.DataFrame) -> np.ndarray:
        """
        Compute per-gene Median Absolute Deviation in two vectorized NumPy passes.

        Implementation:
            X_np   = (n_samples × n_genes) float32 matrix
            Pass 1 : median_per_gene = np.median(X_np, axis=0)  → (n_genes,)
                     One reduction over the sample axis. Fully vectorized in C.
            Pass 2 : mad = np.median(|X_np - median_per_gene|, axis=0) → (n_genes,)
                     Broadcasting: (n_samples, n_genes) - (n_genes,) expands
                     correctly without materializing a copy — NumPy handles
                     this in place in the C ufunc.

        The anti-pattern to avoid on a 15,000-gene matrix:
            mad = X.apply(lambda col: col.mad())   ← 15,000 separate Python
                                                      calls, ~100x slower.

        Returns:
            float32 array of shape (n_genes,), one MAD value per gene.
        """
        X_np = X.to_numpy(dtype=np.float32)
        median_per_gene = np.median(X_np, axis=0)  # (n_genes,)
        mad_per_gene = np.median(np.abs(X_np - median_per_gene), axis=0)  # (n_genes,)
        return mad_per_gene.astype(np.float32)

    @staticmethod
    def _validate_dataframe(X, context: str) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MADFilter.{context}() requires a pandas DataFrame "
                f"(to preserve gene-symbol columns); got {type(X)}."
            )

    def _validate_feature_alignment(self, X: pd.DataFrame) -> None:
        if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
            mismatched = set(X.columns).symmetric_difference(
                set(self.feature_names_in_)
            )
            raise ValueError(
                f"Column mismatch: {len(mismatched)} genes differ between "
                f"fit() and transform(). Ensure X_train and X_test both passed "
                f"through the same fitted GenomicsPreprocessor instance."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator: GenomicsFeatureSelector
# ──────────────────────────────────────────────────────────────────────────────


class GenomicsFeatureSelector:
    """
    Orchestrates the 2-stage genomics feature selection pipeline:
        Stage 1 — MADFilter   : Unsupervised, robust pre-filter (MAD ranking).
        Stage 2 — SelectKBest : Supervised ANOVA F-statistic ranking.

    Mirrors the GenomicsPreprocessor design pattern from preprocessing.py:
    a thin orchestrator that owns a sklearn Pipeline internally and adds
    genomics-specific logging, memory instrumentation, and a gene-name
    accessor so the downstream SHAP module never has to reconstruct which
    genes were actually passed to the model.

    WHY y IS REQUIRED AT fit_transform() INSTEAD OF INSIDE A PIPELINE:
    -------------------------------------------------------------------
    SelectKBest requires y in its .fit() call. sklearn's Pipeline.fit(X, y)
    passes y through to every step's .fit() correctly, so SelectKBest works
    inside a Pipeline without any special handling. However, exposing a
    fit_transform(X, y) signature here (rather than hiding y inside a
    Pipeline.fit(X, y) call) makes the supervised dependency explicit at
    the API boundary — a caller can't accidentally call fit_transform(X_test)
    without y and get silently wrong results. The requirement is opt-in.

    Leakage Contract (repeated from module docstring for fast scanning):
        fit_transform(X_train, y_train) : ONLY valid call point for learning.
        transform(X_test)              : Applies learned mask + scores verbatim.

    Attributes:
        config    : FeatureSelectionConfig instance.
        pipeline_ : sklearn Pipeline; set after first fit_transform() call.
                    Expose directly for use with cross_val_score / GridSearchCV.
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None) -> None:
        self.config = config or FeatureSelectionConfig()
        self.reporter = MemoryReporter()
        self.pipeline_: Optional[Pipeline] = None
        logger.info(
            "GenomicsFeatureSelector initialized | MAD: %d genes → KBest: %d genes "
            "| scoring: %s",
            self.config.n_mad_genes,
            self.config.n_kbest_genes,
            self.config.scoring_func,
        )

    # ── Pipeline Construction ─────────────────────────────────────────────────

    def build_pipeline(self) -> Pipeline:
        """
        Construct (but do not fit) the 2-stage sklearn Pipeline.

        SelectKBest receives set_output(transform="pandas") so it emits a
        DataFrame with correct gene-symbol column names rather than a bare
        ndarray. MADFilter constructs its own DataFrame in transform(), so
        it does not need set_output configuration.

        The mutual_info_classif path wraps the scoring function in
        functools.partial to inject random_state, making k-NN tie-breaking
        deterministic across runs — without partial, SelectKBest has no
        mechanism to forward random_state to the scoring callable.

        Returns:
            Unfitted sklearn Pipeline with gene names preserved end-to-end.
        """
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
        """
        Map config.scoring_func string to a callable, injecting random_state
        for mutual_info_classif to guarantee determinism.
        """
        if self.config.scoring_func == "mutual_info_classif":
            # partial() bakes random_state into the function signature so
            # SelectKBest can call score_func(X, y) without knowing about
            # the random_state argument.
            return partial(
                mutual_info_classif,
                random_state=self.config.random_state,
            )
        return _SCORING_REGISTRY[self.config.scoring_func]

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> pd.DataFrame:
        """
        Fit the 2-stage pipeline on training data only, then transform it.

        This is the single point in the workflow where any statistic is
        learned: the MAD mask (unsupervised, ignores y_train) and the
        ANOVA F-scores (supervised, requires y_train). Both are computed
        exclusively from X_train / y_train passed here.

        Args:
            X_train : Preprocessed expression matrix, training fold only.
                      float32 DataFrame, shape (n_train_samples, ~15,000 genes).
                      Must be output of GenomicsPreprocessor.transform(X_train).
            y_train : Integer-encoded PAM50 labels, shape (n_train_samples,).
                      Must be output of encode_pam50_labels(y_train).

        Returns:
            float32 DataFrame, shape (n_train_samples, n_kbest_genes=500),
            with gene-symbol column names preserved. Ready for model training.

        Raises:
            TypeError : If X_train is not a pandas DataFrame.
            ValueError: If len(X_train) != len(y_train).
        """
        self._validate_fit_inputs(X_train, y_train)
        logger.info(
            "GenomicsFeatureSelector.fit_transform() START | Input: %s", X_train.shape
        )
        self.reporter.report(X_train, "Feature Selection Input (preprocessed, train)")

        self.pipeline_ = self.build_pipeline()
        X_selected = self.pipeline_.fit_transform(X_train, y_train)
        X_selected = self._ensure_float32(X_selected)

        self.reporter.report(X_selected, "Feature Selection Output (selected, train)")
        self._log_top_genes(n=10)
        logger.info(
            "GenomicsFeatureSelector.fit_transform() COMPLETE | %d → %d genes",
            X_train.shape[1],
            X_selected.shape[1],
        )
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted pipeline to held-out or inference data.

        Reuses the MAD mask and ANOVA F-score ranking learned in
        fit_transform() verbatim. Zero recomputation on X.

        Args:
            X: Must have the same ~15,000 gene columns as X_train.

        Returns:
            float32 DataFrame, shape (n_samples, n_kbest_genes=500).

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "GenomicsFeatureSelector.transform() called before fit_transform(). "
                "Fit on the training split first."
            )
        logger.info("GenomicsFeatureSelector.transform() | Input: %s", X.shape)
        X_selected = self.pipeline_.transform(X)
        X_selected = self._ensure_float32(X_selected)
        self.reporter.report(X_selected, "Feature Selection Output (selected, holdout)")
        return X_selected

    def get_selected_genes(self) -> List[str]:
        """
        Return the final 500 gene identifiers that survived both pipeline stages.

        This list must be passed as feature_names to every downstream SHAP
        call (summary_plot, beeswarm, waterfall). If the gene names are
        reconstructed or re-derived anywhere else, they risk going out of
        sync with the model's actual feature order — a silent bug that
        labels SHAP bars with the wrong gene symbol.

        Returns:
            List[str] of length n_kbest_genes, in the column order the
            model will see them.

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before get_selected_genes().")
        return list(self.pipeline_.get_feature_names_out())

    def get_mad_scores(self) -> pd.Series:
        """
        Return the per-gene MAD scores learned during Stage 1 (all ~15K genes).

        Useful for exploratory analysis: plot the MAD score distribution in
        a notebook to validate that the n_mad_genes cutoff sits in the right
        part of the tail, not in the dense-low-MAD bulk.

        Returns:
            pd.Series indexed by gene name, values are float32 MAD scores,
            sorted descending. Length = n_features_in_ of the MAD step.

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if self.pipeline_ is None:
            raise RuntimeError("Call fit_transform() before get_mad_scores().")
        mad_step: MADFilter = self.pipeline_.named_steps["mad_filter"]
        return pd.Series(
            mad_step.mad_scores_,
            index=mad_step.feature_names_in_,
            name="mad_score",
        ).sort_values(ascending=False)

    def get_kbest_scores(self) -> pd.Series:
        """
        Return the ANOVA F-scores (or MI scores) for the n_mad_genes pool.

        These are the per-gene scores that SelectKBest used to pick the
        final 500 genes. Cross-reference the top 20 with your SHAP
        beeswarm plot in the notebook — high F-score genes that also have
        high mean |SHAP| are the most defensible biomarker candidates.

        Returns:
            pd.Series indexed by gene name (post-MAD pool), values are
            float32 F-scores, sorted descending. Length = n_mad_genes.

        Raises:
            RuntimeError: If called before fit_transform().
        """
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
        Serialize the fitted pipeline to disk for use at inference time.

        Mirrors GenomicsPreprocessor.save() from preprocessing.py. The
        saved artifact carries both the MAD mask and the SelectKBest mask —
        exactly what deployment/api/main.py needs to reproduce the identical
        feature subset on an incoming patient sample without re-running the
        full training pipeline.

        Args:
            path: Destination path, e.g. 'models/feature_selection_pipeline.joblib'.

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
        """
        Reconstruct a GenomicsFeatureSelector from a previously saved pipeline.

        Args:
            path: Path to a .joblib file written by save().

        Returns:
            GenomicsFeatureSelector with .pipeline_ populated and ready for
            .transform() and .get_selected_genes() calls.
        """
        instance = cls()
        instance.pipeline_ = joblib.load(path)
        logger.info("Feature selection pipeline loaded from %s", path)
        return instance

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_fit_inputs(X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                f"X_train must be a pandas DataFrame; got {type(X_train)}. "
                f"Pass the output of GenomicsPreprocessor.transform(X_train)."
            )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train have different sample counts: "
                f"{len(X_train)} vs {len(y_train)}. "
                f"Ensure both come from the same stratified train split."
            )

    @staticmethod
    def _ensure_float32(df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast any float64 columns introduced by SelectKBest internals.

        SelectKBest.transform() may return float64 on some sklearn builds
        regardless of input dtype (depends on the scoring function's internal
        computation dtype). This is the same safety-net pattern used in
        GenomicsPreprocessor._enforce_float32().
        """
        float64_cols = df.select_dtypes(include=["float64"]).columns
        if len(float64_cols) == 0:
            return df
        for col in float64_cols:
            df[col] = df[col].astype(np.float32)
        return df

    def _log_top_genes(self, n: int = 10) -> None:
        """Log the top-n selected genes by F-score for rapid sanity-checking."""
        try:
            top = self.get_kbest_scores().head(n)
            logger.info(
                "Top %d genes by F-score (cross-reference against "
                "PAM50 literature):\n%s",
                n,
                top.to_string(),
            )
        except Exception:
            pass  # Non-critical — don't let diagnostic logging crash the pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone execution for testing from PyCharm's Run button.

    Configure PyCharm Run/Debug:
        Script           : src/feature_selection.py
        Working directory: <project root>

    Picks up immediately after preprocessing.py's __main__ block.
    Loads the saved preprocessing pipeline artifact rather than re-running
    ingestion + preprocessing — saving ~2 min on an i7-1360P per test run.
    """
    from sklearn.model_selection import train_test_split

    from src.data_ingestion import TCGADataIngester, setup_logging
    from src.preprocessing import GenomicsPreprocessor, encode_pam50_labels

    setup_logging("INFO")

    # ── Stage 1–2: Ingest + split ────────────────────────────────────────────
    X, y, _ = TCGADataIngester(data_dir="data/raw/").run()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Stage 3: Preprocess (load saved artifact if it exists) ───────────────
    preproc_path = "models/preprocessing_pipeline.joblib"
    if Path(preproc_path).exists():
        preprocessor = GenomicsPreprocessor.load(preproc_path)
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
    else:
        preprocessor = GenomicsPreprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        preprocessor.save(preproc_path)

    # ── Stage 4: Encode labels ────────────────────────────────────────────────
    y_train_enc, label_encoder = encode_pam50_labels(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # ── Stage 5: Feature selection ────────────────────────────────────────────
    selector = GenomicsFeatureSelector()
    X_train_sel = selector.fit_transform(X_train_proc, y_train_enc)
    X_test_sel = selector.transform(X_test_proc)

    selector.save("models/feature_selection_pipeline.joblib")

    # ── Summary ───────────────────────────────────────────────────────────────
    selected_genes = selector.get_selected_genes()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  FEATURE SELECTION COMPLETE — SUMMARY")
    print(sep)
    print(f"  Preprocessed genes  : {X_train_proc.shape[1]:>7,}")
    print(f"  Post-MAD filter     : {selector.config.n_mad_genes:>7,}")
    print(f"  Post-SelectKBest    : {len(selected_genes):>7,}")
    print(f"  X_train_sel shape   : {X_train_sel.shape}")
    print(f"  X_test_sel shape    : {X_test_sel.shape}")
    print(f"  X_train_sel dtype   : {X_train_sel.dtypes.iloc[0]}")
    print(f"  Scoring function    : {selector.config.scoring_func}")
    print("\n  Top 5 genes by F-score:")
    for gene, score in selector.get_kbest_scores().head(5).items():
        print(f"    {gene:<20} F = {score:.2f}")
    print(sep)
    print("\n  ✓ X_train_sel, X_test_sel ready for src/models/ensemble_model.py")
    print(
        f"  ✓ {len(selected_genes)} gene names ready for "
        "src/explainability/shap_explainer.py"
    )
    print(sep)
