"""
shap_explainer.py
=================
Production-ready SHAP explainability module for PAM50 breast cancer
subtype classification, built on the fitted BiomarkerEnsemble.

Computes exact Shapley values for one tree-based base learner (RF or
XGBoost) extracted from the StackingClassifier, generates per-class
beeswarm and mean-|SHAP| bar plots, and exposes a ranked gene table
for biomarker cross-referencing against PAM50 literature.

WHY EXPLAIN A BASE LEARNER, NOT THE FULL STACK?
------------------------------------------------
StackingClassifier was built with passthrough=False (see
ensemble_model.py): the LogisticRegression meta-learner consumes only
the 10 base-learner probability columns (5 RF + 5 XGB), never the 500
genes directly. There is therefore no tree structure mapping genes to
the meta-learner's decision, so shap.TreeExplainer cannot be pointed
at the stack as a whole.

The model-agnostic alternative — treating ensemble.predict_proba as
an opaque function of 500 genes via shap.KernelExplainer — is
theoretically possible but requires thousands of resampled model
evaluations per explained sample. On CPU-only hardware with no GPU,
this is impractical at 880 samples x 500 genes.

Explaining one base learner directly via exact Tree SHAP is therefore
both the practical and principled choice. The limitation is explicit:
these attributions characterize that base learner's decision process,
not a formal decomposition of the stack's blended output. In practice
this is a reasonable proxy — both base learners are trained on the
same 500-gene panel and tend to converge on the same dominant
biological signal, since strong PAM50 markers drive both models.

WHY TreeExplainer's DEFAULT feature_perturbation IS KEPT:
-----------------------------------------------------------
shap.TreeExplainer(model) with no `data=` argument defaults to
feature_perturbation='tree_path_dependent', which needs no separate
background dataset and computes values directly from the tree's
learned split structure. The alternative, 'interventional', requires
an explicit background sample and assumes feature independence when
marginalizing — an assumption gene expression data violates routinely
due to co-regulated pathway modules. The default is both faster and
more defensible here, so it is used as-is rather than exposed as a
half-supported config toggle.

Integration Contract:
    Consumes : A fitted BiomarkerEnsemble (src/models/ensemble_model.py)
               and the float32 DataFrame output of
               GenomicsFeatureSelector.transform() — shape
               (n_samples, 500 genes) — typically X_train_sel.
    Produces : PNG figures in reports/figures/, a CSV biomarker table
               in reports/, and a queryable in-memory SHAP array.
    Used by  : notebooks/04_SHAP_Biomarker_Discovery.ipynb and the
               project README's biomarker validation section.

Usage:
    >>> from src.models.ensemble_model import BiomarkerEnsemble
    >>> from src.explainability.shap_explainer import (
    ...     BiomarkerSHAPExplainer,
    ... )
    >>> ensemble = BiomarkerEnsemble.load("models/ensemble_model.joblib")
    >>> explainer = BiomarkerSHAPExplainer(
    ...     ensemble=ensemble,
    ...     feature_names=selected_genes,
    ...     class_names=list(label_encoder.classes_),
    ... )
    >>> explainer.fit(X_train_sel)
    >>> explainer.generate_all_plots()
    >>> top_genes = explainer.get_top_biomarkers(n_top=20)

Author : [Your Name]
Date   : [Project Date]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.ensemble_model import BiomarkerEnsemble

logger = logging.getLogger(__name__)


# ── Configuration Dataclass ──────────────────────────────────────


@dataclass
class ShapExplainerConfig:
    """
    Typed, injectable configuration for BiomarkerSHAPExplainer.

    Follows the project-wide Config pattern established in
    IngestionConfig, PreprocessingConfig, FeatureSelectionConfig, and
    EnsembleConfig: every field is explicit, IDE-autocompletable, and
    mockable in tests.

    Attributes:
        base_learner_key: Which fitted base learner to explain — 'xgb'
                          or 'rf'. Must be a key in
                          ensemble.stack_.named_estimators_. Default
                          'xgb': XGBoost's depth-6 trees produce exact
                          SHAP values noticeably faster on CPU than an
                          unconstrained-depth Random Forest.
        n_top_genes     : Default row count for get_top_biomarkers().
        figures_dir     : Destination folder for saved PNG plots.
        max_display     : Max genes shown per beeswarm/bar plot.
        max_samples     : Optional cap on samples explained. None uses
                          the full input matrix, as the project spec
                          requires. Set this only if SHAP computation
                          becomes a bottleneck on a larger cohort.
        random_state    : Seed for the max_samples subsample draw.
                          Unused when max_samples is None — Tree SHAP
                          itself is a deterministic, exact algorithm
                          with no randomness of its own.
    """

    base_learner_key: str = "xgb"
    n_top_genes: int = 20
    figures_dir: str = "reports/figures"
    max_display: int = 20
    max_samples: Optional[int] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_top_genes <= 0:
            raise ValueError(f"n_top_genes must be positive; got {self.n_top_genes}.")
        if self.max_display <= 0:
            raise ValueError(f"max_display must be positive; got {self.max_display}.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive; got {self.max_samples}.")


# ── Orchestrator Class ───────────────────────────────────────────


class BiomarkerSHAPExplainer:
    """
    Computes and serves SHAP-based biomarker discovery for one base
    learner extracted from a fitted BiomarkerEnsemble.

    Attributes:
        config           : ShapExplainerConfig controlling behavior.
        feature_names    : 500 gene symbols, in model column order.
        class_names      : PAM50 subtype strings, ordered to match
                            LabelEncoder.classes_ (alphabetical),
                            which is also the column order of both
                            base learners' predict_proba output.
        base_learner_    : The extracted RF or XGB estimator.
        shap_values_     : Set by .fit(). float32 ndarray, shape
                            (n_samples, n_genes, n_classes).
        expected_value_  : Set by .fit(). Base learner's expected
                            output value(s) — scalar, list, or array
                            depending on shap/model version.
        X_explained_     : Set by .fit(). The (possibly subsampled)
                            DataFrame actually passed to TreeExplainer.
        explainer_       : Set by .fit(). The shap.TreeExplainer
                            instance, retained for future extensions
                            (e.g. a waterfall plot on one sample).
    """

    def __init__(
        self,
        ensemble: BiomarkerEnsemble,
        feature_names: List[str],
        class_names: List[str],
        config: Optional[ShapExplainerConfig] = None,
    ) -> None:
        """
        Args:
            ensemble: A fitted BiomarkerEnsemble (ensemble.stack_ must
                      not be None). The base learner is read from
                      ensemble.stack_.named_estimators_.
            feature_names: 500 gene-symbol strings, in the exact
                          column order used to train the ensemble.
                          Pass GenomicsFeatureSelector's
                          get_selected_genes().
            class_names: PAM50 subtype strings in the same order as
                        the LabelEncoder used to train the ensemble,
                        e.g. list(label_encoder.classes_). This order
                        must match the base learners' classes_
                        attribute; both share the same alphabetically-
                        sorted convention from encode_pam50_labels(),
                        so this holds automatically within one
                        pipeline run.
            config: ShapExplainerConfig instance. Uses class defaults
                    if None.

        Raises:
            RuntimeError: If ensemble.stack_ is None (not yet fitted).
            ValueError: If config.base_learner_key is not a key in
                        ensemble.stack_.named_estimators_.
        """
        if ensemble.stack_ is None:
            raise RuntimeError(
                "BiomarkerEnsemble must be fitted before creating a "
                "BiomarkerSHAPExplainer. Call ensemble.fit() first."
            )

        self.config = config or ShapExplainerConfig()
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)

        available = list(ensemble.stack_.named_estimators_.keys())
        if self.config.base_learner_key not in available:
            raise ValueError(
                f"base_learner_key='{self.config.base_learner_key}' "
                f"not found. Available learners: {available}."
            )
        self.base_learner_ = ensemble.stack_.named_estimators_[
            self.config.base_learner_key
        ]

        self.shap_values_: Optional[np.ndarray] = None
        self.expected_value_ = None
        self.X_explained_: Optional[pd.DataFrame] = None
        self.explainer_: Optional[shap.TreeExplainer] = None

        Path(self.config.figures_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "BiomarkerSHAPExplainer initialized | base_learner=%s | "
            "n_genes=%d | n_classes=%d",
            self.config.base_learner_key,
            len(self.feature_names),
            len(self.class_names),
        )

    # ── Public API ────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "BiomarkerSHAPExplainer":
        """
        Compute SHAP values for the base learner on the given samples.

        Args:
            X: float32 DataFrame, shape (n_samples, 500 genes). Must
               have the exact same columns (gene symbols, same order)
               as feature_names passed at construction. Pass
               X_train_sel from GenomicsFeatureSelector.transform().

        Returns:
            self — enables chaining, e.g.
            explainer.fit(X).generate_all_plots()

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If X's columns don't match feature_names.
        """
        self._validate_input(X)
        X_for_shap = self._subsample_if_needed(X)

        if self.config.base_learner_key == "rf":
            rf_depth = getattr(self.base_learner_, "max_depth", "unset")
            if rf_depth is None:
                logger.warning(
                    "Explaining an unconstrained-depth Random Forest. "
                    "TreeExplainer runtime scales with tree depth "
                    "squared; this may take several minutes on "
                    "CPU-only hardware. Consider base_learner_key="
                    "'xgb', or set config.max_samples to bound runtime."
                )

        logger.info(
            "Building shap.TreeExplainer for '%s' on %d samples...",
            self.config.base_learner_key,
            X_for_shap.shape[0],
        )
        explainer = shap.TreeExplainer(self.base_learner_)
        raw_shap = explainer.shap_values(X_for_shap)

        self.shap_values_ = self._normalize_shap_values(
            raw_shap,
            n_samples=X_for_shap.shape[0],
            n_features=X_for_shap.shape[1],
        )
        self.expected_value_ = explainer.expected_value
        self.X_explained_ = X_for_shap
        self.explainer_ = explainer

        logger.info(
            "SHAP values computed | shape=%s (samples, genes, classes)",
            self.shap_values_.shape,
        )
        return self

    def plot_summary_beeswarm(
        self,
        class_index: int,
        save: bool = True,
    ) -> Optional[Path]:
        """
        Generate and optionally save a SHAP beeswarm plot for one class.

        Each point is one sample; horizontal position is the SHAP
        value (impact on that class's predicted probability), color
        encodes the gene's expression level for that sample.

        Args:
            class_index: Index into self.class_names for the PAM50
                        subtype to visualize.
            save: If True, writes a PNG to config.figures_dir.

        Returns:
            Path to the saved PNG, or None if save=False.

        Raises:
            RuntimeError: If called before .fit().
            ValueError: If class_index is out of range.
        """
        return self._render_summary_plot(
            class_index=class_index,
            plot_type="dot",
            filename_prefix="shap_beeswarm",
            title_prefix="SHAP Beeswarm",
            save=save,
        )

    def plot_mean_abs_bar(
        self,
        class_index: int,
        save: bool = True,
    ) -> Optional[Path]:
        """
        Generate and optionally save a mean |SHAP| bar plot for one class.

        Ranks genes by mean absolute SHAP value — a simpler,
        directionless view of global importance, easier to present to
        a non-technical audience than the beeswarm plot.

        Args:
            class_index: Index into self.class_names for the PAM50
                        subtype to visualize.
            save: If True, writes a PNG to config.figures_dir.

        Returns:
            Path to the saved PNG, or None if save=False.

        Raises:
            RuntimeError: If called before .fit().
            ValueError: If class_index is out of range.
        """
        return self._render_summary_plot(
            class_index=class_index,
            plot_type="bar",
            filename_prefix="shap_bar",
            title_prefix="Mean |SHAP|",
            save=save,
        )

    def generate_all_plots(self) -> Dict[str, Optional[Path]]:
        """
        Generate and save beeswarm + bar plots for every PAM50 class.

        Returns:
            Dict mapping "{class_name}_beeswarm" / "{class_name}_bar"
            to the saved PNG Path.

        Raises:
            RuntimeError: If called before .fit().
        """
        self._assert_fitted()
        saved_paths: Dict[str, Optional[Path]] = {}

        for idx, class_name in enumerate(self.class_names):
            saved_paths[f"{class_name}_beeswarm"] = self.plot_summary_beeswarm(
                idx, save=True
            )
            saved_paths[f"{class_name}_bar"] = self.plot_mean_abs_bar(idx, save=True)

        logger.info(
            "Generated %d SHAP figures across %d classes.",
            len(saved_paths),
            len(self.class_names),
        )
        return saved_paths

    def get_mean_abs_shap_by_class(self) -> pd.DataFrame:
        """
        Compute mean(|SHAP|) per gene, per PAM50 class.

        Returns:
            DataFrame indexed by gene symbol, one column per PAM50
            class plus an 'overall' column (mean across all classes),
            sorted descending by 'overall'.

        Raises:
            RuntimeError: If called before .fit().
        """
        self._assert_fitted()

        # Single vectorized reduction over axis=0 (samples) — no
        # Python-level loop over genes or classes.
        mean_abs = np.mean(np.abs(self.shap_values_), axis=0)

        table = pd.DataFrame(
            mean_abs,
            index=self.feature_names,
            columns=self.class_names,
        )
        table["overall"] = table.mean(axis=1)
        return table.sort_values("overall", ascending=False)

    def get_top_biomarkers(
        self,
        n_top: Optional[int] = None,
        class_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the top N genes ranked by mean absolute SHAP value.

        Args:
            n_top: Number of genes to return. Defaults to
                  config.n_top_genes.
            class_name: If given, rank by this PAM50 class's mean
                        |SHAP| column instead of the all-class
                        'overall' average. Must be one of
                        self.class_names.

        Returns:
            DataFrame indexed by gene symbol, columns = each PAM50
            class plus 'overall', sorted by the ranking column and
            limited to n_top rows.

        Raises:
            RuntimeError: If called before .fit().
            ValueError: If class_name is not a recognized PAM50 class.
        """
        self._assert_fitted()
        n = n_top if n_top is not None else self.config.n_top_genes

        table = self.get_mean_abs_shap_by_class()

        if class_name is not None:
            if class_name not in self.class_names:
                raise ValueError(
                    f"class_name='{class_name}' not recognized. "
                    f"Valid options: {self.class_names}."
                )
            table = table.sort_values(class_name, ascending=False)

        return table.head(n)

    def save_biomarker_table(self, path: Optional[str] = None) -> Path:
        """
        Persist the full mean |SHAP| table (all genes, classes) to CSV.

        This is the primary artifact for literature cross-referencing:
        open the CSV, sort by 'overall' or any single PAM50 class
        column, and compare the top rows against known PAM50 markers.

        Args:
            path: Destination CSV path. Defaults to
                  'reports/shap_biomarker_table.csv' (one level above
                  config.figures_dir).

        Returns:
            Path to the saved CSV file.

        Raises:
            RuntimeError: If called before .fit().
        """
        self._assert_fitted()

        if path is not None:
            output_path = Path(path)
        else:
            output_path = (
                Path(self.config.figures_dir).parent / "shap_biomarker_table.csv"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.get_mean_abs_shap_by_class().to_csv(output_path)
        logger.info("Biomarker table saved to %s", output_path)
        return output_path

    def save(self, path: str) -> None:
        """
        Persist the fitted explainer (SHAP values, config, metadata)
        via joblib.

        Caches the most compute-intensive step in the pipeline — SHAP
        value computation, especially costly for deep Random Forests
        — so repeated runs can reload instantly instead of
        recomputing.

        Args:
            path: Destination path, e.g. 'models/shap_explainer.joblib'.

        Raises:
            RuntimeError: If called before .fit().
        """
        self._assert_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("BiomarkerSHAPExplainer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BiomarkerSHAPExplainer":
        """
        Reconstruct a fitted BiomarkerSHAPExplainer from a joblib
        artifact written by .save().

        Args:
            path: Path to a .joblib file written by .save().

        Returns:
            BiomarkerSHAPExplainer with shap_values_ populated and
            ready for plotting / biomarker-extraction calls.
        """
        instance: "BiomarkerSHAPExplainer" = joblib.load(path)
        logger.info("BiomarkerSHAPExplainer loaded from %s", path)
        return instance

    # ── Private Helpers ──────────────────────────────────────────

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate X before computing SHAP values in .fit()."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "BiomarkerSHAPExplainer.fit() requires a pandas "
                f"DataFrame; got {type(X)}."
            )
        if list(X.columns) != self.feature_names:
            raise ValueError(
                "X.columns do not match feature_names passed at "
                "construction. Ensure X is the exact output of "
                "GenomicsFeatureSelector.transform()."
            )

    def _subsample_if_needed(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Bound SHAP compute time by subsampling large training sets.

        TreeExplainer runtime scales roughly linearly with the number
        of samples explained. Disabled by default (max_samples=None
        uses the full input matrix, as required for a complete
        biomarker analysis over X_train_sel).
        """
        limit = self.config.max_samples
        if limit is None or X.shape[0] <= limit:
            return X

        logger.info(
            "Subsampling %d of %d samples for SHAP (max_samples=%d).",
            limit,
            X.shape[0],
            limit,
        )
        return X.sample(n=limit, random_state=self.config.random_state)

    @staticmethod
    def _normalize_shap_values(
        raw_shap,
        n_samples: int,
        n_features: int,
    ) -> np.ndarray:
        """
        Normalize TreeExplainer output across shap/model versions into
        a canonical shape: (n_samples, n_features, n_classes).

        explainer.shap_values() has returned at least three shapes
        across shap versions and model types for multiclass tree
        ensembles: (a) a Python list of length n_classes, each element
        (n_samples, n_features); (b) a single ndarray already shaped
        (n_samples, n_features, n_classes); (c) a single ndarray
        shaped (n_classes, n_samples, n_features). This method detects
        and normalizes all three so every downstream method indexes
        shap_values_ the same way regardless of source.

        Args:
            raw_shap: Direct output of explainer.shap_values(X).
            n_samples: X.shape[0], used to disambiguate axis order.
            n_features: X.shape[1], used to disambiguate axis order.

        Returns:
            float32 ndarray, shape (n_samples, n_features, n_classes).

        Raises:
            ValueError: If the shape cannot be confidently normalized.
        """
        if isinstance(raw_shap, list):
            return np.stack(raw_shap, axis=-1).astype(np.float32)

        arr = np.asarray(raw_shap)

        if arr.ndim == 2:
            return arr[:, :, np.newaxis].astype(np.float32)

        if arr.ndim == 3:
            if arr.shape[0] == n_samples and arr.shape[1] == n_features:
                return arr.astype(np.float32)
            if arr.shape[1] == n_samples and arr.shape[2] == n_features:
                return np.transpose(arr, (1, 2, 0)).astype(np.float32)

        raise ValueError(
            f"Unrecognized SHAP output shape {arr.shape} for "
            f"n_samples={n_samples}, n_features={n_features}. "
            "This may indicate a shap library version incompatibility."
        )

    def _assert_fitted(self) -> None:
        """Raise RuntimeError if .fit() has not been called yet."""
        if self.shap_values_ is None:
            raise RuntimeError(
                "BiomarkerSHAPExplainer is not fitted. Call .fit(X) "
                "before using this method."
            )

    def _validate_class_index(self, class_index: int) -> None:
        """Raise ValueError if class_index is outside the valid range."""
        n_classes = len(self.class_names)
        if not 0 <= class_index < n_classes:
            raise ValueError(
                f"class_index={class_index} out of range. "
                f"Must be in [0, {n_classes - 1}]."
            )

    def _render_summary_plot(
        self,
        class_index: int,
        plot_type: str,
        filename_prefix: str,
        title_prefix: str,
        save: bool,
    ) -> Optional[Path]:
        """
        Shared rendering logic for beeswarm and bar SHAP plots.

        Both public plot methods differ only in shap's plot_type
        argument and the output filename/title — this helper avoids
        duplicating the matplotlib figure lifecycle across two
        nearly-identical methods.

        Args:
            class_index: Index into self.class_names.
            plot_type: 'dot' for beeswarm, 'bar' for mean |SHAP| bars.
            filename_prefix: Prefix for the saved PNG filename.
            title_prefix: Prefix for the matplotlib figure title.
            save: If True, writes a PNG to config.figures_dir.

        Returns:
            Path to the saved PNG, or None if save=False.
        """
        self._assert_fitted()
        self._validate_class_index(class_index)

        class_name = self.class_names[class_index]
        shap_for_class = self.shap_values_[:, :, class_index]

        # plt.close("all") both before and after: shap.summary_plot
        # manages its own current figure/axes internally, and calling
        # this 10x in a row (5 classes x 2 plot types) inside
        # generate_all_plots() will otherwise leak stale figure state
        # between calls — a common gotcha when batch-generating SHAP
        # plots outside of a notebook.
        plt.close("all")
        shap.summary_plot(
            shap_for_class,
            self.X_explained_,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=self.config.max_display,
            show=False,
        )
        plt.title(f"{title_prefix} — {class_name} (PAM50 Subtype)")
        plt.tight_layout()

        output_path = None
        if save:
            filename = f"{filename_prefix}_{class_name.lower()}.png"
            output_path = Path(self.config.figures_dir) / filename
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Plot saved to %s", output_path)

        plt.close("all")
        return output_path


# ── Standalone Entry Point ───────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone execution for testing from PyCharm's Run button.

    Configure PyCharm Run/Debug:
        Script           : src/explainability/shap_explainer.py
        Working directory: <project root>

    BiomarkerEnsemble is already imported at module level above and
    is intentionally not re-imported here.
    """
    from sklearn.model_selection import train_test_split

    from src.data_ingestion import TCGADataIngester, setup_logging
    from src.feature_selection import GenomicsFeatureSelector
    from src.preprocessing import GenomicsPreprocessor, encode_pam50_labels

    setup_logging("INFO")

    # ── Ingest + split ────────────────────────────────────────────
    X, y, _ = TCGADataIngester(data_dir="data/raw/").run()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Preprocess (reuse saved artifact) ─────────────────────────
    preprocessor = GenomicsPreprocessor.load("models/preprocessing_pipeline.joblib")
    X_train_proc = preprocessor.transform(X_train)

    # ── Encode labels ─────────────────────────────────────────────
    y_train_enc, label_encoder = encode_pam50_labels(y_train)

    # ── Feature selection (reuse saved artifact) ──────────────────
    selector = GenomicsFeatureSelector.load("models/feature_selection_pipeline.joblib")
    X_train_sel = selector.transform(X_train_proc)
    selected_genes = selector.get_selected_genes()

    # ── Load fitted ensemble ───────────────────────────────────────
    ensemble = BiomarkerEnsemble.load("models/ensemble_model.joblib")

    # ── SHAP explainability ─────────────────────────────────────────
    class_names = list(label_encoder.classes_)
    explainer = BiomarkerSHAPExplainer(
        ensemble=ensemble,
        feature_names=selected_genes,
        class_names=class_names,
    )
    explainer.fit(X_train_sel)

    figure_paths = explainer.generate_all_plots()
    explainer.save_biomarker_table()
    explainer.save("models/shap_explainer.joblib")

    # ── Summary ─────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  SHAP EXPLAINABILITY — SUMMARY")
    print(sep)
    print(f"  Base learner explained : {explainer.config.base_learner_key}")
    print(f"  SHAP values shape      : {explainer.shap_values_.shape}")
    print(f"  Figures generated      : {len(figure_paths)}")

    print("\n  Top 10 genes (overall mean |SHAP|):")
    top_10 = explainer.get_top_biomarkers(n_top=10)
    for gene, row in top_10.iterrows():
        print(f"    {gene:<15} overall={row['overall']:.4f}")

    if "Basal" in class_names:
        print("\n  Top 5 genes specifically for Basal-like subtype:")
        basal_top = explainer.get_top_biomarkers(n_top=5, class_name="Basal")
        for gene, row in basal_top.iterrows():
            print(f"    {gene:<15} Basal={row['Basal']:.4f}")

    print(sep)
    print("  Saved : reports/figures/*.png")
    print("  Saved : reports/shap_biomarker_table.csv")
    print("  Saved : models/shap_explainer.joblib")
    print(sep)
