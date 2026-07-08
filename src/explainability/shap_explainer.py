"""
shap_explainer.py
=================
SHAP explainability for binary vital-status (Alive vs Dead)
classification, built on the fitted BiomarkerEnsemble.

Computes exact Shapley values for one tree-based base learner (RF
or XGBoost) extracted from the StackingClassifier, generates a
beeswarm and a mean-|SHAP| bar plot, and exposes a ranked,
directional gene table for biomarker cross-referencing.

BINARY SHAPE HANDLING -- WHY THIS DIFFERS FROM THE 5-CLASS VERSION
--------------------------------------------------------------------
The earlier PAM50 version kept a 3D (samples, genes, classes)
array so each of 5 classes could be visualized separately. For a
binary target there is only one independent direction to show --
SHAP toward the positive class is exactly the negative of SHAP
toward the negative class -- so this version collapses directly to
a single signed 2D array, (samples, genes), toward whichever label
LabelEncoder assigned as 1 ('Dead', under this project's
alphabetical convention). A positive mean SHAP value for a gene
means higher expression pushes the model toward predicting Dead;
negative means it pushes toward Alive.

The two base-learner families produce this differently at the shap
library level -- XGBoost's binary:logistic objective has a single
underlying output already, while sklearn's RandomForestClassifier
represents both class probabilities explicitly and has, across
shap versions, returned either a list of 2 arrays or a single 3D
array. _normalize_to_positive_class() detects whichever shape
arrived and returns the correct slice either way.

WHY EXPLAIN A BASE LEARNER, NOT THE FULL STACK
------------------------------------------------
StackingClassifier was built with passthrough=False (see
ensemble_model.py): the LogisticRegression meta-learner consumes
only the 4 base-learner probability columns (2 RF + 2 XGB), never
the ~100 genes directly. There is therefore no tree structure
mapping genes to the meta-learner's decision, so shap.TreeExplainer
cannot be pointed at the stack as a whole. Explaining one base
learner directly via exact Tree SHAP is the practical and
principled choice; the limitation is explicit -- these attributions
characterize that base learner's decision process, not a formal
decomposition of the stack's blended output.

WHY TreeExplainer's DEFAULT feature_perturbation IS KEPT
------------------------------------------------------------
shap.TreeExplainer(model) with no data= argument defaults to
feature_perturbation='tree_path_dependent', which needs no separate
background dataset and computes values directly from the tree's
learned split structure. The alternative, 'interventional', requires
an explicit background sample and assumes feature independence when
marginalizing -- an assumption gene expression data violates
routinely due to co-regulated pathway modules.

Integration Contract:
    Consumes : A fitted BiomarkerEnsemble
               (src/models/ensemble_model.py) and the float32
               DataFrame output of
               GenomicsFeatureSelector.transform() -- shape
               (n_samples, ~100 genes) -- typically X_train_sel.
    Produces : PNG figures in reports/figures/, a CSV biomarker
               table in reports/, and a queryable in-memory SHAP
               array.

Usage:
    >>> from src.models.ensemble_model import BiomarkerEnsemble
    >>> from src.explainability.shap_explainer import (
    ...     BiomarkerSHAPExplainer,
    ... )
    >>> ensemble = BiomarkerEnsemble.load(
    ...     "models/ensemble_model.joblib"
    ... )
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


# ── Configuration Dataclass ──


@dataclass
class ShapExplainerConfig:
    """
    Typed, injectable configuration for BiomarkerSHAPExplainer.

    Attributes:
        base_learner_key: Which fitted base learner to explain --
            'xgb' or 'rf'. Must be a key in
            ensemble.stack_.named_estimators_. Default 'xgb':
            XGBoost's depth-6 trees produce exact SHAP values
            noticeably faster on CPU than an unconstrained-depth
            Random Forest.
        n_top_genes     : Default row count for
            get_top_biomarkers().
        figures_dir     : Destination folder for saved PNG plots.
        max_display     : Max genes shown per beeswarm/bar plot.
        max_samples     : Optional cap on samples explained. None
            uses the full input matrix. Set this only if SHAP
            computation becomes a bottleneck on a larger cohort.
        random_state    : Seed for the max_samples subsample draw.
            Unused when max_samples is None -- Tree SHAP itself is
            a deterministic, exact algorithm with no randomness of
            its own.
    """

    base_learner_key: str = "xgb"
    n_top_genes: int = 20
    figures_dir: str = "reports/figures"
    max_display: int = 20
    max_samples: Optional[int] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_top_genes <= 0:
            raise ValueError(
                "n_top_genes must be positive; got " f"{self.n_top_genes}."
            )
        if self.max_display <= 0:
            raise ValueError(
                "max_display must be positive; got " f"{self.max_display}."
            )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(
                "max_samples must be positive; got " f"{self.max_samples}."
            )


# ── Orchestrator Class ──


class BiomarkerSHAPExplainer:
    """
    Computes and serves SHAP-based biomarker discovery for one base
    learner extracted from a fitted BiomarkerEnsemble, collapsed to
    a single signed direction for the binary vital-status target.

    Attributes:
        config              : ShapExplainerConfig controlling
            behavior.
        feature_names       : Gene symbols, in model column order.
        class_names         : Two PAM50-style labels ordered to
            match LabelEncoder.classes_, e.g. ['Alive', 'Dead'].
        base_learner_       : The extracted RF or XGB estimator.
        positive_label_     : Integer label (0 or 1) treated as the
            positive/event class -- the larger of the two encoded
            values, matching BiomarkerEnsemble.evaluate()'s
            convention.
        positive_idx_       : Index of positive_label_ within
            base_learner_.classes_.
        positive_class_name_: String name of the positive class,
            e.g. 'Dead'.
        negative_class_name_: String name of the negative class,
            e.g. 'Alive'.
        shap_values_        : Set by .fit(). float32 ndarray, shape
            (n_samples, n_genes), signed toward positive_label_.
        expected_value_     : Set by .fit(). Base learner's
            expected output value(s).
        X_explained_        : Set by .fit(). The (possibly
            subsampled) DataFrame actually passed to TreeExplainer.
        explainer_          : Set by .fit(). The shap.TreeExplainer
            instance.
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
            ensemble: A fitted BiomarkerEnsemble
                (ensemble.stack_ must not be None).
            feature_names: Gene-symbol strings, in the exact column
                order used to train the ensemble. Pass
                GenomicsFeatureSelector's get_selected_genes().
            class_names: Exactly 2 labels in the same order as the
                LabelEncoder used to train the ensemble, e.g.
                list(label_encoder.classes_).
            config: ShapExplainerConfig instance. Uses class
                defaults if None.

        Raises:
            RuntimeError: If ensemble.stack_ is None.
            ValueError: If class_names does not have exactly 2
                entries, if config.base_learner_key is not a valid
                key, or if the base learner's classes_ are not
                exactly {0, 1}.
        """
        if ensemble.stack_ is None:
            raise RuntimeError(
                "BiomarkerEnsemble must be fitted before creating "
                "a BiomarkerSHAPExplainer. Call ensemble.fit() "
                "first."
            )
        if len(class_names) != 2:
            raise ValueError(
                "BiomarkerSHAPExplainer expects a binary target; "
                f"got {len(class_names)} class_names: "
                f"{class_names}."
            )

        self.config = config or ShapExplainerConfig()
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)

        available = list(ensemble.stack_.named_estimators_.keys())
        if self.config.base_learner_key not in available:
            raise ValueError(
                f"base_learner_key='{self.config.base_learner_key}'"
                f" not found. Available learners: {available}."
            )
        self.base_learner_ = ensemble.stack_.named_estimators_[
            self.config.base_learner_key
        ]

        model_classes = list(self.base_learner_.classes_)
        if set(model_classes) != {0, 1}:
            raise ValueError(
                "BiomarkerSHAPExplainer assumes labels encoded as "
                f"{{0, 1}}; found classes {model_classes} on the "
                "base learner."
            )
        self.positive_label_ = max(model_classes)
        self.positive_idx_ = model_classes.index(self.positive_label_)
        self.positive_class_name_ = self.class_names[self.positive_label_]
        self.negative_class_name_ = self.class_names[1 - self.positive_label_]

        self.shap_values_: Optional[np.ndarray] = None
        self.expected_value_ = None
        self.X_explained_: Optional[pd.DataFrame] = None
        self.explainer_: Optional[shap.TreeExplainer] = None

        Path(self.config.figures_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "BiomarkerSHAPExplainer initialized | base_learner=%s "
            "| n_genes=%d | positive_class=%s | negative_class=%s",
            self.config.base_learner_key,
            len(self.feature_names),
            self.positive_class_name_,
            self.negative_class_name_,
        )

    # ── Public API ──

    def fit(self, X: pd.DataFrame) -> "BiomarkerSHAPExplainer":
        """
        Compute SHAP values for the base learner on the given
        samples.

        Args:
            X: float32 DataFrame, shape (n_samples, n_genes). Must
                have the exact same columns as feature_names.
                Pass X_train_sel from
                GenomicsFeatureSelector.transform().

        Returns:
            self

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
                    "Explaining an unconstrained-depth Random "
                    "Forest. TreeExplainer runtime scales with "
                    "tree depth squared; this may take a while on "
                    "CPU-only hardware. Consider "
                    "base_learner_key='xgb', or set "
                    "config.max_samples to bound runtime."
                )

        logger.info(
            "Building shap.TreeExplainer for '%s' on %d samples...",
            self.config.base_learner_key,
            X_for_shap.shape[0],
        )
        explainer = shap.TreeExplainer(self.base_learner_)
        raw_shap = explainer.shap_values(X_for_shap)

        self.shap_values_ = self._normalize_to_positive_class(
            raw_shap,
            n_samples=X_for_shap.shape[0],
            n_features=X_for_shap.shape[1],
        )
        self.expected_value_ = explainer.expected_value
        self.X_explained_ = X_for_shap
        self.explainer_ = explainer

        logger.info(
            "SHAP values computed | shape=%s (samples, genes) | " "toward class: %s",
            self.shap_values_.shape,
            self.positive_class_name_,
        )
        return self

    def plot_summary_beeswarm(self, save: bool = True) -> Optional[Path]:
        """
        Generate and optionally save a SHAP beeswarm plot.

        Each point is one sample; horizontal position is the SHAP
        value toward self.positive_class_name_, color encodes the
        gene's expression level for that sample.

        Args:
            save: If True, writes a PNG to config.figures_dir.

        Returns:
            Path to the saved PNG, or None if save=False.

        Raises:
            RuntimeError: If called before .fit().
        """
        title = (
            "SHAP Beeswarm -- Impact on Predicted " f"{self.positive_class_name_} Risk"
        )
        return self._render_summary_plot(
            plot_type="dot",
            filename="shap_beeswarm_vital_status.png",
            title=title,
            save=save,
        )

    def plot_mean_abs_bar(self, save: bool = True) -> Optional[Path]:
        """
        Generate and optionally save a mean |SHAP| bar plot.

        Ranks genes by mean absolute SHAP value -- a directionless
        view of global importance, easier to present to a non-
        technical audience than the beeswarm plot.

        Args:
            save: If True, writes a PNG to config.figures_dir.

        Returns:
            Path to the saved PNG, or None if save=False.

        Raises:
            RuntimeError: If called before .fit().
        """
        title = (
            "Mean |SHAP| -- Top Genes Driving "
            f"{self.positive_class_name_} vs "
            f"{self.negative_class_name_} Prediction"
        )
        return self._render_summary_plot(
            plot_type="bar",
            filename="shap_bar_vital_status.png",
            title=title,
            save=save,
        )

    def generate_all_plots(self) -> Dict[str, Optional[Path]]:
        """
        Generate and save both the beeswarm and bar plots.

        Returns:
            Dict with keys 'beeswarm' and 'bar' mapping to saved
            PNG Paths.

        Raises:
            RuntimeError: If called before .fit().
        """
        self._assert_fitted()
        paths = {
            "beeswarm": self.plot_summary_beeswarm(save=True),
            "bar": self.plot_mean_abs_bar(save=True),
        }
        logger.info("Generated %d SHAP figures.", len(paths))
        return paths

    def get_biomarker_ranking(self) -> pd.DataFrame:
        """
        Compute signed and absolute mean SHAP per gene.

        Returns:
            DataFrame indexed by gene symbol with columns
            'mean_shap' (signed -- positive pushes toward
            positive_class_name_, negative toward
            negative_class_name_) and 'mean_abs_shap' (magnitude,
            used for the default ranking), sorted descending by
            'mean_abs_shap'.

        Raises:
            RuntimeError: If called before .fit().
        """
        self._assert_fitted()
        mean_shap = self.shap_values_.mean(axis=0)
        mean_abs_shap = np.abs(self.shap_values_).mean(axis=0)
        table = pd.DataFrame(
            {"mean_shap": mean_shap, "mean_abs_shap": mean_abs_shap},
            index=self.feature_names,
        )
        return table.sort_values("mean_abs_shap", ascending=False)

    def get_top_biomarkers(
        self,
        n_top: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the top N genes ranked by SHAP contribution.

        Args:
            n_top: Number of genes to return. Defaults to
                config.n_top_genes.
            direction: One of None, 'positive', or 'negative'.
                None ranks by mean_abs_shap (overall importance,
                regardless of direction). 'positive' ranks genes
                most strongly pushing toward
                self.positive_class_name_. 'negative' ranks genes
                most strongly pushing toward
                self.negative_class_name_.

        Returns:
            DataFrame with columns 'mean_shap', 'mean_abs_shap',
            limited to n_top rows.

        Raises:
            RuntimeError: If called before .fit().
            ValueError: If direction is not None/'positive'/
                'negative'.
        """
        self._assert_fitted()
        n = n_top if n_top is not None else self.config.n_top_genes
        table = self.get_biomarker_ranking()

        if direction is None:
            ranked = table
        elif direction == "positive":
            ranked = table.sort_values("mean_shap", ascending=False)
        elif direction == "negative":
            ranked = table.sort_values("mean_shap", ascending=True)
        else:
            raise ValueError(
                "direction must be None, 'positive', or "
                f"'negative'; got {direction!r}."
            )

        return ranked.head(n)

    def save_biomarker_table(self, path: Optional[str] = None) -> Path:
        """
        Persist the full biomarker ranking table (all genes) to
        CSV.

        Args:
            path: Destination CSV path. Defaults to
                'reports/shap_biomarker_table.csv'.

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

        self.get_biomarker_ranking().to_csv(output_path)
        logger.info("Biomarker table saved to %s", output_path)
        return output_path

    def save(self, path: str) -> None:
        """
        Persist the fitted explainer via joblib.

        Args:
            path: Destination path, e.g.
                'models/shap_explainer.joblib'.

        Raises:
            RuntimeError: If called before .fit().
        """
        self._assert_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("BiomarkerSHAPExplainer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BiomarkerSHAPExplainer":
        """Reconstruct a fitted BiomarkerSHAPExplainer from disk."""
        instance: "BiomarkerSHAPExplainer" = joblib.load(path)
        logger.info("BiomarkerSHAPExplainer loaded from %s", path)
        return instance

    # ── Private Helpers ──

    def _normalize_to_positive_class(
        self,
        raw_shap,
        n_samples: int,
        n_features: int,
    ) -> np.ndarray:
        """
        Normalize TreeExplainer output into a single, canonical 2D
        array of SHAP values toward self.positive_label_, shape
        (n_samples, n_features).

        Binary tree_output SHAP shapes vary by model family:
            - XGBClassifier (binary:logistic): a single output per
              sample already -- shap_values(X) returns one
              (n_samples, n_features) array representing
              contribution toward whichever label XGBoost's own
              encoding treats as positive (label 1, under this
              project's LabelEncoder convention).
            - RandomForestClassifier (2 classes): sklearn
              represents both class probabilities explicitly, so
              SHAP has returned either a list of 2 arrays (one per
              class) or a single 3D array with a class axis,
              depending on the installed shap/sklearn version.

        This method detects which shape it received and returns
        exactly the slice corresponding to self.positive_label_,
        regardless of which convention produced it.

        Args:
            raw_shap: Direct output of explainer.shap_values(X).
            n_samples: X.shape[0], used to disambiguate axis order.
            n_features: X.shape[1], used to disambiguate axis
                order.

        Returns:
            float32 ndarray, shape (n_samples, n_features).

        Raises:
            ValueError: If the shape cannot be confidently
                normalized.
        """
        if isinstance(raw_shap, list):
            return raw_shap[self.positive_idx_].astype(np.float32)

        arr = np.asarray(raw_shap)

        if arr.ndim == 2:
            # Single-output convention (binary XGBoost, or some
            # shap/sklearn versions for binary RF). This IS the
            # SHAP-toward-class-1 array already -- no separate
            # class index to select. Verify that assumption holds.
            if self.positive_idx_ != 1:
                logger.warning(
                    "Single-output SHAP array received, but the "
                    "positive class index is %d, not the usual 1. "
                    "Sign of these SHAP values may be inverted "
                    "relative to '%s' -- verify manually.",
                    self.positive_idx_,
                    self.positive_class_name_,
                )
            return arr.astype(np.float32)

        if arr.ndim == 3:
            if arr.shape[0] == n_samples and arr.shape[1] == n_features:
                return arr[:, :, self.positive_idx_].astype(np.float32)
            if arr.shape[1] == n_samples and arr.shape[2] == n_features:
                return arr[self.positive_idx_, :, :].astype(np.float32)

        raise ValueError(
            f"Unrecognized SHAP output shape {arr.shape} for "
            f"n_samples={n_samples}, n_features={n_features}."
        )

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
        Bound SHAP compute time by subsampling large training
        sets. Disabled by default (max_samples=None uses the full
        input matrix).
        """
        limit = self.config.max_samples
        if limit is None or X.shape[0] <= limit:
            return X

        logger.info(
            "Subsampling %d of %d samples for SHAP " "(max_samples=%d).",
            limit,
            X.shape[0],
            limit,
        )
        return X.sample(n=limit, random_state=self.config.random_state)

    def _assert_fitted(self) -> None:
        """Raise RuntimeError if .fit() has not been called yet."""
        if self.shap_values_ is None:
            raise RuntimeError(
                "BiomarkerSHAPExplainer is not fitted. Call "
                ".fit(X) before using this method."
            )

    def _render_summary_plot(
        self,
        plot_type: str,
        filename: str,
        title: str,
        save: bool,
    ) -> Optional[Path]:
        """
        Shared rendering logic for the beeswarm and bar SHAP
        plots.

        Args:
            plot_type: 'dot' for beeswarm, 'bar' for mean |SHAP|
                bars.
            filename: Filename for the saved PNG.
            title: Matplotlib figure title.
            save: If True, writes a PNG to config.figures_dir.

        Returns:
            Path to the saved PNG, or None if save=False.
        """
        self._assert_fitted()

        # plt.close("all") before and after: shap.summary_plot
        # manages its own current figure/axes internally, and
        # calling this twice in a row (beeswarm then bar) inside
        # generate_all_plots() will otherwise leak stale figure
        # state between calls.
        plt.close("all")
        shap.summary_plot(
            self.shap_values_,
            self.X_explained_,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=self.config.max_display,
            show=False,
        )
        plt.title(title)
        plt.tight_layout()

        output_path = None
        if save:
            output_path = Path(self.config.figures_dir) / filename
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Plot saved to %s", output_path)

        plt.close("all")
        return output_path


# ── Standalone Entry Point ──

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.data_ingestion import TCGADataIngester, setup_logging
    from src.feature_selection import GenomicsFeatureSelector
    from src.preprocessing import GenomicsPreprocessor

    setup_logging("INFO")

    X, y, _ = TCGADataIngester(data_dir="data/raw/").run()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Preprocessor: self-healing load, same pattern as
    # ensemble_model.py / feature_selection.py. ──
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

    # ── Feature selector: same self-healing pattern. ──
    sel_path = "models/feature_selection_pipeline.joblib"
    try:
        selector = GenomicsFeatureSelector.load(sel_path)
        X_train_sel = selector.transform(X_train_proc)
        X_test_sel = selector.transform(X_test_proc)
    except AttributeError as exc:
        logger.warning(
            "Failed to unpickle '%s' (%s). Refitting a fresh "
            "selector and overwriting the artifact.",
            sel_path,
            exc,
        )
        selector = GenomicsFeatureSelector()
        X_train_sel = selector.fit_transform(X_train_proc, y_train_enc)
        X_test_sel = selector.transform(X_test_proc)
        selector.save(sel_path)

    selected_genes = selector.get_selected_genes()

    # ── Ensemble: same self-healing pattern. Refitting here costs
    # more than the two steps above, but is still fast at ~100
    # genes / ~960 samples on the i7-1360P. ──
    model_path = "models/ensemble_model.joblib"
    try:
        model = BiomarkerEnsemble.load(model_path)
    except AttributeError as exc:
        logger.warning(
            "Failed to unpickle '%s' (%s). Refitting a fresh "
            "ensemble and overwriting the artifact.",
            model_path,
            exc,
        )
        model = BiomarkerEnsemble()
        model.fit(X_train_sel, y_train_enc)
        model.save(model_path)

    class_names = list(label_encoder.classes_)

    explainer = BiomarkerSHAPExplainer(
        ensemble=model,
        feature_names=selected_genes,
        class_names=class_names,
    )
    explainer.fit(X_train_sel)

    figure_paths = explainer.generate_all_plots()
    explainer.save_biomarker_table()
    explainer.save("models/shap_explainer.joblib")

    sep = "=" * 60
    print(f"\n{sep}")
    print("  SHAP EXPLAINABILITY -- SUMMARY")
    print(sep)
    learner = explainer.config.base_learner_key
    print(f"  Base learner explained : {learner}")
    print(f"  Positive class         : {explainer.positive_class_name_}")
    print(f"  SHAP values shape      : {explainer.shap_values_.shape}")
    print(f"  Figures generated      : {len(figure_paths)}")

    print("\n  Top 10 genes overall (by mean |SHAP|):")
    top_10 = explainer.get_top_biomarkers(n_top=10)
    for gene, row in top_10.iterrows():
        print(
            f"    {gene:<20} mean|SHAP|={row['mean_abs_shap']:.4f} "
            f"| mean_SHAP={row['mean_shap']:+.4f}"
        )

    print("\n  Top 5 genes pushing toward " f"{explainer.positive_class_name_}:")
    top_positive = explainer.get_top_biomarkers(n_top=5, direction="positive")
    for gene, row in top_positive.iterrows():
        print(f"    {gene:<20} mean_SHAP={row['mean_shap']:+.4f}")

    print(sep)
    print("  Saved : reports/figures/*.png")
    print("  Saved : reports/shap_biomarker_table.csv")
    print("  Saved : models/shap_explainer.joblib")
    print(sep)
