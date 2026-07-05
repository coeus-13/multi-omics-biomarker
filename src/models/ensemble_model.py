"""
ensemble_model.py
=================
Production-ready Stacking Ensemble for PAM50 breast cancer subtype
classification using TCGA-BRCA RNA-Seq features.

Implements a three-layer stacking architecture:
    Layer 1 (Base Learners) : RandomForestClassifier + XGBClassifier
    Layer 2 (Meta-Learner)  : LogisticRegression (trained on OOF preds)
    Orchestrator            : BiomarkerEnsemble — fit, predict,
                              evaluate, tune, and joblib serialization.

WHY STACKING OVER A SINGLE MODEL FOR THIS GENOMIC DATASET?
-----------------------------------------------------------
1. COMPLEMENTARY INDUCTIVE BIASES
   Random Forest partitions feature space via axis-aligned splits on
   random gene subsets; it is intrinsically robust to the co-expressed
   gene modules endemic to RNA-Seq data. XGBoost builds trees
   sequentially, correcting prior residuals; it exploits the strongest
   individual gene-subtype signals more aggressively. Their per-sample
   error patterns are partially uncorrelated — the necessary condition
   for stacking to outperform either member alone.

2. VARIANCE REDUCTION IN HIGH DIMENSIONS
   At p/n ≈ 0.57 (500 genes, ~880 training samples), a single deep
   model is high-variance. Stacking's Out-Of-Fold (OOF) protocol
   forces the meta-learner to generalize from held-out base-learner
   predictions, acting as a structural regularizer without dropout
   or explicit weight decay.

3. LOGISTIC REGRESSION AS META-LEARNER
   A deliberately simple meta-learner prevents Layer 2 overfitting.
   Its L2 penalty (C) governs how much weight each base learner's
   probability column receives. A large weight on the XGB column is
   directly interpretable: XGBoost's probability calibration was more
   reliable for those PAM50 classes.

4. NATIVE SKLEARN COMPATIBILITY
   StackingClassifier integrates natively with RandomizedSearchCV,
   cross_val_score, and joblib — no custom training loops needed.

Integration Contract:
    Consumes : GenomicsFeatureSelector.transform() output — float32
               DataFrame, shape (n_samples, 500 genes).
               Integer-encoded y from encode_pam50_labels().
    Produces : Fitted BiomarkerEnsemble exposing predict /
               predict_proba / evaluate / save interface.
    Used by  : src/explainability/shap_explainer.py (via .stack_)
               deployment/api/routers/predict.py (predict_proba)

Usage:
    >>> from src.models.ensemble_model import BiomarkerEnsemble
    >>> model = BiomarkerEnsemble()
    >>> model.fit(X_train_sel, y_train_enc)
    >>> report = model.evaluate(X_test_sel, y_test_enc, class_names)
    >>> model.save("models/ensemble_model.joblib")

Author : [Your Name]
Date   : [Project Date]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EnsembleConfig:
    """
    Typed, injectable configuration for BiomarkerEnsemble.

    Follows the project-wide Config pattern: every parameter is explicit,
    IDE-autocompletable, and mockable in tests by injecting a config with
    reduced estimator counts (e.g. rf_n_estimators=5) to cut fixture
    runtime to under one second.

    Attributes:
        rf_n_estimators     : Trees in the Random Forest. 500 balances
                              variance reduction with i7-1360P RAM (~200 MB
                              for 500 trees on 500 features).
        rf_max_depth        : None = fully grown trees. Set 10-20 to
                              trade recall for speed in development.
        rf_min_samples_leaf : Leaf size floor. 2 provides mild implicit
                              regularization at no tuning cost.
        xgb_n_estimators    : Boosting rounds. 300 at lr=0.05 is a
                              conservative, well-generalizing default.
        xgb_max_depth       : Tree depth. 6 is the XGBoost default;
                              genomic data rarely benefits from deeper.
        xgb_learning_rate   : Step-size shrinkage. 0.05 sacrifices
                              speed for generalization vs the 0.3 default.
        xgb_subsample       : Row subsampling per round. 0.8 adds
                              stochastic regularization.
        xgb_colsample_bytree: Gene subsampling per tree. 0.8 mirrors
                              Random Forest's random subspace method.
        meta_C              : LR inverse regularization. Smaller C =
                              stronger L2 penalty on Layer 2 weights.
        cv_folds            : Folds for OOF stacking and search CV.
        random_state        : Global reproducibility seed.
        n_jobs              : Workers for RF and XGBoost internals.
                              -1 = all P-cores. Use -2 to leave one free
                              for OS tasks during long training runs.
        tune_n_iter         : RandomizedSearchCV candidate count. 30
                              balances exploration with i7-1360P budget
                              (~15-25 min with n_jobs=-1 per base learner).
    """

    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 2
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    meta_C: float = 1.0
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    tune_n_iter: int = 30

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2; got {self.cv_folds}.")
        if not 0.0 < self.xgb_subsample <= 1.0:
            raise ValueError(
                "xgb_subsample must be in (0, 1]; " f"got {self.xgb_subsample}."
            )
        if not 0.0 < self.xgb_colsample_bytree <= 1.0:
            raise ValueError(
                "xgb_colsample_bytree must be in (0, 1]; "
                f"got {self.xgb_colsample_bytree}."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator Class
# ──────────────────────────────────────────────────────────────────────────────


class BiomarkerEnsemble:
    """
    Orchestrates training, evaluation, tuning, and serialization of a
    3-model Stacking Ensemble for PAM50 subtype prediction.

    The internal StackingClassifier is exposed as .stack_ after fitting,
    making this class compatible with any sklearn utility that expects a
    fitted classifier: SHAP TreeExplainer, cross_val_score, permutation
    importance, etc.

    Attributes:
        config : EnsembleConfig controlling all hyperparameters.
        stack_ : Fitted StackingClassifier. Set by .fit() or .tune().
                 Access individual learners via:
                     .stack_.named_estimators_["rf"]
                     .stack_.named_estimators_["xgb"]
                 Both are required by SHAP's TreeExplainer.
        cv_    : Shared StratifiedKFold for OOF stacking and
                 RandomizedSearchCV, ensuring consistent fold assignments
                 across the full pipeline.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        """
        Args:
            config: EnsembleConfig instance. Uses class defaults if None.
        """
        self.config: EnsembleConfig = config or EnsembleConfig()
        self.stack_: Optional[StackingClassifier] = None
        self.cv_ = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        logger.info(
            "BiomarkerEnsemble initialized | RF n_est=%d | "
            "XGB n_est=%d | meta_C=%.2f | cv=%d-fold",
            self.config.rf_n_estimators,
            self.config.xgb_n_estimators,
            self.config.meta_C,
            self.config.cv_folds,
        )

    # ── Private Builder Methods ───────────────────────────────────────────────

    def _build_rf(self) -> RandomForestClassifier:
        """
        Construct the Random Forest base learner.

        class_weight='balanced' scales sample weights inversely
        proportional to PAM50 class frequencies — critical for the
        minority Her2 (~13%) and Normal (~7%) subtypes without the
        added complexity of SMOTE oversampling.
        """
        return RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            class_weight="balanced",
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )

    def _build_xgb(self) -> XGBClassifier:
        """
        Construct the XGBoost base learner.

        tree_method='hist' uses histogram-based approximate splitting,
        which is significantly faster than exact splitting on CPU-only
        hardware (i7-1360P) and has a lower peak memory footprint. This
        is the recommended method for all non-GPU XGBoost fits as of
        XGBoost 2.x.

        eval_metric='mlogloss' is set explicitly to suppress XGBoost's
        default stderr warnings about unspecified metrics. verbosity=0
        suppresses per-round training output that would flood the log
        during the 150 fits inside RandomizedSearchCV.
        """
        return XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbosity=0,
        )

    def _build_meta(self) -> LogisticRegression:
        """
        Construct the Logistic Regression meta-learner.

        solver='lbfgs' handles the 5-class objective natively and
        converges fast on the small OOF prediction matrix (n_train x 10
        cols: 5 RF proba + 5 XGB proba) that the meta-learner trains on.

        max_iter=1000 prevents ConvergenceWarning on the small OOF
        matrix without meaningful compute cost.

        Note: multi_class is omitted — deprecated in sklearn 1.5.
        lbfgs defaults to multinomial for problems with > 2 classes.
        """
        return LogisticRegression(
            C=self.config.meta_C,
            solver="lbfgs",
            max_iter=1000,
        )

    def build_stack(self) -> StackingClassifier:
        """
        Construct an unfitted StackingClassifier from the three learners.

        stack_method='predict_proba' passes each base learner's full
        class-probability vector (5 columns) to the meta-learner rather
        than hard integer predictions. This gives the meta-learner
        calibrated uncertainty signals per PAM50 class — a richer input
        than a single argmax label.

        passthrough=False keeps the meta-learner's input to exactly 10
        features (5 RF proba + 5 XGB proba). Setting passthrough=True
        would append the original 500 gene columns, risking Layer 2
        overfitting on a dataset this size.

        n_jobs=1 at StackingClassifier level prevents CPU over-
        subscription: each base learner already uses n_jobs=-1
        internally. Running the outer OOF folds in parallel on top
        would degrade performance on the i7-1360P.
        """
        return StackingClassifier(
            estimators=[
                ("rf", self._build_rf()),
                ("xgb", self._build_xgb()),
            ],
            final_estimator=self._build_meta(),
            stack_method="predict_proba",
            cv=self.cv_,
            passthrough=False,
            n_jobs=1,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> "BiomarkerEnsemble":
        """
        Fit the Stacking Ensemble on training data.

        StackingClassifier internally runs a 5-fold OOF split on
        (X_train, y_train) to generate the meta-learner's training data.
        This is NOT an additional train/test split of your data — it is
        entirely contained within the training fold you pass in.

        After OOF meta-training, StackingClassifier re-fits each base
        learner on the full X_train before returning, so inference uses
        models trained on all available training data.

        Args:
            X_train : float32 DataFrame, shape (n_train, 500 genes).
                      Must be output of GenomicsFeatureSelector.transform().
            y_train : Integer-encoded PAM50 labels, shape (n_train,).
                      Must be output of encode_pam50_labels().

        Returns:
            self — enables chaining: model.fit(X, y).evaluate(...)

        Raises:
            TypeError : If X_train is not a pandas DataFrame.
            ValueError: If X_train and y_train sample counts differ.
        """
        self._validate_inputs(X_train, y_train, context="fit")
        logger.info(
            "BiomarkerEnsemble.fit() START | X: %s | classes: %s",
            X_train.shape,
            np.unique(y_train).tolist(),
        )
        self.stack_ = self.build_stack()
        self.stack_.fit(X_train, y_train)
        logger.info("BiomarkerEnsemble.fit() COMPLETE")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate hard integer PAM50 class predictions.

        Args:
            X: float32 DataFrame, same 500-gene schema as X_train.

        Returns:
            Integer array, shape (n_samples,). Inverse-transform with
            the LabelEncoder from encode_pam50_labels() to recover
            PAM50 subtype name strings for reporting.

        Raises:
            RuntimeError: If called before .fit() or .tune().
        """
        self._assert_fitted()
        return self.stack_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class probability estimates for each PAM50 subtype.

        Returns the meta-learner's (Logistic Regression) softmax
        probability outputs. Column order matches .stack_.classes_.

        These probabilities are what the FastAPI deployment endpoint
        returns to callers, and what SHAP's TreeExplainer uses to
        compute per-gene attributions for each PAM50 class.

        Args:
            X: float32 DataFrame, same 500-gene schema as X_train.

        Returns:
            float64 array, shape (n_samples, 5). Verify class order with:
                model.stack_.classes_

        Raises:
            RuntimeError: If called before .fit() or .tune().
        """
        self._assert_fitted()
        return self.stack_.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Generate a structured evaluation report on held-out test data.

        Three metrics chosen specifically for PAM50 imbalanced multi-
        class classification:

        1. Macro-F1 : Weights all 5 subtypes equally, penalizing poor
           recall on minority classes (Her2, Normal) as harshly as
           majority classes (LumA). This is the primary headline metric
           for this project — the number that goes on your resume.

        2. Cohen's Kappa : Chance-corrected agreement. More conservative
           than accuracy on imbalanced data, and expected by clinical
           validation and regulatory frameworks.

        3. Per-class report : Precision, Recall, F1 per named PAM50
           class — the consulting deliverable that maps model performance
           to clinically meaningful subtype categories.

        Args:
            X_test     : float32 DataFrame, shape (n_test, 500 genes).
            y_test     : Integer-encoded ground-truth PAM50 labels.
            class_names: PAM50 subtype strings in the same order as
                         LabelEncoder.classes_.
                         e.g. ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']

        Returns:
            Dict with keys:
                'macro_f1'    : float — primary optimization metric.
                'cohen_kappa' : float — chance-corrected agreement.
                'report_str'  : str   — human-readable per-class table.
                'report_dict' : dict  — machine-readable per-class dict.
                'y_pred'      : np.ndarray — integer class predictions.
                'y_proba'     : np.ndarray — (n_test, 5) probabilities.

        Raises:
            RuntimeError: If called before .fit() or .tune().
        """
        self._assert_fitted()

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)
        report_str = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
        )
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True,
        )

        logger.info(
            "Evaluation | Macro-F1: %.4f | Cohen Kappa: %.4f",
            macro_f1,
            kappa,
        )
        logger.info("Per-class report:\n%s", report_str)

        return {
            "macro_f1": macro_f1,
            "cohen_kappa": kappa,
            "report_str": report_str,
            "report_dict": report_dict,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> "BiomarkerEnsemble":
        """
        Run RandomizedSearchCV over the full Stacking Ensemble.

        WHY RANDOMIZED OVER GRID SEARCH?
        RandomizedSearchCV samples n_iter random combinations rather
        than enumerating all candidates. On the i7-1360P, 30 iters x
        5 folds = 150 full ensemble fits (~15-25 min). An equivalent
        GridSearchCV over the same axes requires thousands of fits and
        is impractical on CPU-only hardware.

        JOINT TUNING RATIONALE:
        Base learner hyperparameters are searched jointly with meta-
        learner C. Tuning them independently introduces selection bias:
        the independently-optimal RF may not pair well with the
        independently-optimal XGBoost when their OOF predictions are
        combined. Joint search avoids this.

        After this call, .stack_ is replaced with the best estimator,
        already re-fitted on full (X_train, y_train) via refit=True.
        No additional .fit() call is needed.

        Args:
            X_train : float32 DataFrame, shape (n_train, 500 genes).
            y_train : Integer-encoded PAM50 labels, shape (n_train,).

        Returns:
            self — .stack_ updated in place with the best estimator.

        Raises:
            TypeError : If X_train is not a pandas DataFrame.
            ValueError: If X_train and y_train sample counts differ.
        """
        self._validate_inputs(X_train, y_train, context="tune")

        param_distributions: Dict[str, Any] = {
            "rf__n_estimators": randint(100, 600),
            "rf__max_depth": [None, 5, 10, 15, 20],
            "rf__min_samples_leaf": randint(1, 8),
            "xgb__n_estimators": randint(100, 400),
            "xgb__max_depth": randint(3, 9),
            "xgb__learning_rate": uniform(0.01, 0.29),
            "xgb__subsample": uniform(0.6, 0.4),
            "final_estimator__C": uniform(0.01, 9.99),
        }

        search = RandomizedSearchCV(
            estimator=self.build_stack(),
            param_distributions=param_distributions,
            n_iter=self.config.tune_n_iter,
            scoring="f1_macro",
            cv=self.cv_,
            refit=True,
            n_jobs=1,
            verbose=1,
            random_state=self.config.random_state,
        )

        logger.info(
            "RandomizedSearchCV START | n_iter=%d | cv=%d-fold | " "scoring=f1_macro",
            self.config.tune_n_iter,
            self.config.cv_folds,
        )
        search.fit(X_train, y_train)
        self.stack_ = search.best_estimator_

        logger.info(
            "RandomizedSearchCV COMPLETE | Best macro-F1 (CV): %.4f",
            search.best_score_,
        )
        logger.info("Best params: %s", search.best_params_)
        return self

    def get_rf_feature_importances(
        self,
        feature_names: List[str],
    ) -> pd.Series:
        """
        Extract per-gene Gini importance scores from the fitted RF.

        Useful as a fast first-pass sanity check against SHAP: if the
        top genes by RF importance and the top genes by mean |SHAP|
        overlap significantly, the model has learned consistent signal
        rather than overfitting noise.

        IMPORTANT DISTINCTION FROM SHAP:
        RF importances average Gini impurity reduction across all trees.
        They spread importance across co-expressed gene groups rather
        than crediting the dominant gene. SHAP values are sample-level
        Shapley attributions — more reliable for biomarker discovery.
        Use this for fast iteration; use shap_explainer.py for the
        final publishable biomarker evidence.

        Args:
            feature_names: 500 gene-symbol strings in the exact column
                           order of X_train. Use
                           GenomicsFeatureSelector.get_selected_genes().

        Returns:
            pd.Series, index=gene symbols, values=float importance,
            sorted descending. Sums to 1.0 across all 500 genes.

        Raises:
            RuntimeError: If called before .fit() or .tune().
            ValueError  : If len(feature_names) does not match the
                          model's n_features.
        """
        self._assert_fitted()
        rf: RandomForestClassifier = self.stack_.named_estimators_["rf"]
        importances = rf.feature_importances_

        if len(feature_names) != len(importances):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not "
                f"match the model's n_features ({len(importances)}). "
                "Pass GenomicsFeatureSelector.get_selected_genes()."
            )

        return pd.Series(
            importances,
            index=feature_names,
            name="rf_importance",
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """
        Serialize the full BiomarkerEnsemble to disk via joblib.

        Saves the entire instance — config, cv_, and the fitted
        StackingClassifier — so the FastAPI service can load and call
        .predict_proba() without re-constructing any component. This
        prevents train/serve configuration skew.

        Args:
            path: Destination path, e.g. 'models/ensemble_model.joblib'

        Raises:
            RuntimeError: If called before .fit() or .tune().
        """
        self._assert_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("BiomarkerEnsemble saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BiomarkerEnsemble":
        """
        Reconstruct a BiomarkerEnsemble from a saved .joblib artifact.

        Loads the full instance (not just the inner StackingClassifier),
        so .config, .cv_, and all public methods are immediately
        available without re-construction.

        Args:
            path: Path to a .joblib file written by .save().

        Returns:
            BiomarkerEnsemble with .stack_ populated and ready for
            .predict(), .predict_proba(), and .evaluate() calls.
        """
        instance: "BiomarkerEnsemble" = joblib.load(path)
        logger.info("BiomarkerEnsemble loaded from %s", path)
        return instance

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        """
        Raise RuntimeError if .fit() or .tune() has not been called.

        Uses an explicit None-check rather than sklearn's check_is_fitted
        to match the pattern in GenomicsPreprocessor and
        GenomicsFeatureSelector, keeping the module's dependency surface
        minimal and the error message specific and actionable.
        """
        if self.stack_ is None:
            raise RuntimeError(
                "BiomarkerEnsemble is not fitted. "
                "Call .fit() or .tune() before using this method."
            )

    @staticmethod
    def _validate_inputs(
        X: pd.DataFrame,
        y: np.ndarray,
        context: str,
    ) -> None:
        """
        Validate X and y before fit() or tune().

        Args:
            X      : Feature DataFrame to validate.
            y      : Integer label array to validate.
            context: 'fit' or 'tune' — inserted into error messages.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"BiomarkerEnsemble.{context}() requires a pandas "
                f"DataFrame for X; got {type(X)}. "
                "Pass GenomicsFeatureSelector.transform() output."
            )
        if len(X) != len(y):
            raise ValueError(
                "X and y have different sample counts in "
                f"BiomarkerEnsemble.{context}(): "
                f"{len(X)} vs {len(y)}."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Entry Point
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    """
    Standalone execution for testing from PyCharm's Run button.

    Configure PyCharm Run/Debug:
        Script           : src/models/ensemble_model.py
        Working directory: <project root>

    Loads saved preprocessing and feature-selection artifacts to avoid
    re-running the full ingestion chain on every model iteration.
    Saves ~3-4 minutes per run on the i7-1360P.
    """
    from sklearn.model_selection import train_test_split

    from src.data_ingestion import TCGADataIngester, setup_logging
    from src.feature_selection import GenomicsFeatureSelector
    from src.preprocessing import GenomicsPreprocessor, encode_pam50_labels

    setup_logging("INFO")

    # ── Ingest ────────────────────────────────────────────────────────────────
    X, y, _ = TCGADataIngester(data_dir="data/raw/").run()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Preprocess (load artifact if available) ───────────────────────────────
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

    # ── Encode labels ─────────────────────────────────────────────────────────
    y_train_enc, label_encoder = encode_pam50_labels(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # ── Feature selection (load artifact if available) ────────────────────────
    sel_path = "models/feature_selection_pipeline.joblib"
    if Path(sel_path).exists():
        selector = GenomicsFeatureSelector.load(sel_path)
        X_train_sel = selector.transform(X_train_proc)
        X_test_sel = selector.transform(X_test_proc)
    else:
        selector = GenomicsFeatureSelector()
        X_train_sel = selector.fit_transform(X_train_proc, y_train_enc)
        X_test_sel = selector.transform(X_test_proc)
        selector.save(sel_path)

    selected_genes = selector.get_selected_genes()

    # ── Train base model ──────────────────────────────────────────────────────
    model = BiomarkerEnsemble()
    model.fit(X_train_sel, y_train_enc)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    class_names = list(label_encoder.classes_)
    report = model.evaluate(X_test_sel, y_test_enc, class_names)

    # Optional: uncomment to run HPO (~15-25 min on i7-1360P)
    # model.tune(X_train_sel, y_train_enc)
    # report = model.evaluate(X_test_sel, y_test_enc, class_names)

    # ── RF importances (fast cross-reference against SHAP) ────────────────────
    importances = model.get_rf_feature_importances(selected_genes)

    # ── Persist ───────────────────────────────────────────────────────────────
    model.save("models/ensemble_model.joblib")

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  ENSEMBLE MODEL — SUMMARY")
    print(sep)
    print(f"  Macro-F1 (test) : {report['macro_f1']:.4f}")
    print(f"  Cohen Kappa     : {report['cohen_kappa']:.4f}")
    print("\n  Per-class F1:")
    for cls in class_names:
        f1 = report["report_dict"][cls]["f1-score"]
        print(f"    {cls:<10} : {f1:.4f}")
    print("\n  Top 5 genes by RF importance:")
    for gene, score in importances.head(5).items():
        print(f"    {gene:<20} : {score:.6f}")
    print(sep)
    print("  Saved : models/ensemble_model.joblib")
    print("  Next  : src/explainability/shap_explainer.py")
    print(sep)
