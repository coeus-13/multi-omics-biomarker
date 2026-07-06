"""
ensemble_model.py
=================
Stacking Ensemble for binary vital-status (Alive vs Dead)
classification using the selected biomarker panel.

Layer 1 (Base Learners) : RandomForestClassifier + XGBClassifier
Layer 2 (Meta-Learner)  : LogisticRegression on OOF predictions
Orchestrator            : BiomarkerEnsemble

WHY STACKING FOR THIS TARGET
-------------------------------
1. COMPLEMENTARY INDUCTIVE BIASES
   Random Forest splits on random gene subsets; XGBoost builds
   trees sequentially, correcting prior residuals. Their per-sample
   error patterns are partially uncorrelated -- the condition
   stacking needs to outperform either member alone.

2. DIMENSIONALITY IS NO LONGER THE DRIVING CONCERN
   At n_kbest_genes=100 and roughly 960 training samples, p/n is
   about 0.10 -- a comfortable regime, not the p>>n variance-
   control problem the earlier 500-gene version faced. Stacking is
   kept here for the complementary-bias argument above, not as a
   variance-reduction necessity.

3. scale_pos_weight COMPUTED FROM DATA, NOT FIXED
   Vital-status cohorts are typically imbalanced (more survivors
   than deaths within a given follow-up window). XGBClassifier has
   no class_weight parameter for binary targets; scale_pos_weight
   is its documented mechanism, computed fresh from y_train on
   every fit() call rather than hardcoded.

4. LOGISTIC REGRESSION AS META-LEARNER
   A deliberately simple meta-learner combines the two base
   learners' calibrated P(Dead) estimates without adding meaningful
   overfitting risk of its own.

A NOTE ON WHAT THIS MODEL DOES AND DOES NOT ANSWER
-------------------------------------------------------
This predicts vital status as recorded, without accounting for
follow-up duration or censoring. It is a legitimate binary
classification task, but it is not survival analysis -- a patient
alive at 6 months and one alive at 10 years are treated identically.
If presenting this project, be ready to discuss time-to-event
modeling (Cox proportional hazards, Random Survival Forests) as the
statistically complete next step.

Integration Contract:
    Consumes : GenomicsFeatureSelector.transform() output -- float32
               DataFrame, shape (n_samples, ~100 genes). Integer-
               encoded y from the fitted LabelEncoder (0=Alive,
               1=Dead under default alphabetical ordering -- verify
               via label_encoder.classes_).
    Produces : Fitted BiomarkerEnsemble exposing predict /
               predict_proba / evaluate / save.

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
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ── Configuration Dataclass ──


@dataclass
class EnsembleConfig:
    """
    Typed, injectable configuration for BiomarkerEnsemble.

    Attributes:
        rf_n_estimators      : Trees in the Random Forest.
        rf_max_depth         : None = fully grown trees.
        rf_min_samples_leaf  : Leaf size floor.
        xgb_n_estimators     : Boosting rounds.
        xgb_max_depth        : Tree depth.
        xgb_learning_rate    : Step-size shrinkage.
        xgb_subsample        : Row subsampling per round.
        xgb_colsample_bytree : Gene subsampling per tree.
        meta_C               : LR inverse regularization strength.
        cv_folds             : Folds for OOF stacking and search CV.
        random_state         : Global reproducibility seed.
        n_jobs               : Workers for RF/XGB internals. -1 =
                                all cores.
        tune_n_iter          : RandomizedSearchCV candidate count.
                                At ~100 genes / ~960 samples, a full
                                tuning run now completes in well
                                under a minute on the i7-1360P.
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
                "xgb_subsample must be in (0, 1]; got " f"{self.xgb_subsample}."
            )
        if not 0.0 < self.xgb_colsample_bytree <= 1.0:
            raise ValueError(
                "xgb_colsample_bytree must be in (0, 1]; got "
                f"{self.xgb_colsample_bytree}."
            )


# ── Orchestrator Class ──


class BiomarkerEnsemble:
    """
    Orchestrates training, evaluation, tuning, and serialization of
    a 3-model Stacking Ensemble for binary vital-status prediction.

    Attributes:
        config : EnsembleConfig controlling all hyperparameters.
        stack_ : Fitted StackingClassifier. Set by .fit() or
                 .tune(). Access individual learners via
                 .stack_.named_estimators_["rf"] / ["xgb"] -- both
                 required by SHAP's TreeExplainer later.
        cv_    : Shared StratifiedKFold for OOF stacking and
                 RandomizedSearchCV.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config: EnsembleConfig = config or EnsembleConfig()
        self.stack_: Optional[StackingClassifier] = None
        self.cv_ = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        logger.info(
            "BiomarkerEnsemble initialized | RF n_est=%d | XGB "
            "n_est=%d | meta_C=%.2f | cv=%d-fold",
            self.config.rf_n_estimators,
            self.config.xgb_n_estimators,
            self.config.meta_C,
            self.config.cv_folds,
        )

    # ── Private Builder Methods ──

    def _build_rf(self) -> RandomForestClassifier:
        """
        class_weight='balanced' scales sample weights inversely
        proportional to class frequency -- RandomForestClassifier's
        native mechanism for imbalanced binary targets.
        """
        return RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            class_weight="balanced",
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )

    def _build_xgb(self, scale_pos_weight: float = 1.0) -> XGBClassifier:
        """
        tree_method='hist' is the recommended CPU-only setting for
        XGBoost 2.x. scale_pos_weight is XGBoost's own mechanism for
        imbalanced binary targets -- unlike RandomForestClassifier,
        XGBClassifier has no class_weight parameter, so this is
        computed by the caller from y_train and passed in per fit()
        call rather than fixed at construction time.
        """
        return XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbosity=0,
        )

    def _build_meta(self) -> LogisticRegression:
        """
        Binary logistic regression on the two base learners' OOF
        P(Dead) columns. max_iter=1000 avoids ConvergenceWarning on
        the small OOF matrix at negligible compute cost.
        """
        return LogisticRegression(C=self.config.meta_C, solver="lbfgs", max_iter=1000)

    def build_stack(self, scale_pos_weight: float = 1.0) -> StackingClassifier:
        """
        Construct an unfitted StackingClassifier.

        stack_method='predict_proba' passes each base learner's
        full probability vector to the meta-learner. passthrough=
        False keeps the meta-learner's input to exactly 4 features
        (2 RF + 2 XGB proba columns), avoiding Layer 2 overfitting.
        n_jobs=1 here (each base learner already uses n_jobs=-1
        internally) avoids CPU over-subscription.
        """
        return StackingClassifier(
            estimators=[
                ("rf", self._build_rf()),
                ("xgb", self._build_xgb(scale_pos_weight)),
            ],
            final_estimator=self._build_meta(),
            stack_method="predict_proba",
            cv=self.cv_,
            passthrough=False,
            n_jobs=1,
        )

    @staticmethod
    def _compute_scale_pos_weight(y: np.ndarray) -> float:
        """
        Compute scale_pos_weight = n_negative / n_positive.

        Computed fresh from y on every fit()/tune() call rather
        than hardcoded, so the model adapts automatically if class
        balance shifts (e.g. a different cohort). Falls back to 1.0
        (no reweighting) if y is not binary.

        Args:
            y: Integer-encoded labels.

        Returns:
            float ratio, or 1.0 if y does not have exactly 2
            classes.
        """
        counts = np.bincount(y)
        if len(counts) != 2 or counts[1] == 0:
            return 1.0
        return float(counts[0] / counts[1])

    # ── Public API ──

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "BiomarkerEnsemble":
        """
        Fit the Stacking Ensemble on training data.

        Args:
            X_train : float32 DataFrame, shape (n_train, n_genes).
            y_train : Integer-encoded labels, shape (n_train,).

        Returns:
            self

        Raises:
            TypeError: If X_train is not a pandas DataFrame.
            ValueError: If X_train/y_train sample counts differ.
        """
        self._validate_inputs(X_train, y_train, context="fit")
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        logger.info(
            "BiomarkerEnsemble.fit() START | X: %s | classes: %s "
            "| scale_pos_weight=%.3f",
            X_train.shape,
            np.unique(y_train).tolist(),
            scale_pos_weight,
        )
        self.stack_ = self.build_stack(scale_pos_weight)
        self.stack_.fit(X_train, y_train)
        logger.info("BiomarkerEnsemble.fit() COMPLETE")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate hard integer predictions (0=Alive, 1=Dead)."""
        self._assert_fitted()
        return self.stack_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class probability estimates.

        Returns:
            float64 array, shape (n_samples, 2). Column order
            matches self.stack_.classes_ -- verify before assuming
            column 1 is P(Dead).
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
        Generate a structured evaluation report on held-out data.

        ROC-AUC and PR-AUC are the headline metrics for this binary
        clinical-outcome target: both are threshold-independent, and
        PR-AUC in particular stays informative under the class
        imbalance vital-status cohorts typically show. Macro-F1 and
        Cohen's Kappa are retained as supporting metrics.

        Both metrics are computed against the model's own
        self.stack_.classes_ ordering rather than assuming column 1
        of predict_proba is always the positive class -- this stays
        correct even if a future dataset encodes the positive class
        as 0 instead of 1.

        Args:
            X_test      : float32 DataFrame, shape (n_test, n_genes).
            y_test      : Integer-encoded ground-truth labels.
            class_names : Labels in the same order as
                          label_encoder.classes_, e.g.
                          ['Alive', 'Dead'].

        Returns:
            Dict with keys 'roc_auc', 'pr_auc', 'macro_f1',
            'cohen_kappa', 'report_str', 'report_dict', 'y_pred',
            'y_proba'. 'roc_auc'/'pr_auc' are always present as dict
            keys; their values are None only if the model was not
            fitted on exactly 2 classes.

        Raises:
            RuntimeError: If called before .fit() or .tune().
        """
        self._assert_fitted()

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        roc_auc: Optional[float] = None
        pr_auc: Optional[float] = None
        model_classes = list(self.stack_.classes_)

        if len(model_classes) == 2:
            positive_label = max(model_classes)
            positive_idx = model_classes.index(positive_label)
            y_proba_positive = y_proba[:, positive_idx]
            roc_auc = roc_auc_score(y_test, y_proba_positive)
            pr_auc = average_precision_score(y_test, y_proba_positive)
        else:
            logger.warning(
                "evaluate() expected a binary model for ROC-AUC/"
                "PR-AUC; found %d classes on self.stack_. Skipping "
                "those two metrics.",
                len(model_classes),
            )

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)
        report_str = classification_report(
            y_test, y_pred, target_names=class_names, digits=4
        )
        report_dict = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )

        logger.info(
            "Evaluation | ROC-AUC: %s | PR-AUC: %s | Macro-F1: "
            "%.4f | Cohen Kappa: %.4f",
            f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
            f"{pr_auc:.4f}" if pr_auc is not None else "N/A",
            macro_f1,
            kappa,
        )
        logger.info("Per-class report:\n%s", report_str)

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "macro_f1": macro_f1,
            "cohen_kappa": kappa,
            "report_str": report_str,
            "report_dict": report_dict,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    def tune(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "BiomarkerEnsemble":
        """
        Run RandomizedSearchCV over the full Stacking Ensemble.

        scale_pos_weight is computed once from y_train and fixed
        (not tuned) -- XGBoost's own documentation recommends
        sum(negative)/sum(positive) directly rather than treating it
        as a free hyperparameter.

        After this call, .stack_ holds the best estimator, already
        refit on the full (X_train, y_train) via refit=True.

        Args:
            X_train : float32 DataFrame, shape (n_train, n_genes).
            y_train : Integer-encoded labels, shape (n_train,).

        Returns:
            self

        Raises:
            TypeError: If X_train is not a pandas DataFrame.
            ValueError: If X_train/y_train sample counts differ.
        """
        self._validate_inputs(X_train, y_train, context="tune")
        scale_pos_weight = self._compute_scale_pos_weight(y_train)

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
            estimator=self.build_stack(scale_pos_weight),
            param_distributions=param_distributions,
            n_iter=self.config.tune_n_iter,
            scoring="roc_auc",
            cv=self.cv_,
            refit=True,
            n_jobs=1,
            verbose=1,
            random_state=self.config.random_state,
        )

        logger.info(
            "RandomizedSearchCV START | n_iter=%d | cv=%d-fold | "
            "scoring=roc_auc | scale_pos_weight=%.3f",
            self.config.tune_n_iter,
            self.config.cv_folds,
            scale_pos_weight,
        )
        search.fit(X_train, y_train)
        self.stack_ = search.best_estimator_

        logger.info(
            "RandomizedSearchCV COMPLETE | Best ROC-AUC (CV): %.4f",
            search.best_score_,
        )
        logger.info("Best params: %s", search.best_params_)
        return self

    def get_rf_feature_importances(self, feature_names: List[str]) -> pd.Series:
        """
        Extract per-gene Gini importance from the fitted RF -- a
        fast sanity check against SHAP later.

        Args:
            feature_names: Gene symbols in X_train's column order.
                Use GenomicsFeatureSelector.get_selected_genes().

        Raises:
            RuntimeError: If called before .fit() or .tune().
            ValueError: If feature_names length doesn't match.
        """
        self._assert_fitted()
        rf: RandomForestClassifier = self.stack_.named_estimators_["rf"]
        importances = rf.feature_importances_

        if len(feature_names) != len(importances):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) "
                "does not match model n_features "
                f"({len(importances)})."
            )

        return pd.Series(
            importances, index=feature_names, name="rf_importance"
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """
        Serialize the full BiomarkerEnsemble instance via joblib.

        Raises:
            RuntimeError: If called before .fit() or .tune().
        """
        self._assert_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("BiomarkerEnsemble saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BiomarkerEnsemble":
        """Reconstruct a fitted BiomarkerEnsemble from disk."""
        instance: "BiomarkerEnsemble" = joblib.load(path)
        logger.info("BiomarkerEnsemble loaded from %s", path)
        return instance

    # ── Private Helpers ──

    def _assert_fitted(self) -> None:
        if self.stack_ is None:
            raise RuntimeError(
                "BiomarkerEnsemble is not fitted. Call .fit() or "
                ".tune() before using this method."
            )

    @staticmethod
    def _validate_inputs(X: pd.DataFrame, y: np.ndarray, context: str) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"BiomarkerEnsemble.{context}() requires a pandas "
                f"DataFrame for X; got {type(X)}."
            )
        if len(X) != len(y):
            raise ValueError(
                "X and y have different sample counts in "
                f"BiomarkerEnsemble.{context}(): {len(X)} vs "
                f"{len(y)}."
            )


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

    # ── Preprocessor: same __main__ pickle trap as the selector
    # below -- Log2FPKMTransformer/LowExpressionFilter are also
    # custom classes defined in preprocessing.py, and that file
    # was last run directly. Self-healed proactively here. ──
    preproc_path = "models/preprocessing_pipeline.joblib"
    try:
        preprocessor = GenomicsPreprocessor.load(preproc_path)
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
    except AttributeError as exc:
        logger.warning(
            "Failed to unpickle '%s' (%s). Classic joblib/"
            "__main__ module-path trap: a custom transformer "
            "class was pickled under module '__main__' because "
            "preprocessing.py was last run directly. Refitting a "
            "fresh preprocessor and overwriting the artifact so "
            "this is fixed permanently.",
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

    # ── Feature selector: identical trap. MADFilter was pickled
    # under '__main__' when feature_selection.py was last run
    # directly, so it can't be found now that ensemble_model.py
    # is __main__ instead. ──
    sel_path = "models/feature_selection_pipeline.joblib"
    try:
        selector = GenomicsFeatureSelector.load(sel_path)
        X_train_sel = selector.transform(X_train_proc)
        X_test_sel = selector.transform(X_test_proc)
    except AttributeError as exc:
        logger.warning(
            "Failed to unpickle '%s' (%s). MADFilter was pickled "
            "under module '__main__' instead of "
            "'src.feature_selection'. Refitting a fresh selector "
            "and overwriting the artifact so this is fixed "
            "permanently.",
            sel_path,
            exc,
        )
        selector = GenomicsFeatureSelector()
        X_train_sel = selector.fit_transform(X_train_proc, y_train_enc)
        X_test_sel = selector.transform(X_test_proc)
        selector.save(sel_path)

    selected_genes = selector.get_selected_genes()

    model = BiomarkerEnsemble()
    model.fit(X_train_sel, y_train_enc)

    class_names = list(label_encoder.classes_)
    report = model.evaluate(X_test_sel, y_test_enc, class_names)

    importances = model.get_rf_feature_importances(selected_genes)
    model.save("models/ensemble_model.joblib")

    sep = "=" * 60
    print(f"\n{sep}")
    print("  ENSEMBLE MODEL -- SUMMARY")
    print(sep)
    print(f"  Classes         : {class_names}")
    if report["roc_auc"] is not None:
        print(f"  ROC-AUC (test)  : {report['roc_auc']:.4f}")
        print(f"  PR-AUC (test)   : {report['pr_auc']:.4f}")
    print(f"  Macro-F1 (test) : {report['macro_f1']:.4f}")
    print(f"  Cohen Kappa     : {report['cohen_kappa']:.4f}")
    print("\n  Top 5 genes by RF importance:")
    for gene, score in importances.head(5).items():
        print(f"    {gene:<20} : {score:.6f}")
    print(sep)
    print("  Saved : models/ensemble_model.joblib")
    print("  Next  : update src/explainability/shap_explainer.py")
    print("          for the binary target before rerunning it.")
    print(sep)
