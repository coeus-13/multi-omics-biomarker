"""
main.py
=======
FastAPI service for binary vital-status (Alive vs Dead) prediction,
serving the fitted preprocessing, feature-selection, and ensemble
artifacts as a single request pipeline.

FOUR ARTIFACTS, NOT THREE
----------------------------
models/label_encoder.joblib is loaded alongside the three named in
the request: without it there is no way to map the ensemble's raw
integer prediction (0/1) back to "Alive"/"Dead" strings, which the
/predict response is required to return.

WHY THIS FILE DOES NOT SELF-HEAL THE PICKLE TRAP BY REFITTING
--------------------------------------------------------------------
Every training script's __main__ block (data_ingestion.py,
feature_selection.py, ensemble_model.py) self-heals a poisoned
joblib artifact by refitting fresh and overwriting the file. That
pattern is correct there: those scripts have the raw training data
on hand, refitting takes seconds at this data scale, and they run
in a development/iteration context.

None of that holds for a deployed API. Refitting here would mean
either shipping raw TCGA training data alongside the production
service (a real security/footprint concern) or silently serving
predictions from a model nobody validated -- different random
splits, different scale_pos_weight, no evaluate() report -- without
anyone noticing a swap happened. A production inference service
should fail loudly at startup on a bad artifact, not quietly train
a new one and keep going.

Given the .__module__ overrides added to Log2FPKMTransformer,
LowExpressionFilter, MADFilter, and BiomarkerEnsemble in the
previous step, the underlying bug should not recur regardless of
how those files are executed. _load_artifacts() still catches
AttributeError specifically and raises a clear, actionable
RuntimeError if it ever does -- pointing back at that fix -- rather
than attempting an in-process repair.

POSITIVE-CLASS CONVENTION (CONSISTENT WITH THE REST OF THE PROJECT)
-------------------------------------------------------------------------
Same rule as BiomarkerEnsemble.evaluate() and
BiomarkerSHAPExplainer: positive_label = max(ensemble.stack_.classes_),
looked up dynamically rather than hardcoded as index 1, even though
in practice it always resolves to 1 ("Dead") under this project's
LabelEncoder convention.

WHY /predict IS A SYNC (not async) ENDPOINT
------------------------------------------------
Every call inside it (.transform(), .predict(), .predict_proba())
is synchronous, CPU-bound sklearn/XGBoost work, never I/O. A plain
`def`, not `async def`, tells FastAPI to run it in Starlette's
external threadpool rather than blocking the single asyncio event
loop for the duration of each prediction.

Usage:
    Run directly (see __main__), or from the project root:
        uvicorn src.api.main:app --reload

Author: [Your Name]
Date  : [Project Date]
"""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.preprocessing import LabelEncoder

from src.data_ingestion import setup_logging
from src.feature_selection import GenomicsFeatureSelector
from src.models.ensemble_model import BiomarkerEnsemble
from src.preprocessing import GenomicsPreprocessor

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
API_HOST = "0.0.0.0"
API_PORT = 8000

# Global dictionary state, populated once at startup by lifespan()
# and read by every request handler via the get_ml_models()
# dependency below. Cleared on shutdown.
ml_models: Dict[str, Any] = {}


# ── Pydantic Request / Response Schemas ──


class PatientProfile(BaseModel):
    """Request schema for a single patient's raw expression panel."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "PATIENT-0001",
                "expressions": {
                    "ENSG00000141510": 5.23,
                    "ENSG00000012048": 3.87,
                },
            }
        }
    )

    patient_id: str = Field(
        ...,
        min_length=1,
        description="Caller-supplied patient identifier, echoed "
        "back unchanged in the response.",
    )
    expressions: Dict[str, float] = Field(
        ...,
        description="Mapping of gene identifier to raw, non-"
        "negative FPKM-UQ expression value. Must include every "
        "gene the deployed preprocessor was fit on -- see GET "
        "/health for the exact count.",
    )

    @field_validator("expressions")
    @classmethod
    def validate_expressions(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Reject empty payloads and negative/non-finite values."""
        if not value:
            raise ValueError("expressions must not be empty.")

        invalid = [
            gene for gene, expr in value.items() if not math.isfinite(expr) or expr < 0
        ]
        if invalid:
            preview = invalid[:5]
            raise ValueError(
                f"{len(invalid)} gene(s) have negative or "
                f"non-finite values, e.g. {preview}."
            )
        return value


class PredictionResponse(BaseModel):
    """Response schema returned by POST /predict."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "PATIENT-0001",
                "predicted_status": "Alive",
                "risk_probability": 0.2137,
                "class_probabilities": {"Alive": 0.7863, "Dead": 0.2137},
                "n_genes_received": 5658,
                "n_genes_used_by_model": 100,
            }
        }
    )

    patient_id: str
    predicted_status: str = Field(
        ..., description="Highest-probability class: 'Alive' or 'Dead'."
    )
    risk_probability: float = Field(
        ...,
        description="Model-estimated probability of the 'Dead' "
        "outcome specifically (the positive/event class), "
        "regardless of predicted_status.",
    )
    class_probabilities: Dict[str, float] = Field(
        ..., description="Full probability breakdown, both classes."
    )
    n_genes_received: int
    n_genes_used_by_model: int


class HealthResponse(BaseModel):
    """Response schema returned by GET /health."""

    status: str
    models_loaded: bool
    n_expected_genes: Optional[int] = None
    n_model_input_genes: Optional[int] = None
    classes: Optional[List[str]] = None


# ── Artifact Loading ──


def _load_artifacts(models_dir: Path) -> Dict[str, Any]:
    """
    Load all four fitted artifacts from disk exactly once.

    Fails fast and loudly on any problem -- see the module
    docstring's "WHY THIS FILE DOES NOT SELF-HEAL" section for why
    that is the correct behavior here, unlike the training
    scripts' __main__ blocks.

    Args:
        models_dir: Directory containing the four .joblib files.

    Returns:
        Dict with keys 'preprocessor', 'selector', 'ensemble',
        'label_encoder', 'expected_genes'.

    Raises:
        FileNotFoundError: If any required artifact is missing.
        RuntimeError: If a class module-path mismatch (the
            joblib/__main__ pickle trap) prevents unpickling, or if
            the loaded ensemble's classes are not exactly [0, 1].
    """
    preprocessor_path = models_dir / "preprocessing_pipeline.joblib"
    selector_path = models_dir / "feature_selection_pipeline.joblib"
    ensemble_path = models_dir / "ensemble_model.joblib"
    label_encoder_path = models_dir / "label_encoder.joblib"

    for path in (
        preprocessor_path,
        selector_path,
        ensemble_path,
        label_encoder_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}\n"
                "Run the training pipeline (src/preprocessing.py "
                "-> src/feature_selection.py -> "
                "src/models/ensemble_model.py) before starting "
                "the API."
            )

    logger.info("Loading model artifacts from %s ...", models_dir)

    try:
        preprocessor = GenomicsPreprocessor.load(str(preprocessor_path))
        selector = GenomicsFeatureSelector.load(str(selector_path))
        ensemble = BiomarkerEnsemble.load(str(ensemble_path))
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to unpickle a training artifact due to a "
            "class module-path mismatch (the joblib/__main__ "
            "pickle trap). This should be permanently prevented "
            "by the .__module__ overrides on Log2FPKMTransformer, "
            "LowExpressionFilter, MADFilter, and BiomarkerEnsemble "
            "-- verify those are present, or re-run the training "
            "scripts' self-healing __main__ blocks to regenerate "
            f"a clean artifact.\nOriginal error: {exc}"
        ) from exc

    label_encoder: LabelEncoder = joblib.load(label_encoder_path)

    model_classes = list(ensemble.stack_.classes_)
    if model_classes != [0, 1]:
        raise RuntimeError(
            f"Loaded ensemble has unexpected classes {model_classes}"
            "; this API requires exactly [0, 1]."
        )

    log2_step = preprocessor.pipeline_.named_steps["log2_transform"]
    expected_genes = list(log2_step.feature_names_in_)

    artifacts: Dict[str, Any] = {
        "preprocessor": preprocessor,
        "selector": selector,
        "ensemble": ensemble,
        "label_encoder": label_encoder,
        "expected_genes": expected_genes,
    }

    logger.info(
        "Artifacts loaded | raw genes expected=%d | model input "
        "genes=%d | classes=%s",
        len(expected_genes),
        len(selector.get_selected_genes()),
        list(label_encoder.classes_),
    )
    return artifacts


def get_ml_models() -> Dict[str, Any]:
    """FastAPI dependency exposing the global ml_models dict."""
    return ml_models


# ── API Router & Endpoints ──

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    models: Dict[str, Any] = Depends(get_ml_models),
) -> HealthResponse:
    """Lightweight readiness signal for load balancers / smoke tests."""
    if not models:
        return HealthResponse(status="unavailable", models_loaded=False)

    return HealthResponse(
        status="ok",
        models_loaded=True,
        n_expected_genes=len(models["expected_genes"]),
        n_model_input_genes=len(models["selector"].get_selected_genes()),
        classes=list(models["label_encoder"].classes_),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    profile: PatientProfile,
    models: Dict[str, Any] = Depends(get_ml_models),
) -> PredictionResponse:
    """
    Score one patient through preprocessor -> selector -> ensemble.

    Declared as a plain `def`, not `async def` -- see the module
    docstring's threadpool-offloading rationale.
    """
    if not models:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model artifacts are not loaded. The service "
            "did not start correctly.",
        )

    expected_genes = models["expected_genes"]
    missing = set(expected_genes) - set(profile.expressions.keys())
    if missing:
        preview = sorted(missing)[:10]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Request is missing {len(missing)} gene(s) the "
                f"model requires, e.g. {preview}. Call GET "
                "/health for the total expected gene count."
            ),
        )

    raw_row = pd.DataFrame([profile.expressions], dtype=np.float32)
    raw_row = raw_row.reindex(columns=expected_genes)

    if raw_row.isnull().values.any():
        # Should be unreachable given the missing-gene check above;
        # retained as a defensive guard against key-matching edge
        # cases (e.g. duplicate keys differing only in whitespace).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gene alignment produced unexpected null values.",
        )

    try:
        x_proc = models["preprocessor"].transform(raw_row)
        x_sel = models["selector"].transform(x_proc)
        y_pred = models["ensemble"].predict(x_sel)
        y_proba = models["ensemble"].predict_proba(x_sel)
    except Exception as exc:
        # Intentionally broad: this is the outermost API boundary,
        # the one place an unexpected internal failure must become
        # a safe HTTP response instead of a raw traceback leaking
        # to the caller.
        logger.exception(
            "Pipeline failure while scoring patient_id=%s",
            profile.patient_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model pipeline failed to process the "
            "submitted sample. This has been logged for "
            "investigation.",
        ) from exc

    label_encoder: LabelEncoder = models["label_encoder"]
    model_classes = list(models["ensemble"].stack_.classes_)

    # Same "positive = max(classes)" convention used throughout
    # this project -- see BiomarkerEnsemble.evaluate() and
    # BiomarkerSHAPExplainer -- looked up dynamically rather than
    # hardcoded, even though it always resolves to 1 here.
    positive_label = max(model_classes)
    positive_idx = model_classes.index(positive_label)
    risk_probability = round(float(y_proba[0, positive_idx]), 6)

    predicted_status = str(label_encoder.inverse_transform([int(y_pred[0])])[0])
    class_probabilities = {
        str(class_name): round(float(proba), 6)
        for class_name, proba in zip(label_encoder.classes_, y_proba[0])
    }

    return PredictionResponse(
        patient_id=profile.patient_id,
        predicted_status=predicted_status,
        risk_probability=risk_probability,
        class_probabilities=class_probabilities,
        n_genes_received=len(profile.expressions),
        n_genes_used_by_model=x_sel.shape[1],
    )


# ── Application Factory ──


def create_app(models_dir: Path = MODELS_DIR) -> FastAPI:
    """
    Application factory.

    models_dir is a parameter (not just the module-level constant)
    specifically so a future test suite can point create_app() at
    a directory of small synthetic artifacts without monkeypatching
    module state.

    Builds and returns a fresh FastAPI instance on every call, each
    with its own lifespan closure over models_dir.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging("INFO")
        logger.info("Startup: loading model artifacts...")
        ml_models.clear()
        ml_models.update(_load_artifacts(models_dir))
        logger.info("Startup complete. API ready to accept requests.")

        yield

        logger.info("Shutting down. Releasing model artifacts.")
        ml_models.clear()

    app = FastAPI(
        title="Vital-Status Biomarker Prediction API",
        description=(
            "Predicts binary vital status (Alive vs Dead) from "
            "RNA-Seq expression data using a Random Forest + "
            "XGBoost stacking ensemble."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()


# ── Standalone Entry Point ──

if __name__ == "__main__":
    """
    Run directly from PyCharm's Run button.

    Configure PyCharm Run/Debug:
        Script           : src/api/main.py
        Working directory: <project root>

    Requires src/api/__init__.py to exist (empty file is fine) so
    this module is importable as src.api.main for uvicorn's
    string-based app reference and for --reload to work.
    """
    import uvicorn

    sep = "=" * 60
    print(f"\n{sep}")
    print("  VITAL-STATUS PREDICTION API -- STARTING")
    print(sep)
    print(f"  Host          : {API_HOST}")
    print(f"  Port          : {API_PORT}")
    print(f"  Docs          : http://localhost:{API_PORT}/docs")
    print(f"  Health check  : http://localhost:{API_PORT}/health")
    print(sep)

    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
