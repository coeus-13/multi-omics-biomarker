"""
manual_api_smoke.py
====================
Standalone smoke test for the live FastAPI vital-status prediction
service (src/api/main.py). Run this in a second terminal while the
API is running in the first.

NOT A PYTEST FILE -- NAMED DELIBERATELY TO AVOID DISCOVERY
------------------------------------------------------------------
This is a manual, network-calling smoke test against a live
server. It lives in tests/ for organizational consistency, but is
deliberately named to avoid pytest's default discovery globs
(test_*.py, *_test.py) -- a file matching those patterns would be
imported during `pytest tests/`, and while none of its functions
are named test_* (so nothing would actually run), it's a needless
foot-gun to leave in place. Naming it out of the glob is a
structural guarantee that can't drift, unlike a conftest.py
exclusion rule -- the same lesson learned from the E203 CI/pre-
commit config drift earlier in this project.

THIS IS A WIRING TEST, NOT A SCIENTIFIC ONE
------------------------------------------------
The dummy payload is uniform random noise across every gene, which
looks nothing like real FPKM-UQ expression data (heavily right-
skewed in reality -- most genes near zero, a few highly expressed).
This confirms the HTTP round-trip, Pydantic validation, and the
full preprocessor -> selector -> ensemble chain execute without
error. It says nothing about whether the returned risk_probability
is meaningful. Don't read biology into this response.

DEPENDENCY NOTE
--------------------
requests is not in requirements.txt -- it belongs in
requirements-dev.txt, since the API server itself never imports
it; only this client does.

Usage:
    Terminal 1: python src/api/main.py
    Terminal 2: python tests/manual_api_smoke.py

Author: [Your Name]
Date  : [Project Date]
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests

from src.preprocessing import GenomicsPreprocessor

API_URL = "http://localhost:8000"
PREPROCESSOR_PATH = "models/preprocessing_pipeline.joblib"
RANDOM_SEED = 42
REQUEST_TIMEOUT_SECONDS = 30


def load_expected_genes(preprocessor_path: str) -> List[str]:
    """
    Load the fitted preprocessing pipeline and extract the exact
    raw gene panel it was fit on.

    Mirrors src/api/main.py's own _load_artifacts() so this script
    independently reconstructs the same expected-gene list the
    live API uses -- no dependency on calling /health first.

    Args:
        preprocessor_path: Path to preprocessing_pipeline.joblib.

    Returns:
        Gene identifiers, in the exact column order the
        preprocessor expects.

    Raises:
        FileNotFoundError: If the artifact is missing.
    """
    path = Path(preprocessor_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifact not found: {path}\n"
            "Run the training pipeline before this smoke test."
        )

    preprocessor = GenomicsPreprocessor.load(str(path))
    log2_step = preprocessor.pipeline_.named_steps["log2_transform"]
    return list(log2_step.feature_names_in_)


def build_dummy_payload(
    expected_genes: List[str], patient_id: str, seed: int
) -> Dict[str, Any]:
    """
    Build a synthetic /predict payload: every expected gene mapped
    to a random float in [0.5, 10.0).

    Vectorized generation: one rng.uniform() call over the full
    gene count, not a Python loop calling it once per gene.
    .tolist() converts numpy.float64 -> native Python float, which
    is required for JSON serialization -- requests' json= parameter
    uses the stdlib json module, which does not know how to
    serialize numpy scalar types and will raise TypeError otherwise.

    Args:
        expected_genes: Full raw gene panel, in order.
        patient_id: Identifier to attach to this synthetic patient.
        seed: Seed for reproducibility across repeated runs.

    Returns:
        Dict matching the PatientProfile request schema.
    """
    rng = np.random.default_rng(seed)
    random_values = rng.uniform(0.5, 10.0, size=len(expected_genes))

    return {
        "patient_id": patient_id,
        "expressions": dict(zip(expected_genes, random_values.tolist())),
    }


def check_server_health(base_url: str) -> bool:
    """
    Best-effort GET /health check before attempting /predict, so a
    down server produces a clear message rather than a raw
    connection error buried under a large POST attempt.

    Args:
        base_url: e.g. 'http://localhost:8000'.

    Returns:
        True if the server responded 200 with models_loaded=True.
        Never raises -- failure here is advisory, not fatal;
        send_prediction_request() still runs its own full error
        handling regardless of this result.
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
    except requests.exceptions.RequestException:
        return False

    if response.status_code != 200:
        return False

    try:
        return bool(response.json().get("models_loaded", False))
    except ValueError:
        return False


def send_prediction_request(base_url: str, payload: Dict[str, Any]) -> None:
    """
    POST payload to /predict and print the formatted response.

    Distinguishes connection failure, timeout, and non-200
    responses so the failure mode is obvious from the printed
    message alone.

    Args:
        base_url: e.g. 'http://localhost:8000'.
        payload: Dict matching the PatientProfile request schema.
    """
    url = f"{base_url}/predict"
    n_genes = len(payload["expressions"])
    print(f"POST {url}  ({n_genes:,} genes in payload)")

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError:
        print(
            "\nERROR: could not connect to the API.\n"
            "Is 'src/api/main.py' running and listening on "
            f"{base_url}?"
        )
        return
    except requests.exceptions.Timeout:
        print(
            "\nERROR: request timed out after "
            f"{REQUEST_TIMEOUT_SECONDS}s. The server may be "
            "overloaded or stuck."
        )
        return
    except requests.exceptions.RequestException as exc:
        print(f"\nERROR: request failed unexpectedly: {exc}")
        return

    print(f"Status: {response.status_code}\n")

    try:
        body = response.json()
    except ValueError:
        print("Response was not valid JSON:")
        print(response.text)
        return

    print(json.dumps(body, indent=2))

    if response.status_code != 200:
        print(
            f"\nNOTE: server returned HTTP {response.status_code}, "
            "not 200 -- see 'detail' above for the reason."
        )


def main() -> None:
    print("=" * 60)
    print("  API SMOKE TEST -- /predict")
    print("=" * 60)

    print(f"\nChecking {API_URL}/health ...")
    if check_server_health(API_URL):
        print("Server is up and models are loaded.")
    else:
        print(
            "WARNING: /health check failed or reported "
            "models_loaded=False. Attempting /predict anyway."
        )

    print(f"\nLoading expected gene panel from {PREPROCESSOR_PATH} ...")
    try:
        expected_genes = load_expected_genes(PREPROCESSOR_PATH)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print(f"Loaded {len(expected_genes):,} expected genes.")

    payload = build_dummy_payload(
        expected_genes, patient_id="TEST-001", seed=RANDOM_SEED
    )

    print()
    send_prediction_request(API_URL, payload)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
