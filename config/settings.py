"""
config/settings.py
==================
Centralized project settings loaded from environment variables / .env file.
All path resolution is relative to the project root so the codebase is
portable across dev machines and CI environments.
"""

from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Project-wide settings loaded from .env file at runtime.
    Access via: from config.settings import settings
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Paths
    raw_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    models_dir: Path = Field(default=PROJECT_ROOT / "models")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports" / "figures")

    # Ingestion
    float_dtype: str = Field(default="float32")
    low_memory_mode: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # TCGA-BRCA specific
    expression_file: str = Field(default="TCGA-BRCA.htseq_fpkm-uq.tsv.gz")
    phenotype_file: str = Field(default="TCGA-BRCA.GDC_phenotype.tsv.gz")
    label_column: str = Field(default="paper_BRCA_Subtype_PAM50")
    valid_subtypes: Tuple[str, ...] = Field(
        default=("LumA", "LumB", "Her2", "Basal", "Normal")
    )

    # Modeling
    random_state: int = Field(default=42)
    cv_folds: int = Field(default=5)
    n_top_genes: int = Field(default=500)

    # FastAPI
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)


# Singleton — import this everywhere
settings = Settings()
