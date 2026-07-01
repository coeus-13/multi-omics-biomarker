"""
data_ingestion.py
=================
Production-ready, memory-optimized ingestion module for TCGA-BRCA RNA-Seq data.

Handles the full ETL workflow from raw UCSC Xena TSV files to an aligned
(X, y) pair ready for preprocessing. Engineered specifically for machines
with limited RAM (≤16GB) and no discrete GPU (e.g., Intel i7-1360P).

Memory Budget for TCGA-BRCA:
    Raw float64 matrix (1,100 × 60,000) : ~528 MB  ← AVOIDED
    With float32 at load time            : ~264 MB  ← TARGET
    Peak during transpose + alignment    : ~600 MB  ← MANAGED via gc.collect()
    Final aligned X in memory            : ~264 MB

Optimization Techniques Used:
    1. dtype=np.float32 specified at pd.read_csv() — no post-load copy needed
    2. Two-pass reading: header inspection before full allocation
    3. usecols in phenotype load — skips ~100 irrelevant clinical columns
    4. dtype='category' for string label column (10-100x memory saving)
    5. Explicit del + gc.collect() after each major allocation is freed
    6. Column-level downcast pass (float64 → float32) as a safety net
    7. MemoryReporter instruments every stage for regression detection

Usage:
    >>> from src.data_ingestion import TCGADataIngester, IngestionConfig
    >>> config = IngestionConfig(low_memory_mode=True)
    >>> ingester = TCGADataIngester(data_dir="data/raw/", config=config)
    >>> X, y, metadata = ingester.run()
    >>> print(f"Ready: X={X.shape}, classes={y.unique().tolist()}")

Data Source:
    UCSC Xena Browser → GDC Hub → TCGA-BRCA cohort
    Expression : https://xenabrowser.net → TCGA-BRCA.htseq_fpkm-uq.tsv.gz
    Phenotype  : https://xenabrowser.net → TCGA-BRCA.GDC_phenotype.tsv.gz

Author: [Your Name]
Date  : [Project Date]
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Module-level logger — configured by the caller (see setup_logging() below)
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class IngestionConfig:
    """
    Immutable configuration for the TCGA ingestion pipeline.

    Using a dataclass (rather than kwargs or a dict) enforces type safety,
    enables IDE autocomplete, and makes it trivially mockable in unit tests
    by substituting a config pointing to a synthetic mini-dataset.

    Attributes:
        expression_filename: Filename of the gzipped RNA-Seq TSV from UCSC Xena.
        phenotype_filename:  Filename of the gzipped phenotype TSV from UCSC Xena.
        label_column:        Column name in phenotype file containing PAM50 labels.
        valid_subtypes:      Tuple of canonical PAM50 class strings to retain.
            TCGA uses abbreviated names in the phenotype file:
            'LumA', 'LumB', 'Her2', 'Basal', 'Normal'
        low_memory_mode:     If True, adds extra gc.collect() calls and logs
            memory at every stage. Recommended for ≤16GB systems.
        float_dtype:         NumPy dtype for expression values. float32 is
            sufficient for log-normalized FPKM (range ~0–30) and halves RAM.
        barcode_length:      TCGA sample barcode length to use for alignment.
            UCSC Xena uses 16-char barcodes; phenotype uses 15-char.
            Set to min(len) of your actual barcodes if alignment fails.
    """

    expression_filename: str = "TCGA-BRCA.htseq_fpkm-uq.tsv.gz"
    phenotype_filename: str = "TCGA-BRCA.GDC_phenotype.tsv.gz"
    label_column: str = "paper_BRCA_Subtype_PAM50"
    valid_subtypes: Tuple[str, ...] = ("LumA", "LumB", "Her2", "Basal", "Normal")
    low_memory_mode: bool = True
    float_dtype: type = np.float32
    barcode_length: int = 15  # Trim both sources to this length for alignment


# ──────────────────────────────────────────────────────────────────────────────
# Memory Reporter Utility
# ──────────────────────────────────────────────────────────────────────────────


class MemoryReporter:
    """
    Utility class to instrument DataFrame memory consumption at each pipeline stage.

    Provides consistent, structured logging of shape and memory so memory
    regressions (e.g., an accidental float64 column introduced upstream)
    are immediately visible in logs without manual debugging.

    Example output:
        [MemoryReport] Raw Expression Matrix | Shape: (60483, 1113) | Memory: 511.4 MB
        [MemoryReport] Transposed Matrix     | Shape: (1113, 60483) | Memory: 511.4 MB
        [MemoryReport] Final Aligned X       | Shape: (1084, 60483) | Memory: 248.7 MB
    """

    @staticmethod
    def report(df: pd.DataFrame, label: str = "DataFrame") -> Dict[str, Any]:
        """
        Compute and log memory statistics for a DataFrame.

        Args:
            df:    The DataFrame to profile.
            label: Human-readable name for log output (e.g., 'Raw Expression Matrix').

        Returns:
            Dict with keys: 'total_mb' (float), 'shape' (tuple), 'n_cells' (int),
            'dtype_counts' (dict mapping dtype string → column count).
        """
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = round(mem_bytes / (1024**2), 2)

        # Count columns by dtype for fast dtype-regression detection
        dtype_counts: Dict[str, int] = (
            df.dtypes.value_counts().rename(index=str).to_dict()
        )

        stats: Dict[str, Any] = {
            "total_mb": mem_mb,
            "shape": df.shape,
            "n_cells": df.shape[0] * df.shape[1],
            "dtype_counts": dtype_counts,
        }

        logger.info(
            "[MemoryReport] %-40s | Shape: %-18s | Memory: %7.1f MB | Dtypes: %s",
            label,
            str(df.shape),
            mem_mb,
            dtype_counts,
        )
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# Main Ingestion Class
# ──────────────────────────────────────────────────────────────────────────────


class TCGADataIngester:
    """
    Memory-efficient ETL pipeline for TCGA-BRCA RNA-Seq expression data.

    Orchestrates loading, validation, transposition, label alignment, and
    memory optimization of UCSC Xena formatted files into an (X, y) pair
    ready for the preprocessing module.

    Design Principles:
        - Separation of concerns: each private method does exactly one ETL step.
        - All configuration is injected via IngestionConfig (no hardcoded paths).
        - Memory is profiled at every stage via MemoryReporter.
        - Errors are specific and actionable — they tell the user exactly what
          file is missing and where to download it.

    Attributes:
        data_dir (Path): Resolved absolute path to the raw data directory.
        config (IngestionConfig): Ingestion configuration and parameters.
        reporter (MemoryReporter): Memory instrumentation utility.

    Raises:
        FileNotFoundError: If data_dir or required files are absent.
        ValueError: If the loaded data is empty, malformed, or yields
            fewer than 100 aligned samples.
        MemoryError: If the expression matrix cannot be loaded in available RAM
            (hint: enable low_memory_mode or increase system swap).
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[IngestionConfig] = None,
    ) -> None:
        """
        Initialize the ingester with the data directory path and optional config.

        Args:
            data_dir: Path (relative or absolute) to the folder containing
                TCGA-BRCA raw files. Will be resolved to an absolute Path.
            config:   IngestionConfig instance. If None, uses class defaults.

        Raises:
            FileNotFoundError: If data_dir does not exist on disk.

        Example:
            >>> ingester = TCGADataIngester(data_dir="data/raw/")
            >>> ingester = TCGADataIngester(
            ...     data_dir="data/raw/",
            ...     config=IngestionConfig(low_memory_mode=True, barcode_length=15)
            ... )
        """
        self.data_dir = Path(data_dir).resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Create it with: mkdir -p {self.data_dir}\n"
                f"Then download TCGA-BRCA files from: https://xenabrowser.net"
            )

        self.config = config or IngestionConfig()
        self.reporter = MemoryReporter()

        logger.info(
            "TCGADataIngester initialized | data_dir=%s | low_memory=%s | dtype=%s",
            self.data_dir,
            self.config.low_memory_mode,
            self.config.float_dtype.__name__,
        )

    # ── Private Methods ───────────────────────────────────────────────────────

    def _load_expression_matrix(self) -> pd.DataFrame:
        """
        Load the UCSC Xena RNA-Seq expression matrix into memory as float32.

        UCSC Xena format:
            - Rows    → genes (Ensembl IDs or HUGO symbols)
            - Columns → TCGA sample barcodes
            - Values  → FPKM-UQ normalized expression (non-negative floats)

        After loading, the matrix is transposed to (samples × genes) to match
        the scikit-learn convention of (n_observations, n_features).

        Memory Strategy:
            Pass 1 (2 rows)   : Inspect file structure and sample count → no cost
            Pass 2 (full load): dtype=float32 specified upfront → ~264MB peak
            Transpose         : Creates a new object → peaks at ~528MB briefly
            del original      : Drops to ~264MB
            gc.collect()      : Forces CPython to reclaim the freed ~264MB

        Returns:
            pd.DataFrame of shape (n_samples, n_genes) with float32 values.
            Row index: TCGA sample barcodes (str, trimmed to barcode_length).
            Column names: gene identifiers (Ensembl IDs or HUGO symbols).

        Raises:
            FileNotFoundError: If the expression file is not in data_dir.
            ValueError: If the file loads as an empty DataFrame.
            MemoryError: If RAM is insufficient even for float32 loading.
        """
        expr_path = self.data_dir / self.config.expression_filename

        if not expr_path.exists():
            raise FileNotFoundError(
                f"Expression file not found: {expr_path}\n"
                f"\nDownload steps:\n"
                f"  1. Go to: https://xenabrowser.net/datapages/\n"
                f"  2. Select: GDC TCGA Breast Cancer (BRCA) → Gene Expression RNAseq\n"
                f"  3. Download: TCGA-BRCA.htseq_fpkm-uq.tsv.gz\n"
                f"  4. Place in: {self.data_dir}"
            )

        # ── Pass 1: Header Inspection ─────────────────────────────────────────
        # Read only 2 rows to detect file structure cheaply before committing
        # to a full load. This catches malformed files early.
        logger.info("Pass 1: Inspecting expression matrix header...")
        try:
            header_peek = pd.read_csv(
                expr_path,
                sep="\t",
                compression="gzip",
                nrows=2,
                index_col=0,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot read expression file header. File may be corrupt: {exc}"
            ) from exc

        n_samples_detected = len(header_peek.columns)

        logger.info(
            "Header inspection complete | Detected samples: %d | "
            "Proceeding with full float32 load...",
            n_samples_detected,
        )
        del header_peek
        gc.collect()

        # ── Pass 2: Full Load with float32 ───────────────────────────────────
        # dtype=np.float32 is applied at the C-level CSV parser — this means
        # Python never allocates float64 memory. The savings compound because
        # the 60,000-column shape makes this a large allocation.
        logger.info("Pass 2: Loading full expression matrix as float32...")
        try:
            expr_df = pd.read_csv(
                expr_path,
                sep="\t",
                compression="gzip",
                index_col=0,  # Gene identifiers become row index
                dtype=self.config.float_dtype,  # ← Core memory optimization
                low_memory=False,  # Prevents ambiguous mixed-type inference
            )
        except MemoryError as exc:
            raise MemoryError(
                "Insufficient RAM to load the expression matrix even as float32.\n"
                "Options:\n"
                "  1. Close other applications to free RAM.\n"
                "  2. Increase system swap/page file size.\n"
                "  3. Pre-filter genes to top 20K variance-ranked before loading.\n"
                f"Original error: {exc}"
            ) from exc

        if expr_df.empty:
            raise ValueError(
                f"Expression matrix loaded as empty. "
                f"Check file integrity: {expr_path}"
            )

        self.reporter.report(expr_df, "Raw Expression Matrix (genes × samples)")

        # ── Transpose: genes × samples → samples × genes ─────────────────────
        # .T creates a new DataFrame object. During this call, both the original
        # and its transpose exist simultaneously → brief peak at ~2× matrix size.
        # We immediately delete the original to reclaim memory.
        logger.info(
            "Transposing matrix: (%d genes × %d samples) → (%d samples × %d genes)...",
            expr_df.shape[0],
            expr_df.shape[1],
            expr_df.shape[1],
            expr_df.shape[0],
        )
        expr_transposed = expr_df.T

        # Free the un-transposed copy immediately
        del expr_df
        if self.config.low_memory_mode:
            gc.collect()  # Force CPython to release the memory now, not lazily

        self.reporter.report(expr_transposed, "Transposed Matrix (samples × genes)")

        return expr_transposed

    def _load_phenotype_labels(self) -> pd.DataFrame:
        """
        Load PAM50 subtype labels from the TCGA clinical phenotype file.

        Uses usecols to load only the 2 columns we need (sample barcode + label)
        out of the ~100+ clinical metadata columns in the full phenotype file.
        This avoids loading irrelevant columns into memory.

        Returns:
            pd.DataFrame indexed by TCGA sample barcode, with one column:
            the PAM50 label column as a CategoricalDtype Series.

        Raises:
            FileNotFoundError: If the phenotype file is not in data_dir.
            KeyError: If the PAM50 label column name doesn't exist in the file.
                      This usually means the column was renamed in a newer release.
        """
        pheno_path = self.data_dir / self.config.phenotype_filename

        if not pheno_path.exists():
            raise FileNotFoundError(
                f"Phenotype file not found: {pheno_path}\n"
                f"\nDownload steps:\n"
                f"  1. Go to: https://xenabrowser.net/datapages/\n"
                f"  2. Select: GDC TCGA Breast Cancer (BRCA) → Phenotype\n"
                f"  3. Download: TCGA-BRCA.GDC_phenotype.tsv.gz\n"
                f"  4. Place in: {self.data_dir}"
            )

        logger.info("Loading phenotype file (PAM50 labels only)...")

        # Probe column names before filtering to give a helpful error if the
        # label column doesn't exist in this version of the phenotype file.
        try:
            pheno_columns = pd.read_csv(
                pheno_path,
                sep="\t",
                compression="gzip",
                nrows=0,  # Zero rows — just headers
            ).columns.tolist()
        except Exception as exc:
            raise ValueError(f"Cannot read phenotype file header: {exc}") from exc

        barcode_col = "submitter_id.samples"
        if barcode_col not in pheno_columns:
            # Try alternate barcode column names used in older Xena releases
            barcode_col = next(
                (c for c in pheno_columns if "submitter" in c.lower()),
                None,
            )
            if barcode_col is None:
                raise KeyError(
                    f"Cannot find sample barcode column in phenotype file.\n"
                    f"Available columns: {pheno_columns[:10]} ... "
                    f"(total {len(pheno_columns)})"
                )

        if self.config.label_column not in pheno_columns:
            raise KeyError(
                f"PAM50 label column '{self.config.label_column}' not found.\n"
                f"Available columns: {pheno_columns}\n"
                f"Update IngestionConfig.label_column to match the correct name."
            )

        # Load only the 2 needed columns
        pheno_df = pd.read_csv(
            pheno_path,
            sep="\t",
            compression="gzip",
            usecols=[barcode_col, self.config.label_column],  # ← Selective load
            dtype={self.config.label_column: "category"},  # ← Categorical dtype
        )

        # Set barcode as index, strip trailing whitespace from barcodes
        pheno_df[barcode_col] = pheno_df[barcode_col].str.strip()
        pheno_df = pheno_df.set_index(barcode_col)

        label_dist = pheno_df[self.config.label_column].value_counts()
        logger.info(
            "Phenotype loaded | Shape: %s | PAM50 distribution:\n%s",
            pheno_df.shape,
            label_dist.to_string(),
        )

        return pheno_df

    def _align_samples(
        self,
        expr_df: pd.DataFrame,
        pheno_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align expression matrix and phenotype labels on TCGA sample barcodes.

        Alignment Strategy:
            TCGA barcodes appear in different lengths across UCSC Xena files.
            We trim both indices to config.barcode_length characters before
            joining. The default is 15 characters:
            'TCGA-3C-AAAU-01A-11R-A41B-07' → 'TCGA-3C-AAAU-01A'
            This drops the vial/portion/analyte suffix which is not meaningful
            for sample-level label alignment.

        Args:
            expr_df:  Transposed expression matrix (samples × genes).
            pheno_df: Phenotype DataFrame indexed by TCGA barcode.

        Returns:
            Tuple (X, y) where:
                X: pd.DataFrame, shape (n_aligned_samples, n_genes), float32.
                y: pd.Series, dtype='category', PAM50 labels.

        Raises:
            ValueError: If fewer than 100 samples remain after alignment.
                        Usually indicates a barcode format mismatch.
        """
        logger.info(
            "Aligning barcodes (trimming to %d chars)...",
            self.config.barcode_length,
        )

        # Trim barcodes to ensure consistent length across both sources
        k = self.config.barcode_length
        expr_df.index = expr_df.index.str[:k]
        pheno_df.index = pheno_df.index.str[:k]

        n_expr_before = len(expr_df)

        # Inner join: only keep samples present in BOTH files
        aligned_df = expr_df.join(pheno_df, how="inner")

        n_after_join = len(aligned_df)
        logger.info(
            "Inner join result: %d expression samples → %d aligned with phenotype",
            n_expr_before,
            n_after_join,
        )

        if n_after_join < 100:
            logger.warning(
                "Only %d samples aligned. This is likely a barcode mismatch.\n"
                "Try setting config.barcode_length to a different value (15 or 16).\n"
                "Inspect barcodes with:\n"
                "  print(expr_df.index[:5].tolist())\n"
                "  print(pheno_df.index[:5].tolist())",
                n_after_join,
            )
            raise ValueError(
                f"Alignment produced only {n_after_join} samples (minimum: 100). "
                f"Check barcode format compatibility between your two files."
            )

        # Filter to valid PAM50 subtypes — removes NaN labels and any
        # non-canonical category strings (e.g., 'NA', 'Unknown')
        valid_mask = aligned_df[self.config.label_column].isin(
            self.config.valid_subtypes
        )
        n_dropped = (~valid_mask).sum()

        if n_dropped > 0:
            logger.warning(
                "Dropped %d samples with invalid/missing PAM50 labels "
                "(NaN or not in %s).",
                n_dropped,
                self.config.valid_subtypes,
            )

        aligned_df = aligned_df[valid_mask]

        if len(aligned_df) < 100:
            raise ValueError(
                f"After PAM50 filtering, only {len(aligned_df)} samples remain. "
                f"Check that valid_subtypes in IngestionConfig matches the "
                f"actual label strings in your phenotype file."
            )

        # Split into feature matrix X and label Series y
        X = aligned_df.drop(columns=[self.config.label_column])
        y = aligned_df[
            self.config.label_column
        ].cat.remove_unused_categories()  # Drop categories with 0 samples

        logger.info(
            "Alignment complete | X: %s | y distribution:\n%s",
            X.shape,
            y.value_counts().to_string(),
        )

        return X, y

    def _enforce_float32(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast any float64 columns to float32 as a safety net.

        This handles edge cases where pandas may internally promote dtypes
        during the join operation (e.g., when a float32 column aligns with
        a NaN-filled column, pandas may upcast to float64 to store NaN).

        Iterates column-by-column rather than casting the entire DataFrame
        at once to avoid a temporary full-copy memory spike.

        Args:
            df: Expression DataFrame post-alignment.

        Returns:
            DataFrame with all float columns guaranteed to be float32.
        """
        float64_cols = df.select_dtypes(include=["float64"]).columns
        if len(float64_cols) == 0:
            logger.info("float32 check: all columns already float32. No action needed.")
            return df

        logger.warning(
            "Found %d float64 columns after alignment (likely from join NaN-fill). "
            "Downcasting to float32...",
            len(float64_cols),
        )

        # Column-by-column to avoid peak memory spike from full DataFrame copy
        for col in float64_cols:
            df[col] = df[col].astype(np.float32)

        if self.config.low_memory_mode:
            gc.collect()

        self.reporter.report(df, "Expression Matrix Post-Float32-Enforcement")
        return df

    def _validate_output(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Run sanity checks on the final (X, y) pair before returning to caller.

        Catches common data quality issues that would silently corrupt
        downstream preprocessing or modeling (e.g., all-zero columns,
        unexpected NaN values, wrong dtype).

        Args:
            X: Final feature matrix.
            y: Final label series.

        Raises:
            ValueError: On any failed validation check.
        """
        # Check 1: No NaN values in X
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(
                f"Expression matrix contains {nan_count} NaN values after alignment. "
                f"This may indicate a join issue. Run X.isnull().sum().sort_values() "
                f"to identify affected genes."
            )

        # Check 2: All expression columns are float32
        non_float32 = [col for col in X.columns if X[col].dtype != np.float32]
        if non_float32:
            raise ValueError(
                f"{len(non_float32)} columns are not float32 after enforcement pass. "
                f"First 5: {non_float32[:5]}"
            )

        # Check 3: y has no NaN values
        y_nan = y.isnull().sum()
        if y_nan > 0:
            raise ValueError(
                f"Label series contains {y_nan} NaN values. "
                f"Alignment or filtering logic may have introduced gaps."
            )

        # Check 4: At least 2 distinct classes
        n_classes = y.nunique()
        if n_classes < 2:
            raise ValueError(
                f"Label series has only {n_classes} unique class(es). "
                f"Cannot perform multi-class classification."
            )

        # Warning only: flag if any gene column is all-zero (not an error,
        # but suspicious — may indicate a QC issue in the source data)
        zero_gene_cols = (X == 0).all(axis=0).sum()
        if zero_gene_cols > 0:
            logger.warning(
                "%d gene columns are all-zero across all samples. "
                "These will be removed by VarianceThreshold in preprocessing.",
                zero_gene_cols,
            )

        logger.info("Validation passed: no NaN, correct dtype, %d classes.", n_classes)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Execute the full ingestion pipeline end-to-end.

        Pipeline Stages:
            1. Load expression matrix (Pass 1: inspect, Pass 2: full float32 load)
            2. Load phenotype labels (PAM50 column only)
            3. Align samples via inner join on TCGA barcodes
            4. Free pre-alignment objects from memory
            5. Enforce float32 dtype on all columns (safety net)
            6. Validate final output
            7. Compile and return metadata

        Returns:
            Tuple of three objects:
                X (pd.DataFrame): float32 expression matrix, shape (n_samples, n_genes).
                    Row index: TCGA sample barcodes.
                    Column names: Gene identifiers (Ensembl IDs or HUGO symbols).

                y (pd.Series): Categorical PAM50 subtype labels, shape (n_samples,).
                    dtype: CategoricalDtype with 5 ordered categories.

                metadata (Dict[str, Any]): Pipeline run statistics including:
                    - 'n_samples' (int): Samples in final X.
                    - 'n_genes' (int): Genes/features in final X.
                    - 'class_distribution' (Dict[str, int]): PAM50 counts per class.
                    - 'memory_mb' (float): RAM used by final X in megabytes.
                    - 'dtype_counts' (Dict[str, int]): Column dtype distribution.

        Raises:
            FileNotFoundError: If raw data files are missing.
            ValueError: If alignment or validation fails.
            MemoryError: If RAM is insufficient for float32 loading.

        Example:
            >>> ingester = TCGADataIngester(data_dir="data/raw/")
            >>> X, y, meta = ingester.run()
            >>> print(f"X: {X.shape}, dtype={X.dtypes.iloc[0]}")
            >>> print(f"Memory: {meta['memory_mb']} MB")
            >>> print(f"Classes: {y.value_counts().to_dict()}")
        """
        logger.info("=" * 70)
        logger.info("TCGADataIngester.run() START")
        logger.info("=" * 70)

        # ── Stage 1: Load raw files ───────────────────────────────────────────
        expr_df = self._load_expression_matrix()
        pheno_df = self._load_phenotype_labels()

        # ── Stage 2: Align samples ────────────────────────────────────────────
        X, y = self._align_samples(expr_df, pheno_df)

        # ── Stage 3: Free pre-alignment raw objects ───────────────────────────
        # Critical: both expr_df and pheno_df are superseded by the aligned X, y.
        # Releasing them before the enforcement pass keeps peak memory low.
        del expr_df, pheno_df
        if self.config.low_memory_mode:
            gc.collect()
            logger.info(
                "gc.collect() called after alignment — pre-alignment objects freed."
            )

        # ── Stage 4: Enforce float32 (safety net post-join) ──────────────────
        X = self._enforce_float32(X)

        # ── Stage 5: Validate ─────────────────────────────────────────────────
        self._validate_output(X, y)

        # ── Stage 6: Compile metadata ─────────────────────────────────────────
        memory_stats = self.reporter.report(X, "Final Aligned X")
        metadata: Dict[str, Any] = {
            "n_samples": X.shape[0],
            "n_genes": X.shape[1],
            "class_distribution": y.value_counts().to_dict(),
            "memory_mb": memory_stats["total_mb"],
            "dtype_counts": memory_stats["dtype_counts"],
            "data_dir": str(self.data_dir),
        }

        logger.info("=" * 70)
        logger.info("TCGADataIngester.run() COMPLETE")
        logger.info(
            "  Samples : %d | Genes : %d | Memory : %.1f MB",
            metadata["n_samples"],
            metadata["n_genes"],
            metadata["memory_mb"],
        )
        logger.info("  Classes : %s", metadata["class_distribution"])
        logger.info("=" * 70)

        return X, y, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration Helper
# ──────────────────────────────────────────────────────────────────────────────


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with a clean, structured format.

    Call this once at the top of any script or notebook that imports
    from this module. Downstream loggers (e.g., xgboost, shap) will
    inherit this configuration.

    Args:
        level: Log level string. One of 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Reconfigures if already set (useful in notebooks)
    )
    # Suppress verbose third-party logs that clutter output
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Entry Point (for manual testing in PyCharm Run configuration)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone execution for testing and debugging from PyCharm's Run button.

    Configure PyCharm Run/Debug:
        Script: src/data_ingestion.py
        Working directory: <project root>
    """
    setup_logging("INFO")

    # ── Run ingestion ─────────────────────────────────────────────────────────
    ingester = TCGADataIngester(data_dir="data/raw/")
    X, y, metadata = ingester.run()

    # ── Summary output ────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  INGESTION COMPLETE — SUMMARY")
    print(sep)
    print(f"  X shape     : {X.shape}")
    print(f"  X dtype     : {X.dtypes.iloc[0]}")
    print(f"  y shape     : {y.shape}")
    print(f"  Memory (X)  : {metadata['memory_mb']} MB")
    print("  Classes     :")
    for cls, count in sorted(metadata["class_distribution"].items()):
        pct = 100 * count / y.shape[0]
        bar = "█" * int(pct / 3)
        print(f"    {cls:<10} : {count:4d} ({pct:5.1f}%)  {bar}")
    print(sep)
    print("\n  ✓ X and y are ready to pass to src/preprocessing.py")
    print(sep)
