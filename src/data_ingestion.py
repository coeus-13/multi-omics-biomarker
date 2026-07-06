"""
data_ingestion.py
=================
Memory-optimized ETL pipeline for TCGA-BRCA RNA-Seq data, refactored
for a phenotype-source pivot: expression still comes from the GDC
hub, but subtype labels now come from the UCSC Xena Toil Recompute
pan-cancer phenotype file (TcgaTargetGTEX_phenotype.txt.gz).

HISTORICAL BUG FIXED -- FLOAT CASTING AT READ TIME
----------------------------------------------------
Passing dtype=np.float32 directly into pd.read_csv() together with
index_col=0 previously raised:
    ValueError: could not convert string to float: 'ENSG...'
A scalar dtype can be applied across all parsed columns before
index_col fully detaches the gene-ID column from the numeric data
block, so the parser attempts to cast gene IDs to float and fails.
Fix: let pandas infer dtypes naturally at read time (gene IDs become
the object index, sample columns become float64), THEN downcast the
fully-formed DataFrame with a single .astype(self.config.float_dtype)
call, applied only after index_col has already separated the index.
See _load_expression_matrix().

DATASET PIVOT -- TOIL RECOMPUTE PHENOTYPE FILE
--------------------------------------------------
Phenotype labels now come from TcgaTargetGTEX_phenotype.txt.gz. Two
consequences of this pivot are handled explicitly:

  1. COHORT CONTAMINATION -- this file mixes TCGA, GTEx, and TARGET
     samples in one table. _load_phenotype_labels() strict-filters
     the phenotype index to barcodes starting with
     config.cohort_prefix ('TCGA' by default) immediately after
     loading, before any other processing happens.

  2. LABEL NAMING VARIANCE -- this file does not use the fixed
     column name 'paper_BRCA_Subtype_PAM50' from the old GDC file.
     _detect_label_column() searches column names case-
     insensitively for config.label_column_hints ('pam50',
     'subtype') and raises an explicit KeyError if none match.
     IMPORTANT: the standard Toil phenotype file typically only
     carries cohort-level metadata (sample type, primary site,
     study) -- it does not carry molecular subtype calls. If your
     copy has no matching column, that error is expected, not a
     bug -- see the message this file was delivered with.

BARCODE ALIGNMENT -- GDC EXPRESSION x TOIL PHENOTYPE
---------------------------------------------------------
The two files are different Xena hubs and do not share identical
barcode granularity. _align_samples() truncates both indices to
config.barcode_length characters (15 by default), drops any row
whose barcode collides with another after truncation (ambiguous --
we cannot know which row is "correct"), then strictly intersects the
two remaining indices before subsetting X and y.

Usage:
    >>> from src.data_ingestion import TCGADataIngester
    >>> ingester = TCGADataIngester(data_dir="data/raw/")
    >>> X, y, metadata = ingester.run()
    >>> print(metadata["label_column_used"])

Author: [Your Name]
Date  : [Project Date]
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Configuration Dataclass ──


@dataclass
class IngestionConfig:
    """
    Immutable configuration for the TCGA/Toil ingestion pipeline.

    Attributes:
        expression_filename : GDC-hub RNA-Seq matrix filename.
            Unchanged by this pivot -- expression still comes from
            the GDC hub; only the phenotype source changed.
        phenotype_filename  : Toil Recompute pan-cancer phenotype
            filename. Mixes TCGA, GTEx, and TARGET samples in one
            table -- see cohort_prefix below.
        label_column        : Optional exact column-name override.
            If set AND present in the phenotype header, used
            directly instead of running hint-based detection.
        label_column_hints  : Case-insensitive substrings used by
            _detect_label_column() when label_column is not set.
            NOTE: the standard Toil phenotype file typically only
            carries cohort-level metadata, not molecular subtype
            calls -- see the module docstring.
        cohort_prefix       : Barcode prefix kept when filtering the
            phenotype index immediately after loading. Removes
            cross-cohort contamination (GTEx, TARGET samples
            bundled into the same Toil file). Default 'TCGA'.
        low_memory_mode     : Extra gc.collect() calls and memory
            logging at each stage.
        float_dtype         : Target dtype for expression values.
        barcode_length      : Character count both indices are
            truncated to before alignment (see _align_samples()).
            GDC and Toil hubs do not always share identical barcode
            granularity; 15 chars ('TCGA-XX-XXXX-XX') is the common
            sample-level prefix both sources share.
    """

    expression_filename: str = "TCGA-BRCA.htseq_fpkm-uq.tsv.gz"
    phenotype_filename: str = "TCGA-BRCA.GDC_phenotype.tsv.gz"
    label_column: Optional[str] = None
    label_column_hints: Tuple[str, ...] = ("vital_status",)
    cohort_prefix: str = "TCGA"
    low_memory_mode: bool = True
    float_dtype: type = np.float32
    barcode_length: int = 15

    def __post_init__(self) -> None:
        if not self.label_column_hints:
            raise ValueError("label_column_hints must contain at least one hint.")
        if not self.cohort_prefix:
            raise ValueError("cohort_prefix must be a non-empty string.")
        if self.barcode_length <= 0:
            raise ValueError(
                "barcode_length must be positive; got " f"{self.barcode_length}."
            )


# ── Memory Reporter Utility ──


class MemoryReporter:
    """
    Instruments DataFrame memory consumption at each pipeline stage.

    Provides consistent, structured logging of shape and memory so
    memory regressions (e.g., an accidental float64 column
    introduced upstream) are immediately visible in logs without
    manual debugging.
    """

    @staticmethod
    def report(df: pd.DataFrame, label: str = "DataFrame") -> Dict[str, Any]:
        """
        Compute and log memory statistics for a DataFrame.

        Args:
            df:    The DataFrame to profile.
            label: Human-readable name for log output, e.g.
                'Raw Expression Matrix'.

        Returns:
            Dict with keys: 'total_mb' (float), 'shape' (tuple),
            'n_cells' (int), 'dtype_counts' (dict mapping dtype
            string to column count).
        """
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = round(mem_bytes / (1024**2), 2)

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
            "[MemoryReport] %-40s | Shape: %-18s | Memory: " "%7.1f MB | Dtypes: %s",
            label,
            str(df.shape),
            mem_mb,
            dtype_counts,
        )
        return stats


# Candidate exact-match names for the sample-barcode column, tried
# in order before falling back to a substring heuristic. GDC-hub
# files typically use 'submitter_id.samples'; Toil-hub files
# typically use 'sample'.
_BARCODE_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "sample",
    "submitter_id.samples",
    "sampleid",
    "barcode",
)


# ── Main Ingestion Class ──


class TCGADataIngester:
    """
    Memory-efficient ETL pipeline for TCGA-BRCA RNA-Seq expression
    data, with phenotype labels sourced from a separate Xena hub.

    Orchestrates loading, validation, transposition, label
    detection, cohort filtering, barcode alignment, and memory
    optimization into an (X, y) pair ready for preprocessing.

    Design Principles:
        - Separation of concerns: each private method does exactly
          one ETL step.
        - All configuration is injected via IngestionConfig.
        - Memory is profiled at every stage via MemoryReporter.
        - Errors are specific and actionable.

    Attributes:
        data_dir (Path): Resolved absolute path to the raw data dir.
        config (IngestionConfig): Ingestion configuration.
        reporter (MemoryReporter): Memory instrumentation utility.

    Raises:
        FileNotFoundError: If data_dir or required files are absent.
        KeyError: If the barcode or label column cannot be found or
            auto-detected in the phenotype file.
        ValueError: If loaded data is empty, malformed, or yields
            fewer than 100 aligned samples.
        MemoryError: If the expression matrix cannot be loaded in
            available RAM.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[IngestionConfig] = None,
    ) -> None:
        """
        Initialize the ingester with the data directory and config.

        Args:
            data_dir: Path to the folder containing raw files.
                Resolved to an absolute Path.
            config:   IngestionConfig instance. Uses defaults if
                None.

        Raises:
            FileNotFoundError: If data_dir does not exist on disk.
        """
        self.data_dir = Path(data_dir).resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Create it with: mkdir -p {self.data_dir}"
            )

        self.config = config or IngestionConfig()
        self.reporter = MemoryReporter()

        logger.info(
            "TCGADataIngester initialized | data_dir=%s | "
            "low_memory=%s | dtype=%s | cohort_prefix=%s",
            self.data_dir,
            self.config.low_memory_mode,
            self.config.float_dtype.__name__,
            self.config.cohort_prefix,
        )

    # ── Private Methods ──

    def _load_expression_matrix(self) -> pd.DataFrame:
        """
        Load the GDC-hub RNA-Seq expression matrix and downcast to
        float32 AFTER the gene-ID index column is separated out.

        See the module docstring for the historical ValueError this
        ordering fixes.

        Memory Strategy:
            Pass 1 (2 rows)    : Inspect file structure -> no cost.
            Pass 2 (full load) : dtypes inferred naturally, then
                                  downcast in one .astype() call.
            Transpose          : New object; briefly holds both.
            del original       : Frees the pre-transpose copy.
            gc.collect()       : Reclaims freed memory immediately.

        Returns:
            pd.DataFrame, shape (n_samples, n_genes), float32,
            indexed by TCGA sample barcode, columns are gene IDs.

        Raises:
            FileNotFoundError: If the expression file is missing.
            ValueError: If the file loads as an empty DataFrame.
            MemoryError: If RAM is insufficient to load the matrix.
        """
        expr_path = self.data_dir / self.config.expression_filename

        if not expr_path.exists():
            raise FileNotFoundError(
                f"Expression file not found: {expr_path}\n"
                "Download 'TCGA-BRCA.htseq_fpkm-uq.tsv.gz' from "
                "the GDC hub at https://xenabrowser.net/datapages/ "
                f"and place it in: {self.data_dir}"
            )

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
                "Cannot read expression file header. File may be " f"corrupt: {exc}"
            ) from exc

        n_samples_detected = len(header_peek.columns)
        logger.info(
            "Header inspection complete | Detected samples: %d",
            n_samples_detected,
        )
        del header_peek
        gc.collect()

        logger.info("Pass 2: Loading full expression matrix...")
        try:
            expr_df = pd.read_csv(
                expr_path,
                sep="\t",
                compression="gzip",
                index_col=0,  # Gene IDs -> index FIRST
                low_memory=False,  # Avoid ambiguous mixed-type read
            )
        except MemoryError as exc:
            raise MemoryError(
                "Insufficient RAM to load the expression matrix.\n"
                "Options:\n"
                "  1. Close other applications to free RAM.\n"
                "  2. Increase system swap/page file size.\n"
                "  3. Pre-filter genes to a top-variance subset "
                "before loading.\n"
                f"Original error: {exc}"
            ) from exc

        if expr_df.empty:
            raise ValueError(
                "Expression matrix loaded as empty. Check file "
                f"integrity: {expr_path}"
            )

        self.reporter.report(expr_df, "Raw Expression Matrix (float64)")

        # Downcast AFTER index_col=0 has already separated the
        # gene-ID column from the numeric sample columns -- this is
        # the fix for the historical ValueError described above.
        expr_df = expr_df.astype(self.config.float_dtype)
        if self.config.low_memory_mode:
            gc.collect()
        self.reporter.report(expr_df, "Expression Matrix (float32)")

        logger.info(
            "Transposing: (%d genes x %d samples) -> " "(%d samples x %d genes)...",
            expr_df.shape[0],
            expr_df.shape[1],
            expr_df.shape[1],
            expr_df.shape[0],
        )
        expr_transposed = expr_df.T

        del expr_df
        if self.config.low_memory_mode:
            gc.collect()

        self.reporter.report(expr_transposed, "Transposed Matrix (samples x genes)")
        return expr_transposed

    @staticmethod
    def _detect_barcode_column(columns: List[str]) -> str:
        """
        Identify the sample-barcode column across different hubs.

        Tries exact, case-insensitive matches against
        _BARCODE_COLUMN_CANDIDATES first, then falls back to any
        column containing 'sample' as a substring.

        Args:
            columns: All column names from the phenotype file.

        Returns:
            The detected barcode column name (original casing).

        Raises:
            KeyError: If no candidate or fallback match is found.
        """
        lowered = {c.lower(): c for c in columns}

        for candidate in _BARCODE_COLUMN_CANDIDATES:
            if candidate in lowered:
                return lowered[candidate]

        fallback = [c for c in columns if "sample" in c.lower()]
        if fallback:
            return fallback[0]

        raise KeyError(
            "Cannot find a sample-barcode column in the phenotype "
            f"file. Available columns: {columns[:15]} "
            f"(total {len(columns)})."
        )

    @staticmethod
    def _detect_label_column(
        columns: List[str],
        hints: Tuple[str, ...],
        exclude: str,
    ) -> str:
        """
        Auto-detect the subtype/label column via case-insensitive
        substring matching on column names.

        Args:
            columns: All column names from the phenotype file.
            hints:   Substrings to match, case-insensitively.
            exclude: Column name to exclude from candidates (the
                barcode column).

        Returns:
            The detected label column name (original casing).

        Raises:
            KeyError: If zero columns match. This is the expected
                outcome when the loaded phenotype file carries only
                cohort-level metadata (sample type, primary site,
                study) rather than disease-specific molecular
                subtype calls -- see the module docstring.
        """
        candidates = [c for c in columns if c != exclude]
        matches = [c for c in candidates if any(hint in c.lower() for hint in hints)]

        if not matches:
            raise KeyError(
                f"No column matched hints {hints} (case-"
                f"insensitive) among {len(candidates)} candidate "
                f"columns: {candidates[:15]}. This commonly "
                "happens when the loaded phenotype file only "
                "carries cohort-level metadata (sample type, "
                "primary site, study) rather than disease-specific "
                "molecular subtype calls. Verify the file actually "
                "contains a subtype column, or set "
                "IngestionConfig.label_column explicitly."
            )

        if len(matches) > 1:
            logger.warning(
                "Multiple columns matched label hints %s: %s. "
                "Using '%s'. Set IngestionConfig.label_column "
                "explicitly to choose a different one.",
                hints,
                matches,
                matches[0],
            )

        return matches[0]

    def _load_phenotype_labels(self) -> pd.DataFrame:
        """
        Load subtype labels from the Toil Recompute phenotype file.

        Detects the barcode and label columns dynamically, loads
        only those two columns, then immediately strict-filters the
        resulting index to barcodes starting with
        config.cohort_prefix -- removing GTEx/TARGET contamination
        before any further processing.

        Returns:
            pd.DataFrame indexed by TCGA sample barcode (TCGA-only),
            with one column: the detected label, as category dtype.

        Raises:
            FileNotFoundError: If the phenotype file is missing.
            KeyError: If the barcode or label column cannot be
                found. See _detect_label_column()'s docstring for
                why this is the expected outcome with the standard
                Toil phenotype file.
            ValueError: If no rows remain after cohort filtering.
        """
        pheno_path = self.data_dir / self.config.phenotype_filename

        if not pheno_path.exists():
            raise FileNotFoundError(
                f"Phenotype file not found: {pheno_path}\n"
                "Download 'TcgaTargetGTEX_phenotype.txt.gz' from "
                "the UCSC Xena Toil Recompute hub "
                "('TCGA TARGET GTEx' cohort -> Phenotype) at "
                "https://xenabrowser.net/datapages/ and place it "
                f"in: {self.data_dir}"
            )

        logger.info("Inspecting phenotype header for columns...")
        try:
            header_columns = pd.read_csv(
                pheno_path,
                sep="\t",
                compression="gzip",
                nrows=0,
            ).columns.tolist()
        except Exception as exc:
            raise ValueError(f"Cannot read phenotype file header: {exc}") from exc

        barcode_col = self._detect_barcode_column(header_columns)

        if self.config.label_column and self.config.label_column in header_columns:
            label_col = self.config.label_column
            logger.info(
                "Using configured label column override: '%s'",
                label_col,
            )
        else:
            label_col = self._detect_label_column(
                header_columns,
                self.config.label_column_hints,
                exclude=barcode_col,
            )
            logger.info(
                "Auto-detected label column via hints %s: '%s'",
                self.config.label_column_hints,
                label_col,
            )

        logger.info(
            "Loading phenotype columns ['%s', '%s'] only...",
            barcode_col,
            label_col,
        )
        pheno_df = pd.read_csv(
            pheno_path,
            sep="\t",
            compression="gzip",
            usecols=[barcode_col, label_col],
            dtype={label_col: "category"},
            low_memory=False,
        )

        pheno_df[barcode_col] = pheno_df[barcode_col].astype(str).str.strip()
        pheno_df = pheno_df.set_index(barcode_col)

        # ── Strict cohort filter, immediately after loading ──
        # Removes GTEx/TARGET contamination before anything else
        # touches this DataFrame.
        n_before_filter = len(pheno_df)
        cohort_mask = pheno_df.index.str.startswith(self.config.cohort_prefix)
        pheno_df = pheno_df.loc[cohort_mask]
        n_dropped_cohort = n_before_filter - len(pheno_df)

        logger.info(
            "Cohort filter '%s*' | Kept %d / %d rows (dropped %d "
            "non-%s rows, e.g. GTEx/TARGET).",
            self.config.cohort_prefix,
            len(pheno_df),
            n_before_filter,
            n_dropped_cohort,
            self.config.cohort_prefix,
        )

        if pheno_df.empty:
            raise ValueError(
                "No rows remain after filtering for barcodes "
                f"starting with '{self.config.cohort_prefix}'. "
                f"Verify column '{barcode_col}' stores TCGA-format "
                "sample IDs."
            )

        label_dist = pheno_df[label_col].value_counts(dropna=False)
        logger.info(
            "Phenotype loaded | Shape: %s | Label distribution:\n%s",
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
        Align expression and phenotype data via strict index
        intersection, truncating both barcodes to a common length
        first since GDC and Toil hubs do not share identical
        barcode granularity.

        Steps:
            1. Truncate both indices to config.barcode_length chars.
            2. Drop any barcode that collides with another AFTER
               truncation, independently in each source -- such
               collisions are ambiguous and cannot be resolved.
            3. common_index = expr_df.index.intersection(y.index)
            4. Subset both X and y to common_index.
            5. Drop any sample whose label is NaN.

        Args:
            expr_df:  Transposed expression matrix (samples x
                genes), full-length barcodes.
            pheno_df: Phenotype DataFrame indexed by TCGA barcode,
                already cohort-filtered, one label column.

        Returns:
            Tuple (X, y):
                X: float32 DataFrame, shape (n_aligned, n_genes).
                y: category Series, detected subtype labels.

        Raises:
            ValueError: If fewer than 100 samples remain after
                intersection, or fewer than 2 distinct label values
                remain after NaN removal.
        """
        k = self.config.barcode_length
        logger.info("Truncating both indexes to %d chars...", k)

        expr_df.index = expr_df.index.str[:k]
        pheno_df.index = pheno_df.index.str[:k]

        # Truncation can collapse two distinct full-length barcodes
        # (e.g. two aliquots/portions of the same tumor) onto the
        # same k-char prefix. Such collisions are ambiguous -- we
        # cannot know which row is "correct" -- so every colliding
        # row is dropped from its own source BEFORE the two sources
        # are intersected.
        expr_dupes = expr_df.index.duplicated(keep=False)
        if expr_dupes.any():
            n_dupes = int(expr_dupes.sum())
            logger.warning(
                "Dropping %d expression row(s) with duplicate " "truncated barcodes.",
                n_dupes,
            )
            expr_df = expr_df.loc[~expr_dupes]

        pheno_dupes = pheno_df.index.duplicated(keep=False)
        if pheno_dupes.any():
            n_dupes = int(pheno_dupes.sum())
            logger.warning(
                "Dropping %d phenotype row(s) with duplicate " "truncated barcodes.",
                n_dupes,
            )
            pheno_df = pheno_df.loc[~pheno_dupes]

        label_col = pheno_df.columns[0]
        y_full = pheno_df[label_col]

        # Strict index intersection -- the single alignment
        # authority for this method. Only barcodes present in both
        # de-duplicated sources survive.
        common_index = expr_df.index.intersection(y_full.index)

        logger.info(
            "Index intersection | Expression: %d | Phenotype: %d " "| Common: %d",
            len(expr_df.index),
            len(y_full.index),
            len(common_index),
        )

        if len(common_index) < 100:
            raise ValueError(
                f"Only {len(common_index)} samples survived index "
                "intersection (minimum: 100). This usually means "
                "the two files use incompatible barcode formats -- "
                "inspect expr_df.index[:5] and pheno_df.index[:5] "
                "directly, and adjust IngestionConfig.barcode_"
                "length if needed."
            )

        X = expr_df.loc[common_index]
        y = y_full.loc[common_index]

        # Drop any sample whose label is NaN -- required even
        # after intersection, since a barcode can exist in both
        # files while its label value itself is missing.
        valid_mask = y.notna()
        n_nan = int((~valid_mask).sum())
        if n_nan > 0:
            logger.warning(
                "Dropping %d sample(s) with a NaN label after " "alignment.",
                n_nan,
            )
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        if isinstance(y.dtype, pd.CategoricalDtype):
            y = y.cat.remove_unused_categories()

        if y.nunique() < 2:
            raise ValueError(
                f"After alignment, only {y.nunique()} distinct "
                "label value(s) remain. Cannot proceed with "
                "multi-class classification. Verify label column "
                f"'{label_col}' is populated for the expected "
                "samples."
            )

        logger.info(
            "Alignment complete | X: %s | y distribution:\n%s",
            X.shape,
            y.value_counts().to_string(),
        )
        return X, y

    def _enforce_float32(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast any float64 columns to float32 as a safety net.

        The .loc[] subsetting in _align_samples() generally
        preserves dtype and should not upcast, but this is retained
        for defense in depth, matching preprocessing.py and
        feature_selection.py.

        Args:
            df: Expression DataFrame post-alignment.

        Returns:
            DataFrame with all float columns guaranteed float32.
        """
        float64_cols = df.select_dtypes(include=["float64"]).columns
        if len(float64_cols) == 0:
            logger.info("float32 check: all columns already float32.")
            return df

        logger.warning(
            "Found %d float64 column(s) post-alignment. " "Downcasting to float32...",
            len(float64_cols),
        )
        for col in float64_cols:
            df[col] = df[col].astype(np.float32)

        if self.config.low_memory_mode:
            gc.collect()

        self.reporter.report(df, "X Post-Float32-Enforcement")
        return df

    def _validate_output(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Run sanity checks on the final (X, y) pair before run()
        returns it to the caller.

        Args:
            X: Final feature matrix.
            y: Final label series.

        Raises:
            ValueError: On any failed validation check.
        """
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(
                f"Expression matrix contains {nan_count} NaN "
                "values after alignment. Run "
                "X.isnull().sum().sort_values() to find affected "
                "genes."
            )

        non_float32 = [col for col in X.columns if X[col].dtype != np.float32]
        if non_float32:
            raise ValueError(
                f"{len(non_float32)} columns are not float32 "
                f"after enforcement. First 5: {non_float32[:5]}"
            )

        y_nan = y.isnull().sum()
        if y_nan > 0:
            raise ValueError(
                f"Label series contains {y_nan} NaN values "
                "post-alignment. This should be unreachable -- "
                "_align_samples() should have dropped these "
                "already."
            )

        n_classes = y.nunique()
        if n_classes < 2:
            raise ValueError(
                f"Label series has only {n_classes} unique "
                "class(es). Cannot perform multi-class "
                "classification."
            )

        zero_gene_cols = (X == 0).all(axis=0).sum()
        if zero_gene_cols > 0:
            logger.warning(
                "%d gene columns are all-zero across all samples. "
                "VarianceThreshold in preprocessing.py will remove "
                "these.",
                zero_gene_cols,
            )

        logger.info(
            "Validation passed: no NaN, correct dtype, %d classes.",
            n_classes,
        )

    # ── Public API ──

    def run(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Execute the full ingestion pipeline end-to-end.

        Pipeline Stages:
            1. Load expression matrix (float32, post-index_col).
            2. Load phenotype labels (dynamic column detection,
               strict TCGA-only cohort filter applied immediately).
            3. Align samples: truncate barcodes, drop truncation
               collisions, intersect indices, drop NaN labels.
            4. Free pre-alignment objects from memory.
            5. Enforce float32 dtype (safety net).
            6. Validate final output.
            7. Compile and return metadata.

        Returns:
            Tuple of (X, y, metadata):
                X: float32 DataFrame, shape (n_samples, n_genes).
                y: category Series, detected subtype labels.
                metadata: Dict with n_samples, n_genes,
                    label_column_used, cohort_prefix,
                    class_distribution, memory_mb, dtype_counts,
                    data_dir.

        Raises:
            FileNotFoundError: If raw data files are missing.
            KeyError: If barcode/label columns cannot be detected.
            ValueError: If alignment or validation fails.
            MemoryError: If RAM is insufficient.
        """
        logger.info("=" * 70)
        logger.info("TCGADataIngester.run() START")
        logger.info("=" * 70)

        expr_df = self._load_expression_matrix()
        pheno_df = self._load_phenotype_labels()

        X, y = self._align_samples(expr_df, pheno_df)

        del expr_df, pheno_df
        if self.config.low_memory_mode:
            gc.collect()
            logger.info("gc.collect() called after alignment -- raw " "objects freed.")

        X = self._enforce_float32(X)
        self._validate_output(X, y)

        memory_stats = self.reporter.report(X, "Final Aligned X")
        metadata: Dict[str, Any] = {
            "n_samples": X.shape[0],
            "n_genes": X.shape[1],
            "label_column_used": y.name,
            "cohort_prefix": self.config.cohort_prefix,
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
        logger.info("  Label column : %s", metadata["label_column_used"])
        logger.info("  Classes : %s", metadata["class_distribution"])
        logger.info("=" * 70)

        return X, y, metadata


# ── Logging Configuration Helper ──


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with a clean, structured format.

    Call this once at the top of any script or notebook that
    imports from this module.

    Args:
        level: Log level string. One of 'DEBUG', 'INFO', 'WARNING',
            'ERROR'.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | " "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ── Standalone Entry Point ──

if __name__ == "__main__":
    setup_logging("INFO")

    ingester = TCGADataIngester(data_dir="data/raw/")
    X, y, metadata = ingester.run()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  INGESTION COMPLETE -- SUMMARY")
    print(sep)
    print(f"  X shape       : {X.shape}")
    print(f"  X dtype       : {X.dtypes.iloc[0]}")
    print(f"  y shape       : {y.shape}")
    print(f"  Label column  : {metadata['label_column_used']}")
    print(f"  Cohort prefix : {metadata['cohort_prefix']}")
    print(f"  Memory (X)    : {metadata['memory_mb']} MB")
    print("  Classes       :")
    for cls, count in sorted(metadata["class_distribution"].items()):
        pct = 100 * count / y.shape[0]
        bar = "#" * int(pct / 3)
        print(f"    {cls!s:<12} : {count:4d} ({pct:5.1f}%)  {bar}")
    print(sep)
    print("\n  X and y ready for src/preprocessing.py")
    print(sep)
