"""
Dataset Loader and Benchmark Management for Software Defect Prediction.
Supports PROMISE (NASA MDP), AEEEM, ReLink, and MORPH benchmark suites,
as well as custom user-uploaded CSV files.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


BENCHMARK_CATALOG = {
    "PROMISE": {
        "PC1": {
            "name": "PC1 (Flight Software)",
            "domain": "NASA earth-orbiting satellite flight software (C)",
            "metrics_type": "McCabe & Halstead software metrics",
            "file": "PC1.csv"
        },
        "PC3": {
            "name": "PC3 (Flight Software)",
            "domain": "NASA flight instrument software (C)",
            "metrics_type": "McCabe, Halstead, Line count metrics",
            "file": "PC3.csv"
        },
        "PC4": {
            "name": "PC4 (Flight Software)",
            "domain": "NASA spacecraft instrument flight software (C)",
            "metrics_type": "McCabe, Halstead, Line count metrics",
            "file": "PC4.csv"
        },
        "MW1": {
            "name": "MW1 (Microgravity Experiment)",
            "domain": "Zero-gravity experiment software (C)",
            "metrics_type": "McCabe, Halstead metrics",
            "file": "MW1.csv"
        },
        "MC2": {
            "name": "MC2 (Video Guidance)",
            "domain": "Video guidance system software (C++)",
            "metrics_type": "McCabe, Halstead, Function metrics",
            "file": "MC2.csv"
        },
        "KC1": {
            "name": "KC1 (Storage Management)",
            "domain": "NASA ground system for receiving/processing telemetry (C++)",
            "metrics_type": "McCabe & Halstead metrics",
            "file": "KC1.csv"
        },
        "KC2": {
            "name": "KC2 (Storage Management)",
            "domain": "NASA ground system data processing (C++)",
            "metrics_type": "McCabe & Halstead metrics",
            "file": "KC2.csv"
        },
        "CM1": {
            "name": "CM1 (Spacecraft Instrument)",
            "domain": "NASA spacecraft instrument software (C)",
            "metrics_type": "McCabe, Halstead & Line count metrics",
            "file": "CM1.csv"
        }
    },
    "AEEEM": {
        "EQ": {
            "name": "Equinox (AEEEM)",
            "domain": "Eclipse OSGi core runtime framework (Java)",
            "metrics_type": "61 Source code, Change & Entropy metrics (D'Ambros et al.)",
            "file": "EQ.csv"
        },
        "JDT": {
            "name": "Eclipse JDT (AEEEM)",
            "domain": "Eclipse Java Development Tools (Java)",
            "metrics_type": "61 Source code, Change & Entropy metrics",
            "file": "JDT.csv"
        },
        "LC": {
            "name": "Lucene (AEEEM)",
            "domain": "Apache Lucene search engine library (Java)",
            "metrics_type": "61 Source code, Change & Entropy metrics",
            "file": "LC.csv"
        },
        "ML": {
            "name": "Mylyn (AEEEM)",
            "domain": "Eclipse Mylyn task-focused interface (Java)",
            "metrics_type": "61 Source code, Change & Entropy metrics",
            "file": "ML.csv"
        },
        "PDE": {
            "name": "PDE (AEEEM)",
            "domain": "Eclipse Plugin Development Environment (Java)",
            "metrics_type": "61 Source code, Change & Entropy metrics",
            "file": "PDE.csv"
        }
    },
    "ReLink": {
        "apache": {
            "name": "Apache (ReLink)",
            "domain": "Apache HTTP Server component",
            "metrics_type": "26 Defect-corrected software metrics (Wu et al.)",
            "file": "apache.csv"
        },
        "safe": {
            "name": "Safe (ReLink)",
            "domain": "J2ME mobile password manager application",
            "metrics_type": "26 Defect-corrected software metrics",
            "file": "safe.csv"
        },
        "zxing": {
            "name": "Zxing (ReLink)",
            "domain": "Zxing 1D/2D barcode image processing library",
            "metrics_type": "26 Defect-corrected software metrics",
            "file": "zxing.csv"
        }
    },
    "MORPH": {
        "ar1": {
            "name": "AR1 (Softlab MORPH)",
            "domain": "Embedded controller system software (Turkish software organization)",
            "metrics_type": "29 Software product & defect metrics (Turhan et al.)",
            "file": "ar1.csv"
        },
        "ar3": {
            "name": "AR3 (Softlab MORPH)",
            "domain": "Telecommunication system software",
            "metrics_type": "29 Software product & defect metrics",
            "file": "ar3.csv"
        },
        "ar4": {
            "name": "AR4 (Softlab MORPH)",
            "domain": "Financial transaction software system",
            "metrics_type": "29 Software product & defect metrics",
            "file": "ar4.csv"
        },
        "ar5": {
            "name": "AR5 (Softlab MORPH)",
            "domain": "Industrial automation software module",
            "metrics_type": "29 Software product & defect metrics",
            "file": "ar5.csv"
        },
        "ar6": {
            "name": "AR6 (Softlab MORPH)",
            "domain": "Measurement and calibration instrument software",
            "metrics_type": "29 Software product & defect metrics",
            "file": "ar6.csv"
        }
    }
}

DEFECT_COLUMN_CANDIDATES = [
    "class", "defects", "problems", "bug", "is_defective",
    "defect", "c", "Defective", "bugs", "bug_binary"
]


class DatasetLoader:
    """Manages loading, parsing, and inspection of software defect datasets."""

    def __init__(self, benchmarks_dir: Optional[str] = None):
        if benchmarks_dir is None:
            # Default to data/benchmarks relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.benchmarks_dir = os.path.join(current_dir, "benchmarks")
        else:
            self.benchmarks_dir = benchmarks_dir

    def get_available_suites(self) -> List[str]:
        """Returns the list of benchmark suites supported."""
        return list(BENCHMARK_CATALOG.keys())

    def get_datasets_in_suite(self, suite: str) -> Dict[str, Dict[str, str]]:
        """Returns metadata for all datasets in a given suite."""
        return BENCHMARK_CATALOG.get(suite, {})

    def load_benchmark(self, suite: str, dataset_key: str) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Loads a built-in benchmark dataset.
        Returns:
            X: Feature DataFrame
            y: Binary target Series (0 = Non-defective, 1 = Defective)
            target_col: Name of detected target column
        """
        suite_info = BENCHMARK_CATALOG.get(suite)
        if not suite_info:
            raise ValueError(f"Unknown benchmark suite: '{suite}'. Available: {list(BENCHMARK_CATALOG.keys())}")

        dataset_meta = suite_info.get(dataset_key)
        if not dataset_meta:
            raise ValueError(f"Unknown dataset '{dataset_key}' in suite '{suite}'. Available: {list(suite_info.keys())}")

        file_name = dataset_meta["file"]
        suite_folder = suite.lower()
        file_path = os.path.join(self.benchmarks_dir, suite_folder, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmark file not found at: {file_path}")

        df = pd.read_csv(file_path)
        return self.parse_dataset(df)

    def load_custom_csv(self, file_source: Any, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Loads a custom CSV dataset from file path, StringIO, or uploaded buffer.
        """
        if isinstance(file_source, str) and os.path.exists(file_source):
            df = pd.read_csv(file_source)
        else:
            df = pd.read_csv(file_source)
        return self.parse_dataset(df, target_col=target_col)

    def parse_dataset(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Identifies target defect column, standardizes target to binary {0, 1},
        and cleans feature columns (dropping non-predictive identifiers).
        """
        clean_df = df.copy()

        # Drop common index/id columns if present
        cols_to_drop = [c for c in clean_df.columns if c.lower() in [
            'id', 'unnamed: 0', 'index', 'project', 'version', 'name', 'classname', 'filename'
        ]]
        if cols_to_drop:
            clean_df = clean_df.drop(columns=cols_to_drop)

        # Detect target column if not explicitly supplied
        if target_col is None or target_col not in clean_df.columns:
            target_col = self._detect_target_column(clean_df)

        if not target_col:
            # Fallback to the very last column
            target_col = clean_df.columns[-1]

        y_raw = clean_df[target_col]
        X = clean_df.drop(columns=[target_col])

        # Standardize target to binary 0 and 1
        y = self._binarize_target(y_raw)
        y.name = "defective"

        # Ensure all X features are numeric
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce')

        return X, y, target_col

    def _detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Searches for common software defect target column names."""
        lower_cols = {col.lower().strip(): col for col in df.columns}
        for cand in DEFECT_COLUMN_CANDIDATES:
            if cand in lower_cols:
                return lower_cols[cand]

        # Check last column
        return df.columns[-1] if len(df.columns) > 0 else None

    def _binarize_target(self, target_series: pd.Series) -> pd.Series:
        """
        Converts varied target formats (true/false, Y/N, integer bug counts, strings)
        into standard binary int {0: Non-defective, 1: Defective}.
        """
        s = target_series.copy()
        if np.issubdtype(s.dtype, np.number):
            # Numeric: >0 or ==1 means defective
            return (s > 0).astype(int)

        # String or categorical handling
        s_str = s.astype(str).str.strip().str.lower()
        true_values = {'true', 'y', 'yes', 'bug', 'defective', '1', '1.0', 't', 'positive', 'c'}
        false_values = {'false', 'n', 'no', 'clean', 'non-defective', '0', '0.0', 'f', 'negative'}

        def map_val(val):
            if val in true_values:
                return 1
            if val in false_values:
                return 0
            try:
                num = float(val)
                return 1 if num > 0 else 0
            except ValueError:
                return 0

        return s_str.apply(map_val).astype(int)

    @staticmethod
    def get_dataset_summary(X: pd.DataFrame, y: pd.Series, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Calculates comprehensive summary statistics and imbalance metrics for a dataset.
        """
        n_samples, n_features = X.shape
        counts = y.value_counts().to_dict()
        n_clean = counts.get(0, 0)
        n_defective = counts.get(1, 0)
        defective_pct = (n_defective / n_samples * 100) if n_samples > 0 else 0.0
        imbalance_ratio = (n_clean / n_defective) if n_defective > 0 else 0.0

        missing_count = int(X.isna().sum().sum())
        missing_pct = (missing_count / (n_samples * n_features) * 100) if (n_samples * n_features) > 0 else 0.0

        return {
            "dataset_name": dataset_name,
            "total_modules": n_samples,
            "total_features": n_features,
            "non_defective_count": n_clean,
            "defective_count": n_defective,
            "defective_percentage": round(defective_pct, 2),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "missing_values_count": missing_count,
            "missing_values_percentage": round(missing_pct, 2),
            "feature_names": list(X.columns)
        }
