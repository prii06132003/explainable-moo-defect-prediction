"""
SQLite Experiment Tracking and Audit Trail Database.
Stores empirical metrics, selected feature subsets, optimization parameters,
and execution latencies for reproducible research benchmarking.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


class ExperimentDatabase:
    """Manages persistent SQLite storage of experimental runs and Pareto configurations."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(current_dir), "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "experiments.db")
        else:
            self.db_path = db_path

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database schema if not present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    optimizer TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    total_features INTEGER NOT NULL,
                    selected_features_count INTEGER NOT NULL,
                    selected_features_list TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    roc_auc REAL NOT NULL,
                    mcc REAL NOT NULL,
                    train_time_sec REAL NOT NULL,
                    infer_time_sec REAL NOT NULL,
                    notes TEXT
                )
            """)
            conn.commit()

    def log_experiment(
        self,
        dataset_name: str,
        optimizer: str,
        model_name: str,
        total_features: int,
        selected_features_list: List[str],
        metrics: Dict[str, Any],
        notes: str = ""
    ) -> int:
        """
        Inserts an experiment run into the database.
        Returns the created record ID.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feats_json = json.dumps(selected_features_list)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    timestamp, dataset_name, optimizer, model_name,
                    total_features, selected_features_count, selected_features_list,
                    accuracy, precision, recall, f1_score, roc_auc, mcc,
                    train_time_sec, infer_time_sec, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                dataset_name,
                optimizer,
                model_name,
                total_features,
                len(selected_features_list),
                feats_json,
                metrics.get("accuracy", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("f1_score", 0.0),
                metrics.get("roc_auc", 0.0),
                metrics.get("mcc", 0.0),
                metrics.get("train_time_sec", 0.0),
                metrics.get("infer_time_sec", 0.0),
                notes
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_experiments(self) -> pd.DataFrame:
        """Fetches all logged experiments as a pandas DataFrame."""
        with self._get_connection() as conn:
            query = "SELECT * FROM experiments ORDER BY id DESC"
            df = pd.read_sql_query(query, conn)
            return df

    def get_experiment_history(self) -> pd.DataFrame:
        """Alias for get_all_experiments."""
        return self.get_all_experiments()

    def clear_all(self) -> None:
        """Deletes all experiment records from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments")
            conn.commit()
