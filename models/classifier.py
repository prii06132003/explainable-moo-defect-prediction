"""
Machine Learning Classifier Suite and Evaluation Engine for Software Defect Prediction.
Implements:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Support Vector Machine (SVM)
  5. Gradient Boosting

Computes comprehensive software defect metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MCC, etc.),
confusion matrices, ROC/PR curve points, and handles model persistence.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import joblib


AVAILABLE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


class ModelTrainer:
    """Manages training, cross-validation, and multi-metric benchmarking of classification algorithms."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "SVM": SVC(probability=True, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
        }
        self.trained_models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}

    def get_available_model_names(self) -> List[str]:
        return list(self.models.keys())

    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Fits a single classifier, measures training and inference latency,
        and computes all scientific evaluation metrics.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(self.models.keys())}")

        clf = clone(self.models[model_name])

        # Train with latency measurement
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        # Predict with inference latency measurement
        t1 = time.time()
        y_pred = clf.predict(X_test)
        infer_time = time.time() - t1

        # Predict probabilities if available
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            df_vals = clf.decision_function(X_test)
            y_proba = 1.0 / (1.0 + np.exp(-df_vals))
        else:
            y_proba = y_pred.astype(float)

        # Scientific Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_pred))
        kappa = float(cohen_kappa_score(y_test, y_pred))

        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            auc = acc

        # Confusion Matrix elements
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Degenerate case fallback
            tn, fp, fn, tp = len(y_test), 0, 0, 0

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # ROC Curve Points
        fpr_pts, tpr_pts, _ = roc_curve(y_test, y_proba)
        # Precision-Recall Curve Points
        prec_pts, rec_pts, _ = precision_recall_curve(y_test, y_proba)

        result = {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "mcc": round(mcc, 4),
            "kappa": round(kappa, 4),
            "train_time_sec": round(train_time, 4),
            "infer_time_sec": round(infer_time, 4),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            },
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr_pts],
                "tpr": [round(float(x), 4) for x in tpr_pts]
            },
            "pr_curve": {
                "precision": [round(float(x), 4) for x in prec_pts],
                "recall": [round(float(x), 4) for x in rec_pts]
            },
            "y_pred": y_pred.tolist(),
            "y_proba": [round(float(p), 4) for p in y_proba]
        }

        self.trained_models[model_name] = clf
        self.evaluation_results[model_name] = result
        return result

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """Trains and benchmarks all 5 classifiers, returning a comparison DataFrame."""
        summary_rows = []
        for model_name in self.models:
            res = self.train_single_model(model_name, X_train, y_train, X_test, y_test)
            summary_rows.append({
                "Model": model_name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1-Score": res["f1_score"],
                "ROC-AUC": res["roc_auc"],
                "MCC": res["mcc"],
                "Train Time (s)": res["train_time_sec"],
                "Inference (s)": res["infer_time_sec"]
            })

        return pd.DataFrame(summary_rows)

    def save_model(self, model_name: str, filepath: str) -> None:
        """Serializes trained model to file."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet.")
        joblib.dump(self.trained_models[model_name], filepath)

    def load_model(self, model_name: str, filepath: str) -> Any:
        """Loads serialized model from file."""
        clf = joblib.load(filepath)
        self.trained_models[model_name] = clf
        return clf
