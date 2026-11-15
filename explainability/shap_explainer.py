"""
Explainable AI (XAI) Engine using SHAP (SHapley Additive exPlanations).
Provides Global Feature Importance, Local Prediction Decompositions (Waterfall),
and Actionable Software Engineering Refactoring Guidance based on empirical metric impact.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import shap


# Metric taxonomy mapping software engineering metrics to developer refactoring guidance
METRIC_RECOMMENDATION_RULES = {
    "cyclomatic": {
        "title": "High Cyclomatic Complexity",
        "condition": "increases defect risk",
        "action": "Decompose deeply nested control structures (if/else/switch/loops). Extract complex branch predicates into self-documenting helper functions or employ the Strategy Pattern."
    },
    "loc": {
        "title": "Excessive Module Size (Lines of Code)",
        "condition": "increases defect risk",
        "action": "Module violates Single Responsibility Principle (SRP). Split large classes or monolithic functions into cohesive sub-components and modular services."
    },
    "halstead": {
        "title": "Halstead Cognitive / Mental Effort Overload",
        "condition": "increases defect risk",
        "action": "High density of unique operators and operands increases developer cognitive load. Cache intermediate variables and break expressions into simpler declarative steps."
    },
    "coupling": {
        "title": "High Inter-Class Coupling (CBO / RFC)",
        "condition": "increases defect risk",
        "action": "Module is tightly coupled to external classes. Introduce dependency injection and decouple through abstract interfaces to minimize ripple effects."
    },
    "cohesion": {
        "title": "Lack of Cohesion in Methods (LCOM)",
        "condition": "increases defect risk",
        "action": "Methods inside this class operate on disjoint state variables. Decompose this class into smaller, highly cohesive units."
    },
    "entropy": {
        "title": "High Change Entropy / Code Churn",
        "condition": "increases defect risk",
        "action": "Frequent scattered historical modifications indicate unstable architecture. Perform targeted integration testing and freeze fragile APIs."
    }
}


class ShapExplainerEngine:
    """Manages SHAP model explainability for global feature impact and local defect diagnoses."""

    def __init__(self, model: Any, X_train: pd.DataFrame, model_name: str = "Classifier"):
        self.model = model
        self.X_train = X_train
        self.model_name = model_name
        self.feature_names = list(X_train.columns)
        self.explainer = self._init_explainer()

    def _init_explainer(self) -> Any:
        """Instantiates appropriate SHAP Explainer variant based on algorithm category."""
        # Tree-based algorithms
        tree_models = ["Random Forest", "Decision Tree", "Gradient Boosting"]
        if any(tm.lower() in self.model_name.lower() for tm in tree_models):
            try:
                return shap.TreeExplainer(self.model)
            except Exception:
                pass

        # Linear model
        if "logistic" in self.model_name.lower():
            try:
                # Use a representative background sample (up to 100 rows)
                bg = shap.sample(self.X_train, min(100, len(self.X_train)))
                return shap.LinearExplainer(self.model, bg)
            except Exception:
                pass

        # General Kernel / Permutation Explainer with background sample
        bg_sample = shap.sample(self.X_train, min(50, len(self.X_train)))
        predict_fn = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
        return shap.Explainer(predict_fn, bg_sample)

    def compute_global_importance(self, X_eval: pd.DataFrame, max_eval_samples: int = 150) -> Dict[str, Any]:
        """
        Computes Global SHAP feature importance across test instances.
        Returns sorted feature rankings and distribution statistics.
        """
        eval_sample = X_eval.sample(n=min(len(X_eval), max_eval_samples), random_state=42) if len(X_eval) > max_eval_samples else X_eval

        shap_values = self._get_shap_values(eval_sample)
        # Take positive class (defective = 1) if binary output
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1]
        elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
            vals = shap_values.values[:, :, 1]
        elif hasattr(shap_values, "values"):
            vals = shap_values.values
        else:
            vals = np.array(shap_values)

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        rankings = []
        for idx in sorted_indices:
            feat = self.feature_names[idx]
            rankings.append({
                "feature": feat,
                "importance": round(float(mean_abs_shap[idx]), 4),
                "mean_shap": round(float(np.mean(vals[:, idx])), 4),
                "min_shap": round(float(np.min(vals[:, idx])), 4),
                "max_shap": round(float(np.max(vals[:, idx])), 4)
            })

        return {
            "model_name": self.model_name,
            "evaluated_samples": len(eval_sample),
            "feature_importance_ranking": rankings,
            "raw_shap_matrix": vals.tolist(),
            "feature_names": self.feature_names
        }

    def explain_local_instance(self, instance_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decomposes individual prediction into additive SHAP feature contributions (Waterfall).
        Generates actionable software engineering refactoring advice.
        """
        if len(instance_df) != 1:
            instance_df = instance_df.iloc[[0]]

        # Ensure correct column ordering
        aligned_df = instance_df[self.feature_names]

        shap_values = self._get_shap_values(aligned_df)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
            base_val = float(self.explainer.expected_value[1]) if hasattr(self.explainer, "expected_value") else 0.5
        elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
            vals = shap_values.values[0, :, 1]
            base_val = float(shap_values.base_values[0, 1])
        elif hasattr(shap_values, "values"):
            vals = shap_values.values[0]
            base_val = float(shap_values.base_values[0]) if hasattr(shap_values, "base_values") else 0.5
        else:
            arr = np.array(shap_values)
            vals = arr[0] if arr.ndim > 1 else arr
            base_val = 0.5

        contributions = []
        for i, feat in enumerate(self.feature_names):
            actual_val = float(aligned_df.iloc[0, i])
            shap_impact = float(vals[i])
            contributions.append({
                "feature": feat,
                "actual_value": round(actual_val, 4),
                "shap_value": round(shap_impact, 4),
                "impact_direction": "Increases Defect Risk" if shap_impact > 0 else "Decreases Defect Risk"
            })

        # Sort by absolute SHAP impact
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        # Predict probability
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(aligned_df)[0, 1])
        else:
            prob = float(self.model.predict(aligned_df)[0])

        is_defective = prob >= 0.5

        # Generate software engineering actionable advice
        recommendations = self._generate_se_recommendations(contributions)

        return {
            "prediction": "Defective" if is_defective else "Non-Defective",
            "defect_probability": round(prob, 4),
            "base_value": round(base_val, 4),
            "top_contributors": contributions,
            "actionable_recommendations": recommendations
        }

    def _get_shap_values(self, X_df: pd.DataFrame) -> Any:
        """Internal helper to compute SHAP values safely."""
        try:
            return self.explainer(X_df)
        except Exception:
            try:
                return self.explainer.shap_values(X_df)
            except Exception:
                # Fallback to general explainer
                bg = shap.sample(self.X_train, min(30, len(self.X_train)))
                ex = shap.Explainer(self.model.predict, bg)
                return ex(X_df)

    def _generate_se_recommendations(self, top_contributions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Translates metric SHAP values into actionable developer guidance.
        """
        recommendations = []
        # Examine top risk drivers (positive SHAP values)
        risk_drivers = [c for c in top_contributions if c["shap_value"] > 0]

        for item in risk_drivers[:4]:
            feat_lower = item["feature"].lower()
            matched_rule = None

            if any(k in feat_lower for k in ["v(g)", "ev(g)", "iv(g)", "cyclo", "branch", "complex"]):
                matched_rule = METRIC_RECOMMENDATION_RULES["cyclomatic"]
            elif any(k in feat_lower for k in ["loc", "lines", "sloc", "countline"]):
                matched_rule = METRIC_RECOMMENDATION_RULES["loc"]
            elif any(k in feat_lower for k in ["halstead", "vol", "diff", "effort", "length", "v", "b"]):
                matched_rule = METRIC_RECOMMENDATION_RULES["halstead"]
            elif any(k in feat_lower for k in ["cbo", "rfc", "fanout", "ca", "ce", "call"]):
                matched_rule = METRIC_RECOMMENDATION_RULES["coupling"]
            elif any(k in feat_lower for k in ["lcom", "cohesion"]):
                matched_rule = METRIC_RECOMMENDATION_RULES["cohesion"]
            elif any(k in feat_lower for k in ["churn", "entropy", "commit", "dev"]):
                matched_rule = METRIC_RECOMMENDATION_RULES["entropy"]

            if matched_rule:
                recommendations.append({
                    "feature": item["feature"],
                    "value": str(item["actual_value"]),
                    "title": matched_rule["title"],
                    "recommendation": matched_rule["action"]
                })

        if not recommendations:
            recommendations.append({
                "feature": "General Architecture",
                "value": "N/A",
                "title": "Maintain Clean Code Standards",
                "recommendation": "Metric levels are within safe operating thresholds. Continue regular unit testing, code reviews, and static analysis verification."
            })

        return recommendations
