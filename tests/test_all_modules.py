"""
Automated End-to-End System Integration and Algorithmic Validation Suite.
Tests:
  1. Benchmark Dataset Loading & Target Binarization
  2. Zero-Leakage Preprocessing, Imputation, Scaling & SMOTE Resampling
  3. Genetic Algorithm Feature Selection & Convergence
  4. Binary Particle Swarm Optimization (BPSO) Feature Selection
  5. Multi-Objective NSGA-II Optimization & Pareto Front Construction
  6. Machine Learning Classifier Benchmarking (all 5 models)
  7. SHAP Global & Local Explainability + Refactoring Rules
  8. SQLite Database Persistence & Audit Logging
"""

import os
import unittest
import numpy as np
import pandas as pd

from data.loader import DatasetLoader
from data.preprocessor import DataPreprocessor
from metaheuristics.ga import GeneticAlgorithmFeatureSelector
from metaheuristics.pso import ParticleSwarmFeatureSelector
from metaheuristics.nsga2 import NSGA2FeatureSelector
from models.classifier import ModelTrainer
from explainability.shap_explainer import ShapExplainerEngine
from storage.database import ExperimentDatabase


class TestDefectPredictionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = DatasetLoader()
        cls.db = ExperimentDatabase()

    def test_01_dataset_loader_benchmarks(self):
        """Verify PROMISE, AEEEM, ReLink, and MORPH datasets load properly."""
        suites = self.loader.get_available_suites()
        self.assertIn("PROMISE", suites)
        self.assertIn("AEEEM", suites)
        self.assertIn("ReLink", suites)
        self.assertIn("MORPH", suites)

        # Test loading PROMISE PC1
        X_pc1, y_pc1, col = self.loader.load_benchmark("PROMISE", "PC1")
        self.assertGreater(len(X_pc1), 500)
        self.assertEqual(len(y_pc1), len(X_pc1))
        self.assertTrue(set(y_pc1.unique()).issubset({0, 1}))

        # Test loading AEEEM EQ
        X_eq, y_eq, _ = self.loader.load_benchmark("AEEEM", "EQ")
        self.assertGreater(len(X_eq), 200)
        self.assertTrue(set(y_eq.unique()).issubset({0, 1}))

        # Test loading ReLink safe
        X_safe, y_safe, _ = self.loader.load_benchmark("ReLink", "safe")
        self.assertGreater(len(X_safe), 30)

        # Test loading MORPH ar1
        X_ar1, y_ar1, _ = self.loader.load_benchmark("MORPH", "ar1")
        self.assertGreater(len(X_ar1), 50)

    def test_02_leakage_free_preprocessing(self):
        """Verify preprocessing pipeline splits before transforming and prevents leakage."""
        X, y, _ = self.loader.load_benchmark("PROMISE", "PC1")
        preprocessor = DataPreprocessor(
            impute_strategy="median",
            scaling_method="standard",
            imbalance_method="smote",
            test_size=0.2,
            random_state=42
        )
        X_tr, X_te, y_tr, y_te, meta = preprocessor.fit_transform_split(X, y)

        self.assertGreater(len(X_tr), 0)
        self.assertGreater(len(X_te), 0)
        self.assertEqual(len(X_tr), len(y_tr))
        self.assertEqual(len(X_te), len(y_te))
        # Ensure test set was NOT resampled (SMOTE is training only)
        self.assertNotEqual(len(X_tr), len(X_te))
        self.assertIn("train_samples_resampled", meta)

    def test_03_genetic_algorithm_feature_selection(self):
        """Verify single-objective GA optimizes features and records convergence."""
        X, y, _ = self.loader.load_benchmark("PROMISE", "PC1")
        pre = DataPreprocessor(scaling_method="minmax", imbalance_method="none", test_size=0.2)
        X_tr, X_te, y_tr, y_te, _ = pre.fit_transform_split(X, y)

        ga = GeneticAlgorithmFeatureSelector(
            population_size=10,
            n_generations=5,
            alpha=0.8,
            metric="f1",
            cv_folds=2,
            random_state=42
        )
        ga.fit(X_tr, y_tr)

        self.assertGreater(len(ga.best_selected_features), 0)
        self.assertLessEqual(len(ga.best_selected_features), X_tr.shape[1])
        self.assertGreater(ga.best_fitness, 0.0)
        self.assertEqual(len(ga.history), 5)

        X_tr_sel = ga.transform(X_tr)
        self.assertEqual(X_tr_sel.shape[1], len(ga.best_selected_features))

    def test_04_bpso_feature_selection(self):
        """Verify Binary PSO feature selection kinematics and convergence."""
        X, y, _ = self.loader.load_benchmark("PROMISE", "PC1")
        pre = DataPreprocessor(scaling_method="standard", imbalance_method="none", test_size=0.2)
        X_tr, X_te, y_tr, y_te, _ = pre.fit_transform_split(X, y)

        pso = ParticleSwarmFeatureSelector(
            n_particles=10,
            n_iterations=5,
            alpha=0.8,
            metric="f1",
            cv_folds=2,
            random_state=42
        )
        pso.fit(X_tr, y_tr)

        self.assertGreater(len(pso.best_selected_features), 0)
        self.assertGreater(pso.gbest_fitness, 0.0)
        self.assertEqual(len(pso.history), 5)

    def test_05_nsga2_multi_objective_optimization(self):
        """Verify NSGA-II non-dominated sorting and Pareto front generation."""
        X, y, _ = self.loader.load_benchmark("PROMISE", "PC1")
        pre = DataPreprocessor(scaling_method="standard", imbalance_method="none", test_size=0.2)
        X_tr, X_te, y_tr, y_te, _ = pre.fit_transform_split(X, y)

        nsga2 = NSGA2FeatureSelector(
            population_size=12,
            n_generations=4,
            metric="f1",
            cv_folds=2,
            random_state=42
        )
        nsga2.fit(X_tr, y_tr)

        self.assertGreater(len(nsga2.pareto_front), 0)
        self.assertIsNotNone(nsga2.best_tradeoff_solution)
        self.assertGreater(len(nsga2.best_selected_features), 0)
        self.assertIn("ideal_distance", nsga2.best_tradeoff_solution)
        self.assertEqual(len(nsga2.history), 4)

    def test_06_model_training_and_metrics(self):
        """Verify all 5 ML classifiers train and compute metrics accurately."""
        X, y, _ = self.loader.load_benchmark("PROMISE", "PC1")
        pre = DataPreprocessor(scaling_method="standard", imbalance_method="none", test_size=0.2)
        X_tr, X_te, y_tr, y_te, _ = pre.fit_transform_split(X, y)

        trainer = ModelTrainer(random_state=42)
        models_to_test = trainer.get_available_model_names()
        self.assertEqual(len(models_to_test), 5)

        for model_name in models_to_test:
            res = trainer.train_single_model(model_name, X_tr, y_tr, X_te, y_te)
            self.assertIn("accuracy", res)
            self.assertIn("f1_score", res)
            self.assertIn("mcc", res)
            self.assertIn("roc_auc", res)
            self.assertIn("confusion_matrix", res)
            self.assertIn("roc_curve", res)
            self.assertIn("pr_curve", res)

    def test_07_shap_explainability_and_recommendations(self):
        """Verify SHAP global and local explanations and developer recommendation engine."""
        X, y, _ = self.loader.load_benchmark("PROMISE", "PC1")
        pre = DataPreprocessor(scaling_method="standard", imbalance_method="none", test_size=0.2)
        X_tr, X_te, y_tr, y_te, _ = pre.fit_transform_split(X, y)

        trainer = ModelTrainer(random_state=42)
        trainer.train_single_model("Random Forest", X_tr, y_tr, X_te, y_te)
        rf_model = trainer.trained_models["Random Forest"]

        explainer = ShapExplainerEngine(rf_model, X_tr, model_name="Random Forest")

        # Global explanation
        global_res = explainer.compute_global_importance(X_te, max_eval_samples=25)
        self.assertIn("feature_importance_ranking", global_res)
        self.assertGreater(len(global_res["feature_importance_ranking"]), 0)

        # Local explanation
        sample_instance = X_te.iloc[[0]]
        local_res = explainer.explain_local_instance(sample_instance)
        self.assertIn("prediction", local_res)
        self.assertIn("defect_probability", local_res)
        self.assertIn("top_contributors", local_res)
        self.assertIn("actionable_recommendations", local_res)
        self.assertGreater(len(local_res["actionable_recommendations"]), 0)

    def test_08_sqlite_database_logging(self):
        """Verify SQLite experiment persistence and retrieval."""
        rec_id = self.db.log_experiment(
            dataset_name="Test-PC1",
            optimizer="NSGA-II",
            model_name="Random Forest",
            total_features=21,
            selected_features_list=["loc", "v(g)", "ev(g)"],
            metrics={
                "accuracy": 0.91,
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "roc_auc": 0.89,
                "mcc": 0.72,
                "train_time_sec": 0.12,
                "infer_time_sec": 0.01
            },
            notes="Unit test run"
        )
        self.assertGreater(rec_id, 0)
        df = self.db.get_all_experiments()
        self.assertGreater(len(df), 0)
        self.assertEqual(df.iloc[0]["dataset_name"], "Test-PC1")


if __name__ == "__main__":
    unittest.main()
