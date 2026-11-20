import sys
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from data.loader import DatasetLoader
from data.preprocessor import DataPreprocessor
from models.classifier import ModelTrainer
from metaheuristics.nsga2 import NSGA2FeatureSelector
from explainability.shap_explainer import ShapExplainerEngine


class TestDefectIQUIPages(unittest.TestCase):
    """Automated integration tests for DefectIQ Streamlit pages using AppTest."""

    @classmethod
    def setUpClass(cls):
        loader = DatasetLoader()
        cls.X_raw, cls.y_raw, _ = loader.load_benchmark('PROMISE', 'PC1')

        preprocessor = DataPreprocessor(
            impute_strategy='median',
            remove_duplicates=True,
            scaling_method='standard',
            imbalance_method='smote',
            test_size=0.2,
            random_state=42
        )
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te, _ = preprocessor.fit_transform_split(cls.X_raw, cls.y_raw)

        trainer = ModelTrainer(random_state=42)
        cls.comp_df = trainer.train_all_models(cls.X_tr, cls.y_tr, cls.X_te, cls.y_te)
        cls.trainer = trainer

        nsga = NSGA2FeatureSelector(population_size=10, n_generations=2, random_state=42)
        nsga.fit(cls.X_tr, cls.y_tr)
        cls.nsga = nsga

        clf = trainer.trained_models['Random Forest']
        explainer = ShapExplainerEngine(clf, cls.X_tr, model_name='Random Forest')
        cls.global_res = explainer.compute_global_importance(cls.X_te, max_eval_samples=20)
        cls.explainer = explainer

    def test_all_11_pages_render_without_exceptions(self):
        nav_options = [
            'Overview',
            'Dataset Management',
            'Data Preprocessing',
            'Model Training',
            'Optimization Lab',
            'Pareto Analysis',
            'Defect Prediction',
            'Explainable AI',
            'Performance Evaluation',
            'Experiment History',
            'Settings'
        ]

        for page in nav_options:
            with self.subTest(page=page):
                at = AppTest.from_file('app.py')
                at.session_state['nav_target'] = page
                at.session_state['current_dataset_name'] = 'PC1 (PROMISE)'
                at.session_state['raw_X'] = self.X_raw
                at.session_state['raw_y'] = self.y_raw
                at.session_state['X_train'] = self.X_tr
                at.session_state['X_test'] = self.X_te
                at.session_state['y_train'] = self.y_tr
                at.session_state['y_test'] = self.y_te
                at.session_state['trainer'] = self.trainer
                at.session_state['benchmark_results'] = self.comp_df
                at.session_state['selected_features'] = self.nsga.best_selected_features
                at.session_state['active_feature_subset'] = 'selected'
                at.session_state['optimizer_history'] = self.nsga.history
                at.session_state['pareto_front'] = self.nsga.pareto_front
                at.session_state['knee_solution'] = self.nsga.best_tradeoff_solution
                at.session_state['explainer_engine'] = self.explainer
                at.session_state['current_explained_model'] = 'Random Forest'
                at.session_state['cached_global_shap'] = self.global_res

                at.run(timeout=45)
                self.assertFalse(at.exception, f"Exceptions occurred on page {page}: {[str(e.value) for e in at.exception]}")


if __name__ == '__main__':
    unittest.main()
