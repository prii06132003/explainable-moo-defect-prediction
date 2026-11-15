"""
Binary Particle Swarm Optimization (BPSO) for Feature Selection in Software Defect Prediction.
Uses continuous velocity kinematics mapped through a sigmoid transfer function
to dynamically search the binary feature space.
"""

import time
from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier


class ParticleSwarmFeatureSelector:
    """
    Binary Particle Swarm Optimization (BPSO) for feature selection.
    """

    def __init__(
        self,
        estimator=None,
        n_particles: int = 25,
        n_iterations: int = 25,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        v_max: float = 4.0,
        alpha: float = 0.85,
        metric: str = "f1",
        cv_folds: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            estimator: Base classifier used for fitness evaluation
            n_particles: Number of particles in swarm
            n_iterations: Maximum iteration steps
            w_max: Starting inertia weight
            w_min: Ending inertia weight
            c1: Cognitive acceleration coefficient
            c2: Social acceleration coefficient
            v_max: Velocity clamp limit [-v_max, +v_max]
            alpha: Weight balancing predictive performance vs feature reduction
            metric: Performance metric ('f1', 'mcc', 'auc', 'accuracy')
            cv_folds: Number of stratified cross-validation folds
            random_state: Seed for reproducibility
        """
        self.estimator = estimator if estimator is not None else RandomForestClassifier(n_estimators=50, random_state=random_state)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.alpha = alpha
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state)
        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_fitness: float = -1.0
        self.gbest_metric_score: float = 0.0
        self.best_selected_features: List[str] = []
        self.history: List[Dict[str, Any]] = []

    def _sigmoid(self, v: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(v >= 0, 1.0 / (1.0 + np.exp(-v)), np.exp(v) / (1.0 + np.exp(v)))

    def _evaluate_position(
        self,
        position: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluates a binary feature subset using Stratified K-Fold CV.
        Returns: (fitness, raw_metric_score)
        """
        selected_indices = np.where(position == 1)[0]
        n_selected = len(selected_indices)
        n_total = len(position)

        if n_selected == 0:
            return 0.0, 0.0

        X_sub = X[:, selected_indices]
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in skf.split(X_sub, y):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            clf = clone(self.estimator)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)

            if self.metric == "f1":
                score = f1_score(y_val, y_pred, zero_division=0)
            elif self.metric == "mcc":
                score = matthews_corrcoef(y_val, y_pred)
                score = (score + 1.0) / 2.0
            elif self.metric == "auc":
                try:
                    if hasattr(clf, "predict_proba"):
                        y_prob = clf.predict_proba(X_val)[:, 1]
                        score = roc_auc_score(y_val, y_prob)
                    else:
                        score = roc_auc_score(y_val, y_pred)
                except Exception:
                    score = accuracy_score(y_val, y_pred)
            else:
                score = accuracy_score(y_val, y_pred)

            scores.append(score)

        mean_metric = float(np.mean(scores))
        reduction_ratio = 1.0 - (n_selected / n_total)
        fitness = (self.alpha * mean_metric) + ((1.0 - self.alpha) * reduction_ratio)

        return float(fitness), mean_metric

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback: Optional[Callable[[int, int, float, float, int], None]] = None
    ) -> "ParticleSwarmFeatureSelector":
        """
        Executes Binary PSO search over feature space.
        """
        start_time = time.time()
        X_arr = X.values
        y_arr = y.values
        feature_names = list(X.columns)
        n_features = len(feature_names)

        # Initialize particles
        positions = self.rng.randint(0, 2, size=(self.n_particles, n_features))
        for i in range(self.n_particles):
            if np.sum(positions[i]) == 0:
                positions[i, self.rng.randint(0, n_features)] = 1

        velocities = self.rng.uniform(-self.v_max, self.v_max, size=(self.n_particles, n_features))

        pbest_positions = positions.copy()
        pbest_fitnesses = np.zeros(self.n_particles)
        pbest_metrics = np.zeros(self.n_particles)

        # Initial evaluation
        for i in range(self.n_particles):
            fit_val, metric_val = self._evaluate_position(positions[i], X_arr, y_arr)
            pbest_fitnesses[i] = fit_val
            pbest_metrics[i] = metric_val

        gbest_idx = np.argmax(pbest_fitnesses)
        self.gbest_fitness = pbest_fitnesses[gbest_idx]
        self.gbest_metric_score = pbest_metrics[gbest_idx]
        self.gbest_position = pbest_positions[gbest_idx].copy()
        self.best_selected_features = [
            feature_names[j] for j in range(n_features) if self.gbest_position[j] == 1
        ]

        self.history = []

        # Swarm iterations
        for it in range(self.n_iterations):
            # Dynamic inertia weight
            w = self.w_max - (it / max(1, self.n_iterations - 1)) * (self.w_max - self.w_min)

            for i in range(self.n_particles):
                r1 = self.rng.rand(n_features)
                r2 = self.rng.rand(n_features)

                # Velocity update
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (self.gbest_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # Clamp velocities
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

                # Position update via Sigmoid transfer function
                s_v = self._sigmoid(velocities[i])
                r_pos = self.rng.rand(n_features)
                positions[i] = np.where(r_pos < s_v, 1, 0)

                # Ensure non-empty subset
                if np.sum(positions[i]) == 0:
                    positions[i, self.rng.randint(0, n_features)] = 1

                # Evaluate new position
                fit_val, metric_val = self._evaluate_position(positions[i], X_arr, y_arr)

                # Update personal best
                if fit_val > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit_val
                    pbest_metrics[i] = metric_val
                    pbest_positions[i] = positions[i].copy()

                    # Update global best
                    if fit_val > self.gbest_fitness:
                        self.gbest_fitness = fit_val
                        self.gbest_metric_score = metric_val
                        self.gbest_position = positions[i].copy()
                        self.best_selected_features = [
                            feature_names[j] for j in range(n_features) if self.gbest_position[j] == 1
                        ]

            iter_stat = {
                "iteration": it + 1,
                "gbest_fitness": round(float(self.gbest_fitness), 4),
                "avg_fitness": round(float(np.mean(pbest_fitnesses)), 4),
                "gbest_metric": round(float(self.gbest_metric_score), 4),
                "features_selected_count": len(self.best_selected_features),
                "features_selected_ratio": round(len(self.best_selected_features) / n_features, 3),
                "inertia_weight": round(float(w), 3),
                "elapsed_time_sec": round(time.time() - start_time, 2)
            }
            self.history.append(iter_stat)

            if progress_callback:
                progress_callback(
                    it + 1,
                    self.n_iterations,
                    self.gbest_fitness,
                    self.gbest_metric_score,
                    len(self.best_selected_features)
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filters input DataFrame to only keep BPSO-selected features."""
        if not self.best_selected_features:
            return X
        return X[self.best_selected_features]
