"""
Non-dominated Sorting Genetic Algorithm II (NSGA-II) for Multi-Objective Feature Selection.
Jointly optimizes:
  1. Predictive Performance (Minimize Error = 1 - Metric)
  2. Feature Complexity (Minimize Feature Count / Ratio)
  3. Computational Cost (Minimize Evaluation Time / Latency)

Generates complete non-dominated Pareto fronts, Hypervolume metrics, and Knee-point trade-offs.
"""

import time
from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier


class NSGA2FeatureSelector:
    """
    NSGA-II implementation tailored for Multi-Objective Feature Selection in Software Defect Prediction.
    """

    def __init__(
        self,
        estimator=None,
        population_size: int = 40,
        n_generations: int = 20,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.05,
        metric: str = "f1",
        cv_folds: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            estimator: Base classifier used for objective evaluation
            population_size: Number of candidate feature masks in population
            n_generations: Number of evolutionary generations
            crossover_rate: Crossover probability
            mutation_rate: Bit-flip mutation probability
            metric: Target classification metric ('f1', 'mcc', 'auc', 'accuracy')
            cv_folds: Number of cross-validation folds
            random_state: Seed for reproducibility
        """
        self.estimator = estimator if estimator is not None else RandomForestClassifier(n_estimators=50, random_state=random_state)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state)
        self.pareto_front: List[Dict[str, Any]] = []
        self.best_tradeoff_solution: Optional[Dict[str, Any]] = None
        self.best_selected_features: List[str] = []
        self.history: List[Dict[str, Any]] = []

    def _evaluate_individual(
        self,
        chromosome: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Evaluates the 3 minimization objectives for a chromosome:
          f1: Predictive Error (1.0 - Metric)
          f2: Feature Subset Ratio (N_selected / N_total)
          f3: Computational Cost (Evaluation execution latency in seconds)
        
        Returns: (f1, f2, f3, raw_metric_score)
        """
        start_eval = time.time()
        selected_indices = np.where(chromosome == 1)[0]
        n_selected = len(selected_indices)
        n_total = len(chromosome)

        if n_selected == 0:
            return 1.0, 1.0, 1.0, 0.0

        f2 = n_selected / n_total

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

        raw_metric = float(np.mean(scores))
        f1 = max(0.0, 1.0 - raw_metric)  # Minimize Error
        f3 = time.time() - start_eval    # Minimize Latency (sec)

        return float(f1), float(f2), float(f3), float(raw_metric)

    @staticmethod
    def _dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
        """Returns True if objective vector A dominates B (minimization)."""
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """
        Deb's Fast Non-Dominated Sorting algorithm.
        Returns a list of fronts, where front[0] is Pareto Rank 1.
        """
        N = len(objectives)
        domination_counts = np.zeros(N, dtype=int)
        dominated_solutions = [[] for _ in range(N)]
        fronts: List[List[int]] = [[]]

        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                if self._dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif self._dominates(objectives[q], objectives[p]):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                fronts[0].append(p)

        curr_front = 0
        while len(fronts[curr_front]) > 0:
            next_front = []
            for p in fronts[curr_front]:
                for q in dominated_solutions[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            curr_front += 1
            fronts.append(next_front)

        # Drop the last empty front
        return [f for f in fronts if len(f) > 0]

    @staticmethod
    def _calculate_crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
        """
        Calculates crowding distance for individuals within a Pareto front.
        """
        L = len(front)
        distances = np.zeros(L, dtype=float)
        if L <= 2:
            return np.full(L, np.inf)

        M = objectives.shape[1]
        front_objs = objectives[front]

        for m in range(M):
            sorted_indices = np.argsort(front_objs[:, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            f_min = front_objs[sorted_indices[0], m]
            f_max = front_objs[sorted_indices[-1], m]
            diff = f_max - f_min

            if diff == 0:
                continue

            for i in range(1, L - 1):
                prev_val = front_objs[sorted_indices[i - 1], m]
                next_val = front_objs[sorted_indices[i + 1], m]
                distances[sorted_indices[i]] += (next_val - prev_val) / diff

        return distances

    def _crowded_tournament_select(
        self,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding_dists: np.ndarray
    ) -> np.ndarray:
        """Crowded comparison tournament selection."""
        i, j = self.rng.choice(len(population), size=2, replace=False)
        if ranks[i] < ranks[j]:
            return population[i].copy()
        elif ranks[j] < ranks[i]:
            return population[j].copy()
        else:
            if crowding_dists[i] >= crowding_dists[j]:
                return population[i].copy()
            else:
                return population[j].copy()

    def _crossover_and_mutation(
        self,
        p1: np.ndarray,
        p2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover followed by bit-flip mutation."""
        if self.rng.rand() < self.crossover_rate:
            mask = self.rng.rand(len(p1)) < 0.5
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
        else:
            c1 = p1.copy()
            c2 = p2.copy()

        # Mutation
        for c in [c1, c2]:
            for gene in range(len(c)):
                if self.rng.rand() < self.mutation_rate:
                    c[gene] = 1 - c[gene]
            if np.sum(c) == 0:
                c[self.rng.randint(0, len(c))] = 1

        return c1, c2

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback: Optional[Callable[[int, int, float, int, float], None]] = None
    ) -> "NSGA2FeatureSelector":
        """
        Executes NSGA-II Multi-Objective Feature Selection.
        """
        start_time = time.time()
        X_arr = X.values
        y_arr = y.values
        feature_names = list(X.columns)
        n_features = len(feature_names)

        # Initialize random population
        population = self.rng.randint(0, 2, size=(self.population_size, n_features))
        for i in range(self.population_size):
            if np.sum(population[i]) == 0:
                population[i, self.rng.randint(0, n_features)] = 1

        # Evaluate initial population
        objectives = np.zeros((self.population_size, 3))
        raw_metrics = np.zeros(self.population_size)

        for i in range(self.population_size):
            f1, f2, f3, raw_m = self._evaluate_individual(population[i], X_arr, y_arr)
            objectives[i] = [f1, f2, f3]
            raw_metrics[i] = raw_m

        self.history = []

        for gen in range(self.n_generations):
            # Sort parent population
            fronts = self._fast_non_dominated_sort(objectives)
            ranks = np.zeros(self.population_size, dtype=int)
            crowding_dists = np.zeros(self.population_size, dtype=float)

            for rank_idx, front in enumerate(fronts):
                distances = self._calculate_crowding_distance(objectives, front)
                for f_idx, ind in enumerate(front):
                    ranks[ind] = rank_idx
                    crowding_dists[ind] = distances[f_idx]

            # Generate offspring population Q_t of size N
            offspring = []
            while len(offspring) < self.population_size:
                p1 = self._crowded_tournament_select(population, ranks, crowding_dists)
                p2 = self._crowded_tournament_select(population, ranks, crowding_dists)
                c1, c2 = self._crossover_and_mutation(p1, p2)
                offspring.append(c1)
                if len(offspring) < self.population_size:
                    offspring.append(c2)

            offspring = np.array(offspring)

            # Evaluate offspring
            offspring_objs = np.zeros((self.population_size, 3))
            offspring_metrics = np.zeros(self.population_size)

            for i in range(self.population_size):
                f1, f2, f3, raw_m = self._evaluate_individual(offspring[i], X_arr, y_arr)
                offspring_objs[i] = [f1, f2, f3]
                offspring_metrics[i] = raw_m

            # Combine Parents (P_t) and Offspring (Q_t) -> Size 2N
            combined_pop = np.vstack([population, offspring])
            combined_objs = np.vstack([objectives, offspring_objs])
            combined_metrics = np.concatenate([raw_metrics, offspring_metrics])

            # Non-dominated sorting on 2N solutions
            combined_fronts = self._fast_non_dominated_sort(combined_objs)

            # Select N best individuals for next generation P_{t+1}
            new_pop = []
            new_objs = []
            new_metrics = []

            for front in combined_fronts:
                if len(new_pop) + len(front) <= self.population_size:
                    for ind in front:
                        new_pop.append(combined_pop[ind])
                        new_objs.append(combined_objs[ind])
                        new_metrics.append(combined_metrics[ind])
                else:
                    # Fill remaining spots based on crowding distance
                    needed = self.population_size - len(new_pop)
                    distances = self._calculate_crowding_distance(combined_objs, front)
                    sorted_front = [front[idx] for idx in np.argsort(distances)[::-1]]
                    for ind in sorted_front[:needed]:
                        new_pop.append(combined_pop[ind])
                        new_objs.append(combined_objs[ind])
                        new_metrics.append(combined_metrics[ind])
                    break

            population = np.array(new_pop)
            objectives = np.array(new_objs)
            raw_metrics = np.array(new_metrics)

            # Generation statistics for Rank-1 Pareto solutions
            rank1_front = self._fast_non_dominated_sort(objectives)[0]
            best_f1_in_rank1 = float(np.min(objectives[rank1_front, 0]))  # Min error
            best_metric_val = round(1.0 - best_f1_in_rank1, 4)
            avg_features = round(float(np.mean(objectives[rank1_front, 1]) * n_features), 1)

            # Approximate hypervolume using reference point [1.0, 1.0, 1.0]
            # (Higher hypervolume signifies superior multi-objective convergence and spread)
            hv_approx = float(np.mean(
                (1.0 - np.clip(objectives[rank1_front, 0], 0, 1)) *
                (1.0 - np.clip(objectives[rank1_front, 1], 0, 1))
            ))

            gen_stat = {
                "generation": gen + 1,
                "pareto_solutions_count": len(rank1_front),
                "best_metric": best_metric_val,
                "avg_features_selected": avg_features,
                "hypervolume_2d": round(hv_approx, 4),
                "elapsed_time_sec": round(time.time() - start_time, 2)
            }
            self.history.append(gen_stat)

            if progress_callback:
                progress_callback(
                    gen + 1,
                    self.n_generations,
                    best_metric_val,
                    int(avg_features),
                    hv_approx
                )

        # Final Pareto Front Extraction (Rank 1)
        final_fronts = self._fast_non_dominated_sort(objectives)
        rank1_indices = final_fronts[0]

        self.pareto_front = []
        for idx in rank1_indices:
            mask = population[idx]
            sel_feats = [feature_names[j] for j in range(n_features) if mask[j] == 1]
            sol = {
                "solution_index": int(idx),
                "chromosome": mask.tolist(),
                "selected_features": sel_feats,
                "n_selected": len(sel_feats),
                "feature_ratio": round(float(objectives[idx, 1]), 4),
                "raw_metric": round(float(raw_metrics[idx]), 4),
                "predictive_error": round(float(objectives[idx, 0]), 4),
                "eval_latency_sec": round(float(objectives[idx, 2]), 4),
                # Distance to ideal objective point [0 (error), 0 (features), 0 (latency)]
                "ideal_distance": round(float(np.linalg.norm(objectives[idx])), 4)
            }
            self.pareto_front.append(sol)

        # Sort Pareto front by predictive metric descending
        self.pareto_front.sort(key=lambda s: s["raw_metric"], reverse=True)

        # Select Knee-point / Best Trade-off Solution (closest to ideal [0, 0, 0])
        self.best_tradeoff_solution = min(self.pareto_front, key=lambda s: s["ideal_distance"])
        self.best_selected_features = self.best_tradeoff_solution["selected_features"]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filters input DataFrame to only keep NSGA-II knee-point selected features."""
        if not self.best_selected_features:
            return X
        return X[self.best_selected_features]
