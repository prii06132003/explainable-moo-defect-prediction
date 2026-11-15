"""
Genetic Algorithm (GA) for Metaheuristic Feature Selection in Software Defect Prediction.
Optimizes software metric subsets to maximize predictive performance while minimizing dimensionality.
"""

import time
from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier


class GeneticAlgorithmFeatureSelector:
    """
    Binary Genetic Algorithm implementation for feature selection.
    """

    def __init__(
        self,
        estimator=None,
        population_size: int = 30,
        n_generations: int = 25,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        elitism_count: int = 2,
        alpha: float = 0.85,
        metric: str = "f1",
        cv_folds: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            estimator: Base classifier used for fitness evaluation
            population_size: Number of chromosomes per generation
            n_generations: Maximum iteration cycles
            crossover_rate: Probability of crossover between parent pairs
            mutation_rate: Probability of bit mutation per gene
            tournament_size: Number of individuals competing in tournament selection
            elitism_count: Number of top chromosomes preserved directly
            alpha: Trade-off weight (alpha * metric + (1 - alpha) * feature_reduction)
            metric: 'f1', 'mcc', 'auc', or 'accuracy'
            cv_folds: Number of stratified cross-validation folds for fitness estimation
            random_state: Seed for reproducibility
        """
        self.estimator = estimator if estimator is not None else RandomForestClassifier(n_estimators=50, random_state=random_state)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.alpha = alpha
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state)
        self.best_chromosome: Optional[np.ndarray] = None
        self.best_fitness: float = -1.0
        self.best_metric_score: float = 0.0
        self.best_selected_features: List[str] = []
        self.history: List[Dict[str, Any]] = []

    def _init_population(self, n_features: int) -> np.ndarray:
        """Initializes binary population ensuring at least one feature is selected per chromosome."""
        pop = self.rng.randint(0, 2, size=(self.population_size, n_features))
        for i in range(self.population_size):
            if np.sum(pop[i]) == 0:
                pop[i, self.rng.randint(0, n_features)] = 1
        return pop

    def _evaluate_chromosome(
        self,
        chromosome: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates fitness for a chromosome using Stratified K-Fold CV.
        Returns: (fitness, raw_metric_score)
        """
        selected_indices = np.where(chromosome == 1)[0]
        n_selected = len(selected_indices)
        n_total = len(chromosome)

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
                score = (score + 1.0) / 2.0  # normalize MCC from [-1, 1] to [0, 1]
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

    def _tournament_select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Tournament selection operator."""
        indices = self.rng.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover operator."""
        if self.rng.rand() > self.crossover_rate:
            return p1.copy(), p2.copy()

        mask = self.rng.rand(len(p1)) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)

        # Ensure at least 1 feature is selected
        if np.sum(c1) == 0:
            c1[self.rng.randint(0, len(c1))] = 1
        if np.sum(c2) == 0:
            c2[self.rng.randint(0, len(c2))] = 1

        return c1, c2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit-flip mutation operator."""
        mutated = chromosome.copy()
        for gene_idx in range(len(mutated)):
            if self.rng.rand() < self.mutation_rate:
                mutated[gene_idx] = 1 - mutated[gene_idx]

        if np.sum(mutated) == 0:
            mutated[self.rng.randint(0, len(mutated))] = 1
        return mutated

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback: Optional[Callable[[int, int, float, float, int], None]] = None
    ) -> "GeneticAlgorithmFeatureSelector":
        """
        Executes the Genetic Algorithm feature selection search.
        
        Args:
            X: Input feature DataFrame
            y: Binary target Series
            progress_callback: Optional callback func(gen, max_gen, best_fit, best_metric, n_sel)
        """
        start_time = time.time()
        X_arr = X.values
        y_arr = y.values
        feature_names = list(X.columns)
        n_features = len(feature_names)

        population = self._init_population(n_features)
        self.history = []

        for gen in range(self.n_generations):
            fitnesses = np.zeros(self.population_size)
            metrics = np.zeros(self.population_size)

            for i in range(self.population_size):
                fit_val, metric_val = self._evaluate_chromosome(population[i], X_arr, y_arr)
                fitnesses[i] = fit_val
                metrics[i] = metric_val

            best_gen_idx = np.argmax(fitnesses)
            best_gen_fit = fitnesses[best_gen_idx]
            best_gen_metric = metrics[best_gen_idx]
            n_sel = int(np.sum(population[best_gen_idx]))

            if best_gen_fit > self.best_fitness:
                self.best_fitness = best_gen_fit
                self.best_metric_score = best_gen_metric
                self.best_chromosome = population[best_gen_idx].copy()
                self.best_selected_features = [
                    feature_names[j] for j in range(n_features) if self.best_chromosome[j] == 1
                ]

            gen_stat = {
                "generation": gen + 1,
                "best_fitness": round(float(self.best_fitness), 4),
                "gen_best_fitness": round(float(best_gen_fit), 4),
                "avg_fitness": round(float(np.mean(fitnesses)), 4),
                "best_metric": round(float(self.best_metric_score), 4),
                "features_selected_count": len(self.best_selected_features),
                "features_selected_ratio": round(len(self.best_selected_features) / n_features, 3),
                "elapsed_time_sec": round(time.time() - start_time, 2)
            }
            self.history.append(gen_stat)

            if progress_callback:
                progress_callback(
                    gen + 1,
                    self.n_generations,
                    self.best_fitness,
                    self.best_metric_score,
                    len(self.best_selected_features)
                )

            # Form next generation with Elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = [population[sorted_indices[e]].copy() for e in range(self.elitism_count)]

            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)

            population = np.array(new_population)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filters input DataFrame to only keep GA-selected features."""
        if not self.best_selected_features:
            return X
        return X[self.best_selected_features]
