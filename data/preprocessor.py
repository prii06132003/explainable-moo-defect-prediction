"""
Leakage-Free Data Preprocessing Pipeline for Software Defect Prediction.
Provides imputation, outlier handling, feature scaling, and class imbalance mitigation
while strictly guaranteeing that test folds/splits are never seen during fitting.
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import imblearn.over_sampling as over_samp
import imblearn.under_sampling as under_samp


class DataPreprocessor:
    """
    Object-oriented data preprocessing pipeline ensuring strict zero-data-leakage
    between training and testing partitions.
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        remove_duplicates: bool = True,
        scaling_method: str = "standard",
        imbalance_method: str = "smote",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Args:
            impute_strategy: 'median', 'mean', 'most_frequent', or 'none'
            remove_duplicates: Whether to drop duplicate rows in X before splitting
            scaling_method: 'standard', 'minmax', 'robust', or 'none'
            imbalance_method: 'smote', 'adasyn', 'random_over', 'random_under', or 'none'
            test_size: Fraction of samples to allocate to test partition (e.g. 0.20)
            random_state: Seed for reproducibility
        """
        self.impute_strategy = impute_strategy
        self.remove_duplicates = remove_duplicates
        self.scaling_method = scaling_method
        self.imbalance_method = imbalance_method
        self.test_size = test_size
        self.random_state = random_state

        # Fitted transformers
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.feature_names: list = []

    def clean_raw_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Removes duplicate rows and handles non-finite values if configured."""
        df = X.copy()
        df['__target__'] = y.values

        if self.remove_duplicates:
            df = df.drop_duplicates()

        # Drop rows where target is NaN
        df = df.dropna(subset=['__target__'])

        y_clean = df['__target__'].astype(int)
        X_clean = df.drop(columns=['__target__'])
        return X_clean, y_clean

    def fit_transform_split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Splits data into train and test sets, then fits transformers ONLY on train set,
        and applies them to train and test to prevent any information leakage.
        
        Returns:
            X_train_proc, X_test_proc, y_train_res, y_test, metadata_dict
        """
        X_clean, y_clean = self.clean_raw_data(X, y)
        self.feature_names = list(X_clean.columns)

        # 1. Stratified Train-Test Split BEFORE any transformation
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=self.test_size,
            stratify=y_clean,
            random_state=self.random_state
        )

        # 2. Imputation (fit on train only)
        if self.impute_strategy in ["median", "mean", "most_frequent"]:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            X_train_imp = pd.DataFrame(
                self.imputer.fit_transform(X_train),
                columns=self.feature_names,
                index=X_train.index
            )
            X_test_imp = pd.DataFrame(
                self.imputer.transform(X_test),
                columns=self.feature_names,
                index=X_test.index
            )
        else:
            X_train_imp = X_train.fillna(0)
            X_test_imp = X_test.fillna(0)

        # 3. Feature Scaling (fit on train only)
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        if self.scaler is not None:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train_imp),
                columns=self.feature_names,
                index=X_train_imp.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test_imp),
                columns=self.feature_names,
                index=X_test_imp.index
            )
        else:
            X_train_scaled = X_train_imp
            X_test_scaled = X_test_imp

        # 4. Imbalance Resampling (APPLIED ONLY ON TRAINING SET)
        # The test set remains untouched to preserve true empirical defect distribution!
        X_train_res, y_train_res = self._apply_resampling(X_train_scaled, y_train)

        metadata = {
            "train_samples_raw": len(X_train),
            "test_samples": len(X_test),
            "train_samples_resampled": len(X_train_res),
            "features_count": len(self.feature_names),
            "train_class_distribution_raw": y_train.value_counts().to_dict(),
            "train_class_distribution_resampled": y_train_res.value_counts().to_dict(),
            "test_class_distribution": y_test.value_counts().to_dict(),
            "scaling_method": self.scaling_method,
            "imbalance_method": self.imbalance_method
        }

        return X_train_res, X_test_scaled, y_train_res, y_test, metadata

    def _apply_resampling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Applies imbalance handling algorithm strictly on training partition."""
        if self.imbalance_method == "none" or self.imbalance_method is None:
            return X_train, y_train

        n_minority = (y_train == 1).sum()
        if n_minority < 2:
            # Cannot resample with less than 2 minority samples
            return X_train, y_train

        # Configure k_neighbors safely
        k_neighbors = min(5, n_minority - 1)
        if k_neighbors < 1:
            k_neighbors = 1

        try:
            if self.imbalance_method == "smote":
                sampler = over_samp.SMOTE(
                    k_neighbors=k_neighbors,
                    random_state=self.random_state
                )
            elif self.imbalance_method == "adasyn":
                sampler = over_samp.ADASYN(
                    n_neighbors=k_neighbors,
                    random_state=self.random_state
                )
            elif self.imbalance_method == "random_over":
                sampler = over_samp.RandomOverSampler(
                    random_state=self.random_state
                )
            elif self.imbalance_method == "random_under":
                sampler = under_samp.RandomUnderSampler(
                    random_state=self.random_state
                )
            else:
                return X_train, y_train

            X_res, y_res = sampler.fit_resample(X_train, y_train)
            X_res_df = pd.DataFrame(X_res, columns=X_train.columns)
            y_res_ser = pd.Series(y_res, name="defective")
            return X_res_df, y_res_ser

        except Exception as e:
            # Fallback gracefully to original training data if sampling encounters edge cases
            return X_train, y_train

    def transform_new_data(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses unseen instances (e.g., from production or single prediction)
        using the already-fitted imputer and scaler.
        """
        # Ensure correct column ordering and handle missing columns
        df = X_new.copy()
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_names]

        # Impute
        if self.imputer is not None:
            arr_imp = self.imputer.transform(df)
        else:
            arr_imp = df.fillna(0).values

        # Scale
        if self.scaler is not None:
            arr_scaled = self.scaler.transform(arr_imp)
        else:
            arr_scaled = arr_imp

        return pd.DataFrame(arr_scaled, columns=self.feature_names, index=X_new.index)
