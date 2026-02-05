"""
Baseline models for comparison
"""

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class BaselineSMOTE:
    """Standard SMOTE + AdaBoost"""

    def __init__(self, n_estimators=50, k_neighbors=5):
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.base_estimator = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        # Apply SMOTE
        smote = SMOTE(k_neighbors=self.k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Train AdaBoost
        self.model = AdaBoostClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            random_state=42,
            algorithm="SAMME",
        )
        self.model.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class BaselineSMOTERUS:
    """SMOTE + Random Undersampling + AdaBoost"""

    def __init__(self, n_estimators=50, k_neighbors=5):
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.base_estimator = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        # Apply SMOTE
        smote = SMOTE(k_neighbors=self.k_neighbors, random_state=42)
        X_temp, y_temp = smote.fit_resample(X, y)

        # Apply Random Undersampling
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_temp, y_temp)

        # Train AdaBoost
        self.model = AdaBoostClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            random_state=42,
            algorithm="SAMME",
        )
        self.model.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class PlainAdaBoost:
    """Plain AdaBoost without any resampling"""

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.base_estimator = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        self.model = AdaBoostClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            random_state=42,
            algorithm="SAMME",
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
