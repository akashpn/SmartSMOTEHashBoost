"""
Utility functions for data preparation and results handling
"""

import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    scale: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets and optionally apply standard scaling.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target labels.
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split.
    scale : bool, default=True
        Whether to apply StandardScaler to features.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split (and optionally scaled) datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def save_results_to_csv(results: Iterable[dict], filepath: str) -> None:
    """
    Save a list of result dictionaries to a CSV file.

    Parameters
    ----------
    results : iterable of dict
        Each dict should contain scalar values (metrics, config, etc.).
    filepath : str
        Destination CSV path. Parent directories are created if needed.
    """
    # Convert to list in case a generator is passed
    results = list(results)
    if not results:
        raise ValueError("No results to save.")

    df = pd.DataFrame(results)

    # Ensure parent directory exists
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    df.to_csv(filepath, index=False)

