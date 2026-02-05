"""
Smart-SMOTE HashBoost: Safety-Aware Synthetic Sample Generation
Main algorithm implementation
"""

from collections import Counter
from typing import List

import numpy as np
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier


class SmartSMOTEHashBoost:
    """
    Enhanced version of SMOTEHashBoost with safety-aware sample selection
    """

    def __init__(
        self, n_estimators=50, safety_threshold=0.7, k_neighbors=5, verbose=True
    ):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of boosting iterations
        safety_threshold : float (0-1)
            Minimum safety score for synthetic samples
        k_neighbors : int
            Number of neighbors for SMOTE and safety calculation
        verbose : bool
            Print progress information
        """
        self.n_estimators = n_estimators
        self.safety_threshold = safety_threshold
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.training_stats = {}

    def calculate_safety_score(self, sample, X_minority, X_majority, k=5):
        """
        Calculate safety score for a synthetic sample

        Safety score = proportion of minority class in k nearest neighbors
        Score closer to 1 = safer (more minority neighbors)
        Score closer to 0 = dangerous (more majority neighbors)
        """
        # Combine all data
        X_all = np.vstack([X_minority, X_majority])
        y_all = np.array([0] * len(X_minority) + [1] * len(X_majority))

        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
        distances, indices = nbrs.kneighbors([sample])

        # Count minority neighbors (exclude first index if it's the sample itself)
        neighbor_labels = y_all[indices[0][1:]]
        minority_count = np.sum(neighbor_labels == 0)

        return minority_count / k

    def safe_smote(self, X_minority, X_majority):
        """
        Generate synthetic samples with safety filtering
        """
        if self.verbose:
            print(f"  Original minority samples: {len(X_minority)}")

        # Generate synthetic samples using SMOTE
        smote = SMOTE(k_neighbors=self.k_neighbors, random_state=42)
        X_combined = np.vstack([X_minority, X_majority])
        y_combined = np.array([1] * len(X_minority) + [0] * len(X_majority))

        X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

        # Extract synthetic samples (new ones added by SMOTE)
        synthetic_samples = X_resampled[len(X_combined) :]

        if self.verbose:
            print(f"  Generated synthetic samples: {len(synthetic_samples)}")

        # Filter based on safety scores
        safe_synthetic = []
        safety_scores = []

        for sample in synthetic_samples:
            score = self.calculate_safety_score(
                sample, X_minority, X_majority, k=self.k_neighbors
            )
            safety_scores.append(score)

            if score >= self.safety_threshold:
                safe_synthetic.append(sample)

        # Store statistics
        self.training_stats["total_synthetic"] = len(synthetic_samples)
        self.training_stats["safe_synthetic"] = len(safe_synthetic)
        self.training_stats["filtered_synthetic"] = len(synthetic_samples) - len(
            safe_synthetic
        )
        self.training_stats["avg_safety_score"] = np.mean(safety_scores)
        self.training_stats["safety_scores"] = safety_scores

        if self.verbose:
            print(f"  Safe synthetic samples: {len(safe_synthetic)}")
            print(f"  Filtered samples: {self.training_stats['filtered_synthetic']}")
            print(
                f"  Average safety score: {self.training_stats['avg_safety_score']:.3f}"
            )

        # Combine original minority with safe synthetic
        if len(safe_synthetic) > 0:
            return np.vstack([X_minority, np.array(safe_synthetic)])
        else:
            return X_minority

    def hash_undersample(self, X_majority, n_samples):
        """
        Hash-based undersampling of majority class
        """
        if len(X_majority) <= n_samples:
            return X_majority

        # Determine number of PCA components
        n_components = min(10, X_majority.shape[1], len(X_majority))

        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_majority)

        # Create hash buckets
        n_buckets = min(10, n_samples)
        samples_per_bucket = max(1, n_samples // n_buckets)

        selected_indices = []
        bucket_size = len(X_reduced) // n_buckets

        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = min(start_idx + bucket_size, len(X_reduced))

            if start_idx >= end_idx:
                continue

            bucket_indices = list(range(start_idx, end_idx))
            n_select = min(samples_per_bucket, len(bucket_indices))

            selected = np.random.choice(bucket_indices, n_select, replace=False)
            selected_indices.extend(selected)

        return X_majority[selected_indices]

    def fit(self, X, y):
        """
        Train the Smart-SMOTE HashBoost model
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("TRAINING: Smart-SMOTE HashBoost")
            print("=" * 70)

        # Identify minority and majority classes
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("This implementation supports binary classification only")

        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        X_minority = X[y == minority_class]
        X_majority = X[y == majority_class]

        if self.verbose:
            print(f"\nOriginal Dataset:")
            print(f"  Minority class ({minority_class}): {len(X_minority)} samples")
            print(f"  Majority class ({majority_class}): {len(X_majority)} samples")
            print(f"  Imbalance ratio: {len(X_majority) / len(X_minority):.2f}:1")

        # Apply Safe SMOTE
        if self.verbose:
            print(
                f"\nStep 1: Applying Safe SMOTE (threshold={self.safety_threshold})..."
            )

        X_minority_augmented = self.safe_smote(X_minority, X_majority)

        # Apply Hash-based Undersampling
        if self.verbose:
            print(f"\nStep 2: Hash-based Undersampling...")

        target_majority_size = len(X_minority_augmented)
        X_majority_sampled = self.hash_undersample(X_majority, target_majority_size)

        if self.verbose:
            print(f"  Majority samples after undersampling: {len(X_majority_sampled)}")

        # Create balanced dataset
        X_balanced = np.vstack([X_minority_augmented, X_majority_sampled])
        y_balanced = np.array(
            [minority_class] * len(X_minority_augmented)
            + [majority_class] * len(X_majority_sampled)
        )

        if self.verbose:
            print(f"\nBalanced Dataset:")
            print(f"  Total samples: {len(X_balanced)}")
            print(f"  Class distribution: {Counter(y_balanced)}")

        # Train AdaBoost ensemble
        if self.verbose:
            print(f"\nStep 3: Training AdaBoost ({self.n_estimators} estimators)...")

        self.model = AdaBoostClassifier(
            estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            random_state=42,
            algorithm="SAMME",
        )
        self.model.fit(X_balanced, y_balanced)

        if self.verbose:
            print("\n✓ Training Complete!")
            print("=" * 70 + "\n")

        return self

    def predict(self, X):
        """Make class predictions"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)

    def get_training_stats(self):
        """Return training statistics"""
        return self.training_stats


class CostSensitiveBorderlineSMOTEHashBoost:
    """
    Improved SMOTEHashBoost variant with:
    - Borderline-SMOTE / ADASYN oversampling
    - Hash-based undersampling of majority class
    - Cost-sensitive AdaBoost-style boosting loop
    """

    def __init__(
        self,
        n_estimators: int = 50,
        k_neighbors: int = 5,
        minority_cost: float = 2.0,
        majority_cost: float = 1.0,
        oversampler: str = "borderline",
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of boosting iterations.
        k_neighbors : int
            Number of neighbors for oversampling methods.
        minority_cost : float
            Misclassification cost for minority class (> majority_cost).
        majority_cost : float
            Misclassification cost for majority class.
        oversampler : {"borderline", "adasyn", "smote"}
            Oversampling strategy to use for the minority class.
        verbose : bool
            Whether to print training progress.
        """
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.minority_cost = minority_cost
        self.majority_cost = majority_cost
        self.oversampler = oversampler
        self.verbose = verbose

        # Internal attributes populated during fit
        self.classes_: np.ndarray | None = None
        self.minority_class_: int | None = None
        self.majority_class_: int | None = None
        self.estimators_: List[DecisionTreeClassifier] = []
        self.estimator_weights_: List[float] = []

    def _hash_undersample(self, X_majority: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Hash-based undersampling of majority class.
        Reuses the same PCA-bucket idea as SmartSMOTEHashBoost.
        """
        if len(X_majority) <= n_samples:
            return X_majority

        n_components = min(10, X_majority.shape[1], len(X_majority))
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_majority)

        n_buckets = min(10, n_samples)
        samples_per_bucket = max(1, n_samples // n_buckets)

        selected_indices: List[int] = []
        bucket_size = len(X_reduced) // n_buckets

        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = min(start_idx + bucket_size, len(X_reduced))
            if start_idx >= end_idx:
                continue

            bucket_indices = list(range(start_idx, end_idx))
            n_select = min(samples_per_bucket, len(bucket_indices))
            if n_select <= 0:
                continue

            selected = np.random.choice(bucket_indices, n_select, replace=False)
            selected_indices.extend(selected)

        if not selected_indices:
            # Fallback to random selection if something went wrong
            selected_indices = list(
                np.random.choice(len(X_majority), n_samples, replace=False)
            )

        return X_majority[selected_indices]

    def _oversample_minority(
        self, X_minority: np.ndarray, X_majority: np.ndarray
    ) -> np.ndarray:
        """
        Oversample the minority class using the chosen strategy.
        """
        X_combined = np.vstack([X_minority, X_majority])
        y_combined = np.array([1] * len(X_minority) + [0] * len(X_majority))

        if self.oversampler == "adasyn":
            sampler = ADASYN(n_neighbors=self.k_neighbors, random_state=42)
        elif self.oversampler == "smote":
            sampler = SMOTE(k_neighbors=self.k_neighbors, random_state=42)
        else:
            # default: Borderline-SMOTE (type 1)
            sampler = BorderlineSMOTE(
                k_neighbors=self.k_neighbors, random_state=42, kind="borderline-1"
            )

        X_resampled, y_resampled = sampler.fit_resample(X_combined, y_combined)

        # Keep only minority class samples from resampled set
        X_minority_resampled = X_resampled[y_resampled == 1]
        return X_minority_resampled

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Cost-Sensitive Borderline-SMOTE HashBoost model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("This implementation supports binary classification only")

        self.classes_ = unique_classes
        class_counts = Counter(y)
        self.minority_class_ = min(class_counts, key=class_counts.get)
        self.majority_class_ = max(class_counts, key=class_counts.get)

        if self.verbose:
            print("\n" + "=" * 70)
            print("TRAINING: Cost-Sensitive Borderline-SMOTE HashBoost")
            print("=" * 70)
            print(f"Minority class: {self.minority_class_}")
            print(f"Majority class: {self.majority_class_}")
            print(f"Imbalance ratio: {class_counts[self.majority_class_] / class_counts[self.minority_class_]:.2f}:1")

        # Initial sample weights
        n_samples = len(y)
        sample_weights = np.full(n_samples, 1.0 / n_samples, dtype=float)

        # Class-specific misclassification costs
        class_costs = np.where(
            y == self.minority_class_, self.minority_cost, self.majority_cost
        )

        # Map labels to {-1, +1} for boosting updates
        y_signed = np.where(y == self.minority_class_, 1.0, -1.0)

        self.estimators_ = []
        self.estimator_weights_ = []

        for m in range(self.n_estimators):
            if self.verbose:
                print(f"\n--- Boosting iteration {m + 1}/{self.n_estimators} ---")

            # Split current data into minority/majority
            X_minority = X[y == self.minority_class_]
            X_majority = X[y == self.majority_class_]

            # Oversample minority using chosen strategy
            X_minority_aug = self._oversample_minority(X_minority, X_majority)

            # Hash-based undersampling of majority to match minority size
            target_majority_size = len(X_minority_aug)
            X_majority_sampled = self._hash_undersample(X_majority, target_majority_size)

            # Build balanced training set for this iteration
            X_balanced = np.vstack([X_minority_aug, X_majority_sampled])
            y_balanced = np.array(
                [self.minority_class_] * len(X_minority_aug)
                + [self.majority_class_] * len(X_majority_sampled)
            )

            if self.verbose:
                print(f"  Balanced set size: {len(X_balanced)} "
                      f"(minority={len(X_minority_aug)}, majority={len(X_majority_sampled)})")

            # Train weak learner on balanced set
            stump = DecisionTreeClassifier(max_depth=1, random_state=42)
            stump.fit(X_balanced, y_balanced)
            y_pred = stump.predict(X)

            # Convert predictions to {-1, +1}
            h_signed = np.where(y_pred == self.minority_class_, 1.0, -1.0)

            # Weighted, cost-sensitive error
            misclassified = (y_pred != y).astype(float)
            numerator = np.sum(sample_weights * class_costs * misclassified)
            denominator = np.sum(sample_weights * class_costs)
            error = numerator / max(denominator, 1e-16)

            # If error is too high, stop boosting
            if error <= 0 or error >= 0.5:
                if self.verbose:
                    print(
                        f"  Stopping early at iteration {m + 1}, error={error:.4f} "
                        "(weak learner is no better than random)."
                    )
                break

            alpha = 0.5 * np.log((1.0 - error) / error)

            if self.verbose:
                print(f"  Weighted error: {error:.4f}")
                print(f"  Alpha (learner weight): {alpha:.4f}")

            # Update sample weights (cost-sensitive)
            sample_weights *= np.exp(-alpha * class_costs * y_signed * h_signed)

            # Normalize
            sample_weights_sum = np.sum(sample_weights)
            if sample_weights_sum <= 0:
                break
            sample_weights /= sample_weights_sum

            # Store this weak learner
            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)

        if self.verbose:
            print("\n✓ Training Complete!")
            print(f"Total weak learners used: {len(self.estimators_)}")
            print("=" * 70 + "\n")

        return self

    def _aggregate_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute aggregated ensemble score for each sample.
        """
        if not self.estimators_:
            raise RuntimeError("The model has not been fitted yet.")

        X = np.asarray(X)
        scores = np.zeros(X.shape[0], dtype=float)

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = stump.predict(X)
            h_signed = np.where(y_pred == self.minority_class_, 1.0, -1.0)
            scores += alpha * h_signed

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        """
        scores = self._aggregate_scores(X)
        preds_signed = np.where(scores >= 0, 1.0, -1.0)
        return np.where(
            preds_signed == 1.0, self.minority_class_, self.majority_class_
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using a sigmoid over the ensemble scores.
        """
        scores = self._aggregate_scores(X)
        # Simple probability mapping via logistic sigmoid
        proba_minority = 1.0 / (1.0 + np.exp(-scores))
        proba_majority = 1.0 - proba_minority
        return np.vstack([proba_majority, proba_minority]).T
