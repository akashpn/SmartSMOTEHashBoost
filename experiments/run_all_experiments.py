"""
Main experiment script
Run all experiments and save results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from src.baseline_models import BaselineSMOTE, BaselineSMOTERUS, PlainAdaBoost
from src.evaluation import compare_models, evaluate_model, print_evaluation
from src.smart_smote_hashboost import (
    CostSensitiveBorderlineSMOTEHashBoost,
    SmartSMOTEHashBoost,
)
from src.utils import load_and_prepare_data, save_results_to_csv


def create_synthetic_dataset(n_samples=1000, imbalance_ratio=0.1, random_state=42):
    """Create a synthetic imbalanced dataset for testing"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        flip_y=0.01,
        random_state=random_state,
    )
    return X, y


def run_single_experiment(X_train, X_test, y_train, y_test, safety_threshold=0.7):
    """
    Run experiment with all models
    """
    print("\n" + "🚀 Starting Experiment...")

    all_results = []

    # Model 1: Plain AdaBoost (no resampling)
    print("\n[1/5] Training Plain AdaBoost...")
    model1 = PlainAdaBoost(n_estimators=50)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    y_proba1 = model1.predict_proba(X_test)[:, 1]
    results1 = evaluate_model(y_test, y_pred1, y_proba1, "Plain AdaBoost")
    all_results.append(results1)
    print_evaluation(results1)

    # Model 2: SMOTE + AdaBoost
    print("\n[2/5] Training SMOTE + AdaBoost...")
    model2 = BaselineSMOTE(n_estimators=50, k_neighbors=5)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    y_proba2 = model2.predict_proba(X_test)[:, 1]
    results2 = evaluate_model(y_test, y_pred2, y_proba2, "SMOTE + AdaBoost")
    all_results.append(results2)
    print_evaluation(results2)

    # Model 3: SMOTE + RUS + AdaBoost
    print("\n[3/5] Training SMOTE + RUS + AdaBoost...")
    model3 = BaselineSMOTERUS(n_estimators=50, k_neighbors=5)
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    y_proba3 = model3.predict_proba(X_test)[:, 1]
    results3 = evaluate_model(y_test, y_pred3, y_proba3, "SMOTE + RUS + AdaBoost")
    all_results.append(results3)
    print_evaluation(results3)

    # Model 4: YOUR MODEL - Smart-SMOTE HashBoost
    print(f"\n[4/5] Training Smart-SMOTE HashBoost (threshold={safety_threshold})...")
    model4 = SmartSMOTEHashBoost(
        n_estimators=50, safety_threshold=safety_threshold, k_neighbors=5, verbose=True
    )
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    y_proba4 = model4.predict_proba(X_test)[:, 1]
    results4 = evaluate_model(y_test, y_pred4, y_proba4, "Smart-SMOTE HashBoost")
    all_results.append(results4)
    print_evaluation(results4)

    # Print statistics for Smart-SMOTE
    stats = model4.get_training_stats()
    print(f"\n📊 Smart-SMOTE Statistics:")
    print(f"  Total synthetic generated: {stats['total_synthetic']}")
    print(f"  Safe samples kept:         {stats['safe_synthetic']}")
    print(f"  Unsafe samples filtered:   {stats['filtered_synthetic']}")
    print(f"  Average safety score:      {stats['avg_safety_score']:.3f}")

    # Model 5: Cost-Sensitive Borderline-SMOTE HashBoost
    print("\n[5/5] Training Cost-Sensitive Borderline-SMOTE HashBoost...")
    model5 = CostSensitiveBorderlineSMOTEHashBoost(
        n_estimators=50,
        k_neighbors=5,
        minority_cost=2.0,
        majority_cost=1.0,
        oversampler="borderline",
        verbose=True,
    )
    model5.fit(X_train, y_train)
    y_pred5 = model5.predict(X_test)
    y_proba5 = model5.predict_proba(X_test)[:, 1]
    results5 = evaluate_model(
        y_test, y_pred5, y_proba5, "CS Borderline-SMOTE HashBoost"
    )
    all_results.append(results5)
    print_evaluation(results5)

    # Compare all models
    compare_models(all_results)

    return all_results


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("SMART-SMOTE HASHBOOST - COMPLETE EXPERIMENTS")
    print("=" * 80)

    # Create dataset
    print("\n📦 Creating synthetic imbalanced dataset...")
    X, y = create_synthetic_dataset(n_samples=1000, imbalance_ratio=0.1)

    # Prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        X, y, test_size=0.3, scale=True
    )

    # Run experiments with different safety thresholds
    all_experiments = []

    for threshold in [0.5, 0.6, 0.7, 0.8]:
        print(f"\n\n{'#' * 80}")
        print(f"# EXPERIMENT: Safety Threshold = {threshold}")
        print(f"{'#' * 80}")

        results = run_single_experiment(
            X_train, X_test, y_train, y_test, safety_threshold=threshold
        )

        # Add threshold info to results
        for r in results:
            r["safety_threshold"] = threshold

        all_experiments.extend(results)

    # Save all results
    save_results_to_csv(all_experiments, "results/tables/all_experiments.csv")

    print("\n✅ All experiments completed!")
    print("📁 Results saved to: results/tables/all_experiments.csv")


if __name__ == "__main__":
    main()
