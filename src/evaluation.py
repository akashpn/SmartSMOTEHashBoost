"""
Evaluation metrics and utilities
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_gmean(y_true, y_pred):
    """Calculate G-Mean score"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return np.sqrt(sensitivity * specificity)


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Comprehensive model evaluation

    Returns dictionary with all metrics
    """
    results = {
        "model": model_name,
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "g_mean": calculate_gmean(y_true, y_pred),
    }

    if y_proba is not None:
        results["auc_roc"] = roc_auc_score(y_true, y_proba)
        results["avg_precision"] = average_precision_score(y_true, y_proba)

    return results


def print_evaluation(results):
    """Pretty print evaluation results"""
    print(f"\n{'=' * 60}")
    print(f"Results for: {results['model']}")
    print(f"{'=' * 60}")
    print(f"F1-Score:          {results['f1_score']:.4f}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    print(f"G-Mean:            {results['g_mean']:.4f}")

    if "auc_roc" in results:
        print(f"AUC-ROC:           {results['auc_roc']:.4f}")
    if "avg_precision" in results:
        print(f"Average Precision: {results['avg_precision']:.4f}")

    print(f"{'=' * 60}\n")


def compare_models(results_list):
    """
    Compare multiple models and identify best performer

    Parameters:
    -----------
    results_list : list of dicts
        List of evaluation result dictionaries
    """
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print(f"{'=' * 80}")

    # Print header
    print(
        f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'G-Mean':>8} {'AUC':>8}"
    )
    print(f"{'-' * 80}")

    # Print each model
    for result in results_list:
        auc = result.get("auc_roc", 0)
        print(
            f"{result['model']:<25} "
            f"{result['f1_score']:>8.4f} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>8.4f} "
            f"{result['g_mean']:>8.4f} "
            f"{auc:>8.4f}"
        )

    print(f"{'=' * 80}\n")

    # Find best model for each metric
    best_f1 = max(results_list, key=lambda x: x["f1_score"])
    best_auc = max(results_list, key=lambda x: x.get("auc_roc", 0))

    print("🏆 Best Performers:")
    print(f"  F1-Score: {best_f1['model']} ({best_f1['f1_score']:.4f})")
    print(f"  AUC-ROC:  {best_auc['model']} ({best_auc.get('auc_roc', 0):.4f})\n")
