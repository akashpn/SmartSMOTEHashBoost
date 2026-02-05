"""
Generic runner to apply all models to any CSV dataset.

Usage examples
--------------
python run_csv_experiment.py --csv data/credit.csv --target label
python run_csv_experiment.py --csv data/medical.csv --target outcome --drop id
"""

import argparse
from pathlib import Path

import pandas as pd

from src.baseline_models import BaselineSMOTE, BaselineSMOTERUS, PlainAdaBoost
from src.evaluation import compare_models, evaluate_model, print_evaluation
from src.smart_smote_hashboost import (
    CostSensitiveBorderlineSMOTEHashBoost,
    SmartSMOTEHashBoost,
)
from src.utils import load_and_prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Smart-SMOTE HashBoost and baselines on any CSV dataset."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        help=(
            "Name of the target/label column. "
            "If omitted, the last column in the CSV is used."
        ),
    )
    parser.add_argument(
        "--drop",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of columns to drop (e.g., IDs, text fields).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set proportion (default: 0.3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"\nLoading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Loaded CSV is empty.")

    # Determine target column
    if args.target is None:
        target_col = df.columns[-1]
        print(f"No --target specified; using last column as target: '{target_col}'")
    else:
        target_col = args.target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")

    # Drop optional columns
    if args.drop:
        for col in args.drop:
            if col in df.columns and col != target_col:
                print(f"Dropping column: {col}")
                df = df.drop(columns=[col])

    # Split into features and target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    print(f"\nDataset summary:")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target column: {target_col}")
    print(f"  Classes: {pd.Series(y).value_counts().to_dict()}")

    # Prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        X, y, test_size=args.test_size, scale=True
    )

    all_results = []

    # 1) Plain AdaBoost
    print("\n[1/5] Training Plain AdaBoost...")
    model1 = PlainAdaBoost(n_estimators=50)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    y_proba1 = model1.predict_proba(X_test)[:, 1]
    res1 = evaluate_model(y_test, y_pred1, y_proba1, "Plain AdaBoost")
    print_evaluation(res1)
    all_results.append(res1)

    # 2) SMOTE + AdaBoost
    print("\n[2/5] Training SMOTE + AdaBoost...")
    model2 = BaselineSMOTE(n_estimators=50, k_neighbors=5)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    y_proba2 = model2.predict_proba(X_test)[:, 1]
    res2 = evaluate_model(y_test, y_pred2, y_proba2, "SMOTE + AdaBoost")
    print_evaluation(res2)
    all_results.append(res2)

    # 3) SMOTE + RUS + AdaBoost
    print("\n[3/5] Training SMOTE + RUS + AdaBoost...")
    model3 = BaselineSMOTERUS(n_estimators=50, k_neighbors=5)
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    y_proba3 = model3.predict_proba(X_test)[:, 1]
    res3 = evaluate_model(y_test, y_pred3, y_proba3, "SMOTE + RUS + AdaBoost")
    print_evaluation(res3)
    all_results.append(res3)

    # 4) Smart-SMOTE HashBoost
    print("\n[4/5] Training Smart-SMOTE HashBoost...")
    model4 = SmartSMOTEHashBoost(
        n_estimators=50, safety_threshold=0.7, k_neighbors=5, verbose=True
    )
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    y_proba4 = model4.predict_proba(X_test)[:, 1]
    res4 = evaluate_model(y_test, y_pred4, y_proba4, "Smart-SMOTE HashBoost")
    print_evaluation(res4)
    all_results.append(res4)

    # 5) Cost-Sensitive Borderline-SMOTE HashBoost
    print("\n[5/5] Training CS Borderline-SMOTE HashBoost...")
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
    res5 = evaluate_model(
        y_test, y_pred5, y_proba5, "CS Borderline-SMOTE HashBoost"
    )
    print_evaluation(res5)
    all_results.append(res5)

    # Compare all models
    compare_models(all_results)


if __name__ == "__main__":
    main()

