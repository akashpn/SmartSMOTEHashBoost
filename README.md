# Smart-SMOTE HashBoost

Implementation and extension of the SMOTEHashBoost algorithm for imbalanced binary classification, with a focus on **safety-aware synthetic sample generation** and **hash-based undersampling**.

This project is intended as the base code for a final-year project built on top of the paper *"SMOTEHashBoost: Ensemble Algorithm for Imbalanced Dataset Pattern Classification"*.

## 1. Environment setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Project structure

- `src/`
  - `baseline_models.py` – plain AdaBoost and simple SMOTE-based baselines.
  - `smart_smote_hashboost.py` – Smart-SMOTE HashBoost implementation (safety-aware SMOTE + hash-based undersampling + AdaBoost).
  - `evaluation.py` – metrics (F1, precision, recall, G-mean, AUC, etc.) and comparison utilities.
  - `utils.py` – data loading, preprocessing, and results-saving helpers.
- `experiments/`
  - `run_all_experiments.py` – main entry point to run experiments on a synthetic imbalanced dataset.
- `notebooks/`
  - `03_your_model_experiments.ipynb` – place for interactive exploration, plots, and ablation studies.

## 3. Running the experiments

From the project root:

```bash
python -m experiments.run_all_experiments
```

This will:

- Create a synthetic imbalanced dataset.
- Train and evaluate:
  - Plain AdaBoost
  - SMOTE + AdaBoost
  - SMOTE + Random Undersampling + AdaBoost
  - Smart-SMOTE HashBoost (your improved model)
- Print evaluation metrics and comparison tables.
- Save all experiment results to `results/tables/all_experiments.csv`.

## 4. Extending the project

Suggested extensions you can implement for your final-year project:

- Replace vanilla SMOTE with more advanced oversampling (Borderline-SMOTE, ADASYN, or GAN-based).
- Introduce cost-sensitive boosting so minority misclassifications are penalized more.
- Improve the hash-based undersampling step (e.g., supervised hashing, adaptive bucket sampling).
- Add real-world imbalanced datasets (e.g., medical, fraud, dropout prediction) and run full comparisons.

Use the existing `SmartSMOTEHashBoost` class as your starting point and add new variants alongside it (e.g. `CostSensitiveSMOTEHashBoost`, `BorderlineSMOTEHashBoost`), then wire them into `run_all_experiments.py` and the notebook.

