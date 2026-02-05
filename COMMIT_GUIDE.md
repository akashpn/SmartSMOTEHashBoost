# How to Download Real Datasets & Commit Changes

## Part 1: Download Real Datasets

### Step 1: Run the download script

```bash
python download_datasets.py
```

This will automatically download 3 real imbalanced datasets:
- **Credit Card Default** (`data/credit_card_default.csv`)
- **Breast Cancer Wisconsin** (`data/breast_cancer.csv`)
- **Pima Indians Diabetes** (`data/pima_diabetes.csv`)

### Step 2: Run experiments on downloaded datasets

```bash
# Credit Card Default
python run_csv_experiment.py --csv data/credit_card_default.csv --target default

# Breast Cancer (uses last column automatically)
python run_csv_experiment.py --csv data/breast_cancer.csv

# Pima Diabetes
python run_csv_experiment.py --csv data/pima_diabetes.csv
```

---

## Part 2: Commit Your Changes to Git

### If you DON'T have git initialized yet:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Make your first commit
git commit -m "Initial commit: Smart-SMOTE HashBoost implementation with cost-sensitive Borderline-SMOTE variant"
```

### If you ALREADY have git initialized:

```bash
# Check what files changed
git status

# Add all new/modified files
git add .

# Or add specific files:
git add src/smart_smote_hashboost.py
git add run_csv_experiment.py
git add download_datasets.py
git add requirements.txt
git add README.md

# Commit with a descriptive message
git commit -m "Add cost-sensitive Borderline-SMOTE HashBoost and CSV experiment runner

- Implemented CostSensitiveBorderlineSMOTEHashBoost class
- Added run_csv_experiment.py for generic CSV dataset support
- Added download_datasets.py to fetch real imbalanced datasets
- Updated requirements.txt and README.md
- Added .gitignore for Python projects"
```

### Push to remote (if you have one):

```bash
# If you haven't set up remote yet:
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main

# If remote already exists:
git push
```

---

## Quick Reference: All New Files Created

- `src/smart_smote_hashboost.py` - Updated with `CostSensitiveBorderlineSMOTEHashBoost`
- `run_csv_experiment.py` - Generic CSV runner script
- `download_datasets.py` - Automatic dataset downloader
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules
- `src/utils.py` - Utility functions (data loading, CSV saving)
