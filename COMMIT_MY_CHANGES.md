# How to Commit My Changes

## Files I Created/Modified:

### ✅ New Files I Created:
1. `requirements.txt` - Python dependencies
2. `README.md` - Project documentation  
3. `src/utils.py` - Utility functions (was empty, now has data loading/saving)
4. `run_csv_experiment.py` - Generic CSV experiment runner
5. `download_datasets.py` - Automatic dataset downloader
6. `.gitignore` - Git ignore rules
7. `COMMIT_GUIDE.md` - Documentation (you can delete this if you want)

### ✅ Files I Modified:
1. `src/smart_smote_hashboost.py` - Added `CostSensitiveBorderlineSMOTEHashBoost` class
2. `experiments/run_all_experiments.py` - Added the new model to experiments

---

## Step-by-Step: Commit Everything

### Option 1: If you DON'T have git initialized yet

Open PowerShell/Command Prompt in your project folder and run:

```bash
# Initialize git (if not done already)
git init

# Add all files
git add .

# Commit everything
git commit -m "Add cost-sensitive Borderline-SMOTE HashBoost implementation

- Implemented CostSensitiveBorderlineSMOTEHashBoost class
- Added CSV experiment runner (run_csv_experiment.py)
- Added dataset downloader (download_datasets.py)
- Updated requirements.txt and README.md
- Added utility functions (src/utils.py)"
```

### Option 2: If you ALREADY have git initialized

```bash
# See what changed
git status

# Add all my changes
git add .

# Or add specific files:
git add requirements.txt README.md src/utils.py src/smart_smote_hashboost.py
git add run_csv_experiment.py download_datasets.py .gitignore
git add experiments/run_all_experiments.py

# Commit
git commit -m "Add cost-sensitive Borderline-SMOTE HashBoost and CSV runner"
```

---

## Quick One-Liner (if git is already initialized):

```bash
git add . && git commit -m "Add cost-sensitive Borderline-SMOTE HashBoost implementation"
```

---

That's it! Your changes are now committed to git. 🎉
