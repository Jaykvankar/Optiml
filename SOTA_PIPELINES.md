# Reproducible SOTA candidate pipelines

This repository contains two additional, scratch-trained candidate pipelines for
NEXUS OptiML 2026:

1. **Risk-adjusted matured entity history**
   - [`notebooks/nexus_adjusted_history_kaggle.ipynb`](notebooks/nexus_adjusted_history_kaggle.ipynb)
   - Uses a stable CatBoost anchor, delayed/matured customer-product-store
     histories, and a bounded residual correction.
   - History is frozen before each holdout and the notebook excludes temporary
     target-derived columns from the anchor features.

2. **Constrained XGBoost**
   - [`notebooks/nexus_constrained_xgb_kaggle.ipynb`](notebooks/nexus_constrained_xgb_kaggle.ipynb)
   - Compares a constrained CUDA XGBoost model with an unconstrained control
     and a CatBoost control.
   - Stable and seasonal feature blocks are declared explicitly; the model
     never silently falls back from CUDA.

These are experimental candidates, not guaranteed leaderboard improvements.
The matured-history run already showed that delayed history can be useful
locally, but local AUC and public AUC are different populations. Use the
competition's public result only as an outcome report, not as a tuning target.

## Rules and data handling

The notebooks use only the seven supplied competition CSV files. They do not
download data, call hosted prediction APIs, use pretrained weights, use
external labels, or include competition data in Git. The competition rules
prohibit external data and restrict competition data to registered participants;
each user must obtain the files through the authorized competition input.

The repository is suitable for publication, but the raw data, generated
submissions, model files, and run outputs must remain local and ignored by Git.
If a winning solution must be delivered, retain the exact notebook, package
versions, seed, GPU details, training configuration, submission checksum, and
the applicable OSI-approved source license.

## NVIDIA RTX 3050 / Kaggle setup

The notebooks are self-contained and do not require this repository to be
imported. On Kaggle:

1. Add the private competition dataset through **Add Input**.
2. Enable **GPU**.
3. Open one notebook and use **Save Version -> Save & Run All**.
4. Download the CSV only after the notebook reports `SAVED` and `479995` rows.

On another NVIDIA machine, install the portable dependencies with Python 3.10+:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

An RTX 3050 is supported, but its VRAM is smaller than a T4. Use the notebook's
conservative settings first, close other GPU processes, and reduce the declared
batch/sample settings if CUDA reports out-of-memory. Do not solve memory
pressure by changing the temporal split or using test labels.

Before training, validate the machine:

```powershell
.\.venv\Scripts\python.exe tools\check_sota_environment.py --data-dir data
```

If the user has no dataset, the preflight must fail rather than invent or fetch
data:

```text
Missing ... required competition file(s) in data: ...
```

Place the authorized files in `data\` (or set `DATA_DIR`) and rerun the
preflight. The required names are:

```text
train.csv
test.csv
customers.csv
products.csv
stores.csv
calendar.csv
sample_submission.csv
```

## Reproducibility checklist

For every candidate, preserve:

- notebook version and Git commit;
- Python, CatBoost, XGBoost, PyTorch, and CUDA versions;
- GPU name and VRAM;
- random seed and all temporal cutoffs;
- assessment metrics and monthly diagnostics;
- final CSV SHA-256 and row/column validation.

Never submit a file whose IDs have not been checked against
`sample_submission.csv`. The notebooks perform that check before writing their
output.
