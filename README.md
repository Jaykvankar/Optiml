# OptiML submission workflows

This repository contains a reproducible, local-only implementation of the NEXUS
hierarchical ranking strategy. It does not contain competition data, trained models,
run outputs, or generated submission files.

For the two additional GPU candidates, including RTX 3050 guidance, missing-data
behavior, rule compliance, and reproducibility requirements, see
[`SOTA_PIPELINES.md`](SOTA_PIPELINES.md).

## Workflows

| Model | Notebook | Runtime |
|---|---|---|
| Hierarchical strategy | `python -m optiml.strategy` | Local CPU |
| LightGBM baseline | [`notebooks/lightgbm_colab.ipynb`](notebooks/lightgbm_colab.ipynb) | Legacy notebook |
| CatBoost baseline | [`catboost_workflow/catboost_colab.ipynb`](catboost_workflow/catboost_colab.ipynb) | Legacy notebook |

The notebooks use the shared code in `optiml/` for feature preparation, temporal
validation, model fitting, final refits, and submission checks. Use the notebooks for
the documented end-to-end workflows; the CLI commands below are also available for
local runs.

## Local setup

Use Python 3.10 or newer. On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CatBoost, install its additional dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-catboost.txt
```

For the two portable GPU candidate notebooks, install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Place the seven competition CSVs in a local `data/` folder:
`train.csv`, `test.csv`, `customers.csv`, `products.csv`, `stores.csv`,
`calendar.csv`, and `sample_submission.csv`. These files are intentionally excluded
from Git.

Prepare features once:

```powershell
.\.venv\Scripts\python.exe -m optiml.prepare --data-dir data --output-dir prepared/strategy-v1
```

Run a validation-only LightGBM or CatBoost experiment, using a new output directory
for each run:

```powershell
.\.venv\Scripts\python.exe -m optiml.train --prepared-dir prepared/strategy-v1 --output-dir runs/lightgbm-baseline --model lightgbm --device CPU
.\.venv\Scripts\python.exe -m optiml.train --prepared-dir prepared/strategy-v1 --output-dir runs/catboost-baseline --model catboost --device GPU
```

Run the new local strategy (development folds, July--December assessment, and
specialist coefficient selection):

```powershell
.\.venv\Scripts\python.exe -m optiml.strategy --prepared-dir prepared/strategy-v1 --output-dir runs/strategy-v1
```

After reviewing `runs/strategy-v1/metrics.json`, add `--finalize` to refit on all
labeled rows and write a locally aligned `submission.csv`. The strategy uses CPU by
default; it never uploads data or requires Kaggle or Colab.

Add `--finalize` only after selecting a configuration. Full-data finalization refits
the model and writes `submission.csv` into that run's output directory. For CatBoost
GPU or LightGBM CUDA, run on a compatible Linux/NVIDIA runtime; GPU requests fail
explicitly rather than silently falling back to CPU.

The repository never downloads competition data. Each participant must obtain the
authorized seven CSV files through the competition and place them in `data/`.
Validate the local machine and data layout before a GPU run:

```powershell
.\.venv\Scripts\python.exe tools\check_sota_environment.py --data-dir data
```

Audit a generated submission against its completed run and the supplied sample:

```powershell
.\.venv\Scripts\python.exe -m optiml.audit_submission --csv runs/catboost-final/submission.csv --run-dir runs/catboost-final --sample data/sample_submission.csv
```

## Colab source packages

The LightGBM notebook's source archive can be rebuilt with:

```powershell
.\.venv\Scripts\python.exe tools/package_colab.py
```

For CatBoost, rebuild the notebook and its uploadable source archive with:

```powershell
.\.venv\Scripts\python.exe tools/build_gpu_notebook.py
.\.venv\Scripts\python.exe catboost_workflow/build.py
```

Upload the resulting archive to the Colab runtime as described in the notebook.
Generated archives are local build artifacts and are not committed.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Keep raw data, credentials, prepared Parquet files, run/model artifacts, analysis
outputs, and submission CSVs out of Git. The `.gitignore` covers these local files;
only the notebooks, shared source, build helpers, requirements, tests, and project
documentation belong in the public repository.
