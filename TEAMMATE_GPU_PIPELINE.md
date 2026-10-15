# Teammate RTX 3050 pipeline

[`optiml.teammate_pipeline`](optiml/teammate_pipeline.py) is a new two-view
candidate, separate from the existing history, TabM, and constrained-XGBoost
notebooks:

- `line_view` models transaction and dimension fields.
- `order_view` models target-independent basket context: line count, basket
  means, within-basket deviations, and distinct product/store/category counts.
- A small logit blend is selected on the frozen July--December validation
  horizon. No target-derived history or test labels are used.

The teammate must obtain the authorized competition files independently. They
are intentionally not included in this repository.

## RTX 3050 setup

From the repository root, with the seven CSV files in `data\`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe tools\check_sota_environment.py --data-dir data
```

Run validation first:

```powershell
.\.venv\Scripts\python.exe -m optiml.teammate_pipeline `
  --data-dir data `
  --output-dir runs\teammate-two-view `
  --device GPU `
  --threads 4
```

If the 3050 reports out-of-memory, use the conservative configuration below.
It reduces tree depth and iterations without changing the temporal split or
using any additional data:

```powershell
.\.venv\Scripts\python.exe -m optiml.teammate_pipeline `
  --data-dir data `
  --output-dir runs\teammate-two-view-3050 `
  --device GPU `
  --iterations 550 `
  --depth 6 `
  --threads 2
```

After reviewing `metrics.json`, produce the final aligned submission:

```powershell
.\.venv\Scripts\python.exe -m optiml.teammate_pipeline `
  --data-dir data `
  --output-dir runs\teammate-two-view-final `
  --device GPU `
  --iterations 550 `
  --depth 6 `
  --threads 2 `
  --finalize
```

The output is `runs\teammate-two-view-final\submission.csv`. If the dataset is
absent, the command fails explicitly; it does not download, fabricate, or
silently substitute data. For a dependency/code smoke test without a GPU,
replace `--device GPU` with `--device CPU` and use a local authorized dataset.
