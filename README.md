# OptiML: Jay's Colab LightGBM and teammate's local CatBoost

Work is on **`jay_patch1`**. Keep the current branches **`main` and `jay_patch1`**. No teammate branch is required: your teammate can clone the same code and keep their experiment configurations and outputs in separate folders. Coordinate any code edits before pushing to a shared branch.

## Start here: Google Colab instructions for Jay

1. Open [Google Colab](https://colab.research.google.com/), select **File → Upload notebook**, and upload **`notebooks/lightgbm_colab.ipynb`** from this project.
2. Choose **Runtime → Change runtime type → Python 3**, with **CPU / no hardware accelerator**. This LightGBM baseline runs on CPU; enabling a GPU does not change its implementation.
3. Run the first setup cell. The default `SOURCE_MODE = "upload"` asks for **`dist/optiml_colab_source.zip`**, which is built locally from this project. This works before the branch is pushed to GitHub. The archive contains code, not the competition data.
4. Run the dependency installation cell. If Colab requests a restart, restart before importing pandas/LightGBM, then resume with the configuration cell.
5. In your private Google Drive create **`OptimlData`** containing the seven original CSVs: `train`, `test`, `customers`, `products`, `stores`, `calendar`, and `sample_submission`. Each filename ends in `.csv`.
6. Set the notebook's `DRIVE_DATA` if your folder differs. Run the mount/copy cell to copy the data to `/content/optiml_data`.
7. Run feature preparation, then the **small smoke run**. It samples complete orders from both periods and only verifies the setup. Its AUC is not comparable to a full run.
8. Run the **full baseline** cell for a scored model. The default excludes promotion and includes past-month return statistics.
9. Inspect `metrics.json`: overall AUC, monthly AUC and customer/channel slices. Run the promotion variant or October cutoff by changing the experiment configuration and using a fresh run directory.
10. Once a configuration is selected, set **`FINALIZE = True`** and run a full-data experiment. It validates, refits at the selected iteration count, and writes **`submission.csv`**. All 479,995 test rows, including July boundary rows, must remain present.

The notebook backs up each finished run to **`MyDrive/OptimlRuns/jay/<run_id>`**. Local Colab VM files are temporary, and saved notebooks do not include installed packages or runtime files. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

If you edit the source locally, rebuild the ZIP with `python tools/package_colab.py` before uploading it again. If `jay_patch1` is later published, switch the notebook to `SOURCE_MODE = "github"`; it clones `https://github.com/Jaykvankar/Optiml.git` at that branch. Private repositories need your own authenticated access or the upload method. Never put tokens in the notebook.

## Shared implementation

- `optiml/prepare.py`: validates source IDs, joins dimensions without multiplying rows, derives basket/price/product/calendar features and writes reusable Parquet files.
- `optiml/features.py`: deterministic transformations; latest supplied customer snapshot for region, excluding loyalty and update dates. Product launch/review and store rating/manager-tenure fields are excluded in this baseline. Region normalization fixes case/separators; ambiguous region misspellings remain separate for now.
- `optiml/history.py`: product/store/customer target rates and counts, using earlier months only for training rows. Current-month and sibling-order outcomes are excluded. Validation statistics stay frozen at the fit cutoff. First-month prior is fixed at 0.5. This assumes prior-month labels are available; it is not a claim that return windows have matured.
- `optiml/train.py`: both model families use the same features, chronological splits, train-only category vocabulary, fold manifests and probability checks. LightGBM uses callback-based early stopping. [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)

Raw wall-clock timestamps are preserved without invented timezone conversions. The **maximum timestamp within each order** assigns its fold/month, keeping orders together and keeping cutoff-straddling orders out of the fitting side. The baseline includes late boundary rows in the labeled validation period. The October validation check overlaps the July check; they are not independent folds.

Basket features use all input lines of the same order, assuming the basket is available at prediction time. No other validation/test orders contribute target statistics. Categorical levels come only from fitting rows. Raw `order_id`, `order_line_id`, coupon ID, absolute timestamp and snapshot-update time are excluded as predictors. Customer/product/store IDs remain categorical candidates.

The source feature cache contains **no target-derived history**; histories are rebuilt inside each fit/refit. Only reuse a cache with the same data hashes and feature source. A changed feature implementation needs a new preparation directory.

## Local commands, including your teammate's RTX 3050

Use Python 3.11 or 3.12+ in a project environment. From the repository root on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-catboost.txt
.\.venv\Scripts\python.exe -m optiml.prepare --data-dir nexus-opti-ml-2026-operation-insight --output-dir prepared/baseline-v1
nvidia-smi
.\.venv\Scripts\python.exe -m optiml.train --prepared-dir prepared/baseline-v1 --output-dir runs/teammate/cb_baseline --model catboost --device GPU --threads 4
```

If the `py` launcher is unavailable, use your installed `python` executable for environment creation. On Linux/macOS use `.venv/bin/python` in place of `.venv\Scripts\python.exe`; the CLI arguments are identical. Your teammate needs their own authorized copy of the CSVs, which are not in Git.

The RTX 3050 should first run depth 6, `max_ctr_complexity=1`, one fit at a time, with a working NVIDIA driver. Check actual VRAM with `nvidia-smi`. If memory is exhausted, lower depth to 5 (`--depth 5`) or use a development sample (`--sample-order-fraction 0.05`), then assess final candidates on full data. These are starting settings, not a guarantee that every feature set fits every 3050 variant. CatBoost requests GPU explicitly and can have small numerical differences across repeated GPU runs. [CatBoost GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu)

Jay's equivalent local CPU command is:

```powershell
.\.venv\Scripts\python.exe -m optiml.train --prepared-dir prepared/baseline-v1 --output-dir runs/jay/lgb_baseline --model lightgbm --device CPU --threads 4
```

Useful experiment switches (always choose a new output directory):

| Switch | Meaning |
|---|---|
| `--include-promo` | Test the fading promotion feature against the no-promotion default |
| `--no-history` | Measure incremental value of explicit past-month target statistics |
| `--cutoff 2024-10-01` | Recent-period validation check |
| `--train-months 12` | Fit on the 12 months before the validation cutoff |
| `--iterations 1000` | Cap the trial's boosting iterations |
| `--num-leaves 63` | LightGBM capacity experiment |
| `--depth 8` | CatBoost capacity experiment; higher GPU memory demand |
| `--sample-order-fraction 0.05` | Smoke/development only; changes both fit and validation populations |
| `--finalize` | Validate, refit on all labeled data (or chosen trailing window) and predict test |

Finalization is deliberately rejected for sampled runs. With a trailing window, the final window ends at the last labeled order, retaining the supplied edge rows. Ordinary runs save validation predictions only; a finalizing run also writes test predictions and the submission. The final refit uses validation's selected iteration count, not test labels.

## Comparing results without conflicting work

Jay: prioritize LightGBM, basket/price features and historical statistics. Teammate: prioritize CatBoost GPU, promotion ablation and recent-window tuning. Reproduce the shared baseline before testing changes. Use the same seed and sampling fraction for smoke comparisons and use full validation for final decisions.

Every run saves:

```text
config.json                       # exact CLI configuration
metadata.json                     # data hashes, feature policy, commit and timestamp
environment.txt                   # actual installed versions
source/optiml/                    # exact training source, including uncommitted edits
fold_manifest.parquet             # line/order IDs, month and fit/validation role
schema.json                       # feature list and fit-only categorical levels
metrics.json                      # AUC, monthly/cohort metrics, best iterations, ID hashes
validation_predictions.parquet   # IDs, labels, predictions, month
feature_importance.csv
model.txt or model.cbm             # validation-fitted model
COMPLETE.json                     # written only after the run finishes successfully
```

With `--finalize`, there is also a `final/` model/schema, `test_predictions.parquet` and `submission.csv`. The source/config/prepared-data manifest support reproducing feature construction; model files alone are not a standalone raw-CSV prediction service.

Save your runs under `runs/jay/` and his under `runs/teammate/`, or equivalent private Drive folders. Exchange run directories privately; keep data and artifacts out of Git. Confirm validation-ID hashes match before comparing/blending. Join predictions by `order_line_id`, never row position. Do not overwrite one shared metrics file concurrently.

## Publishing code when you choose to

The notebook upload route works immediately without a push. To make GitHub cloning possible after reviewing the changes:

```powershell
git status
git add .gitignore README.md requirements.txt requirements-catboost.txt optiml tests tools notebooks
git commit -m "Add shared temporal baselines and LightGBM Colab notebook"
git push -u origin jay_patch1
```

No extra branch is needed. Your teammate can clone the published `jay_patch1` and run it locally. Local environment directories, raw data, prepared features, run outputs and source ZIPs are ignored.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Behavior checks cover current/future-label isolation, frozen forecast histories, unknown categories, order overlap and submission alignment. Run the notebook smoke experiment before expensive full trials. A smoke score is infrastructure validation, not a competitive-model claim.

Local verification on the supplied files: all seven CSVs prepared successfully; five behavior tests passed; both LightGBM and CatBoost CPU smoke runs completed. A full-data 50-tree LightGBM integration run scored **0.680255 AUC** on **351,923** validation lines, then refitted and generated a submission with **479,995** rows. The final model reloaded successfully and submission IDs/probabilities passed checks. This was a short integration run, not a tuned competition result. Colab-hosted execution and RTX 3050 GPU execution have not been run here; notebook code syntax and its uploaded-source package were checked locally.
