"""Build the independent CatBoost-only Colab notebook and source upload."""
from pathlib import Path
import json
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
old = json.loads((ROOT / "notebooks/gpu_colab.ipynb").read_text(encoding="utf-8"))


def source(index):
    return "".join(old["cells"][index]["source"])


cells = []


def md(value):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": textwrap.dedent(value).strip().splitlines(keepends=True)})


def code(value):
    cells.append({"cell_type": "code", "metadata": {},
                  "source": textwrap.dedent(value).strip().splitlines(keepends=True),
                  "execution_count": None, "outputs": []})


md("""
# CatBoost GPU experiments — VS Code connected to Colab

This is a separate CatBoost workflow. The existing LightGBM notebooks are unchanged.
Connect this notebook to **Colab → New Colab Server → GPU**. Notebook cells run remotely;
the PowerShell terminal stays on your laptop. Upload `catboost_workflow/source.zip` with
VS Code Explorer's **Upload to Colab** command. You already have an uploaded source ZIP
for the older workflow; upload this separate ZIP too.

Run cells in order. The first full run is validation only. It cannot submit accidentally.
For a direct comparison, retain the July 2024 cutoff and no-promotion historical features.
Previous LightGBM validation AUC was 0.686407; reported Kaggle public score was 0.5656.
We cannot infer why those differ without auditing the exact submitted CSV.
""")
code(source(1))
code("""
from pathlib import Path
import hashlib, os, subprocess, sys, zipfile

SOURCE_ZIP = Path("/content/source.zip")  # Change if Colab Contents shows another path.
if not SOURCE_ZIP.is_file():
    raise FileNotFoundError(f"Upload catboost_workflow/source.zip to Colab, then set SOURCE_ZIP. Missing: {SOURCE_ZIP}")
digest = hashlib.sha256(SOURCE_ZIP.read_bytes()).hexdigest()
REPO = Path(f"/content/OptimlCatBoost_{digest[:10]}")
marker = REPO / ".source_zip.sha256"
if REPO.exists():
    if not marker.exists() or marker.read_text() != digest:
        raise RuntimeError("Source changed: choose a fresh REPO path or restart the Colab runtime before extracting.")
else:
    REPO.mkdir(parents=True)
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        for item in archive.infolist():
            if not (REPO / item.filename).resolve().is_relative_to(REPO.resolve()):
                raise ValueError("Unsafe ZIP path")
        archive.extractall(REPO)
    marker.write_text(digest)
assert (REPO / "optiml/gpu_check.py").is_file()
os.chdir(REPO)
print("CatBoost source ready:", REPO)
""")
md("""
## Install and verify CatBoost GPU

This installs CatBoost and the data libraries. It does not install or compile LightGBM.
The preflight trains a tiny CatBoost model on the selected GPU and fails if the GPU is unavailable.
""")
code("""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "catboost_workflow/requirements.txt")], check=True)
subprocess.run([sys.executable, "-m", "optiml.gpu_check", "--model", "catboost", "--device", "GPU"], cwd=REPO, check=True)
""")
drive = source(6).replace(
    'REPO = Path("/content/OptimlGPU")',
    'if "REPO" not in globals():\n    raise RuntimeError("Run the source setup cell first")',
)
drive = drive.replace('Path("/content/optiml_prepared/gpu-baseline-v1")', 'Path("/content/optiml_prepared/catboost-baseline-v1")')
drive = drive.replace('Path("/content/drive/MyDrive/OptimlRuns") / OWNER', 'Path("/content/drive/MyDrive/OptimlRuns/CatBoost") / OWNER')
md("""
## Mount Drive and prepare the seven competition CSVs

Keep the original CSVs in `MyDrive/OptimlData`. Preparation runs once per Colab runtime.
CatBoost runs back up to `MyDrive/OptimlRuns/CatBoost/jay`, separate from other models.
""")
code(drive)
code(source(7))
md("""
## Audit the earlier public-score submission when its exact CSV is available

This optional audit checks alignment against the prior run. A leaderboard gap has no known
cause until the submitted file and its generating run are compared.
""")
code(source(9))
md("""
## CatBoost experiment runner

Every fit explicitly requests GPU. Logs and results are backed up to Drive. Full validation
runs use the same held-out order lines, allowing an ID-hash check across configurations.
""")
runner = source(11).replace('def run_gpu(label, extra_args, model="lightgbm"):', 'def run_catboost(label, extra_args):')
runner = runner.replace('    device = "CUDA" if model == "lightgbm" else "GPU"', '    model, device = "catboost", "GPU"')
code(runner)
md("""
## Smoke test

This uses 3% of complete orders to check the full pipeline. Its AUC is not comparable to
the full-data baseline.
""")
code('smoke = run_catboost("smoke", ["--iterations", "30", "--patience", "10", "--sample-order-fraction", "0.03"])')
md("""
## Full-data CatBoost baseline

Start here after the smoke test. Keep promotion excluded and history enabled, matching the
reported LightGBM baseline. This validates only; it does not produce a Kaggle CSV.
""")
code("""
common = ["--cutoff", "2024-07-01", "--iterations", "2000", "--patience", "150"]
cb_baseline = run_catboost("baseline_depth6", common + ["--depth", "6"])
baseline_metrics = json.loads((cb_baseline / "metrics.json").read_text())
print("CatBoost validation AUC:", baseline_metrics["auc"])
print("Validation ID hash:", baseline_metrics["validation_ids_sha256"])
""")
md("""
## Compare focused CatBoost trials

After baseline, set `RUN_TRIALS=True`. Run one GPU fit at a time. Depth 5 tests lower
capacity; depth 8 tests higher capacity; the 12-month window tests recency; dropping raw
entity IDs tests whether memorization is hurting future generalization. Keep the July
validation ID hash identical. Compare October–December AUC, not only the aggregate.
""")
code("""
RUN_TRIALS = False
TRIALS = {
    "depth5": ["--depth", "5"],
    "depth8": ["--depth", "8"],
    "last12m": ["--depth", "6", "--train-months", "12"],
    "no_raw_ids": ["--depth", "6", "--drop-features", "customer_id", "product_id", "store_id"],
}
run_paths = {"baseline_depth6": cb_baseline}
trial_specs = {"baseline_depth6": ["--depth", "6"]}
if RUN_TRIALS:
    for name, flags in TRIALS.items():
        run_paths[name] = run_catboost(name, common + flags)
        trial_specs[name] = flags
else:
    print("Review the baseline first, then set RUN_TRIALS=True.")
""")
code("""
import pandas as pd

results = {name: json.loads((path / "metrics.json").read_text()) for name, path in run_paths.items()}
assert len({m["validation_ids_sha256"] for m in results.values()}) == 1
summary = pd.DataFrame({name: {"auc": m["auc"], "best_iterations": m["best_iterations"],
                               "seconds": m["validation_elapsed_seconds"]}
                        for name, m in results.items()}).T.sort_values("auc", ascending=False)
monthly = pd.DataFrame({name: {month: row["auc"] for month, row in m["monthly_auc"].items()
                               if row["rows"] >= 1000} for name, m in results.items()})
display(summary)
display(monthly)
display(pd.DataFrame(results["baseline_depth6"]["cohorts"]).T)
summary.to_csv(DRIVE_RUNS / (cb_baseline.name + "_comparison.csv"))
""")
md("""
## Check recent months for shortlisted models

After comparing July validation, enable this for the best one or two configurations.
It checks drift with an October 2024 cutoff; it is not an independent test set.
""")
code("""
RUN_RECENT_CHECK = False
SHORTLIST = summary.index[:2].tolist()
recent_paths = {}
if RUN_RECENT_CHECK:
    for name in dict.fromkeys(["baseline_depth6", *SHORTLIST]):
        recent_paths[name] = run_catboost(name + "_october", ["--cutoff", "2024-10-01", "--iterations", "2000", "--patience", "150"] + trial_specs[name])
    recent_metrics = {name: json.loads((path / "metrics.json").read_text()) for name, path in recent_paths.items()}
    assert len({m["validation_ids_sha256"] for m in recent_metrics.values()}) == 1
    display(pd.DataFrame({name: {"auc": m["auc"], "best_iterations": m["best_iterations"]}
                          for name, m in recent_metrics.items()}).T)
""")
md("""
## Recovery diagnostic after a weak leaderboard score

Run this only when temporal validation and Kaggle disagree sharply. It compares the same
three feature policies on two fixed six-month forecast blocks. The stable policy removes
target history and high-cardinality entity identifiers/proxies. Select a policy for another
submission only when it behaves consistently on both blocks; do not optimize another deep
tree configuration against the July block.
""")
code("""
RUN_ROLLING_DIAGNOSTIC = False
ROLLING_FOLDS = {
    "jan2024": "2024-01-01",
    "jul2024": "2024-07-01",
}
ROBUST_POLICIES = {
    "entity_history": ["--depth", "5"],
    "no_history": ["--depth", "5", "--no-history"],
    "stable_core": ["--depth", "5", "--no-history", "--drop-features",
                    "customer_id", "product_id", "store_id",
                    "product_supplier_code", "banner_id"],
}
rolling_paths = {}
if RUN_ROLLING_DIAGNOSTIC:
    for fold, cutoff in ROLLING_FOLDS.items():
        for policy, flags in ROBUST_POLICIES.items():
            name = f"{fold}_{policy}"
            rolling_paths[name] = run_catboost(
                name,
                ["--cutoff", cutoff, "--validation-months", "6",
                 "--iterations", "3000", "--patience", "200", *flags],
            )
else:
    print("Set RUN_ROLLING_DIAGNOSTIC=True after recording a weak leaderboard result.")
""")
code("""
if rolling_paths:
    import pandas as pd

    rolling_metrics = {
        name: json.loads((path / "metrics.json").read_text())
        for name, path in rolling_paths.items()
    }
    rolling_summary = pd.DataFrame({
        name: {
            "auc": metric["auc"],
            "best_iterations": metric["best_iterations"],
            "validation_rows": metric["validation_rows"],
        }
        for name, metric in rolling_metrics.items()
    }).T
    rolling_summary[["fold", "policy"]] = rolling_summary.index.to_series().str.split("_", n=1, expand=True)
    display(rolling_summary.sort_values(["policy", "fold"]))
    display(rolling_summary.pivot(index="policy", columns="fold", values="auc"))
""")
md("""
## Final refit and submission audit

Choose only a validated configuration. Finalization refits on all labeled lines and writes
all 479,995 test predictions. The audit checks order IDs and probabilities before submission.
""")
code("""
FINALIZE = False
SELECTED = "baseline_depth6"  # Change after inspecting full and recent validation.
if FINALIZE:
    if SELECTED not in trial_specs:
        raise ValueError("SELECTED has not been evaluated in this session")
    final_run = run_catboost(SELECTED + "_final", common + trial_specs[SELECTED] + ["--finalize"])
    subprocess.run([sys.executable, "-m", "optiml.audit_submission", "--csv", str(final_run / "submission.csv"),
                    "--run-dir", str(final_run), "--sample", str(DATA / "sample_submission.csv")], cwd=REPO, check=True)
    print("Use this exact backed-up CSV:", DRIVE_RUNS / final_run.name / "submission.csv")
else:
    print("Finalization is off during model selection.")
""")

for index, cell in enumerate(cells):
    cell["id"] = f"catboost-{index:02d}"
notebook = {"cells": cells, "metadata": old["metadata"], "nbformat": 4, "nbformat_minor": 5}
notebook["metadata"]["colab"]["name"] = "catboost_colab.ipynb"
(HERE / "catboost_colab.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")

files = [ROOT / "requirements.txt", HERE / "requirements.txt"]
files += sorted((ROOT / "optiml").glob("*.py"))
with zipfile.ZipFile(HERE / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(ROOT).as_posix())
print(HERE / "catboost_colab.ipynb")
print(HERE / "source.zip")
