"""Generate the checked-in Colab notebook from reviewable setup cells."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []

def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip().splitlines(keepends=True)})

def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(text).strip().splitlines(keepends=True), 'execution_count': None, 'outputs': []})

md('''
# OptiML — Jay's LightGBM baseline

**First instructions:** Open this file with Colab's **File → Upload notebook**. Choose
**Runtime → Change runtime type → Python 3, CPU / no accelerator**. Run the setup cells
in order. Keep this notebook in your own Drive.

1. Upload `dist/optiml_colab_source.zip` when the first cell asks (rebuild locally with
   `python tools/package_colab.py` after code edits). This route works without pushing Git.
2. Install packages. Restart the session if requested, then resume at configuration.
3. Put all seven CSVs in `MyDrive/OptimlData`, mount Drive and copy them to runtime storage.
4. Prepare features, run the smoke experiment, then run the full baseline.
5. Turn on `FINALIZE` only for a chosen full-data run to generate `submission.csv`.

Use `main` and `jay_patch1` only. Your teammate runs the same shared pipeline locally
with `--model catboost --device GPU`; see README. Save separate run folders.
No credentials or raw data belong in this notebook or Git.
''')
code('''
from pathlib import Path
import os, subprocess, sys, zipfile

SOURCE_MODE = "upload"  # Or "github" after jay_patch1 is pushed.
REPO = Path("/content/Optiml")
if REPO.exists():
    print("Existing source retained:", REPO)
else:
    if SOURCE_MODE == "upload":
        from google.colab import files
        print("Select dist/optiml_colab_source.zip from your computer")
        uploaded = files.upload()
        names = [name for name in uploaded if name.endswith(".zip")]
        if len(names) != 1:
            raise ValueError("Upload exactly one source ZIP")
        REPO.mkdir(parents=True)
        with zipfile.ZipFile(names[0]) as archive:
            for item in archive.infolist():
                destination = (REPO / item.filename).resolve()
                if not destination.is_relative_to(REPO.resolve()):
                    raise ValueError("Unsafe ZIP path")
            archive.extractall(REPO)
    elif SOURCE_MODE == "github":
        subprocess.run(["git", "clone", "--branch", "jay_patch1",
                        "https://github.com/Jaykvankar/Optiml.git", str(REPO)], check=True)
    else:
        raise ValueError("SOURCE_MODE must be upload or github")
os.chdir(REPO)
assert (REPO / "optiml/train.py").exists(), "Upload the generated source archive"
print("Source ready:", REPO)
''')
code('''
%pip install -q -r /content/Optiml/requirements.txt
''')
md('''
## Configure paths and persistent storage
The next cell is safe to run after a session restart. Change DRIVE_DATA if necessary.
Runtime storage is temporary. Finished runs are copied to your Drive directory.
''')
code('''
from pathlib import Path
from datetime import datetime, timezone
import os, sys, subprocess, shutil, json, uuid
from google.colab import drive

REPO = Path("/content/Optiml")
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
drive.mount("/content/drive")
DRIVE_DATA = Path("/content/drive/MyDrive/OptimlData")
DATA = Path("/content/optiml_data")
PREPARED = Path("/content/optiml_prepared/baseline-v1")
OWNER = "jay"
DRIVE_RUNS = Path("/content/drive/MyDrive/OptimlRuns") / OWNER
DATA.mkdir(parents=True, exist_ok=True)
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
NAMES = ["train", "test", "customers", "products", "stores", "calendar", "sample_submission"]
for name in NAMES:
    source = DRIVE_DATA / f"{name}.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}; place all seven original CSVs in DRIVE_DATA")
    shutil.copy2(source, DATA / source.name)
print("Local CSVs ready:", DATA)
''')
md('''
## Prepare the shared feature table
This reads all original rows, validates joins and creates features with no learned target
statistics. The training command computes histories inside each split. A completed cache
is reused only when its raw-data hashes and feature-source fingerprint match.
''')
code('''
from optiml.prepare import sha256

feature_source_hash = sha256(REPO / "optiml/features.py")
cache_manifest = PREPARED / "manifest.json"
cache_code = PREPARED / "feature_source.sha256"
if cache_manifest.exists():
    manifest = json.loads(cache_manifest.read_text())
    current = {name: sha256(DATA / f"{name}.csv") for name in NAMES}
    if manifest["source_sha256"] != current or not cache_code.exists() or cache_code.read_text() != feature_source_hash:
        raise ValueError("Data/features changed: choose a new PREPARED path and rerun preparation")
    print("Using matching completed cache:", PREPARED)
else:
    subprocess.run([sys.executable, "-m", "optiml.prepare", "--data-dir", str(DATA),
                    "--output-dir", str(PREPARED)], cwd=REPO, check=True)
    cache_code.write_text(feature_source_hash)
''')
md('''
## Run helper
Every call gets a unique directory. Logs stream below. Artifacts are backed up in `finally`,
including partial results on an ordinary Python exception. A hard VM termination can still
lose unsaved local files; the presence of COMPLETE.json distinguishes successful runs.
The default model uses CPU, excludes promotion and includes historical target rates.
''')
code('''
def run_experiment(label, extra_args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + label + "_" + uuid.uuid4().hex[:8]
    local = Path("/content/optiml_runs") / OWNER / run_id
    backup = DRIVE_RUNS / run_id
    command = [sys.executable, "-m", "optiml.train", "--prepared-dir", str(PREPARED),
               "--output-dir", str(local), "--model", "lightgbm", "--device", "CPU",
               "--threads", "2", *extra_args]
    print("Running:", " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=REPO, check=True)
    finally:
        if local.exists():
            shutil.copytree(local, backup, dirs_exist_ok=True)
            print("Backup:", backup)
    return local
''')
md('''
## First run: smoke test
This samples complete orders from BOTH periods. It verifies execution and saving, not
competitive AUC. Do not compare this score with full-validation experiments.
''')
code('''
smoke_run = run_experiment("smoke", ["--iterations", "30", "--patience", "10",
                                     "--sample-order-fraction", "0.03"])
print((smoke_run / "metrics.json").read_text())
''')
md('''
## Full LightGBM experiment
For your first full run leave the defaults below. For ablations change one switch at a time.
The October cutoff overlaps the July validation period. Use distinct runs and check the
validation-ID hashes before comparing. Both teammates should use these same split rules.
`FINALIZE=True` runs validation then fits the selected tree count on all labeled data (or
the selected recent window), producing the submission in sample-submission order.
''')
code('''
EXPERIMENT = "lgb_no_promo_history"
CUTOFF = "2024-07-01"  # Recent check: "2024-10-01"
INCLUDE_PROMO = False
USE_HISTORY = True
TRAIN_MONTHS = None   # Alternative: 12 or 6
ITERATIONS = 2000
FINALIZE = False

options = ["--cutoff", CUTOFF, "--iterations", str(ITERATIONS)]
if INCLUDE_PROMO:
    options.append("--include-promo")
if not USE_HISTORY:
    options.append("--no-history")
if TRAIN_MONTHS is not None:
    options += ["--train-months", str(TRAIN_MONTHS)]
if FINALIZE:
    options.append("--finalize")
full_run = run_experiment(EXPERIMENT, options)
''')
code('''
import pandas as pd

metrics = json.loads((full_run / "metrics.json").read_text())
print("Validation AUC:", metrics["auc"])
print("Best iterations:", metrics["best_iterations"])
print("Validation ID hash:", metrics["validation_ids_sha256"])
display(pd.DataFrame(metrics["monthly_auc"]).T)
display(pd.read_csv(full_run / "feature_importance.csv").head(20))
if (full_run / "submission.csv").exists():
    submission = pd.read_csv(full_run / "submission.csv")
    assert len(submission) == 479995
    assert submission.order_line_id.is_unique
    assert submission.returned.notna().all() and submission.returned.between(0, 1).all()
    display(submission.head())
    print("Submission already backed up to Drive. Download from the Files panel if desired.")
''')
md('''
## What to share with your teammate
Share metrics, config, fold manifest and validation predictions from each useful run.
Use identical validation IDs and join predictions by `order_line_id`. Your teammate can
run `python -m optiml.train --prepared-dir prepared/baseline-v1 --output-dir runs/teammate/cb01 --model catboost --device GPU` locally
after installing `requirements-catboost.txt` and running the same prepare command.
Keep the branch list as `main` and `jay_patch1`; experiment folders do not need Git branches.

References: [LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html),
[Colab runtime FAQ](https://research.google.com/colaboratory/faq.html).
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'colab': {'name': 'lightgbm_colab.ipynb', 'provenance': []}},
            'nbformat': 4, 'nbformat_minor': 5}
for index, cell in enumerate(cells):
    cell['id'] = f'optiml-{index:02d}'
target = ROOT / 'notebooks' / 'lightgbm_colab.ipynb'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
print(target)
