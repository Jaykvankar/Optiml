"""Build a GPU notebook usable with the official VS Code Colab extension."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
baseline = json.loads((ROOT / 'notebooks/lightgbm_colab.ipynb').read_text(encoding='utf-8'))
cells = []

def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(s).strip().splitlines(keepends=True)})

def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(s).strip().splitlines(keepends=True), 'execution_count': None, 'outputs': []})

def baseline_code(marker):
    return next(''.join(c['source']) for c in baseline['cells'] if c['cell_type'] == 'code' and marker in ''.join(c['source']))

md('''
# OptiML GPU experiments — Colab from VS Code

**Start here in VS Code:** You already have Google's Colab extension. Select this notebook's
**Select Kernel → Colab → New Colab Server**, sign in, and choose an available NVIDIA GPU
machine. Choose the remote kernel, not your laptop's `.venv` kernel. The first cell checks it.
In the browser instead: Runtime → Change runtime type → GPU.

The laptop checked for this project has Intel Arc graphics. It is not an NVIDIA CUDA GPU.
Your ordinary VS Code PowerShell terminal remains local; these notebook cells run remotely.

**Upload the updated source:** right-click `dist/optiml_colab_source.zip` in VS Code Explorer
and select **Upload to Colab**. Check the remote Contents panel; use its actual path below.
Do not upload the entire repository/venv/data directory. For the browser, use its Files panel.
This notebook does not use `google.colab.files.upload()`, which has VS Code limitations.

Leave existing Drive run folders intact. We recorded 0.686407 validation AUC from the old CPU
baseline, and you reported 0.5656 on Kaggle. The exact submitted run still needs auditing.
GPU hardware enables different/faster experiments; it does not guarantee a better ranking.

[Colab VS Code guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
''')
code('''
import platform, subprocess

assert platform.system() == "Linux", "Select the remote Colab kernel, not local Windows Python"
subprocess.run(["nvidia-smi"], check=True)
print("Run cells in this remote notebook; local PowerShell does not use this GPU.")
''')
code('''
from pathlib import Path
import hashlib, os, subprocess, sys, zipfile

SOURCE_MODE = "uploaded_zip"  # "github" only after updated GPU code has been pushed.
SOURCE_ZIP = Path("/content/optiml_colab_source.zip")
REPO = Path("/content/OptimlGPU")
if SOURCE_MODE == "uploaded_zip":
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError("Right-click the source ZIP in VS Code Explorer → Upload to Colab, then set SOURCE_ZIP")
    digest = hashlib.sha256(SOURCE_ZIP.read_bytes()).hexdigest()
    marker = REPO / ".source_zip.sha256"
    if REPO.exists():
        if not marker.exists() or marker.read_text() != digest:
            raise RuntimeError("Different source already exists. Choose a fresh REPO directory and restart imports.")
    else:
        REPO.mkdir(parents=True)
        with zipfile.ZipFile(SOURCE_ZIP) as archive:
            for item in archive.infolist():
                if not (REPO / item.filename).resolve().is_relative_to(REPO.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(REPO)
        marker.write_text(digest)
elif SOURCE_MODE == "github":
    if not REPO.exists():
        subprocess.run(["git", "clone", "--branch", "jay_patch1",
                        "https://github.com/Jaykvankar/Optiml.git", str(REPO)], check=True)
else:
    raise ValueError("Unknown SOURCE_MODE")
assert (REPO / "optiml/gpu_check.py").exists(), "This is old source; upload the rebuilt GPU ZIP"
os.chdir(REPO)
print("Remote source:", REPO)
''')
md('''
## Install GPU-capable libraries
For **LightGBM CUDA**, the install cell compiles the installed LightGBM version with CUDA
enabled. This can take several minutes and needs the CUDA compiler supplied by the runtime.
Selecting a GPU or installing an ordinary wheel alone does not enable this CUDA backend.
CatBoost GPU needs no LightGBM compilation: set BUILD_LIGHTGBM_CUDA=False if you only want
CatBoost. Run this before importing either model library. If already imported, restart the
session after installation and rerun setup/configuration (without rebuilding unnecessarily).

[Official LightGBM build instructions](https://github.com/lightgbm-org/LightGBM/tree/main/python-package)
''')
code('''
import importlib.metadata, shutil

BUILD_LIGHTGBM_CUDA = True  # False for a CatBoost-only session.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-catboost.txt")], check=True)
if BUILD_LIGHTGBM_CUDA:
    probe = subprocess.run([sys.executable, "-m", "optiml.gpu_check", "--model", "lightgbm", "--device", "CUDA"],
                           cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if probe.returncode != 0:
        print("CUDA preflight before rebuild:", probe.stdout[-4000:])
        if not shutil.which("nvcc"):
            raise RuntimeError("CUDA compiler not found. Use CatBoost GPU, or select a CUDA-enabled Colab GPU runtime.")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"], check=True)
        installed_version = importlib.metadata.version("lightgbm")
        build_env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL="2")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-cache-dir",
            "--no-binary=lightgbm", "--config-settings=cmake.define.USE_CUDA=ON",
            "--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native",
            f"lightgbm=={installed_version}",
        ], env=build_env, check=True)
    subprocess.run([sys.executable, "-m", "optiml.gpu_check", "--model", "lightgbm", "--device", "CUDA"], cwd=REPO, check=True)
subprocess.run([sys.executable, "-m", "optiml.gpu_check", "--model", "catboost", "--device", "GPU"], cwd=REPO, check=True)
''')
md('''
## Drive, CSVs and shared features
Use the same seven original files under MyDrive/OptimlData. Drive mounting is supported by
current versions of the official extension. If prompted, complete authorization. The Colab
toolbar's Mount Google Drive command can also insert a mount cell.
''')
drive_code = baseline_code('DRIVE_DATA =').replace('Path("/content/Optiml")', 'Path("/content/OptimlGPU")')
drive_code = drive_code.replace('Path("/content/optiml_prepared/baseline-v1")', 'Path("/content/optiml_prepared/gpu-baseline-v1")')
code(drive_code)
code(baseline_code('feature_source_hash ='))
md('''
## Audit the file that scored 0.5656 (optional but recommended)
Set PREVIOUS_RUN to its completed Drive run directory and SUBMITTED_CSV to the actual file
you uploaded, not a newly generated replacement. A pass confirms IDs and scores match that
run, not that its model generalizes. Missing run artifacts must be recovered before matching
can be verified. An unknown cause must not be assumed to be GPU/CPU choice.
''')
code('''
PREVIOUS_RUN = None  # Example: Path("/content/drive/MyDrive/OptimlRuns/jay/<your_final_run>")
SUBMITTED_CSV = None # Path to the exact CSV used for the 0.5656 Kaggle submission.
if PREVIOUS_RUN is not None and SUBMITTED_CSV is not None:
    subprocess.run([sys.executable, "-m", "optiml.audit_submission", "--csv", str(SUBMITTED_CSV),
                    "--run-dir", str(PREVIOUS_RUN), "--sample", str(DATA / "sample_submission.csv")], cwd=REPO, check=True)
else:
    print("Audit pending: set PREVIOUS_RUN and SUBMITTED_CSV when available.")
''')
md('''
## Experiment runner: real GPU, visible logs, separate backups
Default: LightGBM's CUDA backend. CatBoost uses `GPU` as its device name. GPU requests never
fall back to CPU. This helper streams subprocess output to the notebook and saves train.log.
Each run copies its source/config and uses a fresh directory. Only COMPLETE.json marks success.
''')
code('''
def run_gpu(label, extra_args, model="lightgbm"):
    device = "CUDA" if model == "lightgbm" else "GPU"
    subprocess.run([sys.executable, "-m", "optiml.gpu_check", "--model", model, "--device", device], cwd=REPO, check=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + label + "_" + uuid.uuid4().hex[:8]
    local = Path("/content/optiml_gpu_runs") / OWNER / run_id
    local.parent.mkdir(parents=True, exist_ok=True)
    backup = DRIVE_RUNS / run_id
    log_path = local.parent / (run_id + ".log")
    command = [sys.executable, "-u", "-m", "optiml.train", "--prepared-dir", str(PREPARED),
               "--output-dir", str(local), "--model", model, "--device", device,
               "--threads", "2", *extra_args]
    print("Running:", " ".join(command), flush=True)
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if local.exists():
            shutil.copy2(log_path, local / "train.log")
            shutil.copytree(local, backup, dirs_exist_ok=True)
        print("Backup location:", backup)
    return local
''')
md('''
## First GPU competition-data smoke test
This is a setup check. Sampling changes the validation population; do not compare its score
with 0.6864. Change SMOKE_MODEL to catboost if you skipped the LightGBM CUDA build.
''')
code('''
SMOKE_MODEL = "lightgbm"
smoke = run_gpu("gpu_smoke", ["--iterations", "30", "--patience", "10", "--sample-order-fraction", "0.03"], model=SMOKE_MODEL)
''')
md('''
## Full-data baseline on GPU
Use this first, with the old features/split, to measure the GPU backend independently of
feature changes. Do not expect changing hardware alone to improve AUC. No submission yet.
''')
code('''
BASE_MODEL = "lightgbm"  # "catboost" also supported.
common = ["--cutoff", "2024-07-01", "--iterations", "2000", "--patience", "150"]
gpu_baseline = run_gpu("gpu_baseline_no_promo", common, model=BASE_MODEL)
baseline_metrics = json.loads((gpu_baseline / "metrics.json").read_text())
print("Baseline GPU validation AUC:", baseline_metrics["auc"])
''')
md('''
## Controlled model search
Set RUN_SEARCH=True only after the full baseline passes. These are sequential full-data
trials, not parallel GPU jobs. Each changes capacity, fit window, or ID reliance. They share
the same July validation rows; ID hashes and target labels are checked before comparison.
The default search below requires LightGBM CUDA. Edit TRIALS for CatBoost-only sessions.
Promotion remains excluded because its historical signal fades; it can be a separate trial.
''')
code('''
RUN_SEARCH = False
TRIALS = {
    "lgb_leaves63_l2_10": ("lightgbm", ["--num-leaves", "63", "--l2", "10"]),
    "lgb_recent12": ("lightgbm", ["--train-months", "12"]),
    "lgb_no_raw_entity_ids": ("lightgbm", ["--drop-features", "customer_id", "product_id", "store_id"]),
    "catboost_depth6": ("catboost", ["--depth", "6"]),
}
run_paths = {"gpu_baseline": gpu_baseline}
trial_specs = {"gpu_baseline": (BASE_MODEL, [])}
if RUN_SEARCH:
    for name, (model, flags) in TRIALS.items():
        run_paths[name] = run_gpu(name, common + flags, model=model)
        trial_specs[name] = (model, flags)
else:
    print("Search is off. Inspect baseline, then enable RUN_SEARCH for the trials above.")
''')
code('''
import pandas as pd

results = {name: json.loads((path / "metrics.json").read_text()) for name, path in run_paths.items()}
assert len({m["validation_ids_sha256"] for m in results.values()}) == 1
summary = pd.DataFrame({name: {"auc": m["auc"], "best_iterations": m["best_iterations"],
                                 "seconds": m["validation_elapsed_seconds"]}
                        for name, m in results.items()}).T.sort_values("auc", ascending=False)
monthly = pd.DataFrame({name: {month: row["auc"] for month, row in m["monthly_auc"].items() if row["rows"] >= 1000}
                        for name, m in results.items()})
display(summary)
display(monthly)
display(pd.DataFrame(results["gpu_baseline"]["cohorts"]).T)
shortlist = summary.index[:2].tolist()
print("Candidates to check on the recent split:", shortlist)
summary.to_csv(DRIVE_RUNS / (gpu_baseline.name + "_comparison.csv"))
''')
md('''
## Recent-period confirmation before choosing a model
Enable to compare the shortlisted candidates when fitted before October 2024. This split
overlaps the July holdout: it is a drift check, not a fresh independent test. Compare each
candidate's result to the baseline trained at the same October cutoff. Neither score estimates
the hidden leaderboard perfectly. Share both tables before making a final selection.
''')
code('''
RUN_RECENT_CHECK = False
recent_paths = {}
if RUN_RECENT_CHECK:
    for name in dict.fromkeys(["gpu_baseline", *shortlist]):
        model, flags = trial_specs[name]
        recent_paths[name] = run_gpu(name + "_october", ["--cutoff", "2024-10-01", "--iterations", "2000"] + flags, model=model)
    recent_metrics = {name: json.loads((path / "metrics.json").read_text()) for name, path in recent_paths.items()}
    assert len({m["validation_ids_sha256"] for m in recent_metrics.values()}) == 1
    display(pd.DataFrame({name: {"auc": m["auc"], "best_iterations": m["best_iterations"]}
                          for name, m in recent_metrics.items()}).T)
''')
md('''
## Refit the chosen configuration and audit its submission
Choose from configurations actually evaluated; do not select solely on an old public score.
This repeats the July validation to choose iterations, then refits and predicts all test rows.
`submission_manifest.json` records the exact file hash so the next Kaggle score is traceable.
It is not automatically uploaded to Kaggle.
''')
code('''
FINALIZE = False
SELECTED = "gpu_baseline"  # Change after assessing both validation views.
if FINALIZE:
    model, flags = trial_specs[SELECTED]
    final_run = run_gpu(SELECTED + "_final", common + flags + ["--finalize"], model=model)
    subprocess.run([sys.executable, "-m", "optiml.audit_submission", "--csv", str(final_run / "submission.csv"),
                    "--run-dir", str(final_run), "--sample", str(DATA / "sample_submission.csv")], cwd=REPO, check=True)
    print("Submit this exact backed-up file:", DRIVE_RUNS / final_run.name / "submission.csv")
else:
    print("Finalization is off while comparing candidates.")
''')

for i, c in enumerate(cells):
    c['id'] = f'gpu-{i:02d}'
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
      'language_info': {'name': 'python'}, 'colab': {'name': 'gpu_colab.ipynb', 'provenance': []}, 'accelerator': 'GPU'},
      'nbformat': 4, 'nbformat_minor': 5}
target = ROOT / 'notebooks/gpu_colab.ipynb'
target.write_text(json.dumps(nb, indent=2), encoding='utf-8')
print(target)
