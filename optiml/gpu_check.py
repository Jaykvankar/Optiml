"""Run a tiny real GPU fit before starting a long competition experiment."""
import argparse
import json
import platform
import subprocess
import numpy as np
import pandas as pd
from .train import backend_parameters

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['lightgbm', 'catboost'], default='lightgbm')
    p.add_argument('--device', choices=['CUDA', 'GPU'], default='CUDA')
    args = p.parse_args()
    backend = backend_parameters(args.model, args.device)
    if args.device == 'CUDA' and platform.system() != 'Linux':
        p.error('LightGBM CUDA requires Linux. Run this inside the Colab notebook kernel.')
    try:
        hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('No accessible NVIDIA GPU. Select a Colab GPU kernel, not the local Python kernel.') from exc
    print(hardware, flush=True)
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(1024, 4)), columns=list('abcd'))
    y = (x.a + x.b > 0).astype(int)
    x['category'] = pd.Categorical(np.arange(len(x)) % 80)
    if args.model == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=5, num_leaves=7, min_child_samples=10, max_bin=63, n_jobs=2, **backend)
        model.fit(x, y, categorical_feature=['category'])
        version = lgb.__version__
    else:
        import catboost
        x['category'] = x.category.astype(str)
        model = catboost.CatBoostClassifier(iterations=5, depth=4, verbose=False, allow_writing_files=False, **backend)
        model.fit(x, y, cat_features=['category'])
        version = catboost.__version__
    predictions = model.predict_proba(x)[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError('GPU preflight produced invalid predictions')
    print(json.dumps({'gpu_fit_passed': True, 'model': args.model, 'version': version, 'backend': backend}), flush=True)

if __name__ == '__main__':
    main()
