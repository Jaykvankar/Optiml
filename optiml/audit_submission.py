"""Check that the actual submitted CSV matches a completed run's predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .prepare import sha256

def audit_submission(csv_path, run_dir, sample_path):
    submitted = pd.read_csv(csv_path)
    sample_path = Path(sample_path)
    sample = pd.read_parquet(sample_path) if sample_path.suffix == '.parquet' else pd.read_csv(sample_path)
    if list(submitted.columns) != ['order_line_id', 'returned']:
        raise ValueError('CSV must contain exactly order_line_id,returned in that order')
    if len(submitted) != len(sample) or submitted.order_line_id.duplicated().any() or set(submitted.order_line_id) != set(sample.order_line_id):
        raise ValueError('CSV IDs do not match sample submission')
    scores = submitted.returned.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Invalid submitted probabilities')
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    complete = json.loads((run_dir / 'COMPLETE.json').read_text())
    if not config.get('finalize') or not complete.get('finalized'):
        raise ValueError('This run was not a completed full refit')
    reference = pd.read_parquet(run_dir / 'test_predictions.parquet')
    compared = submitted.merge(reference, on='order_line_id', validate='one_to_one')
    if len(compared) != len(submitted) or not np.allclose(compared.returned, compared.prediction, rtol=1e-10, atol=1e-12):
        raise ValueError('CSV probabilities differ from this run when aligned by ID')
    result = {'verified': True, 'rows': len(submitted), 'sha256': sha256(csv_path),
              'same_sample_order': submitted.order_line_id.equals(sample.order_line_id),
              'probability_quantiles': submitted.returned.quantile([0, .01, .5, .99, 1]).to_dict(),
              'run_config': config}
    manifest = run_dir / 'submission_manifest.json'
    if manifest.exists():
        result['matches_original_file_hash'] = result['sha256'] == json.loads(manifest.read_text())['submission_sha256']
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_submission(args.csv, args.run_dir, args.sample), indent=2))

if __name__ == '__main__':
    main()
