"""Build reusable target-independent features: python -m optiml.prepare --help."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .features import VERSION, build_features, dimensions

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Output directory must be new or empty to prevent stale/mixed features')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dims = dimensions(args.data_dir)
    files = ['train', 'test', 'customers', 'products', 'stores', 'calendar', 'sample_submission']
    meta = {'feature_version': VERSION, 'source_sha256': {n: sha256(args.data_dir / f'{n}.csv') for n in files},
            'customer_policy': 'latest supplied snapshot; excludes loyalty/update date',
            'time_policy': 'raw wall clock; max timestamp per order for split/history',
            'excluded_snapshots': ['loyalty_tier', 'review_count', 'seller_rating', 'manager_tenure_years', 'launch_date']}
    train_ids = train_orders = None
    for name in ['train', 'test']:
        print(f'Preparing {name}...', flush=True)
        raw = pd.read_csv(args.data_dir / f'{name}.csv', low_memory=False)
        if name == 'train':
            if raw.returned.isna().any() or not raw.returned.isin([0, 1]).all():
                raise ValueError('Invalid training targets')
            train_ids, train_orders = set(raw.order_line_id), set(raw.order_id)
        elif train_ids.intersection(raw.order_line_id) or train_orders.intersection(raw.order_id):
            raise ValueError('Train/test identifiers overlap')
        features = build_features(raw, dims)
        meta[name + '_rows'] = len(features)
        features.to_parquet(args.output_dir / f'{name}.parquet', index=False)
        if name == 'test':
            sample = pd.read_csv(args.data_dir / 'sample_submission.csv')
            if list(sample.columns) != ['order_line_id', 'returned'] or sample.order_line_id.duplicated().any() or set(sample.order_line_id) != set(features.order_line_id):
                raise ValueError('Invalid sample submission IDs/columns')
            sample.to_parquet(args.output_dir / 'sample_submission.parquet', index=False)
        del raw, features
    (args.output_dir / 'manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

if __name__ == '__main__':
    main()
