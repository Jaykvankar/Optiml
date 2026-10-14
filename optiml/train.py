"""Train a shared temporal baseline and optionally refit for submission."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from .features import encode_categories, feature_columns
from .history import add_history

def id_digest(ids):
    return hashlib.sha256('\n'.join(sorted(ids.astype(str))).encode()).hexdigest()

def split_frame(frame, cutoff, train_months=None, validation_months=None):
    cutoff = pd.Timestamp(cutoff)
    before = frame['_order_time'] < cutoff
    if train_months:
        before &= frame['_order_time'] >= cutoff - pd.DateOffset(months=train_months)
    fit = frame.loc[before].copy()
    validation = frame['_order_time'] >= cutoff
    if validation_months:
        validation &= frame['_order_time'] < cutoff + pd.DateOffset(months=validation_months)
    valid = frame.loc[validation].copy()
    if fit.empty or valid.empty or fit.returned.nunique() != 2 or valid.returned.nunique() != 2:
        raise ValueError('Both temporal partitions must contain both target classes')
    if set(fit.order_id).intersection(valid.order_id):
        raise AssertionError('An order crosses the validation boundary')
    return fit, valid

def select_orders(frame, fraction, seed):
    if fraction == 1:
        return frame
    # Fixed hash threshold: complete orders, same selection for both model families.
    hashed = pd.util.hash_pandas_object(frame.order_id, index=False).to_numpy(dtype='uint64')
    selected = ((hashed ^ np.uint64(seed)) % np.uint64(1_000_003)) < int(fraction * 1_000_003)
    return frame.loc[selected].copy()

def model_inputs(fit, forecast, include_promo, history, family, drop_features=()):
    if history:
        fit, forecast = add_history(fit, forecast)
    cols = feature_columns(fit, include_promo)
    unknown = set(drop_features) - set(cols)
    if unknown:
        raise ValueError(f'Unknown/excluded drop features: {sorted(unknown)}')
    cols = [c for c in cols if c not in drop_features]
    x, z, levels = encode_categories(fit, forecast, cols)
    if family == 'catboost':
        for c in levels:
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
            z[c] = z[c].astype('string').fillna('__MISSING__').astype(str)
    schema = {'features': cols, 'category_levels': levels, 'history': history,
              'history_policy': 'monthly expanding, prior months only, frozen forecast',
              'label_maturity_assumption': 'all prior-month labels available; no event dates supplied',
              'include_promo': include_promo, 'drop_features': list(drop_features)}
    return x, z, schema

def backend_parameters(family, device):
    """Never silently substitute CPU for a requested GPU backend."""
    if family == 'lightgbm':
        if device == 'CPU':
            return {'device_type': 'cpu', 'deterministic': True, 'force_col_wise': True}
        if device in {'GPU', 'CUDA'}:
            return {'device_type': device.lower(), 'gpu_device_id': 0}
    elif family == 'catboost':
        if device == 'CPU':
            return {'task_type': 'CPU'}
        if device == 'GPU':
            return {'task_type': 'GPU', 'devices': '0'}
    raise ValueError(f'Unsupported model/device: {family}/{device}; CatBoost uses GPU, not CUDA, as its CLI value')

def fit_model(x, y, z, labels, schema, args, output, iterations):
    backend = backend_parameters(args.model, args.device)
    print(f'Training {args.model} with backend {backend}', flush=True)
    if args.model == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', n_estimators=iterations,
            learning_rate=args.learning_rate, num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples, reg_lambda=args.l2,
            max_bin=args.max_bin,
            colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
            random_state=args.seed, n_jobs=args.threads, verbosity=-1,
            **backend,
        )
        kwargs = {}
        if z is not None:
            kwargs = {'eval_set': [(z, labels)], 'eval_metric': 'auc',
                      'callbacks': [lgb.early_stopping(args.patience, first_metric_only=True), lgb.log_evaluation(100)]}
        model.fit(x, y, categorical_feature=list(schema['category_levels']), **kwargs)
        model.booster_.save_model(str(output / 'model.txt'))
        importance = model.booster_.feature_importance(importance_type='gain')
        best = model.best_iteration_ or iterations
    else:
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise RuntimeError('Install requirements-catboost.txt for this model') from e
        model = CatBoostClassifier(
            loss_function='Logloss', eval_metric='AUC',
            iterations=iterations, depth=args.depth, learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2, max_ctr_complexity=1, random_seed=args.seed,
            thread_count=args.threads, train_dir=str(output / 'catboost_info'), **backend,
        )
        fit_kwargs = {'eval_set': (z, labels), 'early_stopping_rounds': args.patience} if z is not None else {}
        model.fit(x, y, cat_features=list(schema['category_levels']), verbose=100, **fit_kwargs)
        model.save_model(str(output / 'model.cbm'))
        importance = model.feature_importances_
        best = model.get_best_iteration() + 1 if z is not None else iterations
        best = max(best, 1)
    pd.DataFrame({'feature': x.columns, 'importance': importance}).sort_values('importance', ascending=False).to_csv(output / 'feature_importance.csv', index=False)
    (output / 'schema.json').write_text(json.dumps(schema, indent=2))
    return model, int(best)

def evaluation(valid, pred, fit):
    result = {'auc': float(roc_auc_score(valid.returned, pred)),
              'log_loss': float(log_loss(valid.returned, pred)),
              'fit_rows': len(fit), 'validation_rows': len(valid),
              'fit_positive_rate': float(fit.returned.mean()),
              'validation_positive_rate': float(valid.returned.mean()),
              'fit_ids_sha256': id_digest(fit.order_line_id),
              'validation_ids_sha256': id_digest(valid.order_line_id),
              'monthly_auc': {}, 'cohorts': {}}
    def metric(mask):
        part = valid.loc[mask, 'returned']
        return {'rows': len(part), 'positives': int(part.sum()),
                'auc': float(roc_auc_score(part, pred[np.asarray(mask)])) if part.nunique() == 2 else None}
    for m in sorted(valid['_month'].unique()):
        result['monthly_auc'][m] = metric(valid['_month'].eq(m))
    for c in sorted(valid.channel.unique()):
        result['cohorts']['channel_' + c] = metric(valid.channel.eq(c))
    unseen = ~valid.customer_id.isin(fit.customer_id)
    result['cohorts']['unseen_customer'] = metric(unseen)
    result['cohorts']['seen_customer'] = metric(~unseen)
    return result

def write_submission(sample, test_ids, predictions, path):
    predictions = np.asarray(predictions)
    if len(predictions) != len(test_ids) or not np.isfinite(predictions).all() or not ((predictions >= 0) & (predictions <= 1)).all():
        raise ValueError('Invalid probabilities')
    values = pd.DataFrame({'order_line_id': np.asarray(test_ids), 'returned': predictions})
    if values.order_line_id.duplicated().any() or sample.order_line_id.duplicated().any() or set(values.order_line_id) != set(sample.order_line_id):
        raise ValueError('Submission ID mismatch or duplication')
    output = sample[['order_line_id']].merge(values, how='left', on='order_line_id', sort=False, validate='one_to_one')
    if len(output) != len(sample) or not output.order_line_id.equals(sample.order_line_id.reset_index(drop=True)):
        raise AssertionError('Submission alignment failure')
    output.to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model', choices=['lightgbm', 'catboost'], default='lightgbm')
    parser.add_argument('--device', choices=['CPU', 'GPU', 'CUDA'], default='CPU',
                        help='LightGBM: CPU, GPU (OpenCL), CUDA (Linux NVIDIA). CatBoost: CPU or GPU.')
    parser.add_argument('--cutoff', default='2024-07-01')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--num-leaves', type=int, default=31)
    parser.add_argument('--min-child-samples', type=int, default=200)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--l2', type=float, default=5.0)
    parser.add_argument('--max-bin', type=int, default=255)
    parser.add_argument('--drop-features', nargs='+', default=[], help='Explicit feature ablation; IDs remain available for history')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-months', type=int)
    parser.add_argument('--validation-months', type=int,
                        help='Limit validation to this many months after cutoff; final refit is unaffected')
    parser.add_argument('--include-promo', action='store_true')
    parser.add_argument('--no-history', action='store_true')
    parser.add_argument('--sample-order-fraction', type=float, default=1.0,
                        help='Development only: sample complete orders on BOTH sides; changes validation IDs')
    parser.add_argument('--finalize', action='store_true', help='After validation, refit on labeled data and write submission')
    args = parser.parse_args()
    if (not 0 < args.sample_order_fraction <= 1 or args.iterations < 1 or args.patience < 1
            or (args.train_months is not None and args.train_months < 1)
            or (args.validation_months is not None and args.validation_months < 1)):
        parser.error('Invalid fraction, iterations, patience, or training window')
    if args.finalize and args.sample_order_fraction != 1:
        parser.error('Finalization requires full data (--sample-order-fraction 1)')
    if args.l2 < 0 or args.max_bin < 2:
        parser.error('l2 must be nonnegative and max-bin >= 2')
    backend_parameters(args.model, args.device)
    # Fail before expensive data loading if the chosen model dependency is absent.
    __import__(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    source_root = Path(__file__).resolve().parents[1]
    shutil.copytree(source_root / 'optiml', args.output_dir / 'source' / 'optiml', ignore=shutil.ignore_patterns('__pycache__'))
    versions = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (args.output_dir / 'environment.txt').write_text(versions)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        revision = 'unknown; see saved source'
    meta = json.loads((args.prepared_dir / 'manifest.json').read_text())
    meta.update({'code_sha': revision, 'started_utc': datetime.now(timezone.utc).isoformat()})
    meta['backend'] = backend_parameters(args.model, args.device)
    (args.output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('Loading prepared training features...', flush=True)
    frame = pd.read_parquet(args.prepared_dir / 'train.parquet')
    frame = select_orders(frame, args.sample_order_fraction, args.seed)
    fit, valid = split_frame(frame, args.cutoff, args.train_months, args.validation_months)
    del frame
    gc.collect()
    manifest = pd.concat([fit[['order_line_id', 'order_id', '_month']].assign(role='fit'), valid[['order_line_id', 'order_id', '_month']].assign(role='validation')])
    manifest.to_parquet(args.output_dir / 'fold_manifest.parquet', index=False)
    del manifest
    print(f'Fit {len(fit):,} rows; validate {len(valid):,} rows. Building model inputs...', flush=True)
    x, z, schema = model_inputs(fit, valid, args.include_promo, not args.no_history, args.model, args.drop_features)
    model, best = fit_model(x, fit.returned, z, valid.returned, schema, args, args.output_dir, args.iterations)
    pred = model.predict_proba(z)[:, 1]
    metrics = evaluation(valid, pred, fit)
    metrics.update({'best_iterations': best, 'sample_order_fraction': args.sample_order_fraction,
                    'validation_elapsed_seconds': time.monotonic() - started})
    saved = valid[['order_line_id', 'order_id', '_month', 'returned']].copy()
    saved['prediction'] = pred
    saved.to_parquet(args.output_dir / 'validation_predictions.parquet', index=False)
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    if args.finalize:
        del fit, valid, x, z, saved, pred, model
        gc.collect()
        print('Refitting selected iteration count on labeled history...', flush=True)
        fit = pd.read_parquet(args.prepared_dir / 'train.parquet')
        test = pd.read_parquet(args.prepared_dir / 'test.parquet')
        if args.train_months:
            # Final training origin follows the last labeled order, retaining edge rows.
            origin = fit['_order_time'].max() + pd.Timedelta(nanoseconds=1)
            fit = fit.loc[fit['_order_time'] >= origin - pd.DateOffset(months=args.train_months)].copy()
        final_dir = args.output_dir / 'final'
        final_dir.mkdir()
        x, z, schema = model_inputs(fit, test, args.include_promo, not args.no_history, args.model, args.drop_features)
        model, _ = fit_model(x, fit.returned, None, None, schema, args, final_dir, best)
        pred = model.predict_proba(z)[:, 1]
        sample = pd.read_parquet(args.prepared_dir / 'sample_submission.parquet')
        write_submission(sample, test.order_line_id, pred, args.output_dir / 'submission.csv')
        pd.DataFrame({'order_line_id': test.order_line_id, 'prediction': pred}).to_parquet(args.output_dir / 'test_predictions.parquet', index=False)
        from .prepare import sha256
        (args.output_dir / 'submission_manifest.json').write_text(json.dumps({
            'submission_sha256': sha256(args.output_dir / 'submission.csv'),
            'test_prediction_sha256': sha256(args.output_dir / 'test_predictions.parquet'),
            'rows': len(test), 'config': config, 'best_iterations': best,
            'validation_auc': metrics['auc'],
        }, indent=2))
        (final_dir / 'refit.json').write_text(json.dumps({'rows': len(fit), 'iterations': best, 'test_rows': len(test)}, indent=2))
    (args.output_dir / 'COMPLETE.json').write_text(json.dumps({'elapsed_seconds': time.monotonic() - started, 'finalized': args.finalize}))
    print(f'Run complete: {args.output_dir}', flush=True)

if __name__ == '__main__':
    main()
