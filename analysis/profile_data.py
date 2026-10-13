"""Read-only profiling of the supplied competition CSV files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'nexus-opti-ml-2026-operation-insight'
OUT = ROOT / 'analysis'
result = {}
frames = {}
for name in ['customers', 'products', 'stores', 'calendar', 'train', 'test', 'sample_submission']:
    df = pd.read_csv(DATA / f'{name}.csv', low_memory=False)
    frames[name] = df
    info = {'rows': len(df), 'columns': list(df.columns), 'missing': df.isna().sum().to_dict(),
            'unique': df.nunique().to_dict(), 'dtypes': df.dtypes.astype(str).to_dict()}
    info['numeric'] = df.select_dtypes('number').describe(percentiles=[.01,.5,.95,.99]).to_dict()
    info['categories'] = {c: df[c].value_counts(dropna=False).head(25).to_dict()
                          for c in df.select_dtypes('object') if df[c].nunique() < 100}
    key = df.columns[0]
    info['duplicate_key_rows'] = int(df.duplicated(key).sum())
    info['exact_duplicate_rows'] = int(df.duplicated().sum())
    result[name] = info
    print(name, len(df), len(df.columns), 'duplicate key rows', info['duplicate_key_rows'], flush=True)

tr, te = frames['train'], frames['test']
for name, df in [('train', tr), ('test', te)]:
    # Raw wall clock for diagnostics; timezone interpretation is a modeling decision.
    df['_ts'] = pd.to_datetime(df.order_ts, format='mixed', errors='coerce', utc=True)
    df['_month'] = df['_ts'].dt.strftime('%Y-%m')
    result[name]['date_range'] = [str(df['_ts'].min()), str(df['_ts'].max())]
    result[name]['invalid_ts'] = int(df['_ts'].isna().sum())
    result[name]['monthly'] = (df.groupby('_month').returned.agg(['size','mean']) if name == 'train'
                              else df.groupby('_month').size()).to_dict()
    result[name]['unmatched'] = {key: int((~df[key].isin(frames[table][key])).sum())
                               for key, table in [('customer_id','customers'),('product_id','products'),('store_id','stores')]}
    group = df.groupby('order_id', sort=False)
    sizes = group.size()
    result[name]['order_sizes'] = sizes.describe(percentiles=[.5,.9,.99]).to_dict()
    result[name]['inconsistent_order_fields'] = {c: int((group[c].nunique() > 1).sum())
        for c in ['customer_id','order_ts','channel','store_id','items_in_order','order_value']}
    result[name]['observed_vs_items'] = float((sizes == group.items_in_order.first()).mean())
    result[name]['line_no_gt_items'] = int((df.line_no > df.items_in_order).sum())
    result[name]['order_value_lt_line_gross'] = int((df.order_value < df.unit_price * df.quantity).sum())
    result[name]['calendar_unmatched_raw_date'] = int((~(df['_ts'].dt.tz_localize(None).dt.normalize()
        - pd.to_timedelta(df['_ts'].dt.dayofweek, unit='D')).isin(pd.to_datetime(frames['calendar'].week_start))).sum())
    result[name]['channel_store_mismatch'] = int((df.channel.str.strip().str.lower() !=
        df.store_id.map(frames['stores'].set_index('store_id').channel).str.strip().str.lower()).sum())

result['overlap'] = {c: {'shared_unique':len(set(tr[c]) & set(te[c])),
    'test_unseen_rows':int((~te[c].isin(tr[c])).sum())} for c in ['order_line_id','order_id','customer_id','product_id','store_id']}
result['sample_matches_test_order'] = bool(frames['sample_submission'].order_line_id.equals(te.order_line_id))
result['target'] = tr.returned.value_counts().to_dict()
result['target_rate'] = float(tr.returned.mean())
result['target_by_feature'] = {}
for c in ['channel','payment_method','ship_mode','quantity','discount','gift_wrap','promo_channel_code',
          'return_window_days','warehouse_zone','picker_shift','device_type','coupon_code','tz_hint']:
    result['target_by_feature'][c] = tr.groupby(c, dropna=False).returned.agg(['size','mean']).to_dict('index')
for table, key in [('customers','customer_id'),('products','product_id'),('stores','store_id')]:
    dim = frames[table]
    if table == 'customers':
        dup = dim[dim.duplicated(key, keep=False)]
        result[table]['duplicate_ids'] = int(dup[key].nunique())
        result[table]['conflicting_duplicate_fields'] = {c:int((dup.groupby(key)[c].nunique(dropna=False)>1).sum()) for c in dim.columns if c != key}
        dim = dim.sort_values('record_updated_at').drop_duplicates(key, keep='last')
    indexed = dim.set_index(key)
    for name, df in [('train',tr),('test',te)]:
        for date in ['signup_date','record_updated_at','launch_date','opened_date']:
            if date in dim:
                dt = pd.to_datetime(df[key].map(indexed[date]), errors='coerce', utc=True)
                result[name][f'{date}_after_order'] = int((dt > df['_ts']).sum())
    for c in dim.columns:
        if c != key and (dim[c].nunique() < 50 or c in ['review_count','seller_rating']):
            mapped = tr[key].map(indexed[c])
            if pd.api.types.is_numeric_dtype(mapped) and mapped.nunique() > 20:
                mapped = pd.qcut(mapped, 10, duplicates='drop').astype(str)
            result['target_by_feature'][table+'.'+c] = tr.groupby(mapped,dropna=False).returned.agg(['size','mean']).to_dict('index')

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
(OUT / 'data_profile.json').write_text(json.dumps(clean(result), indent=2), encoding='utf-8')
print('Profile saved:', OUT / 'data_profile.json', flush=True)
