"""Investigate semantic inconsistencies and temporal single-feature diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'nexus-opti-ml-2026-operation-insight'
tr = pd.read_csv(DATA/'train.csv', low_memory=False)
te = pd.read_csv(DATA/'test.csv', low_memory=False)
cu = pd.read_csv(DATA/'customers.csv').sort_values('record_updated_at').drop_duplicates('customer_id',keep='last').set_index('customer_id')
pr = pd.read_csv(DATA/'products.csv').set_index('product_id')
st = pd.read_csv(DATA/'stores.csv').set_index('store_id')
out = {'category_values': sorted(pr.category.unique().tolist()), 'region_values': sorted(cu.region.unique().tolist())}

def auc(y,p):
    y = np.asarray(y)
    ranks = pd.Series(np.asarray(p)).rank(method='average').to_numpy()
    n = y.sum()
    return float((ranks[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))

for name, df in [('train',tr),('test',te)]:
    dt = pd.to_datetime(df.order_ts)
    df['_month'] = dt.dt.strftime('%Y-%m')
    df['customer_type'] = df.customer_id.map(cu.customer_type)
    df['loyalty_tier'] = df.customer_id.map(cu.loyalty_tier)
    df['category'] = df.product_id.map(pr.category).str.strip().str.lower().str.replace(' ','_')
    df['review_count'] = df.product_id.map(pr.review_count)
    df['seller_rating'] = df.store_id.map(st.seller_rating)
    gross = df.unit_price * df.quantity
    ratio = df.order_value/gross
    info = {'value_gross_ratio_quantiles':ratio.quantile([0,.01,.05,.25,.5,.75,.95,.99,1]).to_dict(),
            'value_within_half_unit_of_gross':float(((df.order_value-gross).abs()<=.5).mean()),
            'value_within_half_unit_of_discounted_gross':float(((df.order_value-gross*(1-df.discount)).abs()<=.5).mean()),
            'unit_price_to_catalog_ratio_quantiles':(df.unit_price/df.product_id.map(pr.list_price)).quantile([0,.01,.05,.5,.95,.99,1]).to_dict(),
            'date_tail_counts':dt.dt.strftime('%Y-%m-%d').value_counts().sort_index().tail(5).to_dict(),
            'hour_counts':dt.dt.hour.value_counts().sort_index().to_dict(),
            'tz_by_channel':pd.crosstab(df.channel,df.tz_hint.fillna('MISSING')).to_dict(),
            'timestamp_span_hours':((dt.groupby(df.order_id).max()-dt.groupby(df.order_id).min()).dt.total_seconds()/3600).value_counts().head(15).to_dict(),
            'same_order_line_duplicates':int(df.duplicated(['order_id','line_no']).sum())}
    out[name] = info

out['monthly_promo_rates'] = tr.pivot_table(index='_month',columns='promo_channel_code',values='returned',aggfunc='mean').to_dict()
out['monthly_promo_ordinal_auc'] = {month:auc(g.returned,g.promo_channel_code.str[1:].astype(int))
                                  for month,g in tr.groupby('_month') if len(g)>1000}
out['single_feature_temporal_auc'] = {}
for cutoff in ['2024-07','2024-10']:
    past = tr[tr._month < cutoff]
    future = tr[tr._month >= cutoff]
    scores = {}
    for col in ['promo_channel_code','channel','customer_type','loyalty_tier','category','product_id','customer_id',
                'store_id','banner_id','coupon_code','return_window_days','warehouse_zone','picker_shift','gift_wrap','review_count','seller_rating']:
        agg = past.groupby(col).returned.agg(['sum','count'])
        smooth = (agg['sum']+100*past.returned.mean())/(agg['count']+100)
        pred = future[col].map(smooth).fillna(past.returned.mean())
        scores[col] = auc(future.returned,pred)
    out['single_feature_temporal_auc'][cutoff] = scores
out['binned_target_rates'] = {}
for col in ['unit_price','quantity','discount','freight_cost','order_value','items_in_order','session_clicks','banner_id']:
    bins = pd.qcut(tr[col],10,duplicates='drop')
    out['binned_target_rates'][col] = tr.groupby(bins,observed=True).returned.agg(['size','mean']).to_dict('index')
out['missing_target_rates'] = {col:tr.groupby(tr[col].isna()).returned.agg(['size','mean']).to_dict('index') for col in ['payment_method','ship_mode','discount','freight_cost','session_clicks','coupon_code']}
out['fold_counts'] = {cutoff:{'past_rows':int((tr._month<cutoff).sum()),'future_rows':int((tr._month>=cutoff).sum()),
                            'overlapping_orders':len(set(tr.loc[tr._month<cutoff,'order_id']) & set(tr.loc[tr._month>=cutoff,'order_id']))}
                      for cutoff in ['2024-01','2024-07','2024-10']}

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
(ROOT/'analysis'/'targeted_checks.json').write_text(json.dumps(clean(out),indent=2),encoding='utf-8')
print('Saved targeted_checks.json')
print('Monthly promotion-only AUC:', out['monthly_promo_ordinal_auc'])
