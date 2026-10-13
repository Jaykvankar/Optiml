"""Target-independent joins and features. Raw CSV files are never modified."""
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'baseline-v1'
KEYS = ['order_line_id', 'order_id', 'customer_id', 'product_id', 'store_id']
CATEGORY_TYPOS = {
    'electronicsaccessories': 'electronics_accessories', 'fashion_acessories': 'fashion_accessories',
    'food_beerages': 'food_beverages', 'furnture': 'furniture', 'garde_diy': 'garden_diy',
    'healthbeauty': 'health_beauty', 'home_ecor': 'home_decor', 'kitchendining': 'kitchen_dining',
    'office_upplies': 'office_supplies', 'sports_eisure': 'sports_leisure',
    'technlogy': 'technology', 'toys_ifts': 'toys_gifts',
}

def normalize(s):
    return s.astype('string').str.strip().str.lower().str.replace(r'[\s-]+', '_', regex=True)

def ratio(a, b):
    return a.div(b.where(b != 0)).replace([np.inf, -np.inf], np.nan)

def dimensions(data_dir):
    root = Path(data_dir)
    customer = pd.read_csv(root / 'customers.csv')
    customer['record_updated_at'] = pd.to_datetime(customer.record_updated_at, errors='raise')
    customer = customer.sort_values(['customer_id', 'record_updated_at'], kind='stable').drop_duplicates('customer_id', keep='last')
    # Supplied-snapshot policy for region; mutable loyalty and update time are excluded.
    customer = customer.drop(columns=['record_updated_at', 'loyalty_tier'])
    customer['region'] = normalize(customer.region)
    product = pd.read_csv(root / 'products.csv').drop(columns=['launch_date', 'review_count'])
    product['category'] = normalize(product.category).replace(CATEGORY_TYPOS)
    store = pd.read_csv(root / 'stores.csv').drop(columns=['seller_rating', 'manager_tenure_years'])
    store['region'] = normalize(store.region)
    calendar = pd.read_csv(root / 'calendar.csv')
    calendar['week_start'] = pd.to_datetime(calendar.week_start)
    result = []
    for frame, key, prefix in [(customer, 'customer_id', 'customer'), (product, 'product_id', 'product'), (store, 'store_id', 'store')]:
        if frame[key].duplicated().any():
            raise ValueError(f'Duplicate dimension key: {key}')
        result.append(frame.rename(columns={c: f'{prefix}_{c}' for c in frame if c != key}))
    result.append(calendar)
    return result

def build_features(raw, dims):
    df = raw.copy()
    if df.order_line_id.duplicated().any():
        raise ValueError('Duplicate order_line_id')
    original_ids = df.order_line_id.copy()
    ts = pd.to_datetime(df.order_ts, errors='raise')
    # Raw wall-clock timestamps: no unjustified timezone conversion.
    df['_order_time'] = ts.groupby(df.order_id).transform('max')
    df['_month'] = df['_order_time'].dt.strftime('%Y-%m')
    df['order_timestamp_span_hours'] = (df['_order_time'] - ts.groupby(df.order_id).transform('min')).dt.total_seconds() / 3600
    for dim, key in zip(dims[:3], ['customer_id', 'product_id', 'store_id']):
        if not df[key].isin(dim[key]).all():
            raise ValueError(f'Unmatched {key}')
        df = df.merge(dim, on=key, how='left', sort=False, validate='many_to_one')
    df['week_start'] = ts.dt.normalize() - pd.to_timedelta(ts.dt.dayofweek, unit='D')
    df = df.merge(dims[3], on='week_start', how='left', sort=False, validate='many_to_one', indicator='_calendar_join')
    if not df['_calendar_join'].eq('both').all():
        raise ValueError('Unmatched calendar week')
    if not df.order_line_id.reset_index(drop=True).equals(original_ids.reset_index(drop=True)):
        raise AssertionError('Join changed row order or count')
    df = df.drop(columns=['_calendar_join', 'week_start'])
    df['month_of_year'] = ts.dt.month
    df['day_of_week'] = ts.dt.dayofweek
    df['hour'] = ts.dt.hour
    df['weekend'] = (ts.dt.dayofweek >= 5).astype('int8')
    for field, out in [('customer_signup_date', 'customer_tenure_days'), ('store_opened_date', 'store_age_days')]:
        age = (ts - pd.to_datetime(df[field], errors='raise')).dt.total_seconds() / 86400
        df[out + '_invalid'] = (age < 0).astype('int8')
        df[out] = age.where(age >= 0)
        df = df.drop(columns=field)
    df['customer_store_region_match'] = (df.customer_region == df.store_region).astype('int8')
    df['line_gross_candidate'] = df.unit_price * df.quantity
    df['price_to_list_ratio'] = ratio(df.unit_price, df.product_list_price)
    df['amount_to_gross_ratio'] = ratio(df.order_value, df.line_gross_candidate)
    df['amount_gross_residual'] = df.order_value - df.line_gross_candidate
    df['freight_per_unit'] = ratio(df.freight_cost, df.quantity)
    df['freight_to_gross_ratio'] = ratio(df.freight_cost, df.line_gross_candidate)
    df['implied_discount_residual'] = (1 - df.price_to_list_ratio) - df.discount
    df['product_volume_cm3'] = df.product_length_cm * df.product_width_cm * df.product_height_cm
    df['product_density_kg_cm3'] = ratio(df.product_weight_kg, df.product_volume_cm3)
    df['line_weight_kg'] = df.product_weight_kg * df.quantity
    df['coupon_present'] = df.coupon_code.notna().astype('int8')
    for c in ['payment_method', 'ship_mode', 'discount', 'freight_cost', 'session_clicks', 'product_weight_kg', 'customer_age_band', 'store_size_sqm']:
        df[c + '_missing'] = df[c].isna().astype('int8')
    group = df.groupby('order_id', sort=False)
    df['basket_lines'] = group.order_line_id.transform('size')
    if not df.basket_lines.eq(df.items_in_order).all():
        raise ValueError('Incomplete orders or inconsistent items_in_order')
    df['basket_units'] = group.quantity.transform('sum')
    df['basket_gross_candidate'] = group.line_gross_candidate.transform('sum')
    df['basket_categories'] = group.product_category.transform('nunique')
    df['basket_products'] = group.product_id.transform('nunique')
    df['basket_stores'] = group.store_id.transform('nunique')
    df['basket_mean_discount'] = group.discount.transform('mean')
    df['line_basket_value_share'] = ratio(df.line_gross_candidate, df.basket_gross_candidate)
    df['line_basket_units_share'] = ratio(df.quantity, df.basket_units)
    df['relative_line_position'] = ratio(df.line_no, df.basket_lines)
    for c in ['unit_price', 'quantity', 'order_value', 'freight_cost', 'basket_gross_candidate', 'basket_units', 'product_volume_cm3']:
        df['log1p_' + c] = np.log1p(df[c].where(df[c] >= 0))
    df = df.drop(columns=['order_ts', 'coupon_code', 'store_channel'])
    df['banner_id'] = df.banner_id.astype('string')
    for c in df.select_dtypes(include='number'):
        if c != 'returned':
            df[c] = df[c].astype('float32')
    return df

def feature_columns(frame, include_promo=False):
    excluded = {'returned', 'order_line_id', 'order_id', '_order_time', '_month'}
    if not include_promo:
        excluded.add('promo_channel_code')
    return [c for c in frame if c not in excluded]

def encode_categories(train, other, columns):
    """Fit vocabulary only on fitting rows; unseen categories become missing."""
    x = train[columns].copy()
    z = other[columns].copy()
    levels = {}
    for c in columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            values = sorted(x[c].dropna().astype(str).unique().tolist())
            levels[c] = values
            x[c] = pd.Categorical(x[c], categories=values)
            z[c] = pd.Categorical(z[c].where(z[c].isin(values)), categories=values)
    return x, z, levels
