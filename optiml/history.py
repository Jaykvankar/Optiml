"""Monthly forward target rates; the forecast mapping is frozen at fit cutoff."""
import numpy as np
import pandas as pd

GROUPS = ['product_id', 'store_id', 'customer_id']

def add_history(train, other, alpha=100.0):
    """Training month never sees its own labels; initial prior is fixed at 0.5.

    Assumes labels from prior months are available. No return event dates exist;
    use a separate maturity-gap experiment before claiming production realism.
    """
    train, other = train.copy(), other.copy()
    months = train['_month']
    total_sum, total_count = 0.0, 0
    states = {g: pd.DataFrame(columns=['sum', 'count'], dtype=float) for g in GROUPS}
    for g in GROUPS:
        train[g + '_past_rate'] = np.float32(0.5)
        train[g + '_past_count'] = np.float32(0)
    for month in sorted(months.unique()):
        mask = months.eq(month)
        block = train.loc[mask]
        prior = total_sum / total_count if total_count else 0.5
        for g in GROUPS:
            stats = states[g]
            counts = block[g].map(stats['count']).fillna(0).astype(float)
            sums = block[g].map(stats['sum']).fillna(0).astype(float)
            train.loc[mask, g + '_past_count'] = counts.to_numpy(dtype='float32')
            train.loc[mask, g + '_past_rate'] = ((sums + alpha * prior) / (counts + alpha)).to_numpy(dtype='float32')
            update = block.groupby(g).returned.agg(['sum', 'count'])
            states[g] = stats.add(update, fill_value=0)
        total_sum += float(block.returned.sum())
        total_count += len(block)
    prior = total_sum / total_count
    for g, stats in states.items():
        counts = other[g].map(stats['count']).fillna(0)
        sums = other[g].map(stats['sum']).fillna(0)
        other[g + '_past_count'] = counts.astype('float32')
        other[g + '_past_rate'] = ((sums + alpha * prior) / (counts + alpha)).astype('float32')
    return train, other
