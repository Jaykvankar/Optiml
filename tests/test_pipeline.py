"""Behavior checks for temporal leakage, group isolation and submission alignment."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from optiml.features import encode_categories
from optiml.history import add_history
from optiml.train import split_frame, write_submission

class PipelineTests(unittest.TestCase):
    def history_frame(self):
        return pd.DataFrame({'_month': ['2024-01', '2024-01', '2024-02', '2024-03'],
                             'order_id': ['a', 'a', 'b', 'c'], 'product_id': ['p']*4,
                             'customer_id': ['c']*4, 'store_id': ['s']*4, 'returned': [0, 1, 1, 0]})

    def test_history_cannot_see_same_month_or_future_labels(self):
        frame = self.history_frame()
        future = frame.iloc[[3]].drop(columns='returned')
        original, forecast = add_history(frame, future)
        changed = frame.copy()
        changed.loc[2:, 'returned'] = 1 - changed.loc[2:, 'returned']
        modified, _ = add_history(changed, future)
        for group in ['product_id', 'store_id', 'customer_id']:
            np.testing.assert_array_equal(original.loc[:2, group + '_past_rate'], modified.loc[:2, group + '_past_rate'])
            self.assertEqual(original.loc[0, group + '_past_count'], 0)
            self.assertEqual(original.loc[1, group + '_past_count'], 0)
            self.assertEqual(original.loc[2, group + '_past_count'], 2)
            self.assertEqual(forecast.iloc[0][group + '_past_count'], 4)

    def test_forecast_labels_are_never_read(self):
        frame = self.history_frame()
        other = frame.copy()
        _, first = add_history(frame, other)
        other.returned = 1 - other.returned
        _, second = add_history(frame, other)
        pd.testing.assert_frame_equal(first.filter(like='_past_'), second.filter(like='_past_'))

    def test_unseen_categories_are_not_fitted_on_validation(self):
        x, z, levels = encode_categories(pd.DataFrame({'cat': ['a', 'b']}), pd.DataFrame({'cat': ['new', 'a']}), ['cat'])
        self.assertEqual(levels['cat'], ['a', 'b'])
        self.assertTrue(pd.isna(z.iloc[0]['cat']))
        self.assertEqual(list(x.cat.cat.categories), list(z.cat.cat.categories))

    def test_split_rejects_order_overlap(self):
        df = pd.DataFrame({'_order_time': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-08-01', '2024-09-01']),
                           'order_id': ['same', 'b', 'same', 'd'], 'returned': [0, 1, 0, 1]})
        with self.assertRaises(AssertionError):
            split_frame(df, '2024-07-01')

    def test_submission_reorders_by_id_and_rejects_invalid_scores(self):
        sample = pd.DataFrame({'order_line_id': ['b', 'a'], 'returned': [.5, .5]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'submission.csv'
            write_submission(sample, pd.Series(['a', 'b']), [.1, .9], path)
            saved = pd.read_csv(path)
            self.assertEqual(saved.order_line_id.tolist(), ['b', 'a'])
            np.testing.assert_allclose(saved.returned, [.9, .1])
            with self.assertRaises(ValueError):
                write_submission(sample, pd.Series(['a', 'b']), [.1, np.nan], path)
            with self.assertRaises(ValueError):
                write_submission(sample, pd.Series(['a', 'a']), [.1, .9], path)

if __name__ == '__main__':
    unittest.main()
