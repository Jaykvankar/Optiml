import unittest

import numpy as np
import pandas as pd

from optiml.teammate_pipeline import add_order_context, _blend


class TeammatePipelineTests(unittest.TestCase):
    def test_order_context_is_target_independent_and_row_aligned(self):
        frame = pd.DataFrame({
            "order_line_id": ["a", "b", "c"],
            "order_id": ["o1", "o1", "o2"],
            "product_id": ["p1", "p2", "p3"],
            "store_id": ["s1", "s1", "s2"],
            "unit_price": [10.0, 20.0, 5.0],
            "returned": [0, 1, 0],
        })
        context = add_order_context(frame)
        self.assertEqual(context.order_line_id.tolist(), ["a", "b", "c"])
        self.assertEqual(context.ctx_line_count.tolist(), [2.0, 2.0, 1.0])
        self.assertEqual(context.ctx_distinct_products.tolist(), [2.0, 2.0, 1.0])
        self.assertNotIn("returned", [c for c in context if c.startswith("ctx_")])

    def test_blend_stays_a_probability_and_uses_both_views(self):
        result = _blend(np.array([0.1, 0.9]), np.array([0.9, 0.1]), 0.5)
        np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-6)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())


if __name__ == "__main__":
    unittest.main()
