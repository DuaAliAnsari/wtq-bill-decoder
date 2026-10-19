import unittest

from billdecoder.calc import UnsafeExpression, evaluate, run_calculations

VARS = {"bill": {"current_bill": 3430.24, "total_taxes": 569.62, "units_consumed": 151},
        "prior_units": [141, 138, 197, 239, 203, 179], "all_units": [141, 151]}


class CalcTests(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(evaluate("pct(bill['total_taxes'], bill['current_bill'])", VARS), 16.6058, 3)
        self.assertEqual(evaluate("sum(1 for u in prior_units if u > 200)", VARS), 2)

    def test_comprehension_can_use_outer_variables(self):
        self.assertEqual(evaluate("sum(1 for u in prior_units if u > bill['units_consumed'])", VARS), 4)

    def test_division_by_zero_is_reported_not_raised(self):
        out = run_calculations([{"name": "x", "expr": "1/0"}, {"name": "y", "expr": "2+2"}], VARS)
        self.assertIn("could not be computed", out["x"])
        self.assertEqual(out["y"], 4)

    def test_later_calculations_reuse_earlier_ones(self):
        out = run_calculations([{"name": "a", "expr": "10"}, {"name": "b", "expr": "a * 3"}], VARS)
        self.assertEqual(out["b"], 30)

    def test_unsafe_expressions_blocked(self):
        for bad in ["__import__('os')", "bill.keys()", "().__class__", "[0]*99999999999", "open('x')",
                    "lambda: 1", "bill['x'].__class__", "exec('1')", "2**99999999"]:
            with self.assertRaises((UnsafeExpression, SyntaxError), msg=bad):
                evaluate(bad, VARS)


if __name__ == "__main__":
    unittest.main()
