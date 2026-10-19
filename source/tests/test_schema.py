import unittest

from billdecoder.schema import (empty_level1, normalize_level1, to_date, to_month, to_number, validate_level1)
from tests.fakes import GOLD, MESSY_L1


class NumberDateTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(to_number("Rs. 19,964"), 19964)
        self.assertEqual(to_number("38209CR"), -38209)
        self.assertEqual(to_number("54770CR"), -54770)
        self.assertEqual(to_number("Rs. -0.53"), -0.53)
        self.assertEqual(to_number("39411 / 2"), 39411)
        self.assertEqual(to_number("(125.50)"), -125.5)
        self.assertIsNone(to_number("NOT TO BE PAID"))
        self.assertIsNone(to_number(""))
        self.assertIsNone(to_number(None))
        self.assertIsInstance(to_number(3430.0), int)

    def test_dates(self):
        self.assertEqual(to_date("22nd Sep 2026"), "2026-09-22")
        self.assertEqual(to_date("08 OCT 26"), "2026-10-08")
        self.assertEqual(to_date("21-Apr-2026"), "2026-04-21")
        self.assertEqual(to_date("2026-04-21"), "2026-04-21")
        self.assertIsNone(to_date("32 Apr 2026"))
        self.assertEqual(to_month("MAR 2026"), "2026-03")
        self.assertEqual(to_month("Apr-26"), "2026-04")
        self.assertEqual(to_month("2026-04-07"), "2026-04")


class NormalizeTests(unittest.TestCase):
    def test_messy_model_output_becomes_gold(self):
        self.assertEqual(normalize_level1(MESSY_L1, "KESC_0008"), GOLD)

    def test_key_order_and_all_fields_present(self):
        out = normalize_level1({"provider": "LESCO"}, "LESCO_0005")
        self.assertEqual(list(out), list(empty_level1()))
        self.assertEqual(out["charges"], [])

    def test_zero_and_blank_lines_dropped_duplicates_kept(self):
        out = normalize_level1({"charges": [{"type": "energy", "amount": 0}, {"type": "energy", "amount": 5},
                                            {"type": "energy", "amount": 7}, {"type": "x", "amount": None}]})
        self.assertEqual(out["charges"], [{"type": "energy", "amount": 5}, {"type": "energy", "amount": 7}])

    def test_subsidy_forced_negative(self):
        out = normalize_level1({"charges": [{"type": "subsidy", "amount": 3332}]})
        self.assertEqual(out["charges"], [{"type": "subsidy", "amount": -3332}])

    def test_provider_falls_back_to_bill_id(self):
        self.assertEqual(normalize_level1({}, "KESC_0002")["provider"], "KE")
        self.assertEqual(normalize_level1({"provider": "???"}, "IESCO_0001")["provider"], "IESCO")

    def test_garbage_input(self):
        self.assertEqual(normalize_level1("not a dict"), empty_level1())

    def test_validation(self):
        problems, warnings = validate_level1(normalize_level1(MESSY_L1, "KESC_0008"))
        self.assertEqual(problems, [])
        problems, _ = validate_level1(empty_level1())
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
