import unittest

from billdecoder.answer import allowed_numbers, ungrounded_numbers
from billdecoder.context import derive_facts, normalize_context
from billdecoder.llm import extract_json

L1_IESCO = {"provider": "IESCO", "bill_month": "2026-09", "units_consumed": 158, "current_bill": 3267,
            "total_charges": 2770, "total_taxes": 498, "taxes": [], "arrears": 0,
            "charges": [{"type": "energy", "amount": 6102}, {"type": "subsidy", "amount": -3332},
                        {"type": "fpa", "amount": 267}],
            "payable_within_due_date": 3535, "payable_after_due_date": 3812, "due_date": "2026-10-08",
            "issue_date": "2026-09-26", "reading_date": "2026-09-25"}

L1_NET = {"provider": "LESCO", "bill_month": "2026-07", "units_consumed": 1017, "current_bill": 20726,
          "total_charges": 17367.77, "total_taxes": 3358.23, "taxes": [], "arrears": -75496,
          "charges": [{"type": "energy", "amount": 17367.77}],
          "payable_within_due_date": -54770, "payable_after_due_date": None}
HIST = [("2025-07", 781), ("2025-08", 225), ("2025-09", -142), ("2025-10", -176), ("2025-11", -440),
        ("2025-12", -162), ("2026-01", 359), ("2026-02", -809), ("2026-03", -818), ("2026-04", -402),
        ("2026-05", -1955), ("2026-06", -152)]


class GroundingTests(unittest.TestCase):
    def test_wrong_subsidy_digit_is_caught(self):
        facts = derive_facts(L1_IESCO, normalize_context({}, "2026-09"))
        allowed = allowed_numbers(L1_IESCO, normalize_context({}, "2026-09"), facts, {}, [])
        self.assertEqual(ungrounded_numbers("a government subsidy of PKR -3,322", allowed), ["3,322"])
        self.assertEqual(ungrounded_numbers("a government subsidy of PKR -3,332", allowed), [])

    def test_net_metering_facts(self):
        ctx = normalize_context({"usage_history": [{"month": m, "units": u} for m, u in HIST]}, "2026-07")
        facts = derive_facts(L1_NET, ctx)
        self.assertEqual(facts["prior_months_over_200_units_count"], 3)
        self.assertEqual(facts["prior_months_over_200_units_list"],
                         ["2025-07: 781 units", "2025-08: 225 units", "2026-01: 359 units"])
        self.assertEqual(facts["months_over_200_units_including_current_count"], 4)
        self.assertTrue(facts["current_month_higher_than_every_prior_month"])
        self.assertEqual(facts["highest_month_including_current"], {"month": "2026-07", "units": 1017})
        self.assertEqual(facts["negative_unit_months_count"], 9)
        self.assertEqual(facts["prior_months_average_units_excluding_negative_months"], 455.0)
        self.assertIn("late_payment_note", facts)
        self.assertTrue(any("energy charges" in x for x in facts["line_items_for_explanation"]))
        self.assertTrue(any("combined total" in x for x in facts["line_items_for_explanation"]))
        self.assertEqual(facts["estimated_bill_if_20_percent_more_units_pkr"], 24871.2)

    def test_rounding_and_small_numbers_are_allowed(self):
        facts = derive_facts(L1_IESCO, normalize_context({}, "2026-09"))
        allowed = allowed_numbers(L1_IESCO, normalize_context({}, "2026-09"), facts, {}, [])
        self.assertEqual(ungrounded_numbers("Taxes are about 15.3% (PKR 498) over 12 days in 2026.", allowed), [])


class JsonRepairTests(unittest.TestCase):
    def test_truncated_and_trailing_comma(self):
        self.assertEqual(extract_json('{"a": [1, 2, {"b": "x"},'), {"a": [1, 2, {"b": "x"}]})
        self.assertEqual(extract_json('{"a": 1, "b": [1,2,],}'), {"a": 1, "b": [1, 2]})
        self.assertEqual(extract_json('```json\n{"a": "text that was cut'), {"a": "text that was cut"})

    def test_garbage_still_raises(self):
        with self.assertRaises(ValueError):
            extract_json("sorry, I cannot help")


if __name__ == "__main__":
    unittest.main()
