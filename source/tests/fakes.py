"""A scripted stand-in for the model so the whole pipeline can be tested offline."""
import json
from pathlib import Path

from billdecoder.config import Config
from billdecoder.llm import LLMClient

GOLD = json.loads((Path(__file__).parent / "gold" / "KESC_0008.json").read_text())

# What a model might plausibly return: messy strings, positive subsidy, wrong-case enums.
MESSY_L1 = {
    "provider": "K-Electric", "tariff": "A1-R", "sanctioned_load_kw": "3", "bill_month": "Apr-26",
    "reading_date": "03-Apr-26", "issue_date": "07-Apr-2026", "due_date": "21st Apr 2026",
    "previous_reading": "8,819", "current_reading": 8970, "units_consumed": "151",
    "charges": [{"type": "Fixed Charges", "amount": "900.00"}, {"type": "energy", "amount": "1,054.00"},
                {"type": "energy", "amount": 663.51}, {"type": "Uniform Quarterly Adjustment", "amount": 52.91},
                {"type": "FCA", "amount": 125.27}, {"type": "Additional Surcharge (PHL)", "amount": 64.93},
                {"type": "other", "amount": 0}],
    "total_charges": "2,860.62",
    "taxes": [{"type": "ED", "amount": 29.41}, {"type": "Sales Tax", "amount": 520.21},
              {"type": "MUCT (KMC)", "amount": 20}],
    "total_taxes": 569.62, "current_bill": "Rs. 3,430.24", "arrears": "Rs. -0.53",
    "payable_within_due_date": "Rs. 3,430", "payable_after_due_date": 3716,
}

USAGE = [141, 138, 197, 239, 203, 179, 173, 187, 124, 96, 88, 115, 151]
MONTHS = ["2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11",
          "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]
CONTEXT = {"usage_history": [{"month": m, "units": u} for m, u in zip(MONTHS, USAGE)],
           "split_percentages": {"energy_percent": 83.39, "tax_percent": 16.61}, "message_board": None}


class FakeClient(LLMClient):
    """Overrides the network layer only; caching, parsing and retries run for real."""
    def __init__(self, tmp_cache, fail_for=()):
        cfg = Config("google", "gemini-3.5-flash-lite", "key", cache_dir=Path(tmp_cache), requests_per_minute=6000)
        super().__init__(cfg)
        self.sent = []
        self.fail_for = fail_for

    def _send(self, system, prompt, images, max_tokens):
        self.sent.append(system[:40])
        if "data-extraction engine" in system:
            return "```json\n" + json.dumps(MESSY_L1) + "\n```"          # fenced, as models often do
        if "transcribe supporting details" in system:
            return json.dumps(CONTEXT)
        if "planning step" in system:
            return json.dumps({"plans": [
                {"question_id": "Q1", "interpretation": "tax share of current bill", "calculations": [
                    {"name": "tax_pct", "expr": "round(pct(bill['total_taxes'], bill['current_bill']), 1)"}]},
                {"question_id": "Q2", "interpretation": "previous months above 200", "calculations": [
                    {"name": "count", "expr": "sum(1 for u in prior_units if u > 200)"}]}]})
        if "friendly, precise assistant" in system:
            if '"Q2"' in prompt:  # batch call (not the single-question fallback)
                assert '"tax_pct":16.6' in prompt and '"count":2' in prompt, "computed results must reach the answer step"
            return json.dumps({"answers": [
                {"question_id": "Q1", "answer": "Taxes are PKR 569.62, about 16.6% of the bill."},
                {"question_id": "Q2", "answer": "Two previous months went above 200 units."}]})
        raise AssertionError("unexpected prompt")
