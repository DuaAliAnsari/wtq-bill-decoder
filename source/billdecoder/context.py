"""Supporting bill details (usage history, billing history, notes) and figures derived from them."""
from __future__ import annotations

from datetime import date
from typing import Any

from .calc import mean, median, pct, safe_div
from .schema import to_date, to_month, to_number


def empty_context() -> dict:
    return {"usage_history": [], "billing_history": [], "calculation_lines": [], "meter_readings": [],
            "split_percentages": {"energy_percent": None, "tax_percent": None},
            "late_payment_amounts": [], "message_board": None, "other_notes": []}


def _dicts(value: Any) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def normalize_context(raw: Any, bill_month: str | None) -> dict:
    """Clean the model's transcription: numbers as numbers, months as YYYY-MM, oldest-first, no duplicates."""
    if not isinstance(raw, dict):
        return empty_context()
    ctx = empty_context()

    seen: dict[str, dict] = {}
    for row in _dicts(raw.get("usage_history")):
        month = to_month(row.get("month"))
        if month and month not in seen:
            seen[month] = {"month": month, "units": to_number(row.get("units")),
                           "is_current": month == bill_month}
    ctx["usage_history"] = [seen[m] for m in sorted(seen)]

    seen_b: dict[str, dict] = {}
    for row in _dicts(raw.get("billing_history")):
        month = to_month(row.get("month"))
        if month and month not in seen_b:
            seen_b[month] = {"month": month, "billed_amount": to_number(row.get("billed_amount")),
                             "payment_amount": to_number(row.get("payment_amount")),
                             "payment_date": to_date(row.get("payment_date"))}
    ctx["billing_history"] = [seen_b[m] for m in sorted(seen_b)]

    ctx["calculation_lines"] = [
        {"label": str(r.get("label") or "").strip(), "units": to_number(r.get("units")),
         "rate": to_number(r.get("rate")), "amount": to_number(r.get("amount"))}
        for r in _dicts(raw.get("calculation_lines")) if r.get("label")]
    ctx["meter_readings"] = [
        {"register": str(r.get("register") or "").strip(), "previous": to_number(r.get("previous")),
         "current": to_number(r.get("current")), "units": to_number(r.get("units"))}
        for r in _dicts(raw.get("meter_readings"))]
    split = raw.get("split_percentages") if isinstance(raw.get("split_percentages"), dict) else {}
    ctx["split_percentages"] = {"energy_percent": to_number(split.get("energy_percent")),
                                "tax_percent": to_number(split.get("tax_percent"))}
    ctx["late_payment_amounts"] = [
        {"label": str(r.get("label") or "").strip(), "amount": to_number(r.get("amount"))}
        for r in _dicts(raw.get("late_payment_amounts"))]
    message = raw.get("message_board")
    ctx["message_board"] = str(message).strip() if message not in (None, "") else None
    notes = raw.get("other_notes")
    ctx["other_notes"] = [str(n).strip() for n in notes if str(n).strip()] if isinstance(notes, list) else []
    return ctx


_CHARGE_WORDS = {
    "energy": "energy charges (the cost of the units you used)",
    "fixed": "fixed charges (a standing charge for your connection)",
    "fpa": "fuel price/cost adjustment (changes in the cost of fuel used to generate power)",
    "quarterly_adjustment": "quarterly tariff adjustment",
    "surcharge": "surcharge",
    "meter_rent": "meter/service rent",
    "subsidy": "government subsidy (reduces your bill)",
    "other": "other charge",
}
_TAX_WORDS = {"gst": "sales tax (GST)", "electricity_duty": "electricity duty", "income_tax": "income tax",
              "municipal_tax": "municipal tax", "other_tax": "other tax"}


def _pkr(value) -> str:
    return f"PKR {value:,.2f}"


def _line_items(bill: dict) -> list[str]:
    """Plain-language list of every printed charge/tax line, built in code so none can be skipped."""
    items = [f"{_CHARGE_WORDS.get(c['type'], 'charge')}: {_pkr(c['amount'])}" for c in bill.get("charges") or []]
    if bill.get("total_charges") is not None:
        items.append(f"total charges (before taxes): {_pkr(bill['total_charges'])}")
    if bill.get("taxes"):
        items += [f"{_TAX_WORDS.get(t['type'], 'tax')}: {_pkr(t['amount'])}" for t in bill["taxes"]]
    if bill.get("total_taxes") is not None:
        label = "total taxes" if bill.get("taxes") else "taxes (only a combined total is printed)"
        items.append(f"{label}: {_pkr(bill['total_taxes'])}")
    if bill.get("current_bill") is not None:
        items.append(f"current bill (charges plus taxes): {_pkr(bill['current_bill'])}")
    if bill.get("arrears") is not None:
        kind = "credit carried forward" if bill["arrears"] < 0 else "arrears (unpaid previous dues)"
        items.append(f"{kind}: {_pkr(bill['arrears'])}")
    return items


def _r(value, digits=2):
    return None if value is None else round(value, digits)


def derive_facts(bill: dict, ctx: dict) -> dict:
    """Exact figures computed in code from the extracted data. Nothing here comes from the model."""
    facts: dict[str, Any] = {}
    current_bill, taxes = bill.get("current_bill"), bill.get("total_taxes")
    units = bill.get("units_consumed")

    facts["tax_share_of_current_bill_percent"] = _r(pct(taxes, current_bill))
    facts["energy_charges_share_of_current_bill_percent"] = _r(pct(bill.get("total_charges"), current_bill))
    facts["average_cost_per_unit_all_in_pkr"] = _r(safe_div(current_bill, units)) if units else None
    facts["average_charge_per_unit_before_tax_pkr"] = _r(safe_div(bill.get("total_charges"), units)) if units else None
    within, after = bill.get("payable_within_due_date"), bill.get("payable_after_due_date")
    if within is not None and after is not None:
        facts["late_payment_extra_pkr"] = _r(after - within)
        facts["late_payment_extra_percent"] = _r(pct(after - within, within))
    if bill.get("arrears") is not None and current_bill is not None:
        facts["current_bill_plus_arrears_pkr"] = _r(current_bill + bill["arrears"])
    due, issued, read = bill.get("due_date"), bill.get("issue_date"), bill.get("reading_date")
    try:
        if due and issued:
            facts["days_from_issue_to_due_date"] = (date.fromisoformat(due) - date.fromisoformat(issued)).days
        if due and read:
            facts["days_from_reading_to_due_date"] = (date.fromisoformat(due) - date.fromisoformat(read)).days
    except ValueError:
        pass
    if bill.get("charges"):
        facts["charges_by_line"] = [f'{c["type"]}: {c["amount"]}' for c in bill["charges"]]

    facts["line_items_for_explanation"] = _line_items(bill)
    if units and current_bill is not None:
        facts["next_month_units_if_20_percent_more"] = _r(units * 1.2)
        facts["estimated_bill_if_20_percent_more_units_pkr"] = _r(current_bill * 1.2)
        facts["estimate_note"] = ("estimate = current bill x 1.20, i.e. 20% more units at the same average cost per unit "
                                  "(current bill / units); arrears and credits are not included")
    if after is None:
        facts["late_payment_note"] = ("No payable-after-due-date amount is printed on this bill "
                                      "(it may say the bill must not be paid after the due date).")

    prior = [u["units"] for u in ctx["usage_history"] if not u["is_current"] and u["units"] is not None]
    allu = [u["units"] for u in ctx["usage_history"] if u["units"] is not None]
    facts["prior_months_count"] = len(prior)
    if prior:
        by_month = [(u["units"], u["month"]) for u in ctx["usage_history"]
                    if not u["is_current"] and u["units"] is not None]
        facts["prior_months_average_units"] = _r(mean(prior))
        facts["prior_months_median_units"] = _r(median(prior))
        facts["prior_months_highest"] = {"units": max(by_month)[0], "month": max(by_month)[1]}
        facts["prior_months_lowest"] = {"units": min(by_month)[0], "month": min(by_month)[1]}
        facts["last_three_months_average_units"] = _r(mean(prior[-3:]))
        if units is not None:
            avg = mean(prior)
            facts["current_units_vs_prior_average_difference"] = _r(units - avg)
            facts["current_units_vs_prior_average_percent"] = _r(pct(units - avg, avg))
            facts["current_units_vs_previous_month_difference"] = _r(units - prior[-1])
    if allu:
        facts["all_listed_months_average_units"] = _r(mean(allu))

    prior_rows = [u for u in ctx["usage_history"] if not u["is_current"] and u["units"] is not None]
    if prior_rows:
        over = [u for u in prior_rows if u["units"] > 200]
        facts["prior_months_over_200_units_count"] = len(over)
        facts["prior_months_over_200_units_list"] = [f'{u["month"]}: {u["units"]} units' for u in over]
        if units is not None:
            facts["current_month_over_200_units"] = units > 200
            facts["months_over_200_units_including_current_count"] = len(over) + (1 if units > 200 else 0)
            peak = max(prior_rows, key=lambda u: u["units"])
            facts["current_month_higher_than_every_prior_month"] = units > peak["units"]
            facts["highest_month_including_current"] = (
                {"month": bill.get("bill_month"), "units": units} if units > peak["units"]
                else {"month": peak["month"], "units": peak["units"]})
        negatives = [u for u in prior_rows if u["units"] < 0]
        if negatives:
            non_neg = [u["units"] for u in prior_rows if u["units"] >= 0]
            facts["negative_unit_months_count"] = len(negatives)
            facts["negative_unit_months_note"] = ("Negative units are surplus electricity exported to the grid "
                                                  "(net metering); they are not usage.")
            if non_neg:
                facts["prior_months_average_units_excluding_negative_months"] = _r(mean(non_neg))

    billed = [b["billed_amount"] for b in ctx["billing_history"] if b["billed_amount"] is not None]
    if billed:
        facts["billing_history_average_billed_pkr"] = _r(mean(billed))
        facts["billing_history_highest_billed_pkr"] = max(billed)
        facts["billing_history_lowest_billed_pkr"] = min(billed)
        if current_bill is not None:
            facts["current_bill_vs_history_average_pkr"] = _r(current_bill - mean(billed))
    return {k: v for k, v in facts.items() if v is not None}