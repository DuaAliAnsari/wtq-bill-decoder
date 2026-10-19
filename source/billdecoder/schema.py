"""Level 1 schema: deterministic clean-up of model output so it always matches the required format.

The model reads the bill; this module (not the model) enforces number/date formats, enums,
sign rules and field order. That keeps the output identical from run to run.
"""
from __future__ import annotations

import re
from typing import Any

PROVIDERS = ("KE", "LESCO", "IESCO")
CHARGE_TYPES = ("energy", "fixed", "fpa", "quarterly_adjustment", "surcharge", "meter_rent", "subsidy", "other")
TAX_TYPES = ("gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax")

SCALAR_FIELDS = (
    "provider", "tariff", "sanctioned_load_kw", "bill_month", "reading_date", "issue_date", "due_date",
    "previous_reading", "current_reading", "units_consumed",
)
TRAILING_FIELDS = ("total_charges", "taxes", "total_taxes", "current_bill", "arrears",
                   "payable_within_due_date", "payable_after_due_date")
NUMERIC_FIELDS = ("sanctioned_load_kw", "previous_reading", "current_reading", "units_consumed", "total_charges",
                  "total_taxes", "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date")

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTHS.update({"sept": 9})


# ---------------------------------------------------------------------------- numbers
def to_number(value: Any) -> float | int | None:
    """'Rs. 19,964' -> 19964, '38209CR' -> -38209, '(12.5)' -> -12.5, '' / 'N/A' -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _tidy(float(value))
    text = str(value).strip()
    if not text:
        return None
    negative = False
    upper = text.upper()
    if re.search(r"\bCR\b|CR$|\bCREDIT\b", upper):
        negative = True
    if text.startswith("(") and text.endswith(")"):
        negative = True
    if "/" in text:  # printed oddities such as "39411 / 2": keep the leading figure
        text = text.split("/")[0]
    match = re.search(r"-?\s*\d[\d,]*\.?\d*|-?\s*\.\d+", text.replace("\u2212", "-"))
    if not match:
        return None
    raw = match.group(0).replace(",", "").replace(" ", "")
    try:
        number = float(raw)
    except ValueError:
        return None
    if negative and number > 0:
        number = -number
    return _tidy(number)


def _tidy(number: float) -> float | int:
    number = round(number, 2)
    return int(number) if number == int(number) else number


# ---------------------------------------------------------------------------- dates
def to_date(value: Any) -> str | None:
    """Return YYYY-MM-DD for '22nd Sep 2026', '08 OCT 26', '21-Apr-2026', '2026-04-21'."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        y, mo, d = map(int, m.groups())
        return _fmt_date(y, mo, d)
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*([A-Za-z]{3,9})[\s\-/.,]*(\d{2,4})", text)
    if m:
        d, mon, y = m.groups()
        month = _MONTHS.get(mon[:4].lower()) or _MONTHS.get(mon[:3].lower())
        if month:
            return _fmt_date(_year(y), month, int(d))
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", text)
    if m:  # day-first, as printed on Pakistani bills
        d, mo, y = m.groups()
        return _fmt_date(_year(y), int(mo), int(d))
    return None


def to_month(value: Any) -> str | None:
    """Return YYYY-MM for 'MAR 2026', 'Apr-26', '2026-04' or a full date."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", text)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    m = re.search(r"([A-Za-z]{3,9})[\s\-/.,]*(\d{2,4})", text)
    if m:
        month = _MONTHS.get(m.group(1)[:4].lower()) or _MONTHS.get(m.group(1)[:3].lower())
        if month:
            return f"{_year(m.group(2)):04d}-{month:02d}"
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{2,4})", text)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{_year(m.group(2)):04d}-{int(m.group(1)):02d}"
    return None


def _year(raw: str) -> int:
    y = int(raw)
    return y + 2000 if y < 100 else y


def _fmt_date(y: int, m: int, d: int) -> str | None:
    if not (1 <= m <= 12 and 1 <= d <= 31 and 2000 <= y <= 2100):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


# ---------------------------------------------------------------------------- categories
def normalize_provider(value: Any, fallback: str | None = None) -> str | None:
    text = re.sub(r"[^a-z]", "", str(value or "").lower())
    if text in ("ke", "kesc", "kelectric", "karachielectric") or "kelectric" in text:
        return "KE"
    if "lesco" in text or "lahore" in text:
        return "LESCO"
    if "iesco" in text or "islamabad" in text or "iesc" in text:
        return "IESCO"
    return normalize_provider(fallback) if fallback and fallback != value else None


def provider_from_bill_id(bill_id: str) -> str | None:
    prefix = bill_id.split("_")[0].upper()
    return {"KESC": "KE", "KE": "KE", "LESCO": "LESCO", "IESCO": "IESCO"}.get(prefix)


def _charge_type(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in CHARGE_TYPES:
        return text
    t = text
    if "subsid" in t or "relief" in t:
        return "subsidy"
    if "quarter" in t or t in ("qta", "uqa"):
        return "quarterly_adjustment"
    if "fpa" in t or "fca" in t or "fuel" in t:
        return "fpa"
    if "meter" in t or "service_rent" in t:
        return "meter_rent"
    if "surcharge" in t:
        return "surcharge"
    if "fixed" in t:
        return "fixed"
    if "energy" in t or "variable" in t or "electricity" in t:
        return "energy"
    return "other"


def _tax_type(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in TAX_TYPES:
        return text
    t = text
    if "gst" in t or "sales" in t:
        return "gst"
    if "duty" in t or t == "ed":
        return "electricity_duty"
    if "income" in t:
        return "income_tax"
    if "muct" in t or "municipal" in t or "kmc" in t or "local" in t:
        return "municipal_tax"
    return "other_tax"


def _clean_lines(items: Any, type_fn, subsidy_negative: bool) -> list[dict]:
    """Keep one entry per printed line; drop blanks and zeros; never merge repeated types."""
    out: list[dict] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        amount = to_number(item.get("amount"))
        if amount is None or amount == 0:
            continue
        kind = type_fn(item.get("type"))
        if subsidy_negative and kind == "subsidy" and amount > 0:
            amount = -amount
        out.append({"type": kind, "amount": amount})
    return out


# ---------------------------------------------------------------------------- main entry points
def empty_level1() -> dict:
    """Every required field, all null / empty. Used as a safe fallback."""
    result: dict[str, Any] = {k: None for k in SCALAR_FIELDS}
    result["charges"] = []
    result["total_charges"] = None
    result["taxes"] = []
    for key in TRAILING_FIELDS[2:]:
        result[key] = None
    return _ordered(result)


def _ordered(data: dict) -> dict:
    order = list(SCALAR_FIELDS) + ["charges", "total_charges", "taxes", "total_taxes", "current_bill",
                                   "arrears", "payable_within_due_date", "payable_after_due_date"]
    return {k: data.get(k, [] if k in ("charges", "taxes") else None) for k in order}


def normalize_level1(raw: Any, bill_id: str | None = None) -> dict:
    """Coerce a model's (possibly messy) answer into the exact Level 1 format."""
    if not isinstance(raw, dict):
        return empty_level1()
    data: dict[str, Any] = {}
    data["provider"] = normalize_provider(raw.get("provider"), provider_from_bill_id(bill_id or ""))
    tariff = raw.get("tariff")
    data["tariff"] = str(tariff).strip() if tariff not in (None, "") else None
    data["bill_month"] = to_month(raw.get("bill_month"))
    for key in ("reading_date", "issue_date", "due_date"):
        data[key] = to_date(raw.get(key))
    for key in NUMERIC_FIELDS:
        data[key] = to_number(raw.get(key))
    data["charges"] = _clean_lines(raw.get("charges"), _charge_type, subsidy_negative=True)
    data["taxes"] = _clean_lines(raw.get("taxes"), _tax_type, subsidy_negative=False)
    return _ordered(data)


def validate_level1(data: dict) -> tuple[list[str], list[str]]:
    """Return (problems, warnings). Problems trigger one repair attempt; warnings are only logged."""
    problems: list[str] = []
    warnings: list[str] = []
    for key in ("provider", "bill_month", "due_date"):
        if data.get(key) is None:
            problems.append(f"{key} is missing or not in the required format")
    if data.get("current_bill") is None and data.get("payable_within_due_date") is None:
        problems.append("neither current_bill nor payable_within_due_date was read")
    if data.get("units_consumed") is None:
        warnings.append("units_consumed is null")
    for lines_key, total_key in (("charges", "total_charges"), ("taxes", "total_taxes")):
        total = data.get(total_key)
        lines = data.get(lines_key) or []
        if total is not None and lines:
            diff = abs(sum(x["amount"] for x in lines) - total)
            if diff > 1.5:
                warnings.append(f"sum of {lines_key} differs from {total_key} by {diff:.2f}")
    prev, cur, units = data.get("previous_reading"), data.get("current_reading"), data.get("units_consumed")
    if None not in (prev, cur, units) and abs((cur - prev) - units) > 1.5:
        warnings.append("readings difference does not equal units_consumed (multi-register or multiplier bill?)")
    return problems, warnings
