"""Score Level 1 output against gold JSON files using the competition's rules (exact text/dates, numbers within +/-1)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import NUMERIC_FIELDS, SCALAR_FIELDS

TEXT_FIELDS = ("provider", "tariff", "bill_month", "reading_date", "issue_date", "due_date")


def _num_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1


def _list_score(pred: list, gold: list) -> float:
    """Fraction of gold lines matched one-to-one by (type, amount within +/-1); extra predicted lines are penalised."""
    if not gold and not pred:
        return 1.0
    remaining = list(pred)
    matched = 0
    for g in gold:
        for i, p in enumerate(remaining):
            if p.get("type") == g.get("type") and _num_equal(p.get("amount"), g.get("amount")):
                matched += 1
                del remaining[i]
                break
    return matched / max(len(gold), len(pred))


def score_bill(pred: dict, gold: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key in TEXT_FIELDS:
        scores[key] = float(pred.get(key) == gold.get(key))
    for key in NUMERIC_FIELDS:
        scores[key] = float(_num_equal(pred.get(key), gold.get(key)))
    scores["charges"] = _list_score(pred.get("charges") or [], gold.get("charges") or [])
    scores["taxes"] = _list_score(pred.get("taxes") or [], gold.get("taxes") or [])
    return scores


def evaluate(pred_csv: Path, gold_dir: Path) -> float:
    gold = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in Path(gold_dir).glob("*.json")}
    with open(pred_csv, newline="", encoding="utf-8-sig") as handle:
        preds = {row["bill_id"]: json.loads(row["json"]) for row in csv.DictReader(handle) if row.get("json")}
    totals = []
    for bill_id, gold_json in sorted(gold.items()):
        if bill_id not in preds:
            print(f"{bill_id}: no prediction")
            continue
        scores = score_bill(preds[bill_id], gold_json)
        total = sum(scores.values()) / len(scores)
        totals.append(total)
        print(f"{bill_id}: {total:.1%}")
        for key, value in scores.items():
            if value < 1:
                print(f"    {key}: expected {gold_json.get(key)!r}, got {preds[bill_id].get(key)!r}")
    overall = sum(totals) / len(totals) if totals else 0.0
    print(f"Overall over {len(totals)} gold bill(s): {overall:.1%}")
    return overall


assert set(TEXT_FIELDS) <= set(SCALAR_FIELDS)
