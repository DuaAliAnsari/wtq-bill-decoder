"""Reading the competition's CSV templates and writing the completed result files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .imaging import IMAGE_EXTENSIONS

LEVEL1_COLUMNS = ["bill_id", "json"]
LEVEL2_COLUMNS = ["bill_id", "question_id", "question", "answer"]


class CsvFormatError(ValueError):
    pass


def _read(path: Path, required: list[str]) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise CsvFormatError(f"{path} is missing column(s): {missing}")
        return [dict(row) for row in reader]


def read_level1_template(path: Path) -> list[dict]:
    return _read(path, LEVEL1_COLUMNS)


def read_level2_template(path: Path) -> list[dict]:
    return _read(path, LEVEL2_COLUMNS)


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def to_json_line(data: dict) -> str:
    """One-line JSON; the csv module doubles the quotation marks when it quotes the cell."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def write_level1(path: Path, template_rows: list[dict], results: dict[str, dict]) -> None:
    rows = [{"bill_id": r["bill_id"], "json": to_json_line(results[r["bill_id"]])} for r in template_rows]
    _write(path, LEVEL1_COLUMNS, rows)


def write_level2(path: Path, template_rows: list[dict], answers: dict[tuple[str, str], str]) -> None:
    rows = [{"bill_id": r["bill_id"], "question_id": r["question_id"], "question": r["question"],
             "answer": answers[(r["bill_id"], r["question_id"])]} for r in template_rows]
    _write(path, LEVEL2_COLUMNS, rows)


def make_templates(bills_dir: Path, questions_csv: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build level1/level2 templates for a folder of bills (used for the training bills)."""
    bill_ids = sorted(p.stem for p in Path(bills_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not bill_ids:
        raise FileNotFoundError(f"No bill images found in {bills_dir}")
    with open(questions_csv, newline="", encoding="utf-8-sig") as handle:
        questions = list(csv.DictReader(handle))
    l1 = [{"bill_id": b, "json": ""} for b in bill_ids]
    l2 = [{"bill_id": b, "question_id": q["question_id"], "question": q["question"], "answer": ""}
          for b in bill_ids for q in questions]
    p1, p2 = Path(out_dir) / "level1.csv", Path(out_dir) / "level2.csv"
    _write(p1, LEVEL1_COLUMNS, l1)
    _write(p2, LEVEL2_COLUMNS, l2)
    return p1, p2
