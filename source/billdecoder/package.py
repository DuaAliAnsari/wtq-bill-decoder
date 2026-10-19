"""Pre-submission checks and ZIP builder (submission allows only one upload, so verify everything first)."""
from __future__ import annotations

import csv
import json
import os
import re
import zipfile
from pathlib import Path

from .csv_io import LEVEL1_COLUMNS, LEVEL2_COLUMNS
from .schema import SCALAR_FIELDS

MAX_ZIP_BYTES = 15 * 1024 * 1024
EXCLUDE_DIRS = {"train", "test", "output", ".cache", "logs", "__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache"}
EXCLUDE_FILES = {".env"}
SECRET_PATTERNS = [re.compile(p) for p in (r"AIza[0-9A-Za-z_\-]{30,}", r"sk-ant-[0-9A-Za-z_\-]{20,}",
                                           r"sk-[0-9A-Za-z_\-]{30,}")]
DEMO_EXTENSIONS = (".mp4", ".mov", ".webm")


def validate_outputs(out_dir: Path, expected_bills: int = 10, expected_questions: int = 12) -> list[str]:
    """Return a list of problems (empty list = the CSVs look ready to submit)."""
    problems: list[str] = []
    p1, p2 = out_dir / "level1.csv", out_dir / "level2.csv"
    for path in (p1, p2):
        if not path.exists():
            problems.append(f"missing {path}")
    if problems:
        return problems

    with open(p1, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LEVEL1_COLUMNS:
            problems.append(f"level1.csv columns are {reader.fieldnames}, expected {LEVEL1_COLUMNS}")
        rows1 = list(reader)
    if len(rows1) != expected_bills:
        problems.append(f"level1.csv has {len(rows1)} rows, expected {expected_bills}")
    for row in rows1:
        try:
            data = json.loads(row.get("json") or "")
        except json.JSONDecodeError:
            problems.append(f"level1.csv: {row.get('bill_id')} has empty or invalid JSON")
            continue
        missing = [k for k in (*SCALAR_FIELDS, "charges", "total_charges", "taxes", "total_taxes",
                               "current_bill", "arrears", "payable_within_due_date",
                               "payable_after_due_date") if k not in data]
        if missing:
            problems.append(f"level1.csv: {row['bill_id']} missing fields {missing}")
        if all(data.get(k) is None for k in ("current_bill", "units_consumed", "due_date")):
            problems.append(f"level1.csv: {row['bill_id']} is an all-null fallback (extraction failed?)")

    with open(p2, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LEVEL2_COLUMNS:
            problems.append(f"level2.csv columns are {reader.fieldnames}, expected {LEVEL2_COLUMNS}")
        rows2 = list(reader)
    expected_rows = expected_bills * expected_questions
    if len(rows2) != expected_rows:
        problems.append(f"level2.csv has {len(rows2)} rows, expected {expected_rows}")
    empty = [f"{r['bill_id']}/{r['question_id']}" for r in rows2 if not (r.get("answer") or "").strip()]
    if empty:
        problems.append(f"level2.csv has {len(empty)} empty answers, e.g. {empty[:3]}")
    fallback = [f"{r['bill_id']}/{r['question_id']}" for r in rows2
                if "could not be generated automatically" in (r.get("answer") or "")]
    if fallback:
        problems.append(f"level2.csv has {len(fallback)} fallback answers (model call failed), e.g. {fallback[:3]}")
    return problems


def scan_for_secrets(source_dir: Path) -> list[str]:
    """Look for anything resembling an API key in the files that would be shipped."""
    findings: list[str] = []
    live_key = (os.getenv("API_KEY") or "").strip()
    for path in _source_files(source_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if live_key and len(live_key) > 8 and live_key in text:
            findings.append(f"{path} contains your real API key")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path} looks like it contains an API key")
                break
    return findings


def _source_files(source_dir: Path):
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if name not in EXCLUDE_FILES and not name.endswith(".pyc"):
                yield Path(root) / name


def build_zip(source_dir: Path, out_dir: Path, zip_path: Path, demo: Path | None = None) -> list[str]:
    """Create the submission ZIP; returns problems (the ZIP is only left in place if there are none)."""
    problems: list[str] = []
    for required in ("README.md", ".env.example", "requirements.txt"):
        if not (source_dir / required).exists():
            problems.append(f"source/{required} is missing")
    if demo is not None and demo.suffix.lower() not in DEMO_EXTENSIONS:
        problems.append(f"demo must be one of {DEMO_EXTENSIONS}")
    if problems:
        return problems

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("level1.csv", "level2.csv"):
            archive.write(out_dir / name, f"output/{name}")
        for path in sorted(_source_files(source_dir)):
            archive.write(path, f"source/{path.relative_to(source_dir).as_posix()}")
        if demo is not None:
            archive.write(demo, f"demo/demo{demo.suffix.lower()}")

    size = zip_path.stat().st_size
    if size > MAX_ZIP_BYTES:
        zip_path.unlink()
        problems.append(f"ZIP would be {size / 1048576:.1f} MB; the limit is 15 MB (shrink or trim the demo video)")
    return problems
