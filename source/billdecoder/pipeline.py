"""End-to-end run: templates + bill images in, completed level1.csv / level2.csv out."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .answer import answer_questions, fallback_answer
from .config import Config
from .context import empty_context
from .csv_io import read_level1_template, read_level2_template, write_level1, write_level2
from .extract import extract_context, extract_level1
from .imaging import find_bill_image, load_bill_image
from .llm import LLMClient, QuotaExceededError
from .schema import empty_level1

log = logging.getLogger(__name__)


@dataclass
class BillResult:
    bill_id: str
    level1: dict
    answers: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    quota_exhausted: bool = False


def process_bill(client: LLMClient, image_path: Path, bill_id: str, questions: list[tuple[str, str]]) -> BillResult:
    """Three stages (Level 1, context, answers). A later stage failing never discards an earlier result."""
    image = load_bill_image(image_path)
    level1, warnings = extract_level1(client, image, bill_id)
    try:
        ctx = extract_context(client, image, bill_id, level1.get("bill_month"))
    except QuotaExceededError as exc:
        return BillResult(bill_id, level1, {}, warnings, error=str(exc), quota_exhausted=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("%s: context extraction failed; continuing without history", bill_id)
        warnings.append(f"context extraction failed: {type(exc).__name__}: {exc}")
        ctx = empty_context()
    try:
        answers = answer_questions(client, image, bill_id, level1, ctx, questions) if questions else {}
    except QuotaExceededError as exc:
        return BillResult(bill_id, level1, {}, warnings, error=str(exc), quota_exhausted=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("%s: answering failed; fallback answers will be used", bill_id)
        warnings.append(f"answering failed: {type(exc).__name__}: {exc}")
        answers = {}
    return BillResult(bill_id, level1, answers, warnings)


def run(config: Config, bills_dir: Path, level1_csv: Path, level2_csv: Path, out_dir: Path,
        only: list[str] | None = None, use_cache: bool = True, report_dir: Path = Path("logs")) -> list[BillResult]:
    started = time.time()
    l1_rows = read_level1_template(level1_csv)
    l2_rows = read_level2_template(level2_csv)

    bill_ids: list[str] = []
    for row in [*l1_rows, *l2_rows]:
        if row["bill_id"] not in bill_ids:
            bill_ids.append(row["bill_id"])
    questions_by_bill: dict[str, list[tuple[str, str]]] = {b: [] for b in bill_ids}
    for row in l2_rows:
        questions_by_bill[row["bill_id"]].append((row["question_id"], row["question"]))

    client = LLMClient(config)
    client.use_cache = use_cache
    results: dict[str, BillResult] = {}
    quota_hit = False

    for number, bill_id in enumerate(bill_ids, 1):
        if only and bill_id not in only:
            results[bill_id] = BillResult(bill_id, empty_level1(), error="skipped (--only)")
            continue
        log.info("[%d/%d] %s", number, len(bill_ids), bill_id)
        if quota_hit:
            results[bill_id] = BillResult(bill_id, empty_level1(), error="skipped: API quota exhausted")
            continue
        try:
            path = find_bill_image(bills_dir, bill_id)
            results[bill_id] = process_bill(client, path, bill_id, questions_by_bill[bill_id])
            quota_hit = results[bill_id].quota_exhausted
        except QuotaExceededError as exc:
            quota_hit = True
            log.error("%s", exc)
            results[bill_id] = BillResult(bill_id, empty_level1(), error=str(exc))
        except Exception as exc:  # noqa: BLE001 - one bad bill must never stop the whole run
            log.exception("%s failed", bill_id)
            results[bill_id] = BillResult(bill_id, empty_level1(), error=f"{type(exc).__name__}: {exc}")

    # Always write complete files: a failed bill gets null JSON / a plain fallback sentence, never an empty cell.
    answers: dict[tuple[str, str], str] = {}
    for row in l2_rows:
        res = results[row["bill_id"]]
        text = res.answers.get(row["question_id"]) or fallback_answer(res.level1, None, row["question"])
        answers[(row["bill_id"], row["question_id"])] = text
    write_level1(Path(out_dir) / "level1.csv", l1_rows, {b: r.level1 for b, r in results.items()})
    write_level2(Path(out_dir) / "level2.csv", l2_rows, answers)

    _write_report(Path(report_dir), results, client, time.time() - started)
    failed = [r.bill_id for r in results.values() if r.error and not r.error.startswith("skipped (--only)")]
    log.info("done in %.0fs: %d bills, %d model calls, %d cache hits, %d failed %s",
             time.time() - started, len(results), client.calls_made, client.cache_hits, len(failed), failed or "")
    return list(results.values())


def _write_report(report_dir: Path, results: dict[str, BillResult], client: LLMClient, seconds: float) -> None:
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {"model": f"{client.config.provider}/{client.config.model}", "seconds": round(seconds, 1),
                  "model_calls": client.calls_made, "cache_hits": client.cache_hits,
                  "bills": {b: {"warnings": r.warnings, "error": r.error} for b, r in results.items()}}
        (report_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("could not write run report: %s", exc)
