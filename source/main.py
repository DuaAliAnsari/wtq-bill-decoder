#!/usr/bin/env python3
"""Utility Bill Decoder - command line entry point.

  python main.py run --bills test/bills --level1 test/csv/level1.csv --level2 test/csv/level2.csv --out output
  python main.py bill path/to/KESC_0008.png --ask "How much of my bill is taxes?"
  python main.py make-templates --bills train/bills --out train/csv
  python main.py evaluate --pred output/level1.csv --gold tests/gold
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from billdecoder.config import Config, ConfigError
from billdecoder.csv_io import make_templates
from billdecoder.imaging import load_bill_image
from billdecoder.llm import LLMClient, LLMError

HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read electricity bills and explain them in plain English.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process a folder of bills and write level1.csv and level2.csv")
    run.add_argument("--bills", type=Path, required=True, help="folder with the bill images")
    run.add_argument("--level1", type=Path, required=True, help="level1.csv template")
    run.add_argument("--level2", type=Path, required=True, help="level2.csv template")
    run.add_argument("--out", type=Path, default=Path("output"), help="folder for the completed CSV files")
    run.add_argument("--only", nargs="*", help="process only these bill ids (for quick tests)")
    run.add_argument("--no-cache", action="store_true", help="ignore cached model responses")

    bill = sub.add_parser("bill", help="decode one bill image and optionally answer questions (demo mode)")
    bill.add_argument("image", type=Path)
    bill.add_argument("--ask", action="append", default=[], help="a customer question (repeatable)")
    bill.add_argument("--no-cache", action="store_true")

    tpl = sub.add_parser("make-templates", help="create level1/level2 CSV templates for a folder of bills")
    tpl.add_argument("--bills", type=Path, required=True)
    tpl.add_argument("--out", type=Path, required=True)
    tpl.add_argument("--questions", type=Path, default=HERE / "data" / "sample_questions.csv")

    pk = sub.add_parser("package", help="validate the results and build the submission ZIP")
    pk.add_argument("--out-dir", type=Path, default=Path("output"), help="folder holding level1.csv / level2.csv")
    pk.add_argument("--zip", type=Path, default=Path("submission.zip"))
    pk.add_argument("--demo", type=Path, default=None, help="optional demo video (.mp4/.mov/.webm)")
    pk.add_argument("--bills", type=int, default=10, help="expected number of test bills")
    pk.add_argument("--questions", type=int, default=12, help="expected number of questions per bill")

    ev = sub.add_parser("evaluate", help="score a level1.csv against gold JSON files")
    ev.add_argument("--pred", type=Path, required=True)
    ev.add_argument("--gold", type=Path, default=HERE / "tests" / "gold")
    return parser


def cmd_run(args) -> int:
    from billdecoder.pipeline import run
    config = Config.from_env()
    results = run(config, args.bills, args.level1, args.level2, args.out,
                  only=args.only, use_cache=not args.no_cache)
    failed = [r for r in results if r.error and not r.error.startswith("skipped (--only)")]
    print(f"\nWrote {args.out / 'level1.csv'} and {args.out / 'level2.csv'}")
    if failed:
        print(f"WARNING: {len(failed)} bill(s) had errors (fallback values written): "
              + ", ".join(r.bill_id for r in failed))
        return 1
    return 0


def cmd_bill(args) -> int:
    from billdecoder.answer import answer_questions
    from billdecoder.extract import extract_context, extract_level1
    config = Config.from_env()
    client = LLMClient(config)
    client.use_cache = not args.no_cache
    bill_id = args.image.stem
    image = load_bill_image(args.image)
    level1, warnings = extract_level1(client, image, bill_id)
    print("LEVEL 1 - extracted fields")
    print(json.dumps(level1, indent=2, ensure_ascii=False))
    for w in warnings:
        print(f"  note: {w}")
    if args.ask:
        ctx = extract_context(client, image, bill_id, level1.get("bill_month"))
        questions = [(f"Q{i}", q) for i, q in enumerate(args.ask, 1)]
        answers = answer_questions(client, image, bill_id, level1, ctx, questions)
        print("\nLEVEL 2 - answers")
        for qid, question in questions:
            print(f"\n{qid}. {question}\n   {answers[qid]}")
    return 0


def cmd_package(args) -> int:
    from billdecoder.package import build_zip, scan_for_secrets, validate_outputs
    problems = validate_outputs(args.out_dir, args.bills, args.questions)
    problems += scan_for_secrets(HERE)
    if not problems:
        problems += build_zip(HERE, args.out_dir, args.zip, args.demo)
    if problems:
        print("NOT READY TO SUBMIT:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: wrote {args.zip} ({args.zip.stat().st_size / 1048576:.2f} MB). "
          "Open it and check output/, source/ (and demo/) before uploading.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "bill":
            return cmd_bill(args)
        if args.command == "make-templates":
            p1, p2 = make_templates(args.bills, args.questions, args.out)
            print(f"Created {p1} and {p2}")
            return 0
        if args.command == "package":
            return cmd_package(args)
        if args.command == "evaluate":
            from billdecoder.evaluate import evaluate
            evaluate(args.pred, args.gold)
            return 0
    except (ConfigError, FileNotFoundError, LLMError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
