import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from billdecoder.csv_io import LEVEL1_COLUMNS, LEVEL2_COLUMNS, to_json_line
from billdecoder.package import build_zip, scan_for_secrets, validate_outputs
from billdecoder.schema import normalize_level1
from tests.fakes import MESSY_L1


def _write(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "output"
        self.out.mkdir()
        good = to_json_line(normalize_level1(MESSY_L1, "KESC_0008"))
        _write(self.out / "level1.csv", LEVEL1_COLUMNS, [{"bill_id": f"B{i}", "json": good} for i in range(2)])
        _write(self.out / "level2.csv", LEVEL2_COLUMNS,
               [{"bill_id": f"B{i}", "question_id": f"Q{q}", "question": "q?", "answer": "an answer"}
                for i in range(2) for q in range(1, 4)])
        self.src = self.tmp / "source"
        self.src.mkdir()
        for name in ("README.md", ".env.example", "requirements.txt", "main.py"):
            (self.src / name).write_text("x")

    def test_valid_outputs(self):
        self.assertEqual(validate_outputs(self.out, 2, 3), [])

    def test_detects_wrong_counts_and_empty_answers(self):
        self.assertTrue(validate_outputs(self.out, 10, 12))
        text = (self.out / "level2.csv").read_text().replace("an answer", "", 1)
        (self.out / "level2.csv").write_text(text)
        self.assertTrue(any("empty answers" in p for p in validate_outputs(self.out, 2, 3)))

    def test_detects_null_fallback_json(self):
        from billdecoder.schema import empty_level1
        _write(self.out / "level1.csv", LEVEL1_COLUMNS,
               [{"bill_id": "B0", "json": json.dumps(empty_level1())}, {"bill_id": "B1", "json": "{}"}])
        problems = validate_outputs(self.out, 2, 3)
        self.assertTrue(any("all-null" in p for p in problems))

    def test_secret_scan_and_zip_exclusions(self):
        (self.src / ".env").write_text("API_KEY=AIza" + "a" * 35)
        (self.src / ".cache").mkdir()
        (self.src / ".cache" / "x.txt").write_text("cached")
        self.assertEqual(scan_for_secrets(self.src), [])                    # .env itself is never shipped
        (self.src / "leak.py").write_text('KEY = "AIza' + "b" * 35 + '"')
        self.assertTrue(scan_for_secrets(self.src))
        (self.src / "leak.py").unlink()
        problems = build_zip(self.src, self.out, self.tmp / "s.zip")
        self.assertEqual(problems, [])
        names = zipfile.ZipFile(self.tmp / "s.zip").namelist()
        self.assertIn("output/level1.csv", names)
        self.assertIn("source/README.md", names)
        self.assertFalse(any(".env" == Path(n).name or ".cache" in n for n in names))


if __name__ == "__main__":
    unittest.main()
