import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from billdecoder import pipeline
from billdecoder.config import Config
from billdecoder.csv_io import read_level1_template, to_json_line
from billdecoder.evaluate import evaluate
from billdecoder.llm import LLMClient, extract_json
from tests.fakes import GOLD, FakeClient


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "bills").mkdir()
        for name in ("KESC_0008", "LESCO_0001"):
            Image.new("RGB", (900, 1200), "white").save(self.tmp / "bills" / f"{name}.png")
        (self.tmp / "level1.csv").write_text("bill_id,json\nKESC_0008,\nLESCO_0001,\nIESCO_0099,\n")
        (self.tmp / "level2.csv").write_text(
            "bill_id,question_id,question,answer\n"
            "KESC_0008,Q1,How much is tax?,\nKESC_0008,Q2,Months over 200 units?,\nKESC_0008,Q3,Any other thing?,\n"
            "IESCO_0099,Q1,How much is tax?,\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, out, client=None):
        client = client or FakeClient(self.tmp / "cache")
        with mock.patch.object(pipeline, "LLMClient", lambda cfg: client):
            return pipeline.run(client.config, self.tmp / "bills", self.tmp / "level1.csv",
                                self.tmp / "level2.csv", self.tmp / out, report_dir=self.tmp / "logs"), client

    def test_end_to_end_files_are_complete_and_valid(self):
        _, _ = self._run("out")
        rows = list(csv.DictReader(open(self.tmp / "out" / "level1.csv", newline="", encoding="utf-8")))
        self.assertEqual([r["bill_id"] for r in rows], ["KESC_0008", "LESCO_0001", "IESCO_0099"])  # order kept
        self.assertEqual(json.loads(rows[0]["json"]), GOLD)
        for row in rows:
            data = json.loads(row["json"])                                  # every cell is valid JSON
            self.assertIn("payable_after_due_date", data)
        raw = (self.tmp / "out" / "level1.csv").read_text(encoding="utf-8")
        self.assertIn('{""provider"":""KE""', raw)                          # quotes doubled, one line
        self.assertEqual(len(raw.strip().splitlines()), 4)

        l2 = list(csv.DictReader(open(self.tmp / "out" / "level2.csv", newline="", encoding="utf-8")))
        self.assertEqual(len(l2), 4)
        self.assertTrue(all(r["answer"].strip() for r in l2))               # no empty answers, ever
        self.assertIn("16.6%", l2[0]["answer"])
        self.assertEqual([r["question"] for r in l2][:2], ["How much is tax?", "Months over 200 units?"])

    def test_missing_image_and_missing_answer_get_safe_fallbacks(self):
        results, _ = self._run("out")
        by_id = {r.bill_id: r for r in results}
        self.assertIn("FileNotFoundError", by_id["IESCO_0099"].error)
        l1 = {r["bill_id"]: json.loads(r["json"]) for r in csv.DictReader(open(self.tmp / "out" / "level1.csv"))}
        self.assertIsNone(l1["IESCO_0099"]["current_bill"])                 # null skeleton, not a crash
        l2 = list(csv.DictReader(open(self.tmp / "out" / "level2.csv")))
        q3 = [r for r in l2 if r["question_id"] == "Q3"][0]                 # model skipped Q3 -> filled in
        self.assertTrue(q3["answer"].strip())

    def test_second_run_is_identical_and_uses_cache(self):
        _, first = self._run("run1")
        calls_first = first.calls_made
        _, second = self._run("run2", FakeClient(self.tmp / "cache"))
        self.assertEqual(second.calls_made, 0)
        self.assertGreater(second.cache_hits, 0)
        for name in ("level1.csv", "level2.csv"):
            self.assertEqual((self.tmp / "run1" / name).read_bytes(), (self.tmp / "run2" / name).read_bytes())
        self.assertGreater(calls_first, 0)

    def test_scoring_against_gold(self):
        self._run("out")
        gold_dir = self.tmp / "gold"
        gold_dir.mkdir()
        (gold_dir / "KESC_0008.json").write_text(json.dumps(GOLD))
        self.assertEqual(evaluate(self.tmp / "out" / "level1.csv", gold_dir), 1.0)


class ClientTests(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps'), {"a": [1, 2]})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def _client(self, provider, model):
        return LLMClient(Config(provider, model, "SECRET", cache_dir=Path(tempfile.mkdtemp())))

    def _capture(self, client, reply):
        seen = {}
        def post(url, headers=None, json=None, timeout=None):
            seen.update(url=url, headers=headers, body=json)
            return mock.Mock(status_code=200, headers={}, json=lambda: reply)
        client._session.post = post
        return seen

    def test_google_request_shape(self):
        from billdecoder.imaging import BillImage
        c = self._client("google", "gemini-3.5-flash-lite")
        seen = self._capture(c, {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]})
        self.assertEqual(c.generate_json("sys", "prompt", [BillImage(b"abc")]), {"ok": True})
        self.assertTrue(seen["url"].endswith("/models/gemini-3.5-flash-lite:generateContent"))
        self.assertEqual(seen["headers"]["x-goog-api-key"], "SECRET")
        self.assertEqual(seen["body"]["generationConfig"]["temperature"], 0)
        self.assertEqual(seen["body"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertIn("inline_data", seen["body"]["contents"][0]["parts"][1])

    def test_anthropic_and_openai_request_shapes(self):
        from billdecoder.imaging import BillImage
        c = self._client("anthropic", "claude-haiku-4-5")
        seen = self._capture(c, {"content": [{"type": "text", "text": '{"ok": 1}'}]})
        self.assertEqual(c.generate_json("sys", "p", [BillImage(b"x")]), {"ok": 1})
        self.assertEqual(seen["headers"]["x-api-key"], "SECRET")
        self.assertEqual(seen["body"]["messages"][0]["content"][0]["type"], "image")

        c = self._client("openai", "gpt-5-mini")
        seen = self._capture(c, {"choices": [{"message": {"content": '{"ok": 2}'}}]})
        self.assertEqual(c.generate_json("sys", "p", [BillImage(b"x")]), {"ok": 2})
        self.assertNotIn("temperature", seen["body"])                      # gpt-5 rejects temperature
        self.assertIn("max_completion_tokens", seen["body"])

    def test_disallowed_model_is_rejected(self):
        import os
        from billdecoder.config import ConfigError
        with mock.patch.dict(os.environ, {"MODEL_PROVIDER": "openai", "MODEL_NAME": "gpt-4o", "API_KEY": "k"}):
            with self.assertRaises(ConfigError):
                Config.from_env()


if __name__ == "__main__":
    unittest.main()
