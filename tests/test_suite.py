"""Tests for suite loading and the CLI contract."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from promptreg.cli import EXIT_INVALID, EXIT_OK, EXIT_REGRESSION, main
from promptreg.suite import SuiteError, load_suite


class TestSuiteLoading(unittest.TestCase):
    def _write(self, data):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        if isinstance(data, str):
            handle.write(data)
        else:
            json.dump(data, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_loads_a_bare_list(self):
        path = self._write([{"id": "a", "prompt": "hi"}])
        suite = load_suite(path)
        self.assertEqual(len(suite), 1)
        self.assertEqual(suite.cases[0].id, "a")

    def test_loads_wrapped_object(self):
        path = self._write({"name": "demo", "cases": [{"id": "a", "prompt": "hi"}]})
        self.assertEqual(load_suite(path).name, "demo")

    def test_rejects_missing_id(self):
        path = self._write({"cases": [{"prompt": "hi"}]})
        with self.assertRaises(SuiteError):
            load_suite(path)

    def test_rejects_missing_prompt(self):
        path = self._write({"cases": [{"id": "a"}]})
        with self.assertRaises(SuiteError):
            load_suite(path)

    def test_rejects_duplicate_ids(self):
        path = self._write(
            {"cases": [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}]}
        )
        with self.assertRaises(SuiteError) as ctx:
            load_suite(path)
        self.assertIn("duplicate", str(ctx.exception))

    def test_rejects_empty_cases(self):
        path = self._write({"cases": []})
        with self.assertRaises(SuiteError):
            load_suite(path)

    def test_rejects_invalid_json(self):
        path = self._write("{not json")
        with self.assertRaises(SuiteError):
            load_suite(path)

    def test_rejects_out_of_range_similarity(self):
        path = self._write(
            {"cases": [{"id": "a", "prompt": "x", "expected": "y", "min_similarity": 2}]}
        )
        with self.assertRaises(SuiteError):
            load_suite(path)

    def test_rejects_similarity_without_baseline(self):
        # Otherwise every case would pass silently.
        path = self._write(
            {"cases": [{"id": "a", "prompt": "x", "min_similarity": 0.8}]}
        )
        with self.assertRaises(SuiteError):
            load_suite(path)

    def test_rejects_non_list_contains(self):
        path = self._write({"cases": [{"id": "a", "prompt": "x", "contains": "y"}]})
        with self.assertRaises(SuiteError):
            load_suite(path)

    def test_missing_file(self):
        with self.assertRaises(SuiteError):
            load_suite("definitely-not-here.json")


class TestCli(unittest.TestCase):
    def _suite(self, cases):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"name": "t", "cases": cases}, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_passing_suite_exits_zero(self):
        path = self._suite([{"id": "a", "prompt": "hello", "contains": ["hello"]}])
        code, out, _ = self._run(["run", path, "--provider", "echo"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("1/1 passed", out)

    def test_missing_content_exits_one(self):
        path = self._suite([{"id": "a", "prompt": "hello", "contains": ["goodbye"]}])
        code, out, _ = self._run(["run", path, "--provider", "echo"])
        self.assertEqual(code, EXIT_REGRESSION)
        self.assertIn("FAIL", out)

    def test_bad_suite_exits_two(self):
        code, _, err = self._run(["run", "no-such-file.json", "--provider", "echo"])
        self.assertEqual(code, EXIT_INVALID)
        self.assertIn("error", err)

    def test_check_validates_without_calling(self):
        path = self._suite([{"id": "a", "prompt": "hello"}])
        code, out, _ = self._run(["check", path])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("1 cases", out)

    def test_check_flags_invalid_suite(self):
        path = self._suite([{"id": "a"}])
        code, _, err = self._run(["check", path])
        self.assertEqual(code, EXIT_INVALID)
        self.assertIn("invalid suite", err)

    def test_update_baseline_writes_expected(self):
        path = self._suite([{"id": "a", "prompt": "hello"}])
        code, _, _ = self._run(["run", path, "--provider", "echo", "--update-baseline"])
        self.assertEqual(code, EXIT_OK)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertIn("hello", data["cases"][0]["expected"])

    def test_report_is_written_on_failure(self):
        path = self._suite([{"id": "a", "prompt": "hello", "contains": ["nope"]}])
        report = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        report.close()
        self.addCleanup(os.unlink, report.name)
        code, _, _ = self._run(
            ["run", path, "--provider", "echo", "--report", report.name]
        )
        self.assertEqual(code, EXIT_REGRESSION)
        data = json.loads(Path(report.name).read_text(encoding="utf-8"))
        self.assertEqual(data["failed"], 1)
        self.assertFalse(data["cases"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
