"""Tests for output comparison.

The important cases are the ones where a naive implementation would lie: a
high similarity score on an output that dropped required content, or a low
score on an output that is semantically identical but reformatted.
"""

import unittest

from promptreg.diff import compare, normalize, similarity, unified


class TestNormalize(unittest.TestCase):
    def test_collapses_runs_of_spaces(self):
        self.assertEqual(normalize("a    b"), "a b")

    def test_normalizes_line_endings(self):
        self.assertEqual(normalize("a\r\nb"), "a\nb")

    def test_collapses_excess_blank_lines(self):
        self.assertEqual(normalize("a\n\n\n\nb"), "a\n\nb")

    def test_strips_outer_whitespace(self):
        self.assertEqual(normalize("  x  "), "x")


class TestSimilarity(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertEqual(similarity("hello world", "hello world"), 1.0)

    def test_ignores_case(self):
        self.assertEqual(similarity("Hello", "hello"), 1.0)

    def test_ignores_whitespace_differences(self):
        self.assertEqual(similarity("a  b", "a b"), 1.0)

    def test_empty_against_text_is_zero(self):
        self.assertEqual(similarity("", "something"), 0.0)

    def test_unrelated_text_is_low(self):
        self.assertLess(similarity("the cat sat", "quantum chromodynamics"), 0.5)

    def test_paraphrase_is_not_full_marks(self):
        # Guards against a threshold so low it accepts anything.
        self.assertLess(similarity("send 3 emails", "do not send any emails"), 0.9)


class TestCompare(unittest.TestCase):
    def test_passes_with_no_expectations(self):
        result = compare("c", "anything")
        self.assertTrue(result)
        self.assertEqual(result.score, 1.0)

    def test_requires_contains(self):
        result = compare("c", "Hello there", contains=["goodbye"])
        self.assertFalse(result)
        self.assertIn("missing required text", result.reasons[0])

    def test_contains_is_case_insensitive(self):
        self.assertTrue(compare("c", "Hello World", contains=["hello world"]))

    def test_rejects_forbidden_text(self):
        result = compare("c", "Here is your password", excludes=["password"])
        self.assertFalse(result)
        self.assertIn("forbidden", result.reasons[0])

    def test_reports_every_failure_not_just_the_first(self):
        result = compare("c", "x", contains=["a"], excludes=["x"])
        self.assertEqual(len(result.reasons), 2)

    def test_similarity_threshold_applies_to_baseline(self):
        result = compare("c", "totally different", "the original text", min_similarity=0.9)
        self.assertFalse(result)
        self.assertIn("similarity", result.reasons[0])

    def test_passing_contains_does_not_rescue_low_similarity(self):
        # The layered ordering: content checks and similarity are both hard.
        result = compare(
            "c", "alpha", "alpha beta gamma delta", min_similarity=0.99, contains=["alpha"]
        )
        self.assertFalse(result)

    def test_lenient_similarity_does_not_rescue_missing_content(self):
        # This is the failure mode that makes naive diffing dangerous.
        result = compare(
            "c",
            "The report shows all systems nominal.",
            "The report shows all systems nominal. Do not deploy.",
            min_similarity=0.5,
            contains=["Do not deploy"],
        )
        self.assertFalse(result)

    def test_max_chars(self):
        self.assertFalse(compare("c", "x" * 50, max_chars=10))

    def test_min_chars(self):
        self.assertFalse(compare("c", "x", min_chars=10))

    def test_score_is_exposed_even_when_failing(self):
        result = compare("c", "abc", "abd", min_similarity=0.99)
        self.assertFalse(result)
        self.assertGreater(result.score, 0.0)

    def test_diff_is_populated_on_baseline_failure(self):
        result = compare("c", "new line", "old line", min_similarity=0.9)
        self.assertIn("-old line", result.diff)
        self.assertIn("+new line", result.diff)

    def test_baseline_alone_does_not_fail(self):
        # A baseline with no threshold records output; it does not assert
        # equality. Scoring is opt-in via min_similarity.
        self.assertTrue(compare("c", "new line", "old line"))

    def test_no_diff_when_requested(self):
        result = compare("c", "a", "b", min_similarity=0.9, show_diff=False)
        self.assertEqual(result.diff, "")

    def test_output_is_truncated(self):
        result = compare("c", "y" * 10_000)
        self.assertLessEqual(len(result.output), 2001)


class TestUnified(unittest.TestCase):
    def test_includes_file_labels(self):
        text = unified("a", "b", "mycase")
        self.assertIn("mycase/expected", text)
        self.assertIn("mycase/actual", text)


if __name__ == "__main__":
    unittest.main()
