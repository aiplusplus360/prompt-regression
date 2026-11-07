# prompt-regression

**A prompt is a dependency. This treats it like one — record expected behaviour, run it in CI, fail the build when output drifts.**

[![CI](https://github.com/aiplusplus360/prompt-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/aiplusplus360/prompt-regression/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The problem

You tune a prompt. The demo looks better. A week later a customer says the bot stopped mentioning refunds — and you have no idea which of the last nine commits did it, because nobody diffs model output.

Unit tests don't help: the output is prose. Snapshot tests don't help either, because the model legitimately rephrases every run and the snapshot fails on noise until someone deletes it.

The useful question isn't *"is the output byte-identical?"* — it's **"did the behaviour I depend on change?"**

## What this does

```bash
$ prompt-regression run examples/support-suite.json --provider openai --model gpt-4o-mini

suite    support-reply-guardrails  (3 cases)
provider openai-compatible:gpt-4o-mini

  ok    refund-policy            (score 1.000)
  FAIL  no-pii-in-reply          (score 1.000)
        - contains forbidden text: '4'
  ok    escalates-cancellation   (score 1.000)

2/3 passed
```

Exit code `1`. CI stops. Nobody ships the prompt.

That example is the whole point: **the failing case has a perfect similarity score.** A naive diff-based tool would have called it a pass. It failed because the response echoed a card number, which is a rule, not a distribution.

## How the comparison works

Three layers, cheapest first:

| Layer | Catches | Cost |
|---|---|---|
| **Forbidden text** | leaked PII, banned claims, prompt-injection strings | substring scan |
| **Required text** | dropped policy facts, missing escalation language | substring scan |
| **Similarity threshold** | tone drift, content loss, structural collapse | `difflib` ratio |

Order matters. Required and forbidden text are **hard requirements** — a lenient similarity score can never rescue an output that has dropped mandatory content. That is the specific bug this library exists to avoid:

```python
# Naive: score 0.86, passes, ships the bug.
# This tool: fails on `contains=["Do not deploy"]`.
compare(case, output,
        expected="The report shows all systems nominal. Do not deploy.",
        min_similarity=0.85,
        contains=["Do not deploy"])
```

Failures print a unified diff, so you see what actually changed rather than a number:

```diff
--- refund-policy/expected
+++ refund-policy/actual
-Annual plans can be refunded within 30 days of purchase.
+Annual plans can be refunded within 14 days of purchase.
```

## Usage

```bash
pip install -e .

# no key, no network, no cost — proves the plumbing
prompt-regression run examples/support-suite.json --provider echo

# any OpenAI-compatible endpoint
prompt-regression run suite.json --provider openai --model gpt-4o-mini
prompt-regression run suite.json --provider openai \
  --base-url https://api.deepseek.com/v1 --model deepseek-chat

# seed a new suite in one pass, then commit it
prompt-regression run suite.json --provider openai --model gpt-4o-mini --update-baseline

# validate a suite without paying for a single token
prompt-regression check suite.json
```

Configuration also reads from the environment, so CI needs no flags:

| Variable | Purpose |
|---|---|
| `PROMPTREG_PROVIDER` | `echo` \| `openai` \| `compatible` |
| `PROMPTREG_MODEL` | model name |
| `PROMPTREG_BASE_URL` | endpoint base |
| `OPENAI_API_KEY` | credential |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | all cases passed |
| `1` | at least one regression |
| `2` | invalid suite or invocation |

`2` is separate on purpose: a typo in a case file must never be reported as a model failure.

## Writing a suite

A suite is JSON — no YAML parser, no dependency, and anything can generate it.

```json
{
  "name": "support-reply-guardrails",
  "cases": [
    {
      "id": "refund-policy",
      "system": "You are a support agent. Answer in one sentence.",
      "prompt": "What is the refund window for annual plans?",
      "expected": "Annual plans can be refunded within 30 days of purchase.",
      "contains": ["30 days"],
      "excludes": ["lifetime", "guaranteed"],
      "min_similarity": 0.8,
      "max_chars": 240,
      "min_chars": 10
    }
  ]
}
```

| Field | Required | Effect |
|---|---|---|
| `id` | yes | Unique case name; appears in reports |
| `prompt` | yes | User message |
| `system` | no | System prompt |
| `expected` | no | Baseline for similarity and diffs |
| `contains` | no | Substrings that must appear (case-insensitive) |
| `excludes` | no | Substrings that must not appear |
| `min_similarity` | no | `0..1`. **Requires `expected`** — enforced at load time |
| `max_chars` / `min_chars` | no | Length bounds, normalized |

Validation runs before any model call and rejects duplicate ids, non-numeric thresholds, and a `min_similarity` with no baseline — the last one because it would otherwise pass every case silently.

## Why no SDK

The provider is a `urllib` POST to `/chat/completions`. That means OpenAI, DeepSeek, Groq, Together, Mistral, OpenRouter, vLLM, LM Studio and Ollama all work with the same code path and **zero installed dependencies**. A tool that fails to install is a tool nobody puts in their pipeline.

Adding a provider is one class with a `complete()` method — the protocol is in [`providers.py`](promptreg/providers.py).

## Design notes

- **`temperature=0` by default.** A regression suite that is itself nondeterministic is not a suite.
- **Retries only on transient failures.** `429` and `5xx` back off; a `401` fails immediately, because retrying an auth error three times turns a clear message into a slow one.
- **Length is measured after normalization**, so trailing whitespace never fails a case.
- **The report includes the score even when a case fails**, so you can watch a metric improve while the case still fails on a hard rule.
- **`--update-baseline` only fills empty baselines.** It will not quietly overwrite a baseline you deliberately pinned.

## Development

```bash
python -m unittest discover -s tests -v   # 44 tests, no network
```

The suite is tested on Python 3.9, 3.11 and 3.13 in CI, and CI runs the example suite against the echo provider on every push — if the comparison layer regresses, the build goes red.

## License

[MIT](LICENSE)
