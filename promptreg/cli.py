"""Command line interface.

    prompt-regression run suite.json --provider echo
    prompt-regression run suite.json --provider openai --model gpt-4o-mini
    prompt-regression run suite.json --update-baseline
    prompt-regression check suite.json

Exit codes are the contract with CI:

    0  every case passed
    1  at least one case regressed
    2  the suite or the invocation is invalid

Keeping "invalid" distinct from "regressed" means a typo in a case file never
gets reported as a failed model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .diff import compare
from .providers import Provider, complete_with_retry, get_provider
from .suite import Suite, dump_suite, load_suite

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INVALID = 2


def _write_baseline(suite: Suite, results: dict) -> None:
    """Fill in `expected` from the run, so a suite can be seeded in one pass."""
    changed = 0
    for case in suite.cases:
        output = results.get(case.id)
        if output and not case.expected:
            case.expected = output
            changed += 1
    return changed  # type: ignore[return-value]


def _render_failure(result, stream) -> None:
    print(f"  FAIL  {result.case_id}   (score {result.score:.3f})", file=stream)
    for reason in result.reasons:
        print(f"        - {reason}", file=stream)
    if result.diff:
        for line in result.diff.splitlines():
            print(f"        {line}", file=stream)


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)

    provider: Provider = get_provider(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    print(f"suite    {suite.name}  ({len(suite)} cases)")
    print(f"provider {provider.name}")
    if args.update_baseline:
        print("mode     update baseline")
    print("")

    outputs: dict = {}
    results = []
    errors = 0

    for case in suite:
        try:
            output = complete_with_retry(
                provider, case.prompt, system=case.system, attempts=args.attempts
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a case error
            errors += 1
            print(f"  ERROR {case.id}: {exc}", file=sys.stderr)
            continue

        outputs[case.id] = output

        if args.update_baseline:
            continue

        results.append(
            compare(
                case.id,
                output,
                case.expected,
                contains=case.contains,
                excludes=case.excludes,
                min_similarity=case.min_similarity,
                max_chars=case.max_chars,
                min_chars=case.min_chars,
                show_diff=not args.no_diff,
            )
        )

    if args.update_baseline:
        changed = _write_baseline(suite, outputs)
        Path(args.suite).write_text(
            json.dumps(dump_suite(suite), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"updated {changed} baseline(s) in {args.suite}")
        return EXIT_OK if errors == 0 else EXIT_REGRESSION

    if errors and not results:
        return EXIT_INVALID

    failures = [r for r in results if not r.passed]
    for result in results:
        if result.passed:
            print(f"  ok    {result.case_id}   (score {result.score:.3f})")
        else:
            _render_failure(result, sys.stdout)

    passed = len(results) - len(failures)
    print("")
    print(f"{passed}/{len(results)} passed", end="")
    if errors:
        print(f", {errors} error(s)", end="")
    print()

    if failures or errors:
        if args.report:
            Path(args.report).write_text(
                json.dumps(
                    {
                        "suite": suite.name,
                        "provider": provider.name,
                        "passed": passed,
                        "failed": len(failures),
                        "errors": errors,
                        "cases": [
                            {
                                "id": r.case_id,
                                "passed": r.passed,
                                "score": round(r.score, 4),
                                "reasons": r.reasons,
                            }
                            for r in results
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return EXIT_REGRESSION

    return EXIT_OK


def cmd_check(args: argparse.Namespace) -> int:
    """Validate a suite without calling a model — fast enough for a pre-commit."""
    try:
        suite = load_suite(args.suite)
    except Exception as exc:  # noqa: BLE001
        print(f"invalid suite: {exc}", file=sys.stderr)
        return EXIT_INVALID

    print(f"{suite.name}: {len(suite)} cases, all valid")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-regression",
        description="Run a prompt suite and fail when output regresses.",
    )
    parser.add_argument("--version", action="version", version=__version__)

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite against a provider")
    run.add_argument("suite", help="path to the suite JSON file")
    run.add_argument(
        "--provider",
        default=os.environ.get("PROMPTREG_PROVIDER", "echo"),
        choices=["echo", "openai", "compatible"],
        help="echo needs no network or key (default)",
    )
    run.add_argument("--model", default=os.environ.get("PROMPTREG_MODEL"))
    run.add_argument(
        "--base-url",
        default=os.environ.get("PROMPTREG_BASE_URL"),
        help="any OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1",
    )
    run.add_argument("--api-key", default=None, help="defaults to $OPENAI_API_KEY")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--attempts", type=int, default=3)
    run.add_argument(
        "--update-baseline",
        action="store_true",
        help="record this run's output as the expected value",
    )
    run.add_argument("--no-diff", action="store_true", help="hide unified diffs")
    run.add_argument("--report", help="write a JSON report to this path")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="validate a suite without running it")
    check.add_argument("suite")
    check.set_defaults(func=cmd_check)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - top level guard
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
