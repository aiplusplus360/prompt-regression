"""Output comparison: decide whether two model responses are "the same".

Exact string equality is useless for LLM output — a model will reorder a list
or change "3 items" to "three items" on every run. But the opposite failure is
worse: a similarity threshold loose enough to absorb that noise will happily
accept an answer that has inverted its meaning.

The approach here is layered. Cheap, deterministic checks run first and give a
clear verdict on the cases that matter (missing required content, forbidden
content). Only when those pass does fuzzy similarity come into play, and it is
reported alongside the exact diff so a human can see what actually changed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

# Whitespace is normalized before comparison: models emit " \n" and "\n\n"
# interchangeably and it never means anything.
_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Collapse whitespace without disturbing the words themselves."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    """Ratio in [0, 1]. Case-insensitive, whitespace-normalized."""
    a, b = normalize(a).lower(), normalize(b).lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class CompareResult:
    """The verdict for one case."""

    case_id: str
    passed: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    diff: str = ""
    output: str = ""

    def __bool__(self) -> bool:  # so `if result:` reads naturally
        return self.passed


def _missing(haystack: str, needles: Optional[Iterable[str]]) -> List[str]:
    if not needles:
        return []
    lowered = haystack.lower()
    return [n for n in needles if n.lower() not in lowered]


def _present(haystack: str, needles: Optional[Iterable[str]]) -> List[str]:
    if not needles:
        return []
    lowered = haystack.lower()
    return [n for n in needles if n.lower() in lowered]


def unified(expected: str, actual: str, case_id: str = "case") -> str:
    """Unified diff, so a failure shows what changed rather than just a score."""
    return "".join(
        difflib.unified_diff(
            normalize(expected).splitlines(keepends=True),
            normalize(actual).splitlines(keepends=True),
            fromfile=f"{case_id}/expected",
            tofile=f"{case_id}/actual",
            lineterm="\n",
        )
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def compare(
    case_id: str,
    output: str,
    expected: str = "",
    *,
    contains: Optional[Sequence[str]] = None,
    excludes: Optional[Sequence[str]] = None,
    min_similarity: float = 0.0,
    max_chars: Optional[int] = None,
    min_chars: Optional[int] = None,
    show_diff: bool = True,
) -> CompareResult:
    """Evaluate one output against a case's expectations.

    Every expectation that is present in the case is a hard requirement; the
    similarity threshold only applies when a baseline exists to compare
    against. That ordering matters — a lenient similarity score must never
    rescue an output that has dropped required content.
    """
    reasons: List[str] = []
    live = output or ""

    missing = _missing(live, contains)
    if missing:
        reasons.append(f"missing required text: {', '.join(repr(m) for m in missing)}")

    banned = _present(live, excludes)
    if banned:
        reasons.append(f"contains forbidden text: {', '.join(repr(b) for b in banned)}")

    score = 1.0
    has_baseline = bool(expected and expected.strip())
    if has_baseline:
        score = similarity(expected, live)
        if score < min_similarity:
            reasons.append(
                f"similarity {score:.3f} below threshold {min_similarity:.3f}"
            )

    length = len(normalize(live))
    if max_chars is not None and length > max_chars:
        reasons.append(f"output is {length} chars, over the {max_chars} limit")
    if min_chars is not None and length < min_chars:
        reasons.append(f"output is {length} chars, under the {min_chars} minimum")

    passed = not reasons
    diff_text = ""
    if not passed and show_diff and has_baseline:
        diff_text = unified(expected, live, case_id)

    return CompareResult(
        case_id=case_id,
        passed=passed,
        score=score,
        reasons=reasons,
        diff=diff_text,
        output=_truncate(live, 2000),
    )
