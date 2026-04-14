"""Evaluation runner for the hybrid categorizer.

Usage:
    uv run python -m eval.runner

Reads eval/dataset.jsonl, runs hybrid_categorize on each sample,
computes per-source / per-difficulty / overall accuracy,
writes a markdown report to eval/latest_report.md.

This is NOT a pytest target — it makes real LLM API calls and costs
real (tiny) money. Run it when you want to measure, not on every commit.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scotia_agent.hybrid import hybrid_categorize
from scotia_agent.llm_categorize import VALID_CATEGORIES

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
REPORT_PATH = EVAL_DIR / "latest_report.md"


@dataclass
class EvalRow:
    description: str
    sub_description: str | None
    expected_category: str
    difficulty: str
    notes: str
    aliased: bool = False  # 默认 False，向后兼容旧样本


@dataclass
class EvalResult:
    row: EvalRow
    predicted_category: str
    predicted_source: str
    predicted_confidence: float
    correct: bool


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------
def load_dataset(path: Path = DATASET_PATH) -> list[EvalRow]:
    """Load JSONL, validate every row upfront. Fail loud, fail early."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[EvalRow] = []
    errors: list[str] = []

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                row = EvalRow(
                    description=data["description"],
                    sub_description=data.get("sub_description"),
                    expected_category=data["expected_category"],
                    difficulty=data["difficulty"],
                    notes=data.get("notes", ""),
                )
            except (json.JSONDecodeError, KeyError) as e:
                errors.append(f"line {lineno}: {e}")
                continue

            # Invariant checks — wrong labels poison metrics silently.
            if row.expected_category not in VALID_CATEGORIES:
                errors.append(
                    f"line {lineno}: expected_category {row.expected_category!r} "
                    f"not in VALID_CATEGORIES"
                )
            if row.difficulty not in {"easy", "medium", "hard"}:
                errors.append(
                    f"line {lineno}: difficulty {row.difficulty!r} must be easy|medium|hard"
                )
            rows.append(row)

    if errors:
        raise ValueError("Dataset validation failed:\n  " + "\n  ".join(errors))
    return rows


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_eval(rows: list[EvalRow]) -> list[EvalResult]:
    """Dispatch every row through hybrid_categorize, record outcomes."""
    results: list[EvalResult] = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row.description[:40]:40} ... ", end="", flush=True)
        pred = hybrid_categorize(row.description, row.sub_description)
        correct = pred.category == row.expected_category
        print(
            f"{'✓' if correct else '✗'} {pred.category} (src={pred.source}, conf={pred.confidence:.2f})"
        )
        results.append(
            EvalResult(
                row=row,
                predicted_category=pred.category,
                predicted_source=pred.source,
                predicted_confidence=pred.confidence,
                correct=correct,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _accuracy(results: list[EvalResult]) -> tuple[int, int, float]:
    n = len(results)
    hits = sum(r.correct for r in results)
    return hits, n, (hits / n if n else 0.0)


def build_report(results: list[EvalResult]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    hits, total, overall = _accuracy(results)

    lines: list[str] = []
    lines.append(f"# Eval Report — {timestamp}")
    lines.append("")
    lines.append(f"**Overall accuracy:** {hits}/{total} = **{overall:.1%}**")
    lines.append("")

    # Per-difficulty breakdown
    lines.append("## By difficulty")
    lines.append("")
    lines.append("| Difficulty | Hits | Total | Accuracy |")
    lines.append("|---|---|---|---|")
    by_diff: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_diff[r.row.difficulty].append(r)
    for diff in ("easy", "medium", "hard"):
        bucket = by_diff.get(diff, [])
        h, n, acc = _accuracy(bucket)
        lines.append(f"| {diff} | {h} | {n} | {acc:.1%} |" if n else f"| {diff} | - | 0 | - |")
    lines.append("")

    # Per-source breakdown (did rules or LLM produce this answer?)
    lines.append("## By source")
    lines.append("")
    lines.append("| Source | Hits | Total | Accuracy |")
    lines.append("|---|---|---|---|")
    by_source: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_source[r.predicted_source].append(r)
    for src in ("rule", "llm", "fallback"):
        bucket = by_source.get(src, [])
        h, n, acc = _accuracy(bucket)
        lines.append(f"| {src} | {h} | {n} | {acc:.1%} |" if n else f"| {src} | - | 0 | - |")
    lines.append("")

    # Failures in detail
    failures = [r for r in results if not r.correct]
    if failures:
        lines.append(f"## Failures ({len(failures)})")
        lines.append("")
        lines.append("| Description | Expected | Predicted | Source | Conf | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for r in failures:
            lines.append(
                f"| `{r.row.description}` | {r.row.expected_category} "
                f"| {r.predicted_category} | {r.predicted_source} "
                f"| {r.predicted_confidence:.2f} | {r.row.notes} |"
            )
        lines.append("")

    # Confusion: what categories does the model conflate?
    confusions = Counter(
        (r.row.expected_category, r.predicted_category) for r in results if not r.correct
    )
    if confusions:
        lines.append("## Confusion pairs (expected → predicted)")
        lines.append("")
        for (expected, predicted), count in confusions.most_common():
            lines.append(f"- {expected} → {predicted}: {count}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        rows = load_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} eval samples from {DATASET_PATH.name}")
    print("=" * 60)
    results = run_eval(rows)
    print("=" * 60)

    report = build_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")

    hits, total, overall = _accuracy(results)
    print(f"Overall: {hits}/{total} = {overall:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
