"""Hybrid categorization dispatcher.

Rules first, LLM fallback only on rule miss.

Design rationale (for DESIGN.md):
- Miss-only fallback: keeps the rule layer as a hard contract. If a rule
  fires we trust it (confidence=1.0). Avoids a second invariant on top
  of the already-tested rule ordering.
- We reuse CategoryResult from llm_categorize rather than creating a
  parallel type. The `source` field was designed to identify which layer
  produced the result — adding "rule" to its Literal is the natural
  extension, not a new class.
- llm_categorize already guarantees it never raises and degrades to
  source="fallback" on any failure, so the dispatcher does not need
  its own try/except or fallback construction.
"""

from __future__ import annotations

from scotia_agent.categories import categorize
from scotia_agent.llm_categorize import CategoryResult, llm_categorize

_RULE_MISS = "uncategorized"


def hybrid_categorize(
    description: str,
    sub_description: str | None = None,
) -> CategoryResult:
    """Categorize one transaction using rules first, LLM on miss.

    Args:
        description: Merchant name from the Scotia CSV.
        sub_description: Optional location/sub-detail, passed to the
            LLM layer when rules miss (e.g. 'Hamilton ON').

    Returns:
        CategoryResult with source="rule" | "llm" | "fallback".
        Rule hits carry confidence=1.0 by contract.
    """
    rule_cat = categorize(description)
    if rule_cat != _RULE_MISS:
        return CategoryResult(
            category=rule_cat,
            confidence=1.0,
            reasoning="matched keyword rule",
            raw_response="",
            source="rule",
        )
    return llm_categorize(description, sub_description)
