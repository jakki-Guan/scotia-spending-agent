"""Tests for the hybrid categorization dispatcher.

Dispatch logic is tiny (rule hit -> return; rule miss -> delegate to LLM),
so these tests focus on two contracts:

1. When rules hit, the LLM is NOT called (cost + correctness invariant).
2. When rules miss, llm_categorize is called with the correct arguments
   and its return value is passed through unchanged.

We mock llm_categorize at the hybrid module's binding — not where it's
defined — because hybrid.py imports it by name. Patching the definition
site would not intercept hybrid's call.
"""

from __future__ import annotations

from unittest.mock import patch

from scotia_agent.hybrid import hybrid_categorize
from scotia_agent.llm_categorize import CategoryResult


# ---------------------------------------------------------------------------
# Rule-hit path — LLM must not be invoked
# ---------------------------------------------------------------------------
class TestRuleHitPath:
    def test_rule_hit_returns_source_rule(self):
        # "tim hortons" is in the RULES table -> coffee
        r = hybrid_categorize("tim hortons #0226")
        assert r.source == "rule"
        assert r.category == "coffee"
        assert r.confidence == 1.0

    def test_rule_hit_does_not_call_llm(self):
        """Cost invariant: if we already know the category from rules,
        we must not burn an API call."""
        with patch("scotia_agent.hybrid.llm_categorize") as mock_llm:
            hybrid_categorize("tim hortons #0226")
            mock_llm.assert_not_called()

    def test_rule_hit_reasoning_mentions_rule(self):
        """The reasoning field must identify which layer produced the
        result — downstream UI uses this for the transparency indicator."""
        r = hybrid_categorize("tim hortons #0226")
        assert "rule" in r.reasoning.lower()


# ---------------------------------------------------------------------------
# Rule-miss path — LLM is invoked with correct args, result passes through
# ---------------------------------------------------------------------------
class TestRuleMissPath:
    # A merchant string guaranteed not to match any rule. If your rule
    # table ever grows to include "zzz" we'll need to change this.
    MISS_MERCHANT = "zzz nonexistent merchant 99999"

    def _stub_result(self) -> CategoryResult:
        """What we pretend the LLM returned."""
        return CategoryResult(
            category="restaurant",
            confidence=0.88,
            reasoning="stubbed LLM reasoning",
            raw_response='{"stubbed": true}',
            source="llm",
        )

    def test_rule_miss_delegates_to_llm(self):
        with patch("scotia_agent.hybrid.llm_categorize") as mock_llm:
            mock_llm.return_value = self._stub_result()
            hybrid_categorize(self.MISS_MERCHANT)
            mock_llm.assert_called_once()

    def test_rule_miss_passes_description_through(self):
        with patch("scotia_agent.hybrid.llm_categorize") as mock_llm:
            mock_llm.return_value = self._stub_result()
            hybrid_categorize(self.MISS_MERCHANT, "Hamilton ON")
            # Accept either positional or keyword call style.
            args, kwargs = mock_llm.call_args
            all_args = list(args) + list(kwargs.values())
            assert self.MISS_MERCHANT in all_args
            assert "Hamilton ON" in all_args

    def test_rule_miss_returns_llm_result_unchanged(self):
        """Dispatcher must not mutate what the LLM layer returned."""
        stub = self._stub_result()
        with patch("scotia_agent.hybrid.llm_categorize") as mock_llm:
            mock_llm.return_value = stub
            r = hybrid_categorize(self.MISS_MERCHANT)

        assert r.category == stub.category
        assert r.confidence == stub.confidence
        assert r.reasoning == stub.reasoning
        assert r.source == stub.source

    def test_rule_miss_with_llm_fallback_preserves_source(self):
        """When LLM itself degrades to source='fallback', dispatcher
        must propagate that faithfully — not rewrite it."""
        fallback = CategoryResult(
            category="uncategorized",
            confidence=0.0,
            reasoning="API error",
            raw_response="",
            source="fallback",
        )
        with patch("scotia_agent.hybrid.llm_categorize") as mock_llm:
            mock_llm.return_value = fallback
            r = hybrid_categorize(self.MISS_MERCHANT)
        assert r.source == "fallback"
        assert r.category == "uncategorized"


# ---------------------------------------------------------------------------
# sub_description handling
# ---------------------------------------------------------------------------
class TestSubDescription:
    def test_sub_description_not_passed_to_rules(self):
        """Rules only look at the main description. If a merchant matches
        a rule, the sub_description is not used — no surprise behaviour."""
        # Passing a sub_description that would be weird shouldn't affect
        # a rule hit.
        r = hybrid_categorize("tim hortons #0226", sub_description="Mars Colony")
        assert r.category == "coffee"
        assert r.source == "rule"

    def test_sub_description_defaults_to_none(self):
        """Optional parameter — callers without location data shouldn't
        have to pass anything."""
        with patch("scotia_agent.hybrid.llm_categorize") as mock_llm:
            mock_llm.return_value = CategoryResult(
                category="uncategorized",
                confidence=0.0,
                reasoning="",
                raw_response="",
                source="llm",
            )
            hybrid_categorize("zzz unknown merchant 99999")
            args, kwargs = mock_llm.call_args
            # Second positional arg OR 'sub_description' kwarg should be None
            if len(args) >= 2:
                assert args[1] is None
            else:
                assert kwargs.get("sub_description") is None
