"""Tests for the LLM categorization parsing layer.

Focus: _parse_response and its helper coercers. These are pure functions —
no mocking needed, no API calls, instant to run.

Tests that exercise the full llm_categorize() entry point (which hits
_call_llm) will go in a separate test file / class with mocks.
"""

from __future__ import annotations

import pytest

from scotia_agent.llm_categorize import (
    _coerce_category,
    _coerce_confidence,
    _parse_response,
)


# ---------------------------------------------------------------------------
# _coerce_confidence — clamp and fallback behaviour
# ---------------------------------------------------------------------------
class TestCoerceConfidence:
    def test_valid_float_passes_through(self):
        assert _coerce_confidence(0.73) == 0.73

    def test_above_one_clamps_to_one(self):
        assert _coerce_confidence(1.5) == 1.0

    def test_below_zero_clamps_to_zero(self):
        assert _coerce_confidence(-0.2) == 0.0

    def test_string_number_is_coerced(self):
        """Regex rescue passes strings; we must accept them."""
        assert _coerce_confidence("0.85") == 0.85

    def test_garbage_falls_back_to_default(self):
        assert _coerce_confidence("not a number") == 0.5

    def test_none_falls_back_to_default(self):
        assert _coerce_confidence(None) == 0.5

    def test_custom_default_respected(self):
        assert _coerce_confidence("junk", default=0.1) == 0.1


# ---------------------------------------------------------------------------
# _coerce_category — lowercase, strip, validate against VALID_CATEGORIES
# ---------------------------------------------------------------------------
class TestCoerceCategory:
    def test_known_category_passes(self):
        # "restaurant" is in RULES, so it must be valid.
        assert _coerce_category("restaurant") == "restaurant"

    def test_uppercase_normalized(self):
        assert _coerce_category("RESTAURANT") == "restaurant"

    def test_whitespace_stripped(self):
        assert _coerce_category("  restaurant  ") == "restaurant"

    def test_unknown_category_coerces_to_uncategorized(self):
        """LLM sometimes invents categories not in our taxonomy.
        We must not propagate those downstream."""
        assert _coerce_category("interplanetary_travel") == "uncategorized"

    def test_none_coerces_to_uncategorized(self):
        assert _coerce_category(None) == "uncategorized"

    def test_empty_string_coerces_to_uncategorized(self):
        assert _coerce_category("") == "uncategorized"


# ---------------------------------------------------------------------------
# _parse_response — the primary parsing contract
# ---------------------------------------------------------------------------
class TestParseResponseHappyPath:
    def test_clean_json_parses_normally(self):
        raw = '{"category": "coffee", "confidence": 0.9, "reasoning": "Tim Hortons"}'
        r = _parse_response(raw)
        assert r.category == "coffee"
        assert r.confidence == 0.9
        assert r.reasoning == "Tim Hortons"
        assert r.source == "llm"
        # Rescue marker must NOT appear on clean parses.
        assert "rescued" not in r.reasoning.lower()

    def test_markdown_fenced_json_is_unwrapped(self):
        """Some models wrap JSON in ```json ... ``` despite JSON mode.
        We strip the fences before parsing."""
        raw = '```json\n{"category": "coffee", "confidence": 0.9, "reasoning": "x"}\n```'
        r = _parse_response(raw)
        assert r.category == "coffee"
        assert r.confidence == 0.9


class TestParseResponseRescuePath:
    """Regression tests for malformed-JSON rescue.

    Real production failure mode observed 2026-04-12: DeepSeek occasionally
    emits JSON with stray quotes inside string values, which json.loads
    rejects. Before the fix, these responses were discarded entirely.
    After the fix, we salvage category (required) + confidence and
    reasoning (best effort) and tag reasoning with a rescue marker so
    downstream eval can distinguish clean from rescued results.
    """

    # A response with an unescaped quote inside the reasoning value —
    # this is what actually broke json.loads in real traffic.
    MALFORMED = '{"category": "restaurant", "confidence": 0.95, "reasoning": "a "hotpot" place"}'

    def test_rescues_category(self):
        r = _parse_response(self.MALFORMED)
        assert r.category == "restaurant"

    def test_rescues_confidence_honestly(self):
        """Regression: previously this was force-set to 0.5, losing the
        real signal. We now report whatever the LLM said."""
        r = _parse_response(self.MALFORMED)
        assert r.confidence == 0.95

    def test_marks_reasoning_as_rescued(self):
        """Downstream eval/UI must be able to identify rescued results
        via a predictable marker in the reasoning field."""
        r = _parse_response(self.MALFORMED)
        assert r.reasoning.startswith("[rescued from malformed JSON]")

    def test_source_stays_llm(self):
        """Rescue is still LLM-produced output, not a fallback.
        source='fallback' is reserved for infra failures (API down, etc)."""
        r = _parse_response(self.MALFORMED)
        assert r.source == "llm"

    def test_raw_response_preserved(self):
        """Full original response must be stored for debugging."""
        r = _parse_response(self.MALFORMED)
        assert r.raw_response == self.MALFORMED


class TestParseResponseUnrecoverable:
    def test_no_category_at_all_raises(self):
        """If we can't even regex-extract a category, we give up and
        let the outer llm_categorize() degrade to source='fallback'."""
        raw = "the model returned pure prose with no json at all"
        with pytest.raises(ValueError, match="unparseable"):
            _parse_response(raw)

    def test_rescue_with_unknown_category_coerces_to_uncategorized(self):
        """LLM fabricated a category AND emitted malformed JSON —
        both layers of defense must still hold."""
        raw = '{"category": "quantum_lunch", "confidence": 0.8, "reasoning": "bad "json"'
        r = _parse_response(raw)
        assert r.category == "uncategorized"
