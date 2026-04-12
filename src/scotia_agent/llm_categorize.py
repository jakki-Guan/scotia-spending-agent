"""LLM-based fallback categorizer for transactions the keyword rules miss.

Design decisions (see DESIGN.md for full rationale):
  - Returns a structured CategoryResult, not a bare string. This makes
    failures explicit and reasoning auditable.
  - The prompt sends only category NAMES, not the full keyword dict.
    LLMs already know Costco = groceries; re-listing rules wastes tokens
    and limits generalization.
  - All failures (network, malformed output, invalid category) degrade
    to category='uncategorized' rather than raising. A single bad row
    must never crash a batch run.
  - Retries with exponential backoff for transient errors (5xx, rate limits).
  - The set of valid categories is derived from the rule table, so new
    rule categories automatically become valid LLM outputs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Literal

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError

from scotia_agent.categories import RULES
from scotia_agent.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What categories are valid?  Derived from the rule table so the two
# can never drift apart — adding a new rule category automatically makes
# it a valid LLM output.
# ---------------------------------------------------------------------------

VALID_CATEGORIES: set[str] = {cat for _, cat in RULES} | {"uncategorized"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class CategoryResult(BaseModel):
    """Structured result from LLM categorization.

    `category` is the only field the dispatcher uses; the others exist
    for debugging, eval, and future features (confidence-weighted
    aggregation, audit logs)."""

    category: str = Field(description="One of VALID_CATEGORIES")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = Field(default="", description="LLM's stated rationale")
    raw_response: str = Field(default="", description="Full LLM output for debugging")
    source: Literal["llm", "rule", "fallback"] = Field(
        default="llm",
        description="'rule' = keyword match, 'llm' = model inference, 'fallback' = LLM failed",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# A short, human-readable list of categories. The model uses its world
# knowledge to map merchant -> category; we don't need to teach it that
# Costco is groceries.
_CATEGORY_HINTS = """\
Available categories:
  groceries, restaurant, fast_food, coffee, bubble_tea, dessert, alcohol,
  food_delivery, rideshare, transit, fuel, parking, carshare, car_rental,
  travel, shopping_online, shopping_retail, pharmacy, personal_care, vape,
  entertainment, fitness, gaming, education, telecom, utilities, insurance,
  government_fees, subscription_ai, subscription_media, subscription_cloud,
  subscription_pro, bank_fees, payment, vending, uncategorized
"""

SYSTEM_PROMPT = f"""\
You categorize personal credit card transactions for a spending tracker.
The card is a Canadian Scotiabank Visa; most merchants are in Ontario.

{_CATEGORY_HINTS}

Rules:
- Pick exactly one category from the list above.
- Use 'uncategorized' if the merchant name is too ambiguous to classify confidently.
- Recognize merchant aliases: many Canadian businesses appear under their
  numbered Ontario corporation name (e.g. '1234567 ONTARIO LIMIT' could be
  any small business). If you can't identify it, return 'uncategorized'.
- Recognize East Asian restaurant names (Japanese, Korean, Chinese, Vietnamese)
  which are common in this dataset.

Return ONLY a JSON object with this exact schema, no other text:
{{
  "category": "<one of the categories above>",
  "confidence": <number between 0 and 1>,
  "reasoning": "<one short sentence>"
}}"""


def _build_user_prompt(description: str, sub_description: str | None = None) -> str:
    """Build the per-transaction user message."""
    parts = [f"Merchant: {description}"]
    if sub_description:
        parts.append(f"Location/sub-detail: {sub_description}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Client (lazy singleton — created on first use, shared across calls)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.llm_api_key:
            raise RuntimeError(
                "LLM_API_KEY not configured. Set it in .env or disable LLM "
                "fallback by setting LLM_FALLBACK_ENABLED=false."
            )
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _client


# ---------------------------------------------------------------------------
# The core call (with retry)
# ---------------------------------------------------------------------------

# Retry config — kept as module constants so tests can override them.
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


def _call_llm(description: str, sub_description: str | None) -> str:
    """Make the API call with exponential backoff. Returns raw response text.

    Raises the underlying exception if all retries fail — caller decides
    how to degrade."""
    client = _get_client()
    user_prompt = _build_user_prompt(description, sub_description)

    backoff = INITIAL_BACKOFF_SECONDS
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                # Low temperature for classification — we want consistency,
                # not creativity. Some providers ignore this; that's fine.
                temperature=0.1,
                max_tokens=150,
                # JSON mode where supported (DeepSeek, OpenAI). Failing
                # silently on providers that don't support it is fine —
                # we still parse defensively below.
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""

        except (RateLimitError, APITimeoutError, APIError) as e:
            last_exc = e
            if attempt == MAX_RETRIES:
                logger.warning("LLM call failed after %d attempts: %s", MAX_RETRIES, e)
                raise
            logger.info(
                "LLM call attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                MAX_RETRIES,
                type(e).__name__,
                backoff,
            )
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER

    # Unreachable, but satisfies type checker
    raise RuntimeError(f"unreachable: last={last_exc}")


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------


# Regex to rescue the category even when JSON is malformed.
# Accepts both "category": "restaurant" and "category":"restaurant" etc.
_CATEGORY_REGEX = re.compile(
    r'"category"\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)


def _parse_response(raw: str) -> CategoryResult:
    """Parse and validate the LLM's JSON response.

    Two-stage defensive parsing:
      1. Try json.loads on cleaned text (strips markdown fences).
      2. If that fails, regex-extract just the category field. This
         rescues the ~1-2% of responses where LLM JSON mode emits a
         stray quote or similar mistake but the category itself is correct.
    """
    # Strip ``` fences if present (some models add them despite JSON mode)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()

    # --- Stage 1: strict JSON parse ---
    try:
        data = json.loads(text)
        category = str(data.get("category", "")).strip().lower()
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        reasoning = str(data.get("reasoning", "")).strip()

        if category not in VALID_CATEGORIES:
            logger.warning("LLM returned unknown category %r; coercing to uncategorized", category)
            category = "uncategorized"

        return CategoryResult(
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=raw,
            source="llm",
        )

    except json.JSONDecodeError:
        # --- Stage 2: regex rescue ---
        logger.info("JSON parse failed, attempting regex rescue of category field")
        match = _CATEGORY_REGEX.search(text)
        if not match:
            raise ValueError(f"LLM returned unparseable response: {raw[:200]}") from None

        category = match.group(1).strip().lower()
        if category not in VALID_CATEGORIES:
            logger.warning(
                "Regex rescue found unknown category %r; coercing to uncategorized",
                category,
            )
            category = "uncategorized"

        return CategoryResult(
            category=category,
            confidence=0.5,  # Penalized: we salvaged this, so mark lower confidence
            reasoning="(rescued from malformed JSON)",
            raw_response=raw,
            source="llm",
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def llm_categorize(
    description: str,
    sub_description: str | None = None,
) -> CategoryResult:
    """Classify a transaction using the LLM.

    Never raises for normal failure modes — degrades to a 'fallback'
    CategoryResult with category='uncategorized'. Raises only if the
    module is misconfigured (no API key) AND fallback is disabled.

    Args:
        description: Merchant name from the Scotia CSV.
        sub_description: Optional location/sub-detail field, often
            improves disambiguation (e.g. 'Hamilton ON').
    """
    if not description or not description.strip():
        return CategoryResult(
            category="uncategorized",
            confidence=0.0,
            reasoning="empty description",
            source="fallback",
        )

    if not settings.llm_fallback_enabled:
        return CategoryResult(
            category="uncategorized",
            confidence=0.0,
            reasoning="LLM fallback disabled in config",
            source="fallback",
        )

    try:
        raw = _call_llm(description, sub_description)
        return _parse_response(raw)
    except (APIError, RateLimitError, APITimeoutError) as e:
        logger.error("LLM call failed permanently: %s", e)
        return CategoryResult(
            category="uncategorized",
            confidence=0.0,
            reasoning=f"API error: {type(e).__name__}",
            source="fallback",
        )
    except (ValueError, ValidationError) as e:
        logger.error("LLM response unparseable: %s", e)
        return CategoryResult(
            category="uncategorized",
            confidence=0.0,
            reasoning=f"parse error: {e}",
            source="fallback",
        )
