"""Tests for scotia_agent.categories.

Two layers:
  - Unit tests (synthetic): verify specific merchant strings map to the
    intended category, with particular attention to rule ORDER.
  - Quality gates (real data): assert row-weighted AND dollar-weighted
    hit rates on Jake's actual 1000-row export stay above thresholds.

See README "Categorization Quality Metrics" section for why both
metrics are gated.
"""

from __future__ import annotations

import pytest

from scotia_agent.categories import categorize
from scotia_agent.parser import spending_only

# ---------------------------------------------------------------------------
# Unit tests — specific strings map to specific categories.
# ---------------------------------------------------------------------------

# (description, expected_category) — covers the tricky cases, not every rule.
CASES = [
    # Rule-order-sensitive: ubereats must beat plain uber
    ("uber canada/ubereats", "food_delivery"),
    ("Uber Canada/UberTrip", "rideshare"),
    ("uber* eats", "food_delivery"),
    ("Uber* Trip", "rideshare"),
    # High-frequency merchants (head of the distribution)
    ("james street market", "groceries"),
    ("TIM HORTONS #0226", "coffee"),
    ("PRESTO Autl", "transit"),
    ("Communauto ON", "car_rental"),
    # Subscriptions
    ("claude.ai subscription", "subscription_ai"),
    ("OpenAI *ChatGPT Subscr", "subscription_ai"),
    ("Spotify P40AAA2DCF", "subscription_media"),
    ("google *google one", "subscription_cloud"),
    ("apple.com/bill", "gaming"),
    ("LinkedInPrea *39669006", "subscription_pro"),
    # Telecom / utilities / bank
    ("Rogers ******3330", "telecom"),
    ("Wyse Meter Solutions", "utilities"),
    ("SCOTIA CREDIT CARD PROTEC", "bank_fees"),
    ("Interest Charges-Purchase", "bank_fees"),
    ("PAYMENT FROM - *****07*95", "payment"),
    # Merchant aliasing — the Esso case documented in README
    ("1000503499 ontario limit", "fuel"),
    ("Cannon St Esso", "fuel"),
    # Online shopping
    ("AMZN Mktp CA*B20AY4XF0", "subscription_shopping_online"),
    ("Amazon.ca*BD7WP8WR1", "shopping_online"),
    # Alcohol
    ("LCBO/RAO #0571", "alcohol"),
    # Unknown merchant — should fall through to uncategorized (LLM fallback)
    ("totally unknown merchant xyz", "uncategorized"),
]


@pytest.mark.parametrize("description,expected", CASES)
def test_categorize_specific_merchants(description, expected):
    assert categorize(description) == expected, (
        f"{description!r} got {categorize(description)!r}, expected {expected!r}"
    )


def test_empty_string_is_uncategorized():
    assert categorize("") == "uncategorized"


def test_categorize_is_case_insensitive():
    assert categorize("UBER CANADA/UBEREATS") == "food_delivery"
    assert categorize("uber canada/ubereats") == "food_delivery"


# ---------------------------------------------------------------------------
# Rule ordering invariants — catch the #1 way this module can silently break.
# ---------------------------------------------------------------------------


def test_ubereats_rule_precedes_generic_uber():
    """If someone reorders RULES and puts 'uber' before 'ubereats',
    all Uber Eats txns silently become rideshare. This guards that."""
    from scotia_agent.categories import RULES

    uber_idx = next(i for i, (k, _) in enumerate(RULES) if k == "uber")
    eats_idx = next(i for i, (k, _) in enumerate(RULES) if "eats" in k)
    assert eats_idx < uber_idx, "ubereats rules must come before the generic 'uber' catch-all"


# ---------------------------------------------------------------------------
# Quality gates — real data, row + dollar weighted hit rate.
# ---------------------------------------------------------------------------

# Thresholds deliberately set just below current performance, so we catch
# regressions without false alarms from normal rule tuning.
ROW_HIT_THRESHOLD = 0.90  # current: ~93.7%
DOLLAR_HIT_THRESHOLD = 0.90  # current: ~95.9%
MAX_POSITIVE_GAP_PP = 3.0  # current: gap is -2.2pp; fail if it turns +3pp


def _categorized_spend(df):
    """Return the DataFrame used for quality metrics: debits only,
    excluding bank_fees and payments (neither is real consumer spending)."""
    df = df.copy()
    df["category"] = df["description"].apply(categorize)
    spend = spending_only(df)
    return spend[~spend["category"].isin(["bank_fees", "payment"])]


@pytest.mark.real_data
def test_row_weighted_hit_rate(real_data):
    df, _ = real_data
    spend = _categorized_spend(df)
    row_hit = (spend["category"] != "uncategorized").mean()
    assert row_hit >= ROW_HIT_THRESHOLD, (
        f"Row hit rate dropped to {row_hit:.1%} (threshold {ROW_HIT_THRESHOLD:.0%})"
    )


@pytest.mark.real_data
def test_dollar_weighted_hit_rate(real_data):
    """This is the metric that actually matters — see README."""
    df, _ = real_data
    spend = _categorized_spend(df)
    matched = spend[spend["category"] != "uncategorized"]["amount"].sum()
    total = spend["amount"].sum()
    dollar_hit = matched / total
    assert dollar_hit >= DOLLAR_HIT_THRESHOLD, (
        f"Dollar hit rate dropped to {dollar_hit:.1%} (threshold {DOLLAR_HIT_THRESHOLD:.0%})"
    )


@pytest.mark.real_data
def test_gap_is_not_strongly_positive(real_data):
    """Gap > 0 means we're missing big-ticket transactions (bad).
    A healthy categorizer has gap <= 0. Allow a small positive buffer."""
    df, _ = real_data
    spend = _categorized_spend(df)
    row_hit = (spend["category"] != "uncategorized").mean()
    matched = spend[spend["category"] != "uncategorized"]["amount"].sum()
    dollar_hit = matched / spend["amount"].sum()
    gap_pp = (row_hit - dollar_hit) * 100
    assert gap_pp <= MAX_POSITIVE_GAP_PP, (
        f"Gap became {gap_pp:+.1f}pp — rules are leaking big-ticket $. "
        f"Check which merchants landed in uncategorized."
    )


@pytest.mark.real_data
def test_no_category_is_suspiciously_dominant(real_data):
    """uncategorized should never be the largest category by $."""
    df, _ = real_data
    spend = _categorized_spend(df)
    by_cat = spend.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_category = by_cat.index[0]
    assert top_category != "uncategorized", (
        f"uncategorized is the top spending bucket (${by_cat.iloc[0]:.2f}) — "
        f"rules have a major blind spot."
    )
