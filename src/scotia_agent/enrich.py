"""
Turns the parser's raw DataFrame into an agent-ready DataFrame.

Parser's job: validate + load. No categorization, no derived columns.
This module's job: add the derived columns the agent tools expect.

Called once at agent startup. The enriched DataFrame is then passed
into every tool call for the remainder of the session.
"""

from __future__ import annotations

import pandas as pd

from .hybrid import hybrid_categorize


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns needed by the agent tools.

    Adds:
        month       : 'YYYY-MM' string, derived from date
        category    : str, from hybrid_categorize
        confidence  : float, from hybrid_categorize
        cat_source  : 'rule' | 'llm' | 'fallback'

    Does NOT filter out pending or credit rows. Tools decide per-query
    whether to filter (via parser.spending_only) or see everything
    (e.g. search_transactions wants to see refunds and pending rows).
    """
    if raw_df.empty:
        return raw_df.copy()

    df = raw_df.copy()
    df["month"] = df["date"].dt.strftime("%Y-%m")

    # Run hybrid_categorize row by row. Yes it's an apply — the LLM
    # fallback inside makes vectorization pointless, and the cost is
    # dominated by LLM latency on cache misses, not pandas overhead.
    results = df.apply(
        lambda r: hybrid_categorize(r["description"], r.get("sub_description")),
        axis=1,
    )
    df["category"] = [r.category for r in results]
    df["confidence"] = [r.confidence for r in results]
    df["cat_source"] = [r.source for r in results]

    return df
