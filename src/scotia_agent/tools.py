"""
Pure-function tools exposed to the LLM agent.

Each tool takes an enriched DataFrame (see enrich.prepare_dataframe) plus
business parameters, and returns a JSON-serializable dict. No side
effects, no LLM calls, no I/O.

Expected DataFrame columns (from parser + enrich):
    date             : pd.Timestamp
    description      : str           (raw merchant string)
    sub_description  : str | None
    status           : str           ('posted' | 'pending' | ...)
    txn_type         : str           ('debit' | 'credit')
    amount           : float         (debit positive, credit negative)
    month            : str           ('YYYY-MM', from enrich)
    category         : str           (from enrich)
    confidence       : float         (from enrich)
    cat_source       : str           (from enrich)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from .parser import spending_only

# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _filter_month(df: pd.DataFrame, month: str | None) -> pd.DataFrame:
    if month is None:
        return df
    return df[df["month"] == month]


def _round2(x: float) -> float:
    return round(float(x), 2)


CATEGORY_GROUPS: dict[str, list[str]] = {
    "subscriptions": [
        "subscription_ai",
        "subscription_media",
        "subscription_cloud",
        "subscription_pro",
        "subscription_shopping_online",
    ],
    "dining": [
        "restaurant",
        "fast_food",
        "coffee",
        "bubble_tea",
        "dessert",
        "food_delivery",
        "alcohol",
    ],
    "shopping": [
        "shopping_online",
        "shopping_retail",
        "subscription_shopping_online",
        "pharmacy",
        "personal_care",
    ],
    "transport": [
        "rideshare",
        "transit",
        "fuel",
        "parking",
        "car_rental",
    ],
}


def _normalize_text(value: str | None) -> str:
    """Uppercase and collapse whitespace for stable matching/grouping."""
    return re.sub(r"\s+", " ", str(value or "").strip().upper())


def _merchant_group_key(value: str | None) -> str:
    """Normalize merchant strings to a stable brand-like key.

    Conservative by design: we only strip trailing `#1234`-style store
    suffixes after normalizing case/whitespace. This makes queries like
    "Tim Hortons" match statement rows such as "TIM HORTONS #0226"
    without overreaching into aggressive fuzzy matching.
    """
    text = _normalize_text(value)
    return re.sub(r"\s+#\d+\b$", "", text)


def _filter_merchant(scope: pd.DataFrame, merchant: str) -> pd.DataFrame:
    """Filter rows by normalized merchant-group key.

    Exact key match comes first. If that misses, fall back to substring
    match on the normalized key so short user queries like "Tim Hortons"
    can still match "Tim Hortons #0226".
    """
    key = _merchant_group_key(merchant)
    keys = scope["description"].map(_merchant_group_key)

    exact = scope[keys == key]
    if not exact.empty:
        return exact

    contains = scope[keys.str.contains(key, na=False)]
    return contains


def _filter_month_range(
    df: pd.DataFrame,
    month_from: str | None = None,
    month_to: str | None = None,
) -> pd.DataFrame:
    if month_from is not None:
        df = df[df["month"] >= month_from]
    if month_to is not None:
        df = df[df["month"] <= month_to]
    return df


def _resolve_category_group(
    group: str | None = None,
    categories: list[str] | None = None,
) -> tuple[str, list[str]]:
    if categories:
        return group or "custom", categories

    if group is None:
        raise ValueError("Provide either 'group' or 'categories' for grouped category trend.")

    normalized = group.strip().lower()
    resolved = CATEGORY_GROUPS.get(normalized)
    if resolved is None:
        available = ", ".join(sorted(CATEGORY_GROUPS))
        raise ValueError(f"Unknown category group {group!r}. Available groups: {available}")
    return normalized, resolved


# --------------------------------------------------------------------------
# Tool 1: get_spending_by_category
# --------------------------------------------------------------------------


def get_spending_by_category(
    df: pd.DataFrame,
    month: str | None = None,
) -> dict:
    scope = spending_only(df)
    scope = _filter_month(scope, month)

    by_cat = scope.groupby("category")["amount"].sum().sort_values(ascending=False)

    return {
        "month": month if month else "all",
        "total": _round2(scope["amount"].sum()),
        "by_category": {cat: _round2(amt) for cat, amt in by_cat.items()},
    }


# --------------------------------------------------------------------------
# Tool 2: get_top_merchants
# --------------------------------------------------------------------------


def get_top_merchants(
    df: pd.DataFrame,
    n: int = 10,
    month: str | None = None,
    category: str | None = None,
) -> dict:
    scope = spending_only(df)
    scope = _filter_month(scope, month)
    if category is not None:
        scope = scope[scope["category"] == category]

    # Case-insensitive grouping: "TIM HORTONS" and "Tim Hortons" should
    # collapse. Also strip store suffixes like "#0226" into one group key.
    scope = scope.copy()
    scope["_key"] = scope["description"].map(_merchant_group_key)

    grouped = (
        scope.groupby("_key")
        .agg(
            merchant=("description", "first"),
            amount=("amount", "sum"),
            transactions=("amount", "count"),
        )
        .sort_values("amount", ascending=False)
        .head(n)
    )

    merchants = [
        {
            "merchant": row["merchant"],
            "amount": _round2(row["amount"]),
            "transactions": int(row["transactions"]),
        }
        for _, row in grouped.iterrows()
    ]

    return {
        "month": month if month else "all",
        "category": category if category else "all",
        "merchants": merchants,
    }


# --------------------------------------------------------------------------
# Tool 3: get_monthly_trend
# --------------------------------------------------------------------------


def get_monthly_trend(
    df: pd.DataFrame,
    category: str | None = None,
    merchant: str | None = None,
) -> dict:
    scope = spending_only(df)
    if category is not None:
        scope = scope[scope["category"] == category]
    if merchant is not None:
        scope = _filter_merchant(scope, merchant)

    by_month = scope.groupby("month")["amount"].sum().sort_index()

    return {
        "category": category if category else "all",
        "merchant": merchant if merchant else "all",
        "trend": [{"month": m, "amount": _round2(amt)} for m, amt in by_month.items()],
    }


# --------------------------------------------------------------------------
# Tool 4: get_grouped_category_trend
# --------------------------------------------------------------------------


def get_grouped_category_trend(
    df: pd.DataFrame,
    group: str | None = None,
    categories: list[str] | None = None,
    month_from: str | None = None,
    month_to: str | None = None,
) -> dict:
    scope = spending_only(df)
    resolved_group, resolved_categories = _resolve_category_group(group=group, categories=categories)
    scope = scope[scope["category"].isin(resolved_categories)]
    scope = _filter_month_range(scope, month_from=month_from, month_to=month_to)

    by_month = scope.groupby("month")["amount"].sum().sort_index()
    by_category = scope.groupby("category")["amount"].sum().sort_values(ascending=False)

    return {
        "group": resolved_group,
        "categories": resolved_categories,
        "month_from": month_from if month_from else "all",
        "month_to": month_to if month_to else "all",
        "total": _round2(scope["amount"].sum()),
        "by_category": {cat: _round2(amt) for cat, amt in by_category.items()},
        "trend": [{"month": m, "amount": _round2(amt)} for m, amt in by_month.items()],
    }


# --------------------------------------------------------------------------
# Tool 5: search_transactions
# --------------------------------------------------------------------------


def search_transactions(
    df: pd.DataFrame,
    query: str | None = None,
    merchant: str | None = None,
    category: str | None = None,
    status: str | None = None,
    txn_type: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
) -> dict:
    # Deliberately NOT using spending_only — search is the escape hatch
    # and should be able to find refunds, pending rows, etc.
    scope = df.copy()

    if query is not None:
        q = query.lower()
        mask = (
            scope["description"].str.lower().str.contains(q, na=False)
            | scope["sub_description"].fillna("").str.lower().str.contains(q, na=False)
            | scope["category"].str.lower().str.contains(q, na=False)
        )
        scope = scope[mask]

    if merchant is not None:
        scope = _filter_merchant(scope, merchant)
    if category is not None:
        scope = scope[scope["category"] == category]
    if status is not None:
        scope = scope[scope["status"] == status.lower().strip()]
    if txn_type is not None:
        scope = scope[scope["txn_type"] == txn_type.lower().strip()]

    if min_amount is not None:
        scope = scope[scope["amount"].abs() >= min_amount]
    if max_amount is not None:
        scope = scope[scope["amount"].abs() <= max_amount]

    if date_from is not None:
        scope = scope[scope["date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        scope = scope[scope["date"] <= pd.Timestamp(date_to)]

    scope = scope.sort_values("date", ascending=False)
    total_matches = len(scope)
    returned = scope.head(limit)

    transactions = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "merchant": row["description"],
            "amount": _round2(row["amount"]),
            "category": row["category"],
            "status": row["status"],
            "txn_type": row["txn_type"],
        }
        for _, row in returned.iterrows()
    ]

    return {
        "filters_applied": {
            "query": query,
            "merchant": merchant,
            "category": category,
            "status": status,
            "txn_type": txn_type,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "date_from": date_from,
            "date_to": date_to,
        },
        "total_matches": total_matches,
        "returned": len(transactions),
        "truncated": total_matches > limit,
        "transactions": transactions,
    }


# --------------------------------------------------------------------------
# Schemas and dispatch
# --------------------------------------------------------------------------

ToolFn = Callable[..., dict[str, Any]]


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_spending_by_category",
            "description": (
                "Summarize posted debit spending by category. Use this for questions like "
                "'where did I spend money' or 'how much did I spend on groceries last month'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "string",
                        "description": "Optional month filter in YYYY-MM format, e.g. 2026-04.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_merchants",
            "description": (
                "Return the top merchants by posted debit spending, optionally scoped to a month "
                "or category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of merchants to return. Defaults to 10.",
                        "minimum": 1,
                    },
                    "month": {
                        "type": "string",
                        "description": "Optional month filter in YYYY-MM format, e.g. 2026-04.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter such as coffee or groceries.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_trend",
            "description": (
                "Show month-by-month posted debit spending for a category or merchant. "
                "Merchant matching is normalized, so 'Tim Hortons' can match "
                "'TIM HORTONS #0226'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter such as coffee or alcohol.",
                    },
                    "merchant": {
                        "type": "string",
                        "description": "Optional merchant or brand name, e.g. Tim Hortons.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_grouped_category_trend",
            "description": (
                "Show month-by-month posted debit spending for a grouped category question such as "
                "subscriptions, dining, shopping, or transport. Use this when the user asks about a "
                "high-level spending bucket that spans multiple categories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {
                        "type": "string",
                        "enum": sorted(CATEGORY_GROUPS),
                        "description": (
                            "Named category group such as subscriptions, dining, shopping, "
                            "or transport."
                        ),
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional custom category list. Use this for ad-hoc grouped analysis "
                            "when the built-in groups are not enough."
                        ),
                    },
                    "month_from": {
                        "type": "string",
                        "description": "Optional inclusive month lower bound in YYYY-MM format.",
                    },
                    "month_to": {
                        "type": "string",
                        "description": "Optional inclusive month upper bound in YYYY-MM format.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transactions",
            "description": (
                "Search raw transactions as a fallback lookup tool. Supports text search across "
                "description, sub-description, and category plus explicit filters like status, "
                "transaction type, date range, amount range, merchant, and category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional free-text search over description, sub-description, "
                            "and category."
                        ),
                    },
                    "merchant": {
                        "type": "string",
                        "description": "Optional merchant or brand filter, e.g. Tim Hortons.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional exact category filter.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["posted", "pending"],
                        "description": "Optional transaction status filter.",
                    },
                    "txn_type": {
                        "type": "string",
                        "enum": ["debit", "credit"],
                        "description": "Optional transaction type filter.",
                    },
                    "min_amount": {
                        "type": "number",
                        "description": "Optional minimum absolute transaction amount.",
                    },
                    "max_amount": {
                        "type": "number",
                        "description": "Optional maximum absolute transaction amount.",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional inclusive start date in YYYY-MM-DD format.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional inclusive end date in YYYY-MM-DD format.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return. Defaults to 50.",
                        "minimum": 1,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_REGISTRY: dict[str, ToolFn] = {
    "get_spending_by_category": get_spending_by_category,
    "get_top_merchants": get_top_merchants,
    "get_monthly_trend": get_monthly_trend,
    "get_grouped_category_trend": get_grouped_category_trend,
    "search_transactions": search_transactions,
}


def _parse_tool_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    """Normalize tool-call arguments into a plain dict.

    LLM providers often return tool arguments as a JSON string rather than
    a Python dict. This helper accepts either form so the rest of the
    runtime can stay simple.
    """
    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        parsed = dict(arguments)
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ValueError(f"Tool arguments are not valid JSON: {arguments}") from e
    else:
        raise TypeError(f"Tool arguments must be a dict, JSON string, or None; got {type(arguments).__name__}")

    if not isinstance(parsed, dict):
        raise ValueError(f"Tool arguments must decode to a JSON object; got {type(parsed).__name__}")

    if "df" in parsed:
        raise ValueError("Tool arguments must not include 'df'; the runtime injects the DataFrame.")

    return parsed


def call_tool(
    name: str,
    df: pd.DataFrame,
    arguments: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Dispatch one LLM tool call to the matching pure Python function."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        available = ", ".join(sorted(TOOL_REGISTRY))
        raise ValueError(f"Unknown tool {name!r}. Available tools: {available}")

    parsed_args = _parse_tool_arguments(arguments)
    return tool(df=df, **parsed_args)
