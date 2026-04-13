"""Tests for scotia_agent.tools.

These are hermetic unit tests for the four LLM-facing analytics tools.
They validate the business contract of each tool independently from any
agent framework or live LLM calls.
"""

from __future__ import annotations

import pandas as pd

from scotia_agent.tools import (
    TOOL_REGISTRY,
    TOOLS_SCHEMA,
    call_tool,
    get_monthly_trend,
    get_spending_by_category,
    get_top_merchants,
    search_transactions,
)


def make_enriched_df() -> pd.DataFrame:
    """Small agent-ready DataFrame with debits, credits, and pending rows."""
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-01"),
                "description": "Tim Hortons #0226",
                "sub_description": "Hamilton ON",
                "status": "posted",
                "txn_type": "debit",
                "amount": 5.25,
                "month": "2026-03",
                "category": "coffee",
                "confidence": 1.0,
                "cat_source": "rule",
            },
            {
                "date": pd.Timestamp("2026-03-02"),
                "description": "TIM HORTONS #0226",
                "sub_description": "Hamilton ON",
                "status": "posted",
                "txn_type": "debit",
                "amount": 4.75,
                "month": "2026-03",
                "category": "coffee",
                "confidence": 1.0,
                "cat_source": "rule",
            },
            {
                "date": pd.Timestamp("2026-03-03"),
                "description": "James Street Market",
                "sub_description": "Hamilton ON",
                "status": "posted",
                "txn_type": "debit",
                "amount": 32.10,
                "month": "2026-03",
                "category": "groceries",
                "confidence": 1.0,
                "cat_source": "rule",
            },
            {
                "date": pd.Timestamp("2026-03-04"),
                "description": "Uber Canada/UberTrip",
                "sub_description": "Toronto ON",
                "status": "pending",
                "txn_type": "debit",
                "amount": 18.50,
                "month": "2026-03",
                "category": "rideshare",
                "confidence": 1.0,
                "cat_source": "rule",
            },
            {
                "date": pd.Timestamp("2026-03-05"),
                "description": "Refund Coffee",
                "sub_description": None,
                "status": "posted",
                "txn_type": "credit",
                "amount": -5.25,
                "month": "2026-03",
                "category": "coffee",
                "confidence": 0.8,
                "cat_source": "llm",
            },
            {
                "date": pd.Timestamp("2026-04-01"),
                "description": "Tim Hortons #0226",
                "sub_description": "Hamilton ON",
                "status": "posted",
                "txn_type": "debit",
                "amount": 6.00,
                "month": "2026-04",
                "category": "coffee",
                "confidence": 1.0,
                "cat_source": "rule",
            },
            {
                "date": pd.Timestamp("2026-04-02"),
                "description": "LCBO/RAO #0571",
                "sub_description": "Hamilton ON",
                "status": "posted",
                "txn_type": "debit",
                "amount": 40.00,
                "month": "2026-04",
                "category": "alcohol",
                "confidence": 1.0,
                "cat_source": "rule",
            },
        ]
    )
    return df


class TestGetSpendingByCategory:
    def test_excludes_pending_and_credit_rows(self):
        df = make_enriched_df()
        result = get_spending_by_category(df, month="2026-03")

        assert result["month"] == "2026-03"
        assert result["total"] == 42.10
        assert result["by_category"] == {
            "groceries": 32.10,
            "coffee": 10.00,
        }


class TestGetTopMerchants:
    def test_groups_merchants_case_insensitively(self):
        df = make_enriched_df()
        result = get_top_merchants(df, n=3, month="2026-03")

        assert result["month"] == "2026-03"
        assert result["category"] == "all"
        assert result["merchants"][0] == {
            "merchant": "James Street Market",
            "amount": 32.10,
            "transactions": 1,
        }
        assert result["merchants"][1] == {
            "merchant": "Tim Hortons #0226",
            "amount": 10.00,
            "transactions": 2,
        }

    def test_applies_category_filter_before_grouping(self):
        df = make_enriched_df()
        result = get_top_merchants(df, n=5, category="coffee")

        assert result["category"] == "coffee"
        assert [m["merchant"] for m in result["merchants"]] == ["Tim Hortons #0226"]
        assert result["merchants"][0]["amount"] == 16.00
        assert result["merchants"][0]["transactions"] == 3


class TestGetMonthlyTrend:
    def test_sums_by_month_for_category(self):
        df = make_enriched_df()
        result = get_monthly_trend(df, category="coffee")

        assert result["category"] == "coffee"
        assert result["merchant"] == "all"
        assert result["trend"] == [
            {"month": "2026-03", "amount": 10.00},
            {"month": "2026-04", "amount": 6.00},
        ]

    def test_filters_by_normalized_merchant_string(self):
        df = make_enriched_df()
        result = get_monthly_trend(df, merchant="Tim Hortons #0226")

        assert result["merchant"] == "Tim Hortons #0226"
        assert result["trend"] == [
            {"month": "2026-03", "amount": 10.00},
            {"month": "2026-04", "amount": 6.00},
        ]

    def test_short_brand_name_matches_store_suffixed_rows(self):
        df = make_enriched_df()
        result = get_monthly_trend(df, merchant="Tim Hortons")

        assert result["merchant"] == "Tim Hortons"
        assert result["trend"] == [
            {"month": "2026-03", "amount": 10.00},
            {"month": "2026-04", "amount": 6.00},
        ]


class TestSearchTransactions:
    def test_search_includes_pending_and_credit_rows(self):
        df = make_enriched_df()
        result = search_transactions(df, query="coffee")

        assert result["total_matches"] == 4
        assert result["returned"] == 4
        assert result["truncated"] is False
        assert [txn["status"] for txn in result["transactions"]] == [
            "posted",
            "posted",
            "posted",
            "posted",
        ]
        assert [txn["txn_type"] for txn in result["transactions"]] == [
            "debit",
            "credit",
            "debit",
            "debit",
        ]

    def test_query_searches_sub_description_too(self):
        df = make_enriched_df()
        result = search_transactions(df, query="toronto on")

        assert result["total_matches"] == 1
        assert result["transactions"][0]["merchant"] == "Uber Canada/UberTrip"

    def test_explicit_filters_support_pending_credit_and_category(self):
        df = make_enriched_df()
        result = search_transactions(
            df,
            status="pending",
            txn_type="debit",
            category="rideshare",
        )

        assert result["total_matches"] == 1
        assert result["transactions"][0]["merchant"] == "Uber Canada/UberTrip"

    def test_merchant_filter_uses_normalized_brand_key(self):
        df = make_enriched_df()
        result = search_transactions(df, merchant="Tim Hortons")

        assert result["total_matches"] == 3
        assert {txn["merchant"] for txn in result["transactions"]} == {
            "Tim Hortons #0226",
            "TIM HORTONS #0226",
        }

    def test_respects_limit_and_sorts_newest_first(self):
        df = make_enriched_df()
        result = search_transactions(df, max_amount=50, limit=2)

        assert result["total_matches"] == 7
        assert result["returned"] == 2
        assert result["truncated"] is True
        assert [txn["date"] for txn in result["transactions"]] == [
            "2026-04-02",
            "2026-04-01",
        ]


class TestToolSchemasAndDispatch:
    def test_schema_names_match_registry_keys(self):
        schema_names = {tool["function"]["name"] for tool in TOOLS_SCHEMA}
        assert schema_names == set(TOOL_REGISTRY)

    def test_schema_disables_unknown_parameters(self):
        for tool in TOOLS_SCHEMA:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert params["additionalProperties"] is False

    def test_call_tool_dispatches_dict_arguments(self):
        df = make_enriched_df()
        result = call_tool(
            "get_spending_by_category",
            df,
            {"month": "2026-03"},
        )

        assert result["month"] == "2026-03"
        assert result["total"] == 42.10

    def test_call_tool_accepts_json_string_arguments(self):
        df = make_enriched_df()
        result = call_tool(
            "search_transactions",
            df,
            '{"status": "pending", "txn_type": "debit"}',
        )

        assert result["total_matches"] == 1
        assert result["transactions"][0]["merchant"] == "Uber Canada/UberTrip"

    def test_call_tool_rejects_unknown_tool_name(self):
        df = make_enriched_df()

        try:
            call_tool("does_not_exist", df, {})
        except ValueError as e:
            assert "Unknown tool" in str(e)
        else:
            raise AssertionError("Expected ValueError for unknown tool")

    def test_call_tool_rejects_df_in_user_arguments(self):
        df = make_enriched_df()

        try:
            call_tool("get_top_merchants", df, {"df": "not allowed"})
        except ValueError as e:
            assert "must not include 'df'" in str(e)
        else:
            raise AssertionError("Expected ValueError when user arguments include df")
