"""Tests for the Gradio app helper layer.

These tests intentionally avoid importing Gradio itself. They validate
the pure helper functions that power the UI so the app remains testable
even when the optional `ui` dependency group is not installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import app
from scotia_agent.agent import ToolTrace


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
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


class TestAppHelpers:
    def test_build_dataset_summary_includes_core_metadata(self):
        summary = app.build_dataset_summary(make_df(), [], "data/raw/example.csv")

        assert "example.csv" in summary
        assert "2026-03-01 to 2026-04-02" in summary
        assert "Rows available to the agent: 2" in summary

    def test_render_tool_trace_markdown_handles_empty_trace(self):
        rendered = app.render_tool_trace_markdown([])
        assert "No tools were called." in rendered

    def test_render_tool_trace_markdown_renders_arguments_and_result(self):
        rendered = app.render_tool_trace_markdown(
            [
                ToolTrace(
                    name="get_monthly_trend",
                    arguments={"category": "coffee"},
                    result={"trend": [{"month": "2026-03", "amount": 5.25}]},
                )
            ]
        )
        assert "get_monthly_trend" in rendered
        assert "category=coffee" in rendered
        assert "Latest point: 2026-03 at $5.25." in rendered
        assert "Raw result" in rendered

    def test_render_tool_trace_markdown_summarizes_spending_by_category(self):
        rendered = app.render_tool_trace_markdown(
            [
                ToolTrace(
                    name="get_spending_by_category",
                    arguments={"month": "2026-03"},
                    result={
                        "month": "2026-03",
                        "total": 42.10,
                        "by_category": {
                            "groceries": 32.10,
                            "coffee": 10.00,
                        },
                    },
                )
            ]
        )
        assert "Total spending $42.10." in rendered
        assert "Top categories: groceries $32.10, coffee $10.00." in rendered

    def test_render_tool_trace_markdown_summarizes_grouped_trend(self):
        rendered = app.render_tool_trace_markdown(
            [
                ToolTrace(
                    name="get_grouped_category_trend",
                    arguments={"group": "subscriptions"},
                    result={
                        "group": "subscriptions",
                        "categories": [
                            "subscription_ai",
                            "subscription_media",
                            "subscription_cloud",
                        ],
                        "trend": [
                            {"month": "2026-03", "amount": 32.99},
                            {"month": "2026-04", "amount": 35.00},
                        ],
                    },
                )
            ]
        )
        assert "Grouped trend for subscriptions" in rendered
        assert "Latest point: 2026-04 at $35.00." in rendered

    def test_handle_question_requires_loaded_dataset(self):
        history, trace, status = app.handle_question("How much did I spend?", None, [])

        assert history == []
        assert "No run yet" in trace
        assert "Upload a CSV" in status

    def test_handle_question_updates_history_and_trace(self, monkeypatch):
        fake_result = SimpleNamespace(
            answer="Coffee spending increased slightly.",
            tool_trace=[
                ToolTrace(
                    name="get_monthly_trend",
                    arguments={"category": "coffee"},
                    result={"trend": [{"month": "2026-03", "amount": 5.25}]},
                )
            ],
            iterations=2,
            stop_reason="final_answer",
        )

        class FakeAgent:
            def __init__(self, df):
                self.df = df

            def ask(self, question):
                return fake_result

        monkeypatch.setattr(app, "SpendingAgent", FakeAgent)

        history, trace, status = app.handle_question(
            "Has my coffee spending increased recently?",
            {"df": make_df(), "csv_path": "x.csv", "errors": []},
            [],
        )

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Coffee spending increased slightly."
        assert "get_monthly_trend" in trace
        assert "Answered in 2 iteration(s)" in status
        assert "Run Status" in status

    def test_build_status_markdown_wraps_status_message(self):
        rendered = app.build_status_markdown("Loaded successfully.", heading="Run Status")
        assert rendered.startswith("### Run Status")
        assert "Loaded successfully." in rendered
