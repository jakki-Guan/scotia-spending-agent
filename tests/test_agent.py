"""Tests for the tool-calling spending agent."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from scotia_agent.agent import SpendingAgent, format_tool_trace


def make_agent_df() -> pd.DataFrame:
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
                "description": "Starbucks 123",
                "sub_description": "Toronto ON",
                "status": "posted",
                "txn_type": "debit",
                "amount": 4.50,
                "month": "2026-04",
                "category": "coffee",
                "confidence": 1.0,
                "cat_source": "rule",
            },
        ]
    )


def make_tool_call(name: str, arguments: dict | str, tool_id: str = "call_1"):
    raw_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=raw_arguments),
    )


def make_response(*, content: str = "", tool_calls=None):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=list(tool_calls or []),
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Fake client ran out of scripted responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class TestSpendingAgent:
    def test_agent_can_chain_multiple_tools_before_answering(self):
        df = make_agent_df()
        client = FakeClient(
            [
                make_response(
                    tool_calls=[
                        make_tool_call(
                            "get_monthly_trend",
                            {"category": "coffee"},
                            tool_id="call_trend",
                        )
                    ]
                ),
                make_response(
                    tool_calls=[
                        make_tool_call(
                            "get_top_merchants",
                            {"category": "coffee", "month": "2026-04", "n": 1},
                            tool_id="call_top",
                        )
                    ]
                ),
                make_response(
                    content=(
                        "Coffee spending rose from 5.25 in 2026-03 to 10.50 in 2026-04, "
                        "driven mainly by Tim Hortons."
                    )
                ),
            ]
        )
        agent = SpendingAgent(df, client=client, model="fake-model")

        result = agent.ask("Has my coffee spending increased recently, and where is it going?")

        assert result.stop_reason == "final_answer"
        assert result.iterations == 3
        assert [step.name for step in result.tool_trace] == [
            "get_monthly_trend",
            "get_top_merchants",
        ]
        assert "Coffee spending rose" in result.answer

        second_call_messages = client.completions.calls[1]["messages"]
        assert any(message["role"] == "tool" for message in second_call_messages)

    def test_agent_can_answer_without_tools(self):
        df = make_agent_df()
        client = FakeClient([make_response(content="You spent 15.75 total on coffee in these rows.")])
        agent = SpendingAgent(df, client=client, model="fake-model")

        result = agent.ask("How much coffee did I spend?")

        assert result.answer == "You spent 15.75 total on coffee in these rows."
        assert result.tool_trace == []
        assert result.stop_reason == "final_answer"

    def test_agent_records_tool_errors_in_trace_instead_of_crashing(self):
        df = make_agent_df()
        client = FakeClient(
            [
                make_response(
                    tool_calls=[
                        make_tool_call(
                            "search_transactions",
                            '{"status": "pending"',
                            tool_id="call_bad_json",
                        )
                    ]
                ),
                make_response(content="I hit a tool error while searching, so I cannot confirm pending rows."),
            ]
        )
        agent = SpendingAgent(df, client=client, model="fake-model")

        result = agent.ask("Do I have any pending transactions?")

        assert result.stop_reason == "final_answer"
        assert len(result.tool_trace) == 1
        assert result.tool_trace[0].name == "search_transactions"
        assert result.tool_trace[0].error is not None
        assert result.tool_trace[0].arguments == {"_raw": '{"status": "pending"'}

        rendered = format_tool_trace(result.tool_trace)
        assert "error:" in rendered
