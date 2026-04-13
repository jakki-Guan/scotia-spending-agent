"""Tool-calling spending analysis agent.

This module turns the project's pure data tools into a conversational
analyst. The agent does not compute spending metrics itself; it decides
which tool(s) to call, feeds those structured results back to the LLM,
and asks it to synthesize a grounded answer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from .config import settings
from .enrich import prepare_dataframe
from .parser import load_transactions, spending_only
from .tools import TOOLS_SCHEMA, call_tool

SYSTEM_PROMPT = """\
You are Scotia Spending Agent, a careful personal spending analyst.

You answer questions about one user's Scotiabank credit card data.
Use tools for factual claims. Do not invent numbers.

How to work:
- Prefer structured tool evidence over free-form guessing.
- Start broad, then drill down: category summary -> top merchants -> trend -> transaction search.
- Use search_transactions as the fallback lookup tool for pending rows, credits/refunds,
  sub-description/location questions, or when you need exact transactions.
- Multiple tool calls are allowed when the question needs comparison, drill-down, or evidence.
- If the data is insufficient, say so plainly.
- Keep final answers concise but insightful. Mention the main evidence behind your conclusion.
"""


@dataclass
class ToolTrace:
    """One executed tool call inside an agent run."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class AgentRunResult:
    """Final structured result from one user question."""

    question: str
    answer: str
    tool_trace: list[ToolTrace]
    iterations: int
    stop_reason: str


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Convert an SDK chat message object into a plain message dict."""
    payload: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", "") or "",
    }

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]

    return payload


def _build_dataset_context(df: pd.DataFrame) -> str:
    """Summarize the currently loaded dataset for the model."""
    if df.empty:
        return "Dataset is empty."

    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    all_months = sorted(df["month"].dropna().unique().tolist())
    spend = spending_only(df)
    categories = sorted(df["category"].dropna().unique().tolist())

    month_preview = ", ".join(all_months[:8])
    if len(all_months) > 8:
        month_preview += ", ..."

    category_preview = ", ".join(categories[:12])
    if len(categories) > 12:
        category_preview += ", ..."

    return (
        "Loaded transaction dataset summary:\n"
        f"- Date range: {min_date} to {max_date}\n"
        f"- Total rows: {len(df)}\n"
        f"- Posted debit rows: {len(spend)}\n"
        f"- Available months: {month_preview}\n"
        f"- Known categories: {category_preview}\n"
        "- Amount convention: debit is positive spending, credit is negative money in.\n"
        "- Search tool can inspect pending rows and credits; analytics tools focus on posted debit spending."
    )


def _safe_trace_arguments(raw_arguments: str | None) -> dict[str, Any]:
    """Best-effort decode of raw tool arguments for trace/debug output."""
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"_raw": raw_arguments}
    return parsed if isinstance(parsed, dict) else {"_raw": raw_arguments}


class SpendingAgent:
    """Multi-step tool-calling agent for spending questions."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        client: OpenAI | Any | None = None,
        model: str | None = None,
        max_iterations: int = 6,
    ):
        self.df = df
        self.client = client or OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = model or settings.llm_model
        self.max_iterations = max_iterations
        self.dataset_context = _build_dataset_context(df)

    def _create_completion(self, messages: list[dict[str, Any]]) -> Any:
        """Send one model turn with tool definitions attached."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1,
        )

    def ask(self, question: str) -> AgentRunResult:
        """Answer one user question, using tools as needed."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": self.dataset_context},
            {"role": "user", "content": question},
        ]
        trace: list[ToolTrace] = []

        for iteration in range(1, self.max_iterations + 1):
            response = self._create_completion(messages)
            message = response.choices[0].message
            messages.append(_message_to_dict(message))

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                answer = (getattr(message, "content", "") or "").strip()
                if not answer:
                    answer = "I could not produce a grounded answer from the available data."
                return AgentRunResult(
                    question=question,
                    answer=answer,
                    tool_trace=trace,
                    iterations=iteration,
                    stop_reason="final_answer",
                )

            for tool_call in tool_calls:
                name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                try:
                    result = call_tool(name=name, df=self.df, arguments=raw_arguments)
                    trace.append(
                        ToolTrace(
                            name=name,
                            arguments=_safe_trace_arguments(raw_arguments),
                            result=result,
                        )
                    )
                    tool_content = json.dumps(result, ensure_ascii=True)
                except Exception as e:  # noqa: BLE001 - tool error should feed back to model
                    error_payload = {
                        "error": type(e).__name__,
                        "message": str(e),
                    }
                    trace.append(
                        ToolTrace(
                            name=name,
                            arguments=_safe_trace_arguments(raw_arguments),
                            error=str(e),
                        )
                    )
                    tool_content = json.dumps(error_payload, ensure_ascii=True)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": tool_content,
                    }
                )

        return AgentRunResult(
            question=question,
            answer=(
                "I stopped after reaching the tool-iteration limit. "
                "Please ask a narrower question or inspect the tool trace."
            ),
            tool_trace=trace,
            iterations=self.max_iterations,
            stop_reason="max_iterations",
        )


def format_tool_trace(tool_trace: list[ToolTrace]) -> str:
    """Render a readable tool trace for CLI or debugging output."""
    if not tool_trace:
        return "No tools were called."

    lines: list[str] = []
    for i, step in enumerate(tool_trace, start=1):
        lines.append(f"{i}. {step.name}({json.dumps(step.arguments, ensure_ascii=True)})")
        if step.error:
            lines.append(f"   error: {step.error}")
        else:
            lines.append(f"   result: {json.dumps(step.result, ensure_ascii=True)}")
    return "\n".join(lines)


def load_agent_dataframe(csv_path: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load CSV and enrich it into the agent-ready DataFrame."""
    raw_df, errors = load_transactions(csv_path)
    return prepare_dataframe(raw_df), errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for local debugging of the agent loop."""
    parser = argparse.ArgumentParser(description="Run the Scotia spending agent on a CSV.")
    parser.add_argument("question", help="Natural-language question to ask about the data.")
    parser.add_argument("--csv", required=True, help="Path to the Scotia CSV export.")
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the tool trace before the final answer.",
    )
    args = parser.parse_args(argv)

    df, errors = load_agent_dataframe(args.csv)
    agent = SpendingAgent(df)
    result = agent.ask(args.question)

    if errors:
        print(f"Validation warnings: {len(errors)} row(s) failed schema checks and were excluded.")
    if args.show_trace:
        print(format_tool_trace(result.tool_trace))
        print()
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
