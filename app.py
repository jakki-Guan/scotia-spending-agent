"""Gradio UI for the Scotia Spending Agent.

This app is intentionally thin: upload a Scotia CSV, ask a question,
show the final answer, and expose the tool trace so users can see how
the agent reasoned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scotia_agent.agent import SpendingAgent, ToolTrace, load_agent_dataframe

APP_CSS = """
.gradio-container {
  max-width: 1500px !important;
}

#hero-copy p {
  color: #d7d3cb;
  max-width: 900px;
}

#sidebar-card,
#conversation-card,
#trace-card {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015));
}

#status-card {
  border: 1px solid rgba(255, 153, 51, 0.22);
  border-radius: 14px;
  background: rgba(255, 153, 51, 0.08);
  padding: 10px 14px;
}

#dataset-card {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.02);
  padding: 4px 10px;
}
"""


def _empty_session() -> dict[str, Any]:
    return {
        "csv_path": None,
        "df": None,
        "errors": [],
    }


def build_dataset_summary(df, errors: list[dict[str, Any]], csv_path: str | None = None) -> str:
    """Render a short markdown summary of the loaded dataset."""
    title = "### Dataset Loaded"
    source = f"- Source: `{Path(csv_path).name}`\n" if csv_path else ""

    if df is None or df.empty:
        return f"{title}\n\n{source}- No valid transaction rows were loaded."

    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    months = sorted(df["month"].dropna().unique().tolist())
    month_preview = ", ".join(months[:8])
    if len(months) > 8:
        month_preview += ", ..."

    categories = sorted(df["category"].dropna().unique().tolist())
    category_preview = ", ".join(categories[:10])
    if len(categories) > 10:
        category_preview += ", ..."

    warnings = ""
    if errors:
        warnings = f"- Validation warnings: {len(errors)} row(s) were excluded\n"

    return (
        f"{title}\n\n"
        f"{source}"
        f"- Rows available to the agent: {len(df)}\n"
        f"- Date range: {min_date} to {max_date}\n"
        f"- Months: {month_preview}\n"
        f"- Categories: {category_preview}\n"
        f"{warnings}"
    )


def build_status_markdown(status: str, heading: str = "Status") -> str:
    """Render a compact status card."""
    return f"### {heading}\n\n{status}"


def _format_currency(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_arguments(arguments: dict[str, Any]) -> str:
    """Render tool arguments in a compact human-readable form."""
    if not arguments:
        return "none"
    parts = [f"{key}={value}" for key, value in arguments.items()]
    return ", ".join(parts)


def _truncate_text(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_tool_step(step: ToolTrace) -> str:
    """Return a short, readable summary for one tool result."""
    if step.error:
        return f"Tool failed: {step.error}"

    result = step.result or {}

    if step.name == "get_spending_by_category":
        by_category = result.get("by_category", {})
        top_categories = list(by_category.items())[:3]
        top_text = ", ".join(f"{cat} {_format_currency(amount)}" for cat, amount in top_categories)
        return (
            f"Total spending {_format_currency(result.get('total'))}. "
            f"Top categories: {top_text or 'none'}."
        )

    if step.name == "get_top_merchants":
        merchants = result.get("merchants", [])
        if not merchants:
            return "No merchants matched the current filters."
        leader = merchants[0]
        return (
            f"Found {len(merchants)} merchant(s). "
            f"Top merchant: {leader.get('merchant')} at {_format_currency(leader.get('amount'))}."
        )

    if step.name == "get_monthly_trend":
        trend = result.get("trend", [])
        if not trend:
            return "No monthly trend data matched the current filters."
        latest = trend[-1]
        return (
            f"{len(trend)} month(s) in trend. "
            f"Latest point: {latest.get('month')} at {_format_currency(latest.get('amount'))}."
        )

    if step.name == "get_grouped_category_trend":
        trend = result.get("trend", [])
        categories = result.get("categories", [])
        latest = trend[-1] if trend else {}
        category_text = ", ".join(categories[:4])
        if len(categories) > 4:
            category_text += ", ..."
        latest_text = (
            f" Latest point: {latest.get('month')} at {_format_currency(latest.get('amount'))}."
            if latest
            else ""
        )
        return (
            f"Grouped trend for {result.get('group')} across {len(categories)} categories "
            f"({category_text}).{latest_text}"
        )

    if step.name == "search_transactions":
        total = result.get("total_matches", 0)
        returned = result.get("returned", 0)
        truncated = result.get("truncated", False)
        suffix = " Results were truncated." if truncated else ""
        return f"Matched {total} transaction(s), showing {returned}.{suffix}"

    return "Tool completed successfully."


def render_tool_trace_markdown(tool_trace: list[ToolTrace]) -> str:
    """Render the agent's tool trace as readable markdown."""
    if not tool_trace:
        return "### Tool Trace\n\nNo tools were called."

    lines = ["### Tool Trace", ""]
    for i, step in enumerate(tool_trace, start=1):
        lines.append(f"**{i}. `{step.name}`**")
        lines.append(f"- Arguments: `{_format_arguments(step.arguments)}`")
        lines.append(f"- Summary: {_summarize_tool_step(step)}")
        if step.error:
            raw_error = _truncate_text(step.error)
            lines.append(f"- Error detail: `{raw_error}`")
        else:
            raw_result = _truncate_text(json.dumps(step.result, ensure_ascii=True))
            lines.append(f"- Raw result: `{raw_result}`")
        lines.append("")
    return "\n".join(lines)


def handle_upload(file_path: str | None) -> tuple[dict[str, Any], str, list[dict[str, str]], str, str]:
    """Load the CSV and reset the app state for a fresh session."""
    if not file_path:
        return (
            _empty_session(),
            "### Dataset\n\nUpload a Scotia CSV to begin.",
            [],
            "### Tool Trace\n\nNo run yet.",
            build_status_markdown("No file selected."),
        )

    df, errors = load_agent_dataframe(file_path)
    session = {
        "csv_path": file_path,
        "df": df,
        "errors": errors,
    }
    summary = build_dataset_summary(df, errors, file_path)
    if errors:
        status_text = (
            f"Loaded `{Path(file_path).name}` with {len(errors)} validation warning(s). "
            "The excluded rows will not be used by the agent."
        )
    else:
        status_text = f"Loaded `{Path(file_path).name}` successfully and the dataset is ready for questions."
    return session, summary, [], "### Tool Trace\n\nNo run yet.", build_status_markdown(
        status_text,
        heading="Run Status",
    )


def handle_question(
    question: str,
    session: dict[str, Any] | None,
    history: list[dict[str, str]] | None,
) -> tuple[list[dict[str, str]], str, str]:
    """Run one user question through the agent and update the chat history."""
    history = list(history or [])
    if not question or not question.strip():
        return history, "### Tool Trace\n\nNo run yet.", build_status_markdown("Please enter a question.")

    if not session or session.get("df") is None:
        return (
            history,
            "### Tool Trace\n\nNo run yet.",
            build_status_markdown("Upload a CSV before asking questions."),
        )

    agent = SpendingAgent(session["df"])
    result = agent.ask(question)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result.answer})

    trace_md = render_tool_trace_markdown(result.tool_trace)
    status = build_status_markdown(
        f"Answered in {result.iterations} iteration(s); stop reason: `{result.stop_reason}`.",
        heading="Run Status",
    )
    return history, trace_md, status


def clear_chat() -> tuple[list[dict[str, str]], str, str]:
    """Clear only the conversational part of the session."""
    return (
        [],
        "### Tool Trace\n\nNo run yet.",
        build_status_markdown("Chat cleared. Dataset is still loaded.", heading="Run Status"),
    )


def build_demo():
    """Construct the Gradio Blocks UI.

    Imported lazily so the rest of the module stays testable even when the
    optional `ui` dependency group is not installed.
    """
    try:
        import gradio as gr
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Gradio is not installed. Run `uv sync --group ui` before launching the app."
        ) from e

    with gr.Blocks(title="Scotia Spending Agent") as demo:
        session_state = gr.State(_empty_session())
        history_state = gr.State([])

        gr.Markdown(
            """
            # Scotia Spending Agent
            Upload a Scotiabank CSV, then ask a question about your spending.  
            The app answers with structured tool evidence rather than pure free-form chat.
            """,
            elem_id="hero-copy",
        )

        with gr.Row():
            with gr.Column(scale=1, elem_id="sidebar-card"):
                gr.Markdown(
                    """
                    ### 1. Load Data
                    Upload your Scotia CSV once. The app will validate rows, prepare categories,
                    and make the dataset available to the agent.
                    """
                )
                file_input = gr.File(label="Upload Scotia CSV", type="filepath")
                load_button = gr.Button("Load CSV", variant="primary")
                dataset_summary = gr.Markdown(
                    "### Dataset\n\nUpload a Scotia CSV to begin.",
                    elem_id="dataset-card",
                )
                status_box = gr.Markdown(
                    build_status_markdown("No file loaded yet."),
                    elem_id="status-card",
                )

            with gr.Column(scale=2, elem_id="conversation-card"):
                chatbot = gr.Chatbot(label="Conversation", height=460)
                question_box = gr.Textbox(
                    label="2. Ask a question",
                    placeholder="e.g. Has my coffee spending increased recently?",
                    lines=2,
                )
                with gr.Row():
                    ask_button = gr.Button("Ask", variant="primary")
                    clear_button = gr.Button("Clear Chat")
                with gr.Accordion("How The Agent Worked", open=True, elem_id="trace-card"):
                    trace_box = gr.Markdown("### Tool Trace\n\nNo run yet.")

        example_questions = gr.Examples(
            examples=[
                ["How much did I spend on coffee last month?"],
                ["Has my coffee spending increased recently?"],
                ["What do my dining costs look like each month?"],
                ["What are my subscription costs for each month?"],
                ["What were my top merchants in 2026-03?"],
                ["Do I have any pending transactions?"],
            ],
            inputs=question_box,
            label="Example questions",
        )

        load_button.click(
            fn=handle_upload,
            inputs=file_input,
            outputs=[session_state, dataset_summary, chatbot, trace_box, status_box],
        )

        ask_button.click(
            fn=handle_question,
            inputs=[question_box, session_state, history_state],
            outputs=[chatbot, trace_box, status_box],
        ).then(
            fn=lambda history: history,
            inputs=chatbot,
            outputs=history_state,
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=question_box,
        )

        question_box.submit(
            fn=handle_question,
            inputs=[question_box, session_state, history_state],
            outputs=[chatbot, trace_box, status_box],
        ).then(
            fn=lambda history: history,
            inputs=chatbot,
            outputs=history_state,
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=question_box,
        )

        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, trace_box, status_box],
        ).then(
            fn=lambda: [],
            inputs=None,
            outputs=history_state,
        )

    return demo


def main() -> None:
    """Launch the Gradio app."""
    demo = build_demo()
    demo.launch(css=APP_CSS)


if __name__ == "__main__":
    main()
