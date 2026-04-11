# Scotia Spending Agent

> LLM-powered personal finance agent that analyzes Scotiabank transactions with tool-calling reasoning.

An end-to-end project demonstrating the path from raw bank CSVs to an interactive AI agent that can reason about spending patterns, call analysis tools on demand, and answer natural-language questions about personal finances.

## Status

🚧 **Phase 0: Environment setup — Complete**  
⏳ Phase 1: Core pipeline (parser → categorizer → visualizer)  
⏳ Phase 2: LLM integration with tool calling  
⏳ Phase 3: Gradio UI  
⏳ Phase 4: Deployment to Hugging Face Spaces  

## Project Vision

Not another LLM wrapper. The goal is a genuine **agent architecture**: the LLM decides which analysis tools to call, interprets their structured output, and composes insights — transparently showing its reasoning to the user.

## Tech Stack

- **Python 3.11** managed by [uv](https://github.com/astral-sh/uv)
- **pandas** + **matplotlib** for data processing and visualization
- **Anthropic Claude API** / **Ollama** (dual backend) for LLM reasoning
- **Gradio** for the interactive UI
- **pytest** + **ruff** for quality and style
- Runs in **WSL2 Ubuntu** for Linux-native development

## Project Structure

```
scotia-spending-agent/
├── src/                    # Source code (parser, categorizer, visualizer, agent, tools)
├── data/
│   ├── raw/                # Real bank CSVs — gitignored
│   └── sample_anonymized.csv  # Anonymized sample for demos
├── output/
│   ├── charts/             # Generated visualizations
│   └── reports/            # Generated text reports
├── tests/                  # pytest test suite
├── notebooks/              # Jupyter exploration
├── .vscode/                # Shared VS Code settings and extensions
├── pyproject.toml          # Project config + dependency groups
└── uv.lock                 # Locked dependency versions
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

```bash
git clone https://github.com/<your-username>/scotia-spending-agent.git
cd scotia-spending-agent
uv sync                    # Install core dependencies
uv sync --group dev        # Add dev tools (pytest, ruff, jupyter)
```

### Run

_Coming in Phase 1._

## Design Decisions

_Full rationale lives in [DESIGN.md](DESIGN.md) — to be added as the project grows._

Key early choices:
- **uv over conda/poetry**: speed, single tool, reproducible lockfile
- **Dependency groups**: core vs `llm` vs `ui` vs `dev` — lean production, rich development
- **WSL2 over native Windows**: matches Linux deployment targets, avoids Python tooling friction
- **Ruff over black + flake8 + isort**: one tool, 100x faster, same output

## Roadmap

See `PROJECT_CONTEXT.md` (local) for the full phased plan.

## License

MIT