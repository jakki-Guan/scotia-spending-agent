"""One-shot LLM connectivity check.

Verifies that API key, base URL, and model are wired up correctly.
Run this once after setting up .env to confirm the LLM layer works
before writing any real categorization code.
"""

from openai import OpenAI

from scotia_agent.config import settings


def main() -> None:
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY not set — check your .env file")

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        max_tokens=10,
    )

    print(f"Endpoint:  {settings.llm_base_url}")
    print(f"Model:     {resp.model}")
    print(f"Response:  {resp.choices[0].message.content!r}")
    print(f"Tokens:    in={resp.usage.prompt_tokens}, out={resp.usage.completion_tokens}")
    print("\n✅ LLM backend is reachable.")


if __name__ == "__main__":
    main()
