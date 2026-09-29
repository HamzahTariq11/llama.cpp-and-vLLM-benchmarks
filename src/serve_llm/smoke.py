"""Send one chat request to an OpenAI-compatible endpoint and print the reply.

Usage:
    uv run python -m serve_llm.smoke "Hello" [--base-url URL] [--model NAME]
"""

import argparse

import httpx

# 127.0.0.1 rather than localhost: llama-server binds IPv4 only, and on Windows
# "localhost" tries ::1 first, adding a slow fallback to every connection.
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen2.5-1.5b-instruct"


def chat(
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 128,
    client: httpx.Client | None = None,
) -> str:
    """Send a single-turn chat completion request and return the reply text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    http = client or httpx.Client(timeout=120)
    try:
        response = http.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    finally:
        if client is None:
            http.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(chat(args.prompt, base_url=args.base_url, model=args.model))


if __name__ == "__main__":
    main()
