"""Manual demo: makes a REAL OpenAI API call. Requires OPENAI_API_KEY.

Run with:  python examples\\demo_openai_provider.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from churro.prompts import load_system_prompt
from churro.providers.openai_provider import DEFAULT_MODEL, OpenAIProvider
from churro.providers.provider import (
    AuthenticationError,
    MissingAPIKeyError,
    ProviderError,
)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it in your environment and retry.")
        return 1

    provider = OpenAIProvider(model=DEFAULT_MODEL)
    system_prompt = load_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Look at the CHURRO project. Tell me one obvious bug.",},
    ]

    print(f"Calling OpenAI Chat Completions with model {provider.model}...")
    try:
        reply = provider.send(messages)
    except MissingAPIKeyError as exc:
        print(f"Config error: {exc}")
        return 1
    except AuthenticationError as exc:
        print(f"Authentication error: {exc}")
        return 1
    except ProviderError as exc:
        print(f"Provider error: {exc}")
        return 1

    print("\n--- Model reply ---\n")
    print(reply)


if __name__ == "__main__":
    raise SystemExit(main())