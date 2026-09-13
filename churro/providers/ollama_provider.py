import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from churro.providers.openai_compat import (
    extract_text,
    parse_completion_response,
    translate_messages,
    translate_tools,
)
from churro.providers.provider import (
    APIRequestError,
    ChatMessage,
    Provider,
    ProviderError,
    UnexpectedResponseError,
)
from churro.providers.responses import ProviderResponse, ToolResultMessage
from churro.providers.tool_definition import ToolDefinition

DEFAULT_BASE_URL = "http://localhost:11434/v1"

_ENV_BASE_URL = "OLLAMA_BASE_URL"
_ENV_MODEL = "OLLAMA_MODEL"

PostFn = Callable[[str, dict[str, Any], float], Any]


def _http_post(url: str, payload: dict[str, Any], timeout: float) -> Any:
    """Post a JSON payload and return the parsed JSON response body."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise APIRequestError(f"Ollama API request failed (HTTP {exc.code}): {exc}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise APIRequestError(
            f"Ollama is unavailable. Is the Ollama server running at this URL? {exc}"
        ) from exc

    try:
        return json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise UnexpectedResponseError("Ollama returned a non-JSON response body") from exc


class OllamaProvider(Provider):
    """Tool-calling provider backed by a local Ollama server.

    Talks to Ollama's OpenAI-compatible endpoint over plain HTTP using only
    the Python standard library -- no third-party SDK is required. The base
    URL and model are configurable through constructor arguments or the
    ``OLLAMA_BASE_URL`` / ``OLLAMA_MODEL`` environment variables.

    ``OllamaProvider(model="llama3.1")`` picks any model you already pulled
    locally. ``base_url`` defaults to ``http://localhost:11434/v1``.

    The ``client`` argument is an injectable callable
    ``(url, payload, timeout) -> response`` used by tests; it defaults to a
    stdlib ``urllib`` POST.
    """

    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        client: PostFn | None = None,
        timeout_seconds: float = 300.0,
    ):
        self.model = model or os.getenv(_ENV_MODEL) or ""
        if not self.model:
            raise ProviderError(
                f"{_ENV_MODEL} is not set. Set it in your environment or pass "
                "model='...' when constructing OllamaProvider."
            )

        base = base_url or os.getenv(_ENV_BASE_URL) or DEFAULT_BASE_URL
        self.base_url = base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._post = client if client is not None else _http_post

    def send(self, messages: list[ChatMessage]) -> str:
        if not messages:
            raise ValueError("send() requires a non-empty list of messages")

        response = self._post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [dict(message) for message in messages],
                "stream": False,
            },
            self.timeout_seconds,
        )
        return extract_text(response, "Ollama")

    def send_normalized(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        tool_results: list[ToolResultMessage] | None = None,
    ) -> ProviderResponse:
        """Send normalized messages using OpenAI's tool-calling wire format,
        which Ollama implements natively, and return a normalized response.

        ``client``-level provider errors (connection refused, HTTP errors) are
        mapped to :class:`APIRequestError`; malformed tool calls and
        non-JSON arguments produce controlled errors. The method never
        executes tools, never touches the ToolRegistry, and never parses
        chat-handoff state.
        """
        if not messages:
            raise ValueError("send_normalized() requires a non-empty list of messages")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": translate_messages(messages, tool_results),
            "stream": False,
        }
        if tools:
            payload["tools"] = translate_tools(tools)

        response = self._raw_respond(payload)
        return parse_completion_response(response, "Ollama")

    def _raw_respond(self, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}/chat/completions"
        try:
            return self._post(url, payload, self.timeout_seconds)
        except ProviderError:
            raise
        except Exception as exc:
            raise APIRequestError(f"Ollama request failed: {exc}") from exc