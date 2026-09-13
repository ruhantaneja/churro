import os
from typing import Any

from churro.providers.openai_compat import (
    extract_text,
    parse_completion_response,
    translate_messages,
    translate_tools,
)
from churro.providers.provider import (
    APIRequestError,
    AuthenticationError,
    ChatMessage,
    MissingAPIKeyError,
    Provider,
    ProviderError,
)
from churro.providers.responses import ProviderResponse, ToolResultMessage
from churro.providers.tool_definition import ToolDefinition

DEFAULT_MODEL = "gpt-4o"

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError as _OpenAIAuthenticationError,
        OpenAI,
        OpenAIError,
        PermissionDeniedError,
    )
except ImportError:  # pragma: no cover - exercised only when SDK is absent
    OpenAI = None
    OpenAIError = Exception
    _OpenAIAuthenticationError = Exception
    PermissionDeniedError = Exception
    APIConnectionError = Exception
    APITimeoutError = Exception

_ENV_KEY = "OPENAI_API_KEY"


class OpenAIProvider(Provider):
    """Chat Completions provider backed by the official openai SDK.

    Default model ``gpt-4o`` remains generally available through the API
    (the OpenAI retirement plan announced 2026-01-29 affects ChatGPT only).
    Override with ``OpenAIProvider(model="...")``.

    All OpenAI wire-format translation lives in
    :mod:`churro.providers.openai_compat`, shared with
    :class:`~churro.providers.ollama_provider.OllamaProvider`.
    """

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: Any | None = None,
    ):
        self.model = model
        self._client = client if client is not None else self._build_client(api_key)

    def _build_client(self, client_api_key: str | None) -> Any:
        if OpenAI is None:
            raise ProviderError(
                "The 'openai' package is not installed. Run: pip install openai"
            )

        api_key = client_api_key or os.getenv(_ENV_KEY)
        if not api_key:
            raise MissingAPIKeyError(
                f"{_ENV_KEY} is not set. Set it in your environment before using OpenAI."
            )

        return OpenAI(api_key=api_key)

    def send(self, messages: list[ChatMessage]) -> str:
        if not messages:
            raise ValueError("send() requires a non-empty list of messages")

        response = self._create_completion(messages)
        return extract_text(response, "OpenAI")

    def send_normalized(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        tool_results: list[ToolResultMessage] | None = None,
    ) -> ProviderResponse:
        """Send messages with the OpenAI tool-calling format and return a
        normalized :class:`ProviderResponse`.

        Normalized messages are translated into native OpenAI messages:
        ordinary system/user/assistant texts pass through, an assistant
        message carrying ``tool_calls`` becomes an OpenAI assistant message
        with a native ``tool_calls`` array, and ``tool_results`` become
        native ``tool`` messages attached to their call IDs. Translation is
        delegated to :mod:`churro.providers.openai_compat` (shared with the
        Ollama adapter). The method never executes tools, never touches the
        ToolRegistry, and never parses chat-handoff state -- those are CHURRO
        core responsibilities.
        """
        if not messages:
            raise ValueError("send_normalized() requires a non-empty list of messages")

        extra: dict[str, Any] | None = None
        if tools:
            extra = {
                "tools": translate_tools(tools),
                "tool_choice": "auto",
            }

        native_messages = translate_messages(messages, tool_results)
        response = self._create_completion(native_messages, extra)
        return parse_completion_response(response, "OpenAI")

    def _create_completion(
        self,
        messages: list[ChatMessage],
        extra: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if extra:
            kwargs.update(extra)
        try:
            return self._client.chat.completions.create(**kwargs)
        except (_OpenAIAuthenticationError, PermissionDeniedError) as exc:
            raise AuthenticationError(
                f"OpenAI rejected the API key or credentials: {exc}"
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise APIRequestError(f"OpenAI network request failed: {exc}") from exc
        except OpenAIError as exc:
            raise APIRequestError(f"OpenAI API request failed: {exc}") from exc