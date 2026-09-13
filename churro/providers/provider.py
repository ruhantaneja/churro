from abc import ABC, abstractmethod
from typing import Literal, NotRequired, TypedDict

from churro.providers.responses import (
    ProviderResponse,
    ToolCall,
    ToolResultMessage,
    normalize_finish_reason,
)
from churro.providers.tool_definition import ToolDefinition


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    tool_calls: NotRequired[list[ToolCall]]


class ProviderError(Exception):
    """Base class for every provider error surfaced to CHURRO."""


class MissingAPIKeyError(ProviderError):
    """No API key was configured for the provider."""


class AuthenticationError(ProviderError):
    """The provider rejected the supplied credentials."""


class APIRequestError(ProviderError):
    """The provider request failed (network, rate limit, server error...)."""


class UnexpectedResponseError(ProviderError):
    """The provider returned a response CHURRO does not understand."""


class ToolArgumentsError(ProviderError):
    """A tool call's arguments could not be interpreted.

    String args that are not valid JSON, JSON that is not an object, or
    arguments of the wrong type all raise this controlled error instead
    of surfacing a raw parsing exception.
    """


class Provider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def send(self, messages: list[ChatMessage]) -> str:
        """Send standard chat messages and return the model's text response.

        Responsibilities end here: no state mutation, no handoffs, no
        response parsing. The caller owns everything downstream.
        """

    def send_normalized(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        tool_results: list[ToolResultMessage] | None = None,
    ) -> ProviderResponse:
        """Send messages (optionally with tool definitions) and return a
        :class:`~churro.providers.responses.ProviderResponse`.

        ``tool_results`` carries the normalized results of a previous round
        of tool calls so a provider that supports tool calling can include
        them in its native request. The base implementation keeps full
        backwards compatibility: it calls :meth:`send` and wraps the plain
        text into a ``ProviderResponse``; the ``tools`` and ``tool_results``
        arguments are accepted but unused by text-only providers. Providers
        that understand tool calling should override this and translate
        their native response, including parsed ``ToolCall`` objects, so
        CHURRO's core never depends on provider types.
        """
        text = self.send(messages)
        return ProviderResponse(
            text=text, finish_reason=normalize_finish_reason("stop")
        )