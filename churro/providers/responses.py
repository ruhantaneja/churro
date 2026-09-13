from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FinishReason = Literal["tool_calls", "stop", "length", "content_filter"]


class ToolCall(BaseModel):
    """A normalized tool invocation produced by a provider.

    ``arguments`` is the structured argument object (never the raw JSON
    string) -- provider adapters are responsible for parsing.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Normalized token accounting; fields are optional where a provider
    does not report them."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ProviderResponse(BaseModel):
    """A provider response normalized for CHURRO's core.

    ``text`` carries the assistant's plain-text message. ``tool_calls``
    carries normalized ``ToolCall`` objects (empty when the model only
    talked). ``finish_reason`` is normalized via
    :func:`normalize_finish_reason`.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason | None = None
    usage: Usage | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ToolResultMessage(BaseModel):
    """A provider-neutral representation of one tool call's result.

    This is the result-side counterpart of ``ToolCall``: it carries the
    original call identity, the tool name, a human/model-readable
    ``content`` string, and the success flag. It deliberately contains no
    provider-specific message fields -- a provider adapter translates it
    into its native message format later.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    content: str = ""
    success: bool = True


_FINISH_REASON_ALIASES = {
    "function_call": "tool_calls",
    "tools": "tool_calls",
    "max_tokens": "length",
    "eos_token": "length",
    "end_turn": "stop",
    "stop_reason": "stop",
}


def normalize_finish_reason(raw: Any) -> FinishReason | None:
    """Lowercase and apply provider-neutral aliases to a finish reason.

    Unknown values are passed through lowercased so a future provider's
    wording never blocks normalization; None/blank stay None.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if not value:
        return None
    return _FINISH_REASON_ALIASES.get(value, value)  # type: ignore[return-value]