"""OpenAI-compatible wire-format translation shared by provider adapters.

Both :class:`OpenAIProvider` (the native OpenAI API) and
:class:`OllamaProvider` (Ollama's OpenAI-compatible endpoint) speak this
wire format, so the pure data translation lives here once. Everything in
this module is dependency-free: no SDK and no I/O. CHURRO core modules
(agent, tools) never import it.
"""

import json
from typing import Any

from churro.providers.provider import ToolArgumentsError, UnexpectedResponseError
from churro.providers.responses import (
    ProviderResponse,
    ToolCall,
    ToolResultMessage,
    Usage,
    normalize_finish_reason,
)
from churro.providers.tool_definition import ToolDefinition

_VALID_ROLES = ("system", "user", "assistant")


def get_attr_or_key(obj: Any, key: str) -> Any:
    """Attribute-or-dict accessor that tolerates both real SDK objects and
    plain dicts in test fakes."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def translate_tools(definitions: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Translate ``ToolDefinition`` objects into the native tools array."""
    return [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.input_schema,
            },
        }
        for definition in definitions
    ]


def extract_tool_call(call: Any) -> tuple[str, str, dict[str, Any]]:
    call_id = get_attr_or_key(call, "id")
    name = get_attr_or_key(call, "name")
    arguments = get_attr_or_key(call, "arguments")
    return (
        call_id if isinstance(call_id, str) else "",
        name if isinstance(name, str) else "",
        arguments if isinstance(arguments, dict) else {},
    )


def translate_tool_call(tool_call: Any) -> dict[str, Any]:
    """Translate a normalized ``ToolCall`` into a native function call."""
    call_id, name, arguments = extract_tool_call(tool_call)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


def translate_tool_result(result: ToolResultMessage) -> dict[str, Any]:
    """Translate a normalized tool result into a native tool message."""
    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": result.content,
    }


def translate_messages(
    messages: list[Any],
    tool_results: list[ToolResultMessage] | None,
) -> list[dict[str, Any]]:
    """Translate the normalized conversation log into native messages,
    interleaving each assistant tool-call turn with its matching tool
    results by call ID."""
    results_by_id: dict[str, ToolResultMessage] = {}
    for result in tool_results or []:
        results_by_id[result.call_id] = result

    built: list[dict[str, Any]] = []
    emitted: set[str] = set()

    for message in messages:
        role = get_attr_or_key(message, "role")
        content = get_attr_or_key(message, "content")
        tool_calls = get_attr_or_key(message, "tool_calls")

        if isinstance(tool_calls, list) and tool_calls:
            built.append(
                {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else "",
                    "tool_calls": [translate_tool_call(call) for call in tool_calls],
                }
            )
            for call in tool_calls:
                call_id, _, _ = extract_tool_call(call)
                result = results_by_id.get(call_id)
                if result is not None:
                    built.append(translate_tool_result(result))
                    emitted.add(call_id)
        else:
            built.append(
                {
                    "role": role if role in _VALID_ROLES else "user",
                    "content": content if isinstance(content, str) else "",
                }
            )

    for result in tool_results or []:
        if result.call_id not in emitted:
            built.append(translate_tool_result(result))

    return built


def extract_text(raw: Any, provider_label: str) -> str:
    """Return the first choice's text content for legacy ``send()`` paths.

    Raises ``UnexpectedResponseError`` when the shape is wrong or the
    content is empty/non-text.
    """
    choices = get_attr_or_key(raw, "choices")
    if not isinstance(choices, list) or not choices:
        raise UnexpectedResponseError(
            f"Unexpected response shape from {provider_label}; expected "
            "response.choices as a non-empty list"
        )
    try:
        choice = choices[0]
        message = get_attr_or_key(choice, "message")
    except (AttributeError, TypeError, IndexError) as exc:
        raise UnexpectedResponseError(
            f"Unexpected response shape from {provider_label}; expected "
            "response.choices[0].message"
        ) from exc
    if message is None:
        raise UnexpectedResponseError(
            f"Unexpected response shape from {provider_label}; expected "
            "response.choices[0].message"
        )

    content = get_attr_or_key(message, "content")
    if not isinstance(content, str) or content == "":
        raise UnexpectedResponseError(
            f"{provider_label} returned non-text or empty content in "
            "response.choices[0]"
        )
    return content


def parse_tool_calls(raw: Any, provider_label: str) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise UnexpectedResponseError(
            f"{provider_label} returned message.tool_calls that is not a list"
        )
    return [parse_tool_call(item, provider_label) for item in raw]


def parse_tool_call(raw_item: Any, provider_label: str) -> ToolCall:
    function = get_attr_or_key(raw_item, "function")
    call_id = get_attr_or_key(raw_item, "id")
    name = get_attr_or_key(function, "name")
    arguments_text = get_attr_or_key(function, "arguments")

    if not isinstance(call_id, str) or not call_id or call_id == "":
        raise UnexpectedResponseError(
            f"{provider_label} returned a tool call missing a non-empty 'id'"
        )
    if not isinstance(name, str) or not name or name == "":
        raise UnexpectedResponseError(
            f"{provider_label} returned a tool call missing a non-empty "
            "'function.name'"
        )

    return ToolCall(
        id=call_id,
        name=name,
        arguments=parse_arguments(name, arguments_text),
    )


def parse_arguments(name: str, raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        raise ToolArgumentsError(
            f"Tool call {name!r} arguments must be a JSON string, "
            f"got {type(raw).__name__}"
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolArgumentsError(
            f"Tool call {name!r} arguments are not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ToolArgumentsError(
            f"Tool call {name!r} arguments must be a JSON object, "
            f"got {type(parsed).__name__}"
        )
    return parsed


def parse_usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    try:
        return Usage(
            prompt_tokens=get_attr_or_key(raw, "prompt_tokens"),
            completion_tokens=get_attr_or_key(raw, "completion_tokens"),
            total_tokens=get_attr_or_key(raw, "total_tokens"),
        )
    except (TypeError, ValueError):
        return None


def parse_completion_response(raw: Any, provider_label: str) -> ProviderResponse:
    """Normalize a native completion response into a ``ProviderResponse``.

    Handles text, ``finish_reason``, ``usage``, and parsed ``ToolCall``
    objects; raises controlled errors for malformed shapes, missing tool
    call fields, and invalid tool arguments.
    """
    choices = get_attr_or_key(raw, "choices")
    if not isinstance(choices, list) or not choices:
        raise UnexpectedResponseError(
            f"Unexpected response shape from {provider_label}; expected "
            "response.choices as a non-empty list"
        )

    try:
        choice = choices[0]
        message = get_attr_or_key(choice, "message")
    except (AttributeError, TypeError, IndexError) as exc:
        raise UnexpectedResponseError(
            f"Unexpected response shape from {provider_label}; expected "
            "response.choices[0].message"
        ) from exc
    if message is None:
        raise UnexpectedResponseError(
            f"Unexpected response shape from {provider_label}; expected "
            "response.choices[0].message"
        )

    content = get_attr_or_key(message, "content")
    text = content if isinstance(content, str) else ""
    tool_calls = parse_tool_calls(get_attr_or_key(message, "tool_calls"), provider_label)
    finish_reason = normalize_finish_reason(get_attr_or_key(choice, "finish_reason"))
    usage = parse_usage(get_attr_or_key(raw, "usage"))

    return ProviderResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )