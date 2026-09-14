"""Offline tests for the Ollama provider.

Ollama is never contacted: the ``client`` callable is injected with
scripted responses, and the stdlib ``urllib`` error branches are exercised
by monkeypatching ``urllib.request.urlopen``. Nothing here requires a
running Ollama server.
"""

import os
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import churro.providers.ollama_provider as ollama_module
from churro.agent.runner import AgentRunner
from churro.providers.ollama_provider import OllamaProvider
from churro.providers.provider import (
    APIRequestError,
    Provider,
    ProviderError,
    ToolArgumentsError,
    UnexpectedResponseError,
)
from churro.providers.responses import ToolCall, ToolResultMessage
from churro.providers.tool_definition import ToolDefinition
from churro.tools import ToolRegistry
from churro.tools.tool import Tool, ToolArgs, ToolResult

DEFAULT_BASE = "http://localhost:11434/v1"

ADDER_DEF = ToolDefinition(
    name="adder",
    description="Adds two integers together.",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
)


class AdderArgs(ToolArgs):
    a: int
    b: int


class AdderTool(Tool):
    name = "adder"
    description = "Adds two integers together."
    args_model = AdderArgs

    def _run(self, args: AdderArgs) -> ToolResult:
        return ToolResult(success=True, output=str(args.a + args.b))


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def completion(content="", tool_calls=None, finish_reason="stop", usage=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}], "usage": usage}


def usage(prompt, completion, total):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


class RecordingClient:
    def __init__(self, responses=None):
        self.responses = responses if responses is not None else [completion("hi")]
        self.recorded = []

    def __call__(self, url, payload, timeout):
        self.recorded.append({"url": url, "payload": payload, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@contextmanager
def env_as(**updates):
    saved = {}
    keys = set(updates) | {"OLLAMA_MODEL", "OLLAMA_BASE_URL"}
    for key in keys:
        saved[key] = os.environ.get(key)
        os.environ.pop(key, None)
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


# ----------------------------------------------------------------- config
def test_default_base_url():
    with env_as():
        provider = OllamaProvider(model="llama3")
        assert provider.base_url == DEFAULT_BASE


def test_custom_base_url_strips_trailing_slash():
    with env_as():
        provider = OllamaProvider(model="llama3", base_url="http://127.0.0.1:11434/v1/")
        assert provider.base_url == "http://127.0.0.1:11434/v1"


def test_base_url_from_environment():
    with env_as(OLLAMA_BASE_URL="http://192.168.1.50:8000/v1"):
        provider = OllamaProvider(model="llama3")
        assert provider.base_url == "http://192.168.1.50:8000/v1"


def test_model_from_constructor():
    with env_as():
        provider = OllamaProvider(model="llama3.1")
        assert provider.model == "llama3.1"


def test_model_from_environment():
    with env_as(OLLAMA_MODEL="llama3-env"):
        provider = OllamaProvider()
        assert provider.model == "llama3-env"


def test_missing_model_raises_provider_error():
    with env_as():
        try:
            OllamaProvider()
        except ProviderError as exc:
            assert "OLLAMA_MODEL" in str(exc)
        else:
            raise AssertionError("expected ProviderError")


def test_is_a_provider_with_ollama_name():
    assert isinstance(OllamaProvider(model="llama3"), Provider)
    assert OllamaProvider(model="llama3").name == "ollama"


# ------------------------------------------------------------------- send
def test_send_posts_to_chat_completions_and_returns_text():
    with env_as():
        client = RecordingClient([completion("hello there")])
        provider = OllamaProvider(model="llama3", client=client)
        text = provider.send([{"role": "user", "content": "say hi"}])
        assert text == "hello there"
        assert len(client.recorded) == 1
        recorded = client.recorded[0]
        assert recorded["url"] == f"{DEFAULT_BASE}/chat/completions"
        assert recorded["payload"]["model"] == "llama3"
        assert recorded["payload"]["stream"] is False
        assert recorded["payload"]["messages"] == [{"role": "user", "content": "say hi"}]


def test_send_empty_messages_raises():
    with env_as():
        provider = OllamaProvider(model="llama3", client=RecordingClient())
        try:
            provider.send([])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_send_empty_text_raises():
    with env_as():
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("")]))
        try:
            provider.send([{"role": "user", "content": "hi"}])
        except UnexpectedResponseError:
            pass
        else:
            raise AssertionError("expected UnexpectedResponseError")


# ------------------------------------------------------- send_normalized
def test_send_normalized_returns_structured_text():
    with env_as():
        provider = OllamaProvider(
            model="llama3", client=RecordingClient([completion("hi", finish_reason="stop")])
        )
        result = provider.send_normalized([{"role": "user", "content": "hi"}])
        assert result.text == "hi"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage is None


def test_send_normalized_empty_messages_raises():
    with env_as():
        provider = OllamaProvider(model="llama3", client=RecordingClient())
        try:
            provider.send_normalized([])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_send_normalized_accepts_empty_text_when_no_tools():
    with env_as():
        provider = OllamaProvider(
            model="llama3", client=RecordingClient([completion("", finish_reason="stop")])
        )
        result = provider.send_normalized([{"role": "user", "content": "do nothing"}])
        assert result.text == ""
        assert result.tool_calls == []
        assert result.finish_reason == "stop"


def test_finish_reason_aliases_normalized():
    with env_as():
        client = RecordingClient([completion("", tool_calls=[tool_call("c", "f", "{}")], finish_reason="function_call")])
        provider = OllamaProvider(model="llama3", client=client)
        result = provider.send_normalized([{"role": "user", "content": "use a tool"}])
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1


def test_usage_parsed():
    with env_as():
        client = RecordingClient(
            [completion("done", finish_reason="stop", usage=usage(10, 3, 13))]
        )
        provider = OllamaProvider(model="llama3", client=client)
        result = provider.send_normalized([{"role": "user", "content": "work"}])
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 3
        assert result.usage.total_tokens == 13


def test_missing_model_not_sent_but_never_uses_keys():
    with env_as():
        client = RecordingClient([completion("ok")])
        provider = OllamaProvider(model="tiny", client=client)
        provider.send_normalized([{"role": "user", "content": "hi"}])
        assert "tools" not in client.recorded[0]["payload"]
        assert "tool_choice" not in client.recorded[0]["payload"]


# ------------------------------------------------------------- tool calls
def test_one_tool_call_parsed():
    with env_as():
        calls = [tool_call("call_1", "adder", '{"a": 2, "b": 3}')]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=calls)]))
        result = provider.send_normalized([{"role": "user", "content": "add"}])
        assert len(result.tool_calls) == 1
        actual = result.tool_calls[0]
        assert actual.id == "call_1"
        assert actual.name == "adder"
        assert actual.arguments == {"a": 2, "b": 3}


def test_multiple_tool_calls_parsed_in_order():
    with env_as():
        calls = [
            tool_call("call_1", "adder", '{"a": 1, "b": 1}'),
            tool_call("call_2", "adder", '{"a": 10, "b": 20}'),
        ]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=calls)]))
        result = provider.send_normalized([{"role": "user", "content": "add"}])
        assert [c.id for c in result.tool_calls] == ["call_1", "call_2"]
        assert result.tool_calls[1].arguments == {"a": 10, "b": 20}


def test_tool_call_json_argument_types_preserved():
    with env_as():
        calls = [tool_call("c", "f", '{"text": "x", "count": 3, "tags": ["a", "b"]}')]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=calls)]))
        result = provider.send_normalized([{"role": "user", "content": "go"}])
        assert result.tool_calls[0].arguments == {"text": "x", "count": 3, "tags": ["a", "b"]}


def test_empty_arguments_parse_to_empty_dict():
    with env_as():
        calls = [tool_call("c", "f", "")]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=calls)]))
        result = provider.send_normalized([{"role": "user", "content": "go"}])
        assert result.tool_calls[0].arguments == {}


def test_malformed_json_arguments_raise_tool_arguments_error():
    with env_as():
        calls = [tool_call("c", "adder", '{"a":')]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=calls)]))
        try:
            provider.send_normalized([{"role": "user", "content": "go"}])
        except ToolArgumentsError as exc:
            assert "adder" in str(exc)
        else:
            raise AssertionError("expected ToolArgumentsError")


def test_non_object_json_arguments_raise_tool_arguments_error():
    with env_as():
        calls = [tool_call("c", "f", "[1, 2, 3]")]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=calls)]))
        try:
            provider.send_normalized([{"role": "user", "content": "go"}])
        except ToolArgumentsError:
            pass
        else:
            raise AssertionError("expected ToolArgumentsError")


def test_missing_tool_call_id_raises():
    with env_as():
        bad = [{"type": "function", "function": {"name": "adder", "arguments": "{}"}}]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=bad)]))
        try:
            provider.send_normalized([{"role": "user", "content": "go"}])
        except UnexpectedResponseError as exc:
            assert "'id'" in str(exc)
        else:
            raise AssertionError("expected UnexpectedResponseError")


def test_missing_tool_call_name_raises():
    with env_as():
        bad = [{"id": "c", "type": "function", "function": {"arguments": "{}"}}]
        provider = OllamaProvider(model="llama3", client=RecordingClient([completion("", tool_calls=bad)]))
        try:
            provider.send_normalized([{"role": "user", "content": "go"}])
        except UnexpectedResponseError as exc:
            assert "function.name" in str(exc)
        else:
            raise AssertionError("expected UnexpectedResponseError")


# ------------------------------------------------------------ outbound
def test_tool_definitions_translated_into_payload():
    with env_as():
        client = RecordingClient([completion("ok")])
        provider = OllamaProvider(model="llama3", client=client)
        provider.send_normalized(
            [{"role": "user", "content": "add"}],
            tools=[ADDER_DEF],
        )
        payload = client.recorded[0]["payload"]
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "adder",
                    "description": "Adds two integers together.",
                    "parameters": ADDER_DEF.input_schema,
                },
            }
        ]
        assert "tool_choice" not in payload


def test_system_message_passthrough():
    with env_as():
        client = RecordingClient([completion("ok")])
        provider = OllamaProvider(model="llama3", client=client)
        provider.send_normalized([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])
        messages = client.recorded[0]["payload"]["messages"]
        assert messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ]


def test_assistant_tool_call_turn_built_natively():
    with env_as():
        client = RecordingClient([completion("ok")])
        provider = OllamaProvider(model="llama3", client=client)
        turn = ToolCall(id="call_1", name="adder", arguments={"a": 2, "b": 3})
        provider.send_normalized(
            [{"role": "assistant", "content": "", "tool_calls": [turn]}]
        )
        messages = client.recorded[0]["payload"]["messages"]
        assert messages == [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "adder", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            }
        ]


def test_tool_result_becomes_native_tool_message():
    with env_as():
        client = RecordingClient([completion("ok")])
        provider = OllamaProvider(model="llama3", client=client)
        result = ToolResultMessage(call_id="call_1", tool_name="adder", content="5", success=True)
        provider.send_normalized(
            [{"role": "user", "content": "add"}, {"role": "assistant", "content": "", "tool_calls": [ToolCall(id="call_1", name="adder", arguments={"a": 2, "b": 3})]}],
            tool_results=[result],
        )
        messages = client.recorded[0]["payload"]["messages"]
        assert {"role": "tool", "tool_call_id": "call_1", "content": "5"} in messages


def test_tool_results_attached_in_call_order_when_supplied_out_of_order():
    with env_as():
        client = RecordingClient([completion("ok")])
        provider = OllamaProvider(model="llama3", client=client)
        calls = [
            ToolCall(id="call_1", name="adder", arguments={"a": 1, "b": 1}),
            ToolCall(id="call_2", name="adder", arguments={"a": 2, "b": 2}),
        ]
        results = [
            ToolResultMessage(call_id="call_2", tool_name="adder", content="4", success=True),
            ToolResultMessage(call_id="call_1", tool_name="adder", content="2", success=True),
        ]
        provider.send_normalized(
            [{"role": "user", "content": "add"}, {"role": "assistant", "content": "", "tool_calls": calls}],
            tool_results=results,
        )
        messages = client.recorded[0]["payload"]["messages"]
        indexes = [messages.index({"role": "tool", "tool_call_id": call_id, "content": content})
                   for call_id, content in (("call_1", "2"), ("call_2", "4"))]
        assert indexes == sorted(indexes)
        assert indexes[0] < indexes[1]


def test_inputs_are_not_mutated():
    with env_as():
        client = RecordingClient([completion("", tool_calls=[tool_call("c", "f", '{"x": 1}')])])
        provider = OllamaProvider(model="llama3", client=client)
        original = [{"role": "user", "content": "hi"}]
        definitions = [ADDER_DEF]
        provider.send_normalized(original, tools=definitions)
        assert original == [{"role": "user", "content": "hi"}]
        assert definitions == [ADDER_DEF]


def test_no_per_adapter_formatting_duplication_in_source():
    repo = Path(__file__).resolve().parents[1]
    ollama_text = (repo / "churro" / "providers" / "ollama_provider.py").read_text(encoding="utf-8")
    assert "import urllib" in ollama_text
    assert "import ollama" not in ollama_text
    assert "from ollama" not in ollama_text
    assert "import openai" not in ollama_text
    assert "from openai" not in ollama_text


# ------------------------------------------------------------ failures
def test_no_choices_raises_unexpected_response_error():
    with env_as():
        provider = OllamaProvider(model="llama3", client=RecordingClient([{"choices": []}]))
        try:
            provider.send_normalized([{"role": "user", "content": "hi"}])
        except UnexpectedResponseError as exc:
            assert "Ollama" in str(exc)
        else:
            raise AssertionError("expected UnexpectedResponseError")


def test_missing_message_raises_unexpected_response_error():
    with env_as():
        provider = OllamaProvider(model="llama3", client=RecordingClient([{"choices": [{"finish_reason": "stop"}]}]))
        try:
            provider.send_normalized([{"role": "user", "content": "hi"}])
        except UnexpectedResponseError as exc:
            assert "choices[0].message" in str(exc)
        else:
            raise AssertionError("expected UnexpectedResponseError")


def test_http_error_maps_to_api_request_error():
    with env_as():
        provider = OllamaProvider(model="llama3")
        original = urllib.request.urlopen

        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError("http://x", 500, "Internal Server Error", None, None)

        try:
            urllib.request.urlopen = fake_urlopen
            try:
                provider.send([{"role": "user", "content": "hi"}])
            except APIRequestError as exc:
                assert "HTTP 500" in str(exc)
            else:
                raise AssertionError("expected APIRequestError")
        finally:
            urllib.request.urlopen = original


def test_connection_error_maps_to_api_request_error():
    with env_as():
        provider = OllamaProvider(model="llama3")
        original = urllib.request.urlopen

        def fake_urlopen(request, timeout):
            raise urllib.error.URLError("connection refused")

        try:
            urllib.request.urlopen = fake_urlopen
            try:
                provider.send([{"role": "user", "content": "hi"}])
            except APIRequestError as exc:
                assert "unavailable" in str(exc)
            else:
                raise AssertionError("expected APIRequestError")
        finally:
            urllib.request.urlopen = original


def test_non_json_body_maps_to_unexpected_response_error():
    with env_as():
        provider = OllamaProvider(model="llama3")

        class BadBody:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"definitely not json"

        original = urllib.request.urlopen

        def fake_urlopen(request, timeout):
            return BadBody()

        try:
            urllib.request.urlopen = fake_urlopen
            try:
                provider.send([{"role": "user", "content": "hi"}])
            except UnexpectedResponseError as exc:
                assert "non-JSON" in str(exc)
            else:
                raise AssertionError("expected UnexpectedResponseError")
        finally:
            urllib.request.urlopen = original


def test_client_plain_exception_wrapped_in_api_request_error():
    with env_as():
        def exploding(url, payload, timeout):
            raise TimeoutError("timed out")

        provider = OllamaProvider(model="llama3", client=exploding)
        try:
            provider.send_normalized([{"role": "user", "content": "hi"}])
        except APIRequestError as exc:
            assert "Ollama request failed" in str(exc)
        else:
            raise AssertionError("expected APIRequestError")


def test_client_provider_error_passes_through_unwrapped():
    with env_as():
        original = APIRequestError("already mapped")

        def raising(url, payload, timeout):
            raise original

        provider = OllamaProvider(model="llama3", client=raising)
        try:
            provider.send_normalized([{"role": "user", "content": "hi"}])
        except APIRequestError as exc:
            assert exc is original
        else:
            raise AssertionError("expected APIRequestError")


def test_keyboard_interrupt_propagates_unwrapped():
    """Step 18.9: KeyboardInterrupt must not be wrapped in APIRequestError.

    Ctrl+C during a long Ollama generation should surface as
    KeyboardInterrupt so ``_command_agent`` can display the correct
    "interrupted by Ctrl+C" message instead of swallowing it.
    """
    with env_as():

        def raising(url, payload, timeout):
            raise KeyboardInterrupt("aborted")

        provider = OllamaProvider(model="llama3", client=raising)
        try:
            provider.send_normalized([{"role": "user", "content": "hi"}])
        except KeyboardInterrupt as exc:
            assert "aborted" in str(exc)
        except APIRequestError:
            raise AssertionError(
                "KeyboardInterrupt should propagate, not be wrapped in APIRequestError"
            )
        else:
            raise AssertionError("expected KeyboardInterrupt")


# ------------------------------------------------------ agent integration
def test_agent_runner_ollama_full_tool_round():
    with env_as():
        first = completion(
            "",
            tool_calls=[tool_call("call_1", "adder", '{"a": 2, "b": 3}')],
            finish_reason="tool_calls",
        )
        second = completion("5", finish_reason="stop")
        client = RecordingClient([first, second])

        provider = OllamaProvider(model="llama3", client=client)
        registry = ToolRegistry()
        registry.register(AdderTool())
        runner = AgentRunner(provider=provider, registry=registry, max_iterations=4)

        result = runner.run(messages=[{"role": "user", "content": "add 2 and 3"}])

        assert result.completed is True
        assert result.final_text == "5"
        assert len(client.recorded) == 2
        assert len(result.tool_executions) == 1
        assert len(result.tool_results) == 1
        assert result.tool_results[0].call_id == "call_1"
        assert result.tool_results[0].content == "5"
        assert result.tool_results[0].success is True

        second_messages = client.recorded[1]["payload"]["messages"]
        assert second_messages == [
            {"role": "user", "content": "add 2 and 3"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "adder", "arguments": '{"a": 2, "b": 3}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "5"},
        ]


TEST_FUNCTIONS = [
    test_default_base_url,
    test_custom_base_url_strips_trailing_slash,
    test_base_url_from_environment,
    test_model_from_constructor,
    test_model_from_environment,
    test_missing_model_raises_provider_error,
    test_is_a_provider_with_ollama_name,
    test_send_posts_to_chat_completions_and_returns_text,
    test_send_empty_messages_raises,
    test_send_empty_text_raises,
    test_send_normalized_returns_structured_text,
    test_send_normalized_empty_messages_raises,
    test_send_normalized_accepts_empty_text_when_no_tools,
    test_finish_reason_aliases_normalized,
    test_usage_parsed,
    test_missing_model_not_sent_but_never_uses_keys,
    test_one_tool_call_parsed,
    test_multiple_tool_calls_parsed_in_order,
    test_tool_call_json_argument_types_preserved,
    test_empty_arguments_parse_to_empty_dict,
    test_malformed_json_arguments_raise_tool_arguments_error,
    test_non_object_json_arguments_raise_tool_arguments_error,
    test_missing_tool_call_id_raises,
    test_missing_tool_call_name_raises,
    test_tool_definitions_translated_into_payload,
    test_system_message_passthrough,
    test_assistant_tool_call_turn_built_natively,
    test_tool_result_becomes_native_tool_message,
    test_tool_results_attached_in_call_order_when_supplied_out_of_order,
    test_inputs_are_not_mutated,
    test_no_per_adapter_formatting_duplication_in_source,
    test_no_choices_raises_unexpected_response_error,
    test_missing_message_raises_unexpected_response_error,
    test_http_error_maps_to_api_request_error,
    test_connection_error_maps_to_api_request_error,
    test_non_json_body_maps_to_unexpected_response_error,
    test_client_plain_exception_wrapped_in_api_request_error,
    test_client_provider_error_passes_through_unwrapped,
    test_keyboard_interrupt_propagates_unwrapped,
    test_agent_runner_ollama_full_tool_round,
]


def main():
    failures = 0
    for test in TEST_FUNCTIONS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner reports
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(TEST_FUNCTIONS) - failures}/{len(TEST_FUNCTIONS)} Ollama provider tests passed.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()