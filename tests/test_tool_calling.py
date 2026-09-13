"""Offline tests for normalized provider responses and tool calling.

Everything here mocks the OpenAI SDK client; nothing requires network
access or an API key. The core-safety test imports only the normalized
types and asserts OpenAI types never leak into CHURRO's core.
"""

import tempfile
from pathlib import Path

import churro.providers.openai_provider as openai_module
import churro.providers.responses as responses_module
from churro.providers.openai_provider import OpenAIProvider
from churro.providers.provider import (
    APIRequestError,
    Provider,
    ToolArgumentsError,
    UnexpectedResponseError,
)
from churro.providers.responses import (
    ProviderResponse,
    ToolCall,
    Usage,
    normalize_finish_reason,
)
from churro.providers.tool_definition import ToolDefinition
from churro.tools import ReadFileTool

DEFAULT_SYSTEM = [{"role": "system", "content": "You are a coding assistant."}]
USER_MSG = [{"role": "user", "content": "hi"}]
TOOL_MSG = [{"role": "user", "content": "read the file"}]


# ---------------------------------------------------------------- fakes
class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChoice:
    def __init__(self, content="", tool_calls=None, finish_reason="stop"):
        self.message = FakeMessage(content, tool_calls)
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, prompt=5, completion=7, total=12):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class FakeCompletion:
    def __init__(self, content="", tool_calls=None, finish_reason="stop", usage=None):
        self.choices = [FakeChoice(content, tool_calls, finish_reason)]
        self.usage = usage


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        item = self.owner.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeChat:
    def __init__(self, owner):
        self.completions = FakeCompletions(owner)


class FakeClient:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses if responses is not None else [FakeCompletion()]
        self._chat = None

    @property
    def chat(self):
        return FakeChat(self)


class FakeAuthError(Exception):
    pass


class FakeConnectionError(Exception):
    pass


class FakePlainError(Exception):
    pass


def make_provider(responses=None, **kwargs):
    return OpenAIProvider(client=FakeClient(responses=responses), **kwargs)


ROOT_DIR = Path(tempfile.mkdtemp())


def read_file_definition():
    tool = ReadFileTool(ROOT_DIR)
    return ToolDefinition.from_tool(tool)


# -------------------------------------------------------------- tests
def test_text_only_provider_response():
    provider = make_provider(
        responses=[FakeCompletion("Hello there", finish_reason="stop", usage=FakeUsage())]
    )
    resp = provider.send_normalized(USER_MSG)
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "Hello there"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert resp.usage == Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12)


def test_one_normalized_tool_call():
    raw_call = FakeToolCall("call_1", "read_file", '{"path": "test.py"}')
    provider = make_provider(
        responses=[FakeCompletion("", [raw_call], finish_reason="tool_calls")]
    )
    resp = provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "call_1"
    assert tc.name == "read_file"
    assert isinstance(tc.arguments, dict)
    assert tc.arguments == {"path": "test.py"}


def test_multiple_normalized_tool_calls():
    calls = [
        FakeToolCall("a", "read_file", '{"path": "a.py"}'),
        FakeToolCall("b", "list_files", "{}"),
    ]
    provider = make_provider(responses=[FakeCompletion("", calls, "tool_calls")])
    resp = provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
    assert [(tc.id, tc.name) for tc in resp.tool_calls] == [
        ("a", "read_file"),
        ("b", "list_files"),
    ]
    assert resp.tool_calls[0].arguments == {"path": "a.py"}
    assert resp.tool_calls[1].arguments == {}


def test_tool_call_arguments_are_structured_not_string():
    raw_call = FakeToolCall("c1", "search_files", '{"pattern": "*.py", "recursive": true}')
    provider = make_provider(responses=[FakeCompletion("", [raw_call], "tool_calls")])
    resp = provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
    args = resp.tool_calls[0].arguments
    assert isinstance(args, dict)
    assert args == {"pattern": "*.py", "recursive": True}
    assert args["recursive"] is True


def test_malformed_tool_arguments_controlled_error():
    calls = [FakeToolCall("c1", "write_file", '{"path": broken')]
    provider = make_provider(responses=[FakeCompletion("", calls, "tool_calls")])
    try:
        provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
        raise AssertionError("expected ToolArgumentsError")
    except ToolArgumentsError:
        pass


def test_non_object_tool_arguments_controlled_error():
    calls = [FakeToolCall("c1", "write_file", '"[1, 2, 3]"')]
    provider = make_provider(responses=[FakeCompletion("", calls, "tool_calls")])
    try:
        provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
        raise AssertionError("expected ToolArgumentsError")
    except ToolArgumentsError:
        pass


def test_non_string_tool_arguments_controlled_error():
    calls = [FakeToolCall("c1", "write_file", {"path": "x.py"})]
    provider = make_provider(responses=[FakeCompletion("", calls, "tool_calls")])
    try:
        provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
        raise AssertionError("expected ToolArgumentsError")
    except ToolArgumentsError:
        pass


def test_missing_tool_call_id_raises():
    calls = [{"function": {"name": "read_file", "arguments": "{}"}}]
    provider = make_provider(responses=[FakeCompletion("", calls, "tool_calls")])
    try:
        provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
        raise AssertionError("expected UnexpectedResponseError")
    except UnexpectedResponseError:
        pass


def test_missing_tool_call_function_raises():
    calls = [{"id": "c1"}]
    provider = make_provider(responses=[FakeCompletion("", calls, "tool_calls")])
    try:
        provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
        raise AssertionError("expected UnexpectedResponseError")
    except UnexpectedResponseError:
        pass


def test_response_text_preserved_with_tool_call():
    calls = [FakeToolCall("c1", "read_file", '{"path": "test.py"}')]
    provider = make_provider(
        responses=[FakeCompletion("I will read it now", calls, "tool_calls")]
    )
    resp = provider.send_normalized(TOOL_MSG, tools=[read_file_definition()])
    assert resp.text == "I will read it now"
    assert len(resp.tool_calls) == 1


def test_no_tool_calls_when_none_supplied():
    provider = make_provider(responses=[FakeCompletion("plain answer")])
    resp = provider.send_normalized(USER_MSG)
    assert resp.tool_calls == []
    assert not resp.has_tool_calls
    assert resp.text == "plain answer"


def test_finish_reason_normalization_aliases():
    assert normalize_finish_reason("function_call") == "tool_calls"
    assert normalize_finish_reason("tools") == "tool_calls"
    assert normalize_finish_reason("max_tokens") == "length"
    assert normalize_finish_reason("eos_token") == "length"
    assert normalize_finish_reason("end_turn") == "stop"
    assert normalize_finish_reason("stop") == "stop"
    assert normalize_finish_reason(None) is None
    assert normalize_finish_reason("") is None
    assert normalize_finish_reason("weird_new_value") == "weird_new_value"


def test_finish_reason_normalized_from_provider():
    provider = make_provider(
        responses=[FakeCompletion("===", finish_reason="function_call")]
    )
    resp = provider.send_normalized(USER_MSG)
    assert resp.finish_reason == "tool_calls"


def test_tooldefinition_contains_fields():
    td = ToolDefinition(name="read_file", description="reads a file",
                        input_schema={"type": "object"})
    assert td.name == "read_file"
    assert td.description == "reads a file"
    assert td.input_schema == {"type": "object"}


def test_tooldefinition_from_existing_tool_reuses_input_schema():
    tool = ReadFileTool(ROOT_DIR)
    td = ToolDefinition.from_tool(tool)
    assert td.name == tool.name == "read_file"
    assert td.description == tool.description
    assert td.input_schema == tool.input_schema
    assert "properties" in td.input_schema


def test_openai_tool_format_sent_to_sdk():
    provider = make_provider(responses=[FakeCompletion("ok")])
    tool_def = read_file_definition()
    provider.send_normalized(TOOL_MSG, tools=[tool_def])
    kwargs = provider._client.calls[0]
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["messages"] == TOOL_MSG
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.input_schema,
            },
        }
    ]


def test_provider_send_still_returns_plain_text():
    provider = make_provider(responses=[FakeCompletion("still text")])
    assert provider.send(USER_MSG) == "still text"
    assert "tools" not in provider._client.calls[0]


def test_core_consumes_normalized_without_openai_types():
    assert "openai" not in responses_module.__name__
    assert not hasattr(responses_module, "OpenAI")

    resp = ProviderResponse(
        text="",
        tool_calls=[ToolCall(id="call_1", name="read_file",
                             arguments={"path": "test.py"})],
        finish_reason="tool_calls",
    )

    def plan(response):
        return [ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in response.tool_calls]

    planned = plan(resp)
    assert len(planned) == 1
    assert planned[0].name == "read_file"
    assert planned[0].arguments == {"path": "test.py"}


def test_existing_error_handling_continues_in_send_normalized():
    original_auth = openai_module._OpenAIAuthenticationError
    original_conn = openai_module.APIConnectionError
    original_err = openai_module.OpenAIError
    try:
        openai_module._OpenAIAuthenticationError = FakeAuthError
        openai_module.APIConnectionError = FakeConnectionError
        openai_module.OpenAIError = FakePlainError

        provider = make_provider(responses=[FakeAuthError("nope")])
        try:
            provider.send_normalized(USER_MSG)
            raise AssertionError("expected AuthenticationError")
        except Exception as exc:
            from churro.providers.provider import AuthenticationError
            assert isinstance(exc, AuthenticationError)

        provider = make_provider(responses=[FakeConnectionError("boom")])
        try:
            provider.send_normalized(USER_MSG)
            raise AssertionError("expected APIRequestError")
        except APIRequestError:
            pass

        provider = make_provider(responses=[FakePlainError("500")])
        try:
            provider.send_normalized(USER_MSG)
            raise AssertionError("expected APIRequestError")
        except APIRequestError:
            pass
    finally:
        openai_module._OpenAIAuthenticationError = original_auth
        openai_module.APIConnectionError = original_conn
        openai_module.OpenAIError = original_err


class EchoProvider(Provider):
    name = "echo"
    model = "echo-1"

    def send(self, messages):
        return "echo:" + messages[-1]["content"]


def test_base_provider_default_normalization_for_backwards_compat():
    provider = EchoProvider()
    resp = provider.send_normalized(
        [{"role": "user", "content": "hi"}],
    )
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "echo:hi"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"

    # tools are tolerated and ignored by the text-only default
    resp2 = provider.send_normalized(
        [{"role": "user", "content": "hi"}],
        tools=[ToolDefinition(name="x", description="d", input_schema={})],
    )
    assert resp2.text == "echo:hi"


def test_usage_is_none_when_absent():
    provider = make_provider(responses=[FakeCompletion("no usage", usage=None)])
    resp = provider.send_normalized(USER_MSG)
    assert resp.usage is None


TEST_FUNCTIONS = [
    test_text_only_provider_response,
    test_one_normalized_tool_call,
    test_multiple_normalized_tool_calls,
    test_tool_call_arguments_are_structured_not_string,
    test_malformed_tool_arguments_controlled_error,
    test_non_object_tool_arguments_controlled_error,
    test_non_string_tool_arguments_controlled_error,
    test_missing_tool_call_id_raises,
    test_missing_tool_call_function_raises,
    test_response_text_preserved_with_tool_call,
    test_no_tool_calls_when_none_supplied,
    test_finish_reason_normalization_aliases,
    test_finish_reason_normalized_from_provider,
    test_tooldefinition_contains_fields,
    test_tooldefinition_from_existing_tool_reuses_input_schema,
    test_openai_tool_format_sent_to_sdk,
    test_provider_send_still_returns_plain_text,
    test_core_consumes_normalized_without_openai_types,
    test_existing_error_handling_continues_in_send_normalized,
    test_base_provider_default_normalization_for_backwards_compat,
    test_usage_is_none_when_absent,
]


def main():
    failed = 0
    for func in TEST_FUNCTIONS:
        try:
            func()
            print(f"PASS {func.__name__}")
        except Exception:
            import traceback

            failed += 1
            print(f"FAIL {func.__name__}")
            traceback.print_exc()
    total = len(TEST_FUNCTIONS)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())