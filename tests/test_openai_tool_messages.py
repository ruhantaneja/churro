"""Offline tests for OpenAI native tool-result formatting (Step 13).

The OpenAI provider adapter translates normalized ChatMessage /
ToolCall / ToolResultMessage into native OpenAI conversation messages.
Tests use the in-memory FakeClient and fake completions only -- no API
key, no network, no real SDK calls.
"""

from pathlib import Path

import churro.agent.runner as runner_module
import churro.providers.responses as responses_module
import churro.tools.result_messages as result_messages_module
import churro.tools.executor as executor_module
from churro.providers.openai_provider import OpenAIProvider
from churro.providers.responses import (
    ProviderResponse,
    ToolCall,
    ToolResultMessage,
)
from churro.tools import ToolArgs, ToolRegistry, ToolResult
from churro.tools.tool import Tool


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


class FakeCompletion:
    def __init__(self, content="", tool_calls=None, finish_reason="stop"):
        self.choices = [FakeChoice(content, tool_calls, finish_reason)]
        self.usage = None


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


def make_provider(responses=None):
    return OpenAIProvider(client=FakeClient(responses=responses))


def call(call_id, name="read_file", arguments=None):
    return ToolCall(id=call_id, name=name, arguments=arguments or {})


def result(call_id, content, success=True, name="read_file"):
    return ToolResultMessage(call_id=call_id, tool_name=name, content=content,
                             success=success)


def send(messages, tool_results=None, responses=None):
    provider = make_provider(responses)
    provider.send_normalized(messages, tool_results=tool_results)
    return provider._client.calls[-1]["messages"]


# -------------------------------------------------------------- tests
def test_normal_user_message_formatting():
    native = send([{"role": "user", "content": "hi"}])
    assert native == [{"role": "user", "content": "hi"}]


def test_normal_assistant_message_formatting():
    native = send([{"role": "assistant", "content": "thinking..."}])
    assert native == [{"role": "assistant", "content": "thinking..."}]


def test_normal_system_message_formatting():
    native = send([{"role": "system", "content": "be helpful"}])
    assert native == [{"role": "system", "content": "be helpful"}]


def test_assistant_tool_call_message_formatting():
    native = send(
        [{"role": "assistant", "content": "", "tool_calls": [call("call_1")]}],
    )
    assert native == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }
    ]


def test_tool_call_arguments_serialized_to_json_string():
    native = send(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call("c1", "search_files", {"pattern": "*.py", "recursive": True})
                ],
            }
        ],
    )
    tool_calls = native[0]["tool_calls"]
    assert tool_calls[0]["function"]["arguments"] == '{"pattern": "*.py", "recursive": true}'


def test_empty_arguments_serialized_to_empty_object():
    native = send(
        [{"role": "assistant", "content": "", "tool_calls": [call("c1")]}],
    )
    assert native[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_one_tool_result_formatting():
    native = send(
        [{"role": "assistant", "content": "", "tool_calls": [call("call_1")]}],
        tool_results=[result("call_1", "file contents here")],
    )
    assert native[1] == {"role": "tool", "tool_call_id": "call_1",
                         "content": "file contents here"}


def test_multiple_tool_results_ordering():
    native = send(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("a"), call("b"), call("c", "no_arg")],
            }
        ],
        tool_results=[result("a", "r-a"), result("b", "r-b"), result("c", "r-c", name="no_arg")],
    )
    tool_messages = [m for m in native if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b", "c"]
    assert [m["content"] for m in tool_messages] == ["r-a", "r-b", "r-c"]


def test_tool_results_attached_to_correct_call_ids():
    # results arrive out of order; they must still attach to the right call
    native = send(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("a"), call("b")],
            }
        ],
        tool_results=[result("b", "content-b"), result("a", "content-a")],
    )
    tool_messages = [m for m in native if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert tool_messages[0]["content"] == "content-a"
    assert tool_messages[1]["content"] == "content-b"


def test_failed_tool_result_formatted_like_any_other():
    native = send(
        [{"role": "assistant", "content": "", "tool_calls": [call("call_1")]}],
        tool_results=[result("call_1", "tool failed: boom", success=False)],
    )
    tool_messages = [m for m in native if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert tool_messages[0]["content"] == "tool failed: boom"


def test_empty_content_tool_result_preserved_verbatim():
    native = send(
        [{"role": "assistant", "content": "", "tool_calls": [call("c1")]}],
        tool_results=[result("c1", "")],
    )
    tool_messages = [m for m in native if m["role"] == "tool"]
    assert tool_messages[0]["content"] == ""


def test_sequential_tool_iterations_interleaved():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "", "tool_calls": [call("a"), call("b")]},
        {"role": "assistant", "content": "", "tool_calls": [call("c")]},
    ]
    tool_results = [
        result("a", "r-a"),
        result("b", "r-b"),
        result("c", "r-c", name="no_arg"),
    ]
    native = send(messages, tool_results=tool_results)
    roles = [m["role"] for m in native]
    assert roles == ["user", "assistant", "tool", "tool", "assistant", "tool"]
    assert [m["tool_call_id"] for m in native[2:4]] == ["a", "b"]
    assert native[5]["tool_call_id"] == "c"


def test_no_accidental_duplication():
    messages = [{"role": "assistant", "content": "", "tool_calls": [call("a"), call("b")]}]
    tool_results = [result("a", "x"), result("b", "y")]
    native = send(messages, tool_results=tool_results)
    tool_messages = [m for m in native if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert len({m["tool_call_id"] for m in tool_messages}) == 2


def test_no_mutation_of_normalized_messages():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1")]},
    ]
    snapshot = [dict(m) for m in messages]
    tool_results = [result("c1", "data")]
    send(messages, tool_results=tool_results)
    assert messages[0] == snapshot[0]
    assert messages[1]["role"] == "assistant"
    assert isinstance(messages[1]["tool_calls"][0], ToolCall)
    assert messages[1]["tool_calls"][0].id == "c1"
    assert tool_results[0].content == "data"
    assert tool_results[0].call_id == "c1"


def test_no_tool_messages_when_no_results():
    native = send(
        [{"role": "assistant", "content": "", "tool_calls": [call("a")]}],
        tool_results=[],
    )
    assert all(m["role"] != "tool" for m in native)
    assert len(native) == 1


def test_leftover_tool_result_without_matching_call():
    native = send(
        [{"role": "user", "content": "los?"}],
        tool_results=[result("orphan", "stray data")],
    )
    tool_messages = [m for m in native if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "orphan"
    # the plain user message stays first
    assert native[0]["role"] == "user"


def test_text_only_messages_passthrough_in_send_normalized():
    native = send(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        tool_results=None,
    )
    assert native == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_tool_results_are_fed_back_through_send():
    # the native tool message payload reaches the wire (kwargs["messages"])
    provider = make_provider()
    provider.send_normalized(
        [{"role": "assistant", "content": "", "tool_calls": [call("k1")]}],
        tool_results=[result("k1", "payload")],
    )
    wire = provider._client.calls[0]["messages"]
    assert wire[1] == {"role": "tool", "tool_call_id": "k1", "content": "payload"}


# -------------------------------------------------- end-to-end via runner
class AdderArgs(ToolArgs):
    a: int
    b: int


class AdderTool(Tool):
    name = "adder"
    description = "adds two integers"
    args_model = AdderArgs

    def _run(self, args: AdderArgs) -> ToolResult:
        return ToolResult(success=True, output=str(args.a + args.b))


class FailTool(Tool):
    name = "fail"
    description = "always fails"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=False, error="boom")


def tool_response(call_id, name, arguments):
    return ProviderResponse(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


def test_agent_runner_with_openai_adapter_preserves_tool_turn():
    from churro.agent import AgentRunner

    registry = ToolRegistry()
    registry.register(AdderTool())
    responses = [
        FakeCompletion(content="", tool_calls=[FakeToolCall("call_1", "adder", '{"a": 2, "b": 3}')], finish_reason="tool_calls"),
        FakeCompletion(content="5", finish_reason="stop"),
    ]
    provider = make_provider(responses)
    runner = AgentRunner(provider=provider, registry=registry, max_iterations=5)
    result = runner.run(messages=[{"role": "user", "content": "add"}])
    assert result.completed is True
    assert result.final_text == "5"
    # the second wire request contained the assistant tool-call turn plus
    # the matching native tool message
    second = provider._client.calls[1]["messages"]
    roles = [m["role"] for m in second]
    assert "assistant" in roles
    assert "tool" in roles
    assistant_turn = next(m for m in second if m["role"] == "assistant" and "tool_calls" in m)
    assert assistant_turn["tool_calls"][0]["function"]["name"] == "adder"
    tool_message = next(m for m in second if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "5"


def test_agent_runner_failed_tool_is_sent_back_as_tool_message():
    from churro.agent import AgentRunner

    registry = ToolRegistry()
    registry.register(FailTool())
    responses = [
        FakeCompletion(content="", tool_calls=[FakeToolCall("call_9", "fail", "{}")], finish_reason="tool_calls"),
        FakeCompletion(content="ok", finish_reason="stop"),
    ]
    provider = make_provider(responses)
    runner = AgentRunner(provider=provider, registry=registry, max_iterations=5)
    result = runner.run(messages=[{"role": "user", "content": "go"}])
    assert result.completed is True
    second = provider._client.calls[1]["messages"]
    tool_message = next(m for m in second if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_9"
    assert "boom" in tool_message["content"]


def test_agent_runner_sequential_iterations_interleaved_natively():
    from churro.agent import AgentRunner

    registry = ToolRegistry()
    registry.register(AdderTool())
    responses = [
        FakeCompletion(content="", tool_calls=[FakeToolCall("c1", "adder", '{"a": 1, "b": 1}')], finish_reason="tool_calls"),
        FakeCompletion(content="", tool_calls=[FakeToolCall("c2", "adder", '{"a": 2, "b": 2}')], finish_reason="tool_calls"),
        FakeCompletion(content="final", finish_reason="stop"),
    ]
    provider = make_provider(responses)
    runner = AgentRunner(provider=provider, registry=registry, max_iterations=5)
    result = runner.run(messages=[{"role": "user", "content": "go"}])
    assert result.completed is True
    third = provider._client.calls[2]["messages"]
    roles = [m["role"] for m in third]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]
    assert third[2]["tool_call_id"] == "c1"
    assert third[4]["tool_call_id"] == "c2"


def test_agent_runner_reports_result_log_with_assistant_turn():
    from churro.agent import AgentRunner

    registry = ToolRegistry()
    registry.register(AdderTool())
    responses = [
        FakeCompletion(content="", tool_calls=[FakeToolCall("c1", "adder", '{"a": 1, "b": 1}')], finish_reason="tool_calls"),
        FakeCompletion(content="2", finish_reason="stop"),
    ]
    provider = make_provider(responses)
    runner = AgentRunner(provider=provider, registry=registry, max_iterations=5)
    result = runner.run(messages=[{"role": "user", "content": "go"}])
    assert result.completed is True
    assert result.messages == [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "adder", {"a": 1, "b": 1})]},
    ]


# ------------------------------------------- core isolation assertions
def test_core_modules_contain_no_openai_imports():
    for module in (runner_module, responses_module, result_messages_module, executor_module):
        assert "openai" not in vars(module)
        assert not hasattr(module, "OpenAI")


def test_openai_formatting_lives_only_in_openai_provider():
    repo = Path(__file__).resolve().parents[1]
    compat_text = (repo / "churro" / "providers" / "openai_compat.py").read_text(
        encoding="utf-8"
    )
    openai_text = (repo / "churro" / "providers" / "openai_provider.py").read_text(
        encoding="utf-8"
    )
    ollama_text = (repo / "churro" / "providers" / "ollama_provider.py").read_text(
        encoding="utf-8"
    )
    runner_text = (repo / "churro" / "agent" / "runner.py").read_text(encoding="utf-8")
    responses_text = (repo / "churro" / "providers" / "responses.py").read_text(
        encoding="utf-8"
    )
    executor_text = (repo / "churro" / "tools" / "executor.py").read_text(
        encoding="utf-8"
    )
    result_text = (repo / "churro" / "tools" / "result_messages.py").read_text(
        encoding="utf-8"
    )
    assert '"tool_call_id"' in compat_text
    assert "role\": \"tool" in compat_text
    for text in (runner_text, responses_text, executor_text, result_text):
        assert "tool_call_id" not in text
        assert "role\": \"tool" not in text
    assert "from churro.providers.openai_compat import" in openai_text
    assert "from churro.providers.openai_compat import" in ollama_text


TEST_FUNCTIONS = [
    test_normal_user_message_formatting,
    test_normal_assistant_message_formatting,
    test_normal_system_message_formatting,
    test_assistant_tool_call_message_formatting,
    test_tool_call_arguments_serialized_to_json_string,
    test_empty_arguments_serialized_to_empty_object,
    test_one_tool_result_formatting,
    test_multiple_tool_results_ordering,
    test_tool_results_attached_to_correct_call_ids,
    test_failed_tool_result_formatted_like_any_other,
    test_empty_content_tool_result_preserved_verbatim,
    test_sequential_tool_iterations_interleaved,
    test_no_accidental_duplication,
    test_no_mutation_of_normalized_messages,
    test_no_tool_messages_when_no_results,
    test_leftover_tool_result_without_matching_call,
    test_text_only_messages_passthrough_in_send_normalized,
    test_tool_results_are_fed_back_through_send,
    test_agent_runner_with_openai_adapter_preserves_tool_turn,
    test_agent_runner_failed_tool_is_sent_back_as_tool_message,
    test_agent_runner_sequential_iterations_interleaved_natively,
    test_agent_runner_reports_result_log_with_assistant_turn,
    test_core_modules_contain_no_openai_imports,
    test_openai_formatting_lives_only_in_openai_provider,
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