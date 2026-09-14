"""Offline tests for normalized tool-result messages (Step 11).

Converts ToolExecutionResult / ToolExecutionBatch into the provider-neutral
ToolResultMessage. Uses fake tools and fake ToolResults only -- no API key,
no network, no provider SDK.
"""

import tempfile
from pathlib import Path

import churro.tools.result_messages as result_messages_module
from churro.providers.responses import ProviderResponse, ToolCall, ToolResultMessage
from churro.tools import (
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolArgs,
)
from churro.tools.executor import ToolExecutionBatch, ToolExecutionResult
from churro.tools.result_messages import (
    _FAILURE_OUTPUT_LIMIT_CHARS,
    result_content,
    result_message_from_execution,
    result_messages_from_batch,
)
from churro.tools.tool import Tool


# ---------------------------------------------------------------- fakes
class AdderArgs(ToolArgs):
    a: int
    b: int


class AdderTool(Tool):
    name = "adder"
    description = "adds two integers"
    args_model = AdderArgs

    def _run(self, args: AdderArgs) -> ToolResult:
        return ToolResult(success=True, output=str(args.a + args.b))


class FailAlwaysTool(Tool):
    name = "fail_always"
    description = "always returns a failed result"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=False, error="boom")


class EmptyOutputTool(Tool):
    name = "empty_output"
    description = "succeeds with no output"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=True, output="")


class EmptyErrorTool(Tool):
    name = "empty_error"
    description = "fails with no error text"
    args_model = ToolArgs

    def _run(self, args) -> ToolResult:
        return ToolResult(success=False, error="")


ROOT_DIR = Path(tempfile.mkdtemp())


def execution(call_id, tool_name, result):
    return ToolExecutionResult(call_id=call_id, tool_name=tool_name, result=result)


def ok(call_id, tool_name, output):
    return execution(call_id, tool_name, ToolResult(success=True, output=output))


def bad(call_id, tool_name, error):
    return execution(call_id, tool_name, ToolResult(success=False, error=error))


def run_tools(tool_calls, *tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    response = ProviderResponse(tool_calls=tool_calls, finish_reason="tool_calls")
    return ToolExecutor(registry).execute(response)


# -------------------------------------------------------------- tests
def test_successful_execution_to_message():
    message = result_message_from_execution(ok("c1", "adder", "5"))
    assert isinstance(message, ToolResultMessage)
    assert message.success is True
    assert message.content == "5"


def test_failed_execution_to_message():
    message = result_message_from_execution(bad("c1", "fail_always", "boom"))
    assert isinstance(message, ToolResultMessage)
    assert message.success is False
    assert message.content == "boom"


def test_call_id_preserved():
    message = result_message_from_execution(ok("call-xyz", "adder", "1"))
    assert message.call_id == "call-xyz"


def test_tool_name_preserved():
    message = result_message_from_execution(ok("c1", "adder", "1"))
    assert message.tool_name == "adder"


def test_success_output_preserved():
    message = result_message_from_execution(
        ok("c1", "search_files", "churro/main.py:10:match")
    )
    assert message.content == "churro/main.py:10:match"
    assert message.success is True


def test_error_information_preserved():
    message = result_message_from_execution(
        bad("c1", "read_file", "File not found: nope.py")
    )
    assert message.content == "File not found: nope.py"
    assert message.success is False


def test_multiple_execution_results_preserve_ordering():
    batch = ToolExecutionBatch(
        executions=[
            ok("c1", "adder", "first"),
            ok("c2", "adder", "second"),
            ok("c3", "adder", "third"),
        ]
    )
    messages = result_messages_from_batch(batch)
    assert [m.content for m in messages] == ["first", "second", "third"]
    assert [m.call_id for m in messages] == ["c1", "c2", "c3"]


def test_empty_execution_batch():
    messages = result_messages_from_batch(ToolExecutionBatch())
    assert messages == []


def test_mixed_success_and_failure_results():
    batch = ToolExecutionBatch(
        executions=[
            ok("c1", "adder", "3"),
            bad("c2", "fail_always", "boom"),
            ok("c3", "adder", "7"),
        ]
    )
    messages = result_messages_from_batch(batch)
    assert [m.success for m in messages] == [True, False, True]
    assert [m.content for m in messages] == ["3", "boom", "7"]


def test_original_execution_result_not_mutated():
    original = ok("c1", "adder", "5")
    snapshot = original.model_copy(deep=True)
    result_message_from_execution(original)
    assert original == snapshot
    assert original.result.output == "5"


def test_original_batch_not_mutated():
    batch = ToolExecutionBatch(
        executions=[ok("c1", "adder", "1"), bad("c2", "fail_always", "boom")]
    )
    snapshot = batch.model_copy(deep=True)
    result_messages_from_batch(batch)
    assert batch == snapshot


def test_no_provider_sdk_imports():
    assert "openai" not in vars(result_messages_module)
    assert not hasattr(result_messages_module, "OpenAI")
    assert "anthropic" not in vars(result_messages_module)
    assert "gemini" not in vars(result_messages_module)
    assert "ollama" not in vars(result_messages_module)


def test_conversion_works_with_fake_tools():
    batch = run_tools(
        [
            ToolCall(id="c1", name="adder", arguments={"a": 2, "b": 3}),
            ToolCall(id="c2", name="fail_always", arguments={}),
        ],
        AdderTool(),
        FailAlwaysTool(),
    )
    messages = result_messages_from_batch(batch)
    assert len(messages) == 2
    assert messages[0] == ToolResultMessage(
        call_id="c1", tool_name="adder", content="5", success=True
    )
    assert messages[1] == ToolResultMessage(
        call_id="c2", tool_name="fail_always", content="boom", success=False
    )


def test_result_content_is_deterministic():
    r = ToolResult(success=True, output="stable")
    assert result_content(r) == result_content(r) == "stable"
    f = ToolResult(success=False, error="stable-error")
    assert result_content(f) == result_content(f) == "stable-error"


def test_success_with_empty_output_uses_placeholder():
    message = result_message_from_execution(ok("c1", "empty_output", ""))
    assert message.success is True
    assert message.content == "(tool returned no output)"


def test_failure_with_empty_error_uses_placeholder():
    message = result_message_from_execution(bad("c1", "empty_error", ""))
    assert message.success is False
    assert message.content == "(tool failed with no error details)"


def test_multiple_different_names_and_ids():
    batch = ToolExecutionBatch(
        executions=[
            ok("id-1", "read_file", "contents"),
            bad("id-2", "write_file", "permission denied"),
            ok("id-3", "run_tests", "All passed"),
        ]
    )
    messages = result_messages_from_batch(batch)
    assert [(m.call_id, m.tool_name) for m in messages] == [
        ("id-1", "read_file"),
        ("id-2", "write_file"),
        ("id-3", "run_tests"),
    ]


def test_message_has_only_neutral_fields():
    message = result_message_from_execution(ok("c1", "adder", "5"))
    assert set(message.model_dump().keys()) == {"call_id", "tool_name", "content", "success"}


def test_empty_batch_from_real_executor():
    batch = run_tools([], AdderTool())
    assert result_messages_from_batch(batch) == []


def test_batch_conversion_through_executor_cycle():
    batch = run_tools(
        [
            ToolCall(id="c1", name="unknown_tool", arguments={}),
            ToolCall(id="c2", name="adder", arguments={"a": 1, "b": 1}),
        ],
        AdderTool(),
    )
    messages = result_messages_from_batch(batch)
    assert len(messages) == 2
    assert messages[0].success is False
    assert "unknown_tool" in messages[0].content
    assert messages[1].success is True
    assert messages[1].content == "2"


# ---- failure content combines error and captured output ----


def test_failure_with_error_and_output_contains_both():
    result = ToolResult(
        success=False,
        error="Test suite executed but tests failed: 1 of 24 file(s) exited non-zero",
        output="Failed: tests/test_x.py\nModuleNotFoundError: No module named 'x'",
    )
    content = result_content(result)
    assert "Test suite executed but tests failed" in content
    assert "tests/test_x.py" in content
    assert "ModuleNotFoundError" in content


def test_failure_with_error_appears_before_details():
    result = ToolResult(
        success=False,
        error="Command exited with code 1",
        output="error: undefined symbol foo\nbuild failed",
    )
    content = result_content(result)
    assert content.index("Command exited with code 1") < content.index("undefined symbol")
    assert content.startswith("Command exited with code 1")


def test_failure_with_error_only():
    result = ToolResult(success=False, error="Some failure", output="")
    assert result_content(result) == "Some failure"


def test_failure_with_output_only():
    result = ToolResult(success=False, error="", output="actual stderr/stdout")
    assert result_content(result) == "actual stderr/stdout"


def test_failure_with_neither_uses_placeholder():
    assert result_content(ToolResult(success=False)) == "(tool failed with no error details)"


def test_successful_result_unchanged():
    result = ToolResult(success=True, output="all good")
    assert result_content(result) == "all good"
    assert result_content(ToolResult(success=True)) == "(tool returned no output)"


def test_failure_content_is_deterministic_and_ordered():
    result = ToolResult(
        success=False,
        error="boom",
        output="first\ndetails",
    )
    first = result_content(result)
    second = result_content(result)
    assert first == second
    assert first == "boom\n\nfirst\ndetails"


def test_failure_output_not_duplicated_when_identical_to_error():
    result = ToolResult(success=False, error="same text", output="same text")
    assert result_content(result) == "same text"


def test_large_failure_output_respects_limit():
    big = "x" * (_FAILURE_OUTPUT_LIMIT_CHARS + 500)
    result = ToolResult(
        success=False,
        error="too much output",
        output=big,
    )
    content = result_content(result)
    expected_marker = (
        f"\n... [output truncated at {_FAILURE_OUTPUT_LIMIT_CHARS} characters]"
    )
    assert expected_marker in content
    details = content[content.index("\n\n") + 2:]
    assert details.startswith("x" * _FAILURE_OUTPUT_LIMIT_CHARS)
    assert len(details) <= _FAILURE_OUTPUT_LIMIT_CHARS + len(expected_marker)


def test_tool_result_not_mutated():
    result = ToolResult(
        success=False,
        error="boom",
        output="  details\n",
    )
    snapshot = result.model_copy(deep=True)
    result_content(result)
    assert result == snapshot
    assert result.output == "  details\n"
    assert result.error == "boom"


TEST_FUNCTIONS = [
    test_successful_execution_to_message,
    test_failed_execution_to_message,
    test_call_id_preserved,
    test_tool_name_preserved,
    test_success_output_preserved,
    test_error_information_preserved,
    test_multiple_execution_results_preserve_ordering,
    test_empty_execution_batch,
    test_mixed_success_and_failure_results,
    test_original_execution_result_not_mutated,
    test_original_batch_not_mutated,
    test_no_provider_sdk_imports,
    test_conversion_works_with_fake_tools,
    test_result_content_is_deterministic,
    test_success_with_empty_output_uses_placeholder,
    test_failure_with_empty_error_uses_placeholder,
    test_multiple_different_names_and_ids,
    test_message_has_only_neutral_fields,
    test_empty_batch_from_real_executor,
    test_batch_conversion_through_executor_cycle,
    test_failure_with_error_and_output_contains_both,
    test_failure_with_error_appears_before_details,
    test_failure_with_error_only,
    test_failure_with_output_only,
    test_failure_with_neither_uses_placeholder,
    test_successful_result_unchanged,
    test_failure_content_is_deterministic_and_ordered,
    test_failure_output_not_duplicated_when_identical_to_error,
    test_large_failure_output_respects_limit,
    test_tool_result_not_mutated,
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